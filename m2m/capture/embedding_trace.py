"""Explicit source-bound default-v2 weight-only INT8 embedding support.

Only the actual lookup primitive receives quantization permission. Surrounding
module arithmetic is checked by the ordinary exact caller-DAG correspondence.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from types import FunctionType

import torch


@dataclass(frozen=True, slots=True)
class Int8WeightOnlyEmbeddingPolicy:
    """Explicit opt-in to public default-v2 per-row weight-only encoding.

    This grants neither custom module arithmetic nor source/application accuracy.
    No callbacks, state exemptions or policy fields are accepted.
    """


def _error(message):
    from m2m.capture.dynamic_linear_trace import DynamicLinearLineageError

    raise DynamicLinearLineageError(message)


def options(module):
    result = []
    for name in (
        "num_embeddings",
        "embedding_dim",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
        "training",
    ):
        value = getattr(module, name)
        if value is None:
            encoded = ("none",)
        elif type(value) is bool:
            encoded = ("bool", value)
        elif type(value) is int:
            encoded = ("int", value)
        elif type(value) is float and math.isfinite(value):
            encoded = ("float64_bits", struct.pack(">d", value).hex())
        else:
            _error("unsupported embedding module control kind")
        result.append((name, encoded))
    return tuple(result)


def admit(module, policy):
    if type(policy) is not Int8WeightOnlyEmbeddingPolicy:
        _error("explicit Int8WeightOnlyEmbeddingPolicy required")
    from m2m.capture.embedding_forward import observe

    observe(module)
    weight = module.weight
    if (
        module.training
        or module.max_norm is not None
        or module._forward_hooks
        or module._forward_pre_hooks
        or type(weight) is not torch.nn.Parameter
        or weight.dtype is not torch.float32
        or weight.requires_grad
        or module._parameters.get("weight") is not weight
        or "forward" in module.__dict__
        or getattr(module.forward, "__self__", None) is not module
        or not isinstance(getattr(module.forward, "__func__", None), FunctionType)
    ):
        _error(
            "only source-owned readonly inference FP32 embedding behavior is supported"
        )
    if (
        type(module.num_embeddings) is not int
        or type(module.embedding_dim) is not int
        or module.num_embeddings <= 0
        or module.embedding_dim <= 0
        or tuple(weight.shape) != (module.num_embeddings, module.embedding_dim)
        or type(module.scale_grad_by_freq) is not bool
        or type(module.sparse) is not bool
        or (
            module.padding_idx is not None
            and (
                type(module.padding_idx) is not int
                or not 0 <= module.padding_idx < module.num_embeddings
            )
        )
    ):
        _error("embedding controls and exact weight geometry are unsupported")
    return options(module)


def validate_site(site):
    from m2m.capture.embedding_forward import validate

    validate(
        site["module"], site["embedding_forward_contract"], wrapper=site["wrapper"]
    )
    if (
        type(site.get("embedding_policy")) is not Int8WeightOnlyEmbeddingPolicy
        or site["module"].training
        or site["module"].max_norm is not None
        or options(site["module"]) != site["embedding_options"]
    ):
        _error("source embedding controls or permission changed")
    weight = site["module"].weight
    if type(weight) is not torch.nn.Parameter:
        from torchao.quantization import Int8Tensor

        if type(weight) is not Int8Tensor or any(
            name in weight.__dict__
            for name in (
                "dequantize",
                "__tensor_flatten__",
                "__torch_dispatch__",
                "__torch_function__",
            )
        ):
            _error("embedding tensor has an unsupported owner or instance callback")


def boundary(exported, site, inside, sig, byname, rows, *, source):
    """Select one real lookup; leave all subclass pre/post arithmetic unowned."""
    if sig.outputs[0].name not in byname or sig.inputs[0].name not in byname:
        _error("embedding invocation inside a context is unsupported")
    incoming = byname[sig.inputs[0].name]
    outgoing = byname[sig.outputs[0].name]
    primitives = [
        node for node in inside if node.target is torch.ops.aten.embedding.default
    ]
    if len(primitives) != 1 or outgoing not in inside:
        _error("source embedding module needs one actual lookup and owned output")
    primitive = primitives[0]
    schema = torch.ops.aten.embedding.default._schema
    if (
        len(primitive.args) < 2
        or len(primitive.args) > len(schema.arguments)
        or primitive.args[1] is not incoming
        or set(primitive.kwargs) - {a.name for a in schema.arguments[2:]}
        or any(
            a.name in primitive.kwargs for a in schema.arguments[: len(primitive.args)]
        )
    ):
        _error("embedding schema, index operand or controls are unsupported")
    # Every supplied or default control is exact; the final caller key checks
    # source/selected equality of kinds/bits rather than Python coercion.
    for argument in schema.arguments[len(primitive.args) :]:
        if argument.name not in primitive.kwargs and not argument.has_default_value():
            _error("embedding call dropped a required schema operand")
    value = incoming.meta.get("val")
    if not isinstance(value, torch.Tensor) or value.dtype not in (
        torch.int32,
        torch.int64,
    ):
        _error("embedding indices must retain their actual integer tensor type")
    # Prove that the captured module output actually observes this lookup.
    seen = set()

    def reaches(node):
        if node is primitive:
            return True
        if not isinstance(node, torch.fx.Node) or node in seen or node not in inside:
            return False
        seen.add(node)
        return any(reaches(parent) for parent in node.all_input_nodes)

    if not reaches(outgoing):
        _error("embedding module output does not observe its lookup")
    weight = primitive.args[0]
    specs = {s.arg.name: s for s in exported.graph_signature.input_specs}
    spec = specs.get(weight.name) if isinstance(weight, torch.fx.Node) else None
    expected = site["weight"] if source else site["module"].weight
    if (
        not isinstance(weight, torch.fx.Node)
        or weight.op != "placeholder"
        or spec is None
        or exported.state_dict.get(spec.target) is not expected
    ):
        _error(
            "embedding lookup weight is not the exact source/selected module parameter"
        )
    return {
        "input": rows[incoming.name]["id"],
        "output": rows[primitive.name]["id"],
        "inside": [rows[primitive.name]["id"]],
        "weight": rows[weight.name]["id"],
        "module_output": rows[outgoing.name]["id"],
        "operator_schema": str(schema),
    }


def expected_weight(site):
    from torchao.quantization import Int8Tensor, Int8WeightOnlyConfig
    from torchao.quantization.granularity import PerRow

    config = Int8WeightOnlyConfig(set_inductor_config=False)
    if (
        config.version != 2
        or config.group_size is not None
        or config.granularity != PerRow(-1)
    ):
        _error("default-v2 per-row weight-only embedding recipe unavailable")
    validate_site(site)
    return Int8Tensor.from_hp(site["weight"], granularity=config.granularity)


def verify_field_geometry(actual, expected):
    if (
        type(actual) is not torch.Tensor
        or type(expected) is not torch.Tensor
        or actual.device != expected.device
        or actual.storage_offset() != expected.storage_offset()
        or actual.untyped_storage().nbytes() != expected.untyped_storage().nbytes()
    ):
        _error(
            "embedding weight-field geometry or storage capacity differs from public recipe"
        )


def frozen_metadata(value):
    """Closed exact type/bits grammar for default embedding tensor metadata."""
    if value is None:
        return ("none",)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", value)
    if type(value) is float:
        return ("float64_bits", struct.pack(">d", value).hex())
    if type(value) is str:
        return ("str", value)
    if isinstance(value, torch.dtype):
        return ("torch_dtype", str(value))
    if type(value) in (list, tuple):
        return (
            "list" if type(value) is list else "tuple",
            tuple(frozen_metadata(v) for v in value),
        )
    if type(value) is dict and all(type(k) is str for k in value):
        return ("dict", tuple((k, frozen_metadata(v)) for k, v in value.items()))
    _error("unsupported embedding recipe metadata kind")


def _checked_span(device, pointer, capacity):
    """An actual storage byte interval, including empty/null and size_t bounds."""
    limit = (1 << (8 * struct.calcsize("P"))) - 1
    if (
        not isinstance(device, torch.device)
        or type(pointer) is not int
        or type(capacity) is not int
        or not 0 <= pointer <= limit
        or not 0 <= capacity <= limit - pointer
        or (capacity and not pointer)
    ):
        _error("unproved embedding storage byte interval")
    return device, pointer, capacity


def _storage_span(value):
    if type(value) not in (torch.Tensor, torch.nn.Parameter):
        _error("unproved embedding storage owner")
    storage = value.untyped_storage()
    return _checked_span(value.device, storage.data_ptr(), storage.nbytes())


def _overlap(left, right):
    # Distinct StorageImpl owners (e.g. separate from_numpy views) may point
    # into the same allocation. Tensor start pointers and _cdata are insufficient.
    ld, lp, ln = left
    rd, rp, rn = right
    return ld == rd and bool(ln and rn) and max(lp, rp) < min(lp + ln, rp + rn)


def verify_source_weight_uses(source, sites, boundaries):
    nodes = {n.meta.get("_m2m_node_id"): n for n in source.graph.nodes}
    admitted = {nodes[b["output"]] for b in boundaries.values()}
    for path, site in sites.items():
        if "embedding_policy" in site and any(
            u not in admitted for u in nodes[boundaries[path]["weight"]].users
        ):
            _error("source embedding weight has an unclosed unrelated use")


def verify_weight_uses(source, selected, sites, before, after):
    """No transformed parameter can exempt an unrelated numerical state read."""
    old_nodes = {n.meta.get("_m2m_node_id"): n for n in source.graph.nodes}
    new_nodes = {n.meta.get("_m2m_node_id"): n for n in selected.graph.nodes}
    original_calls = {old_nodes[b["output"]] for b in before.values()}
    selected_calls = {new_nodes[i] for b in after.values() for i in b["inside"]}
    for path, site in sites.items():
        if "embedding_policy" not in site:
            continue
        original = old_nodes[before[path]["weight"]]
        if any(user not in original_calls for user in original.users):
            _error("source embedding weight has an unclosed unrelated use")
        selected_weight = new_nodes[after[path]["weight"]]
        for user in selected_weight.users:
            if user in selected_calls:
                continue
            if (
                user.target is torch.ops.export.access_subclass_inner_tensor.default
                and all(child in selected_calls for child in user.users)
            ):
                continue
            _error("selected embedding weight has an unclosed unrelated use")
    # Fresh embedding fields may not alias source state/examples or unrelated
    # selected state. Source weight ties across declared sites remain recorded
    # as value relationships; selected public encodings are checked separately.
    original = list(source.state_dict.values()) + list(source.constants.values())
    original += [
        v
        for v in torch.utils._pytree.tree_leaves(source.example_inputs)
        if isinstance(v, torch.Tensor)
    ]
    original += [
        v
        for v in torch.utils._pytree.tree_leaves(selected.example_inputs)
        if isinstance(v, torch.Tensor)
    ]
    forbidden = [
        _storage_span(v)
        for v in original
        if type(v) in (torch.Tensor, torch.nn.Parameter)
    ]
    for site in sites.values():
        if "embedding_policy" not in site:
            continue
        weight = site["module"].weight
        names, _ = weight.__tensor_flatten__()
        storage = [_storage_span(getattr(weight, n)) for n in names]
        if any(
            _overlap(span, other)
            for index, span in enumerate(storage)
            for other in storage[index + 1 :] + forbidden
        ):
            _error(
                "embedding selected fields alias original state, input or each other"
            )
        for value in [*selected.state_dict.values(), *selected.constants.values()]:
            if value is weight:
                continue
            flatten = getattr(value, "__tensor_flatten__", None)
            leaves = [getattr(value, n) for n in flatten()[0]] if flatten else [value]
            if any(
                isinstance(v, torch.Tensor)
                and any(_overlap(_storage_span(v), span) for span in storage)
                for v in leaves
            ):
                _error("embedding selected fields alias unrelated selected state")
