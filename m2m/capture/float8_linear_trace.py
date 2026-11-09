"""Fixed opt-in public-TorchAO finite per-row FP8 Linear correspondence.

This capability binds a recipe and guards, not numerical accuracy or target
support. It neither quantizes a module nor supplies a caller callback.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import torch

RECIPE_ID = "ao_e4m3fn_perrow_torch_maxabs_lb_1e-12_f32_finite_v1"
RECIPE_KIND = "finite_float8_linear_v1"


@dataclass(frozen=True, slots=True)
class FiniteFloat8LinearPolicy:
    """Explicit permission for this one recipe; no configurable fields."""


def _error(message):
    from m2m.capture.dynamic_linear_trace import DynamicLinearLineageError

    raise DynamicLinearLineageError(message)


def modules(model, policy, selected):
    if policy is None:
        if type(selected) is not tuple or selected:
            _error("FP8 source modules require an explicit finite FP8 capability")
        return ()
    if (
        type(policy) is not FiniteFloat8LinearPolicy
        or type(selected) is not tuple
        or not selected
    ):
        _error(
            "explicit FiniteFloat8LinearPolicy and ordinary nonempty module tuple required"
        )
    owned = {id(module): module for module in model.modules()}
    if len({id(module) for module in selected}) != len(selected) or any(
        type(module) is not torch.nn.Linear or owned.get(id(module)) is not module
        for module in selected
    ):
        _error("FP8 modules must be distinct actual owned source Linear identities")
    return selected


def guard_input(value):
    """Graph-safe equivalent of the finite-only source domain, before Linear.

    This deliberately changes guard encoding from a Python pre-hook, and is
    placed in BOTH fresh original and selected exports. It does not change the
    public AO quantizer or remove its domain requirement. Exception type is not
    claimed identical to a caller's Python hook.
    """
    if value.dtype is not torch.float32:
        _error("finite FP8 source activation must be FP32")
    torch._assert_async(
        torch.isfinite(value).all(), "finite FP8 source activation required"
    )


def validate_site(site):
    from torchao.quantization import Float8Tensor

    if type(site.get("float8_policy")) is not FiniteFloat8LinearPolicy:
        _error("finite FP8 source capability type changed")
    module = site["module"]
    if module.training or module._forward_hooks or module._forward_pre_hooks:
        _error("finite FP8 source controls or hooks changed")
    value = module.weight
    if type(value) not in (torch.nn.Parameter, Float8Tensor):
        _error("unsupported finite FP8 selected owner")
    if type(value) is Float8Tensor:
        expected_keys = {
            "qdata",
            "scale",
            "block_size",
            "mm_config",
            "act_quant_kwargs",
            "kernel_preference",
        }
        # nn.Parameter on a subclass sets this builtin marker; it cannot grant
        # any recipe permission or replace an arbitrary method.
        if "_is_param" in vars(value):
            if vars(value)["_is_param"] is not True:
                _error("unsupported FP8 parameter marker")
            expected_keys.add("_is_param")
        if "_dynamo_static_input_type" in vars(value):
            if (
                type(vars(value)["_dynamo_static_input_type"]) is not str
                or vars(value)["_dynamo_static_input_type"] != "unguarded"
            ):
                _error("unsupported FP8 export cache marker")
            expected_keys.add("_dynamo_static_input_type")
        if set(vars(value)) != expected_keys:
            _error("unsupported finite FP8 owner fields or callbacks")


def expected_weight(site):
    from torchao.float8.inference import Float8MMConfig
    from torchao.quantization import Float8Tensor
    from torchao.quantization.granularity import PerRow
    from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
    from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
        QuantizeTensorToFloat8Kwargs,
    )

    validate_site(site)
    source = site["weight"]
    if source.dtype is not torch.float32 or not bool(torch.isfinite(source).all()):
        _error("finite FP8 recipe requires original finite FP32 weights")
    config = Float8MMConfig(emulate=False, use_fast_accum=False, pad_inner_dim=False)
    activation = QuantizeTensorToFloat8Kwargs(
        torch.float8_e4m3fn,
        PerRow(),
        mm_config=config,
        hp_value_lb=1e-12,
        hp_value_ub=None,
        kernel_preference=KernelPreference.TORCH,
    )
    return Float8Tensor.from_hp(
        source,
        float8_dtype=torch.float8_e4m3fn,
        granularity=PerRow(),
        mm_config=config,
        hp_value_lb=1e-12,
        hp_value_ub=None,
        kernel_preference=KernelPreference.TORCH,
        act_quant_kwargs=activation,
    )


def frozen_metadata(value):
    """Closed type/kind/bit grammar; unknown metadata never invokes callbacks."""
    from torchao.float8.inference import Float8MMConfig
    from torchao.quantization.granularity import PerRow
    from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
    from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
        QuantizeTensorToFloat8Kwargs,
    )

    if value is None:
        return ("none",)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", value)
    if type(value) is float and math.isfinite(value):
        return ("finite_f64_bits", struct.pack(">d", value).hex())
    if type(value) is torch.dtype:
        return ("torch_dtype", str(value))
    if type(value) in (tuple, list):
        return (
            "tuple" if type(value) is tuple else "list",
            tuple(frozen_metadata(v) for v in value),
        )
    if type(value) is dict and all(type(k) is str for k in value):
        return ("dict", tuple((k, frozen_metadata(v)) for k, v in value.items()))
    if type(value) is PerRow and set(vars(value)) == {"dim"} and type(value.dim) is int:
        return ("PerRow", value.dim)
    if (
        type(value) is Float8MMConfig
        and value._fields == ("emulate", "use_fast_accum", "pad_inner_dim")
        and all(type(v) is bool for v in value)
    ):
        return ("Float8MMConfig", tuple(value))
    if type(value) is KernelPreference and value is KernelPreference.TORCH:
        return ("KernelPreference", "TORCH")
    if type(value) is QuantizeTensorToFloat8Kwargs and set(vars(value)) == {
        "float8_dtype",
        "granularity",
        "mm_config",
        "hp_value_lb",
        "hp_value_ub",
        "kernel_preference",
    }:
        return (
            "QuantizeTensorToFloat8Kwargs",
            tuple((k, frozen_metadata(v)) for k, v in vars(value).items()),
        )
    _error("unsupported finite FP8 recipe metadata kind or extra fields")


def verify_field_geometry(actual, expected):
    if (
        type(actual) is not torch.Tensor
        or type(expected) is not torch.Tensor
        or actual.device != expected.device
        or actual.storage_offset() != expected.storage_offset()
        or actual.untyped_storage().nbytes() != expected.untyped_storage().nbytes()
    ):
        _error("finite FP8 field geometry or capacity differs from public recipe")


def input_metadata_assertions(selected, sites, boundaries):
    """Prove AO's unannotated metadata check of an exact declared input.

    Current export can place this assertion outside the annotation even though
    its dtype/device/layout are precisely the existing typed input ABI. No
    assertion of values, sizes, strides or another operand is exempted.
    """
    nodes = {n.meta.get("_m2m_node_id"): n for n in selected.graph.nodes}
    result = {}
    target = torch.ops.aten._assert_tensor_metadata.default
    for path, site in sites.items():
        if "float8_policy" not in site:
            continue
        incoming = nodes[boundaries[path]["input"]]
        for node in incoming.users:
            if node.target is not target or node.meta.get("custom", {}).get(
                "m2m_dynamic_site"
            ):
                continue
            value = incoming.meta.get("val")
            if (
                node.op != "call_function"
                or node.args != (incoming,)
                or type(node.kwargs)
                is not torch.fx.immutable_collections.immutable_dict
                and type(node.kwargs) is not dict
                or set(node.kwargs) != {"dtype", "device", "layout"}
                or node.kwargs["dtype"] is not torch.float32
                or node.kwargs["layout"] is not torch.strided
                or type(node.kwargs["device"]) is not torch.device
                or not isinstance(value, torch.Tensor)
                or value.dtype is not torch.float32
                or value.layout is not torch.strided
                or node.kwargs["device"] != value.device
                or node.users
                or node.meta.get("val", "missing") is not None
            ):
                _error("unproved FP8 source-input metadata assertion")
            schema = target._schema
            if (
                schema.name != "aten::_assert_tensor_metadata"
                or schema.overload_name != ""
                or schema.returns
                or any(a.alias_info is not None for a in schema.arguments)
            ):
                _error("unproved FP8 metadata assertion schema/effects")
            result[node.meta["_m2m_node_id"]] = boundaries[path]["input"]
    return result


def verify_source_weight_uses(source, sites, boundaries):
    nodes = {n.meta.get("_m2m_node_id"): n for n in source.graph.nodes}
    calls = {nodes[b["output"]] for b in boundaries.values()}
    for path, site in sites.items():
        if "float8_policy" in site and any(
            u not in calls for u in nodes[boundaries[path]["weight"]].users
        ):
            _error("finite FP8 source weight has an unclosed unrelated use")


def verify_weight_uses(source, selected, sites, before, after):
    """The recipe may reencode only actual Linear reads, with fresh fields."""
    from torchao.quantization import Float8Tensor, Int8Tensor

    from m2m.capture.embedding_trace import _overlap, _storage_span

    verify_source_weight_uses(source, sites, before)
    nodes = {n.meta.get("_m2m_node_id"): n for n in selected.graph.nodes}
    admitted = {nodes[i] for b in after.values() for i in b["inside"]}
    originals = [
        *source.state_dict.values(),
        *source.constants.values(),
        *torch.utils._pytree.tree_leaves(source.example_inputs),
        *torch.utils._pytree.tree_leaves(selected.example_inputs),
    ]
    forbidden = [
        _storage_span(v)
        for v in originals
        if type(v) in (torch.Tensor, torch.nn.Parameter)
    ]
    for site in sites.values():
        if "float8_policy" not in site:
            continue
        weight = site["module"].weight
        specs = {spec.arg.name: spec for spec in selected.graph_signature.input_specs}
        owners = [
            node
            for node in selected.graph.nodes
            if node.op == "placeholder"
            and node.name in specs
            and selected.state_dict.get(specs[node.name].target) is weight
        ]
        if len(owners) != 1:
            _error("finite FP8 selected parameter owner is not unique")
        for user in owners[0].users:
            if user in admitted:
                continue
            if (
                user.target is torch.ops.export.access_subclass_inner_tensor.default
                and all(child in admitted for child in user.users)
            ):
                continue
            _error("finite FP8 selected weight has an unclosed unrelated use")
        names, _ = type(weight).__tensor_flatten__(weight)
        spans = [_storage_span(getattr(weight, n)) for n in names]
        if any(
            _overlap(span, other)
            for i, span in enumerate(spans)
            for other in spans[i + 1 :] + forbidden
        ):
            _error("finite FP8 fields alias source state, inputs or each other")
        for value in [*selected.state_dict.values(), *selected.constants.values()]:
            if value is weight:
                continue
            if type(value) in (torch.Tensor, torch.nn.Parameter):
                leaves = [value]
            elif type(value) in (Int8Tensor, Float8Tensor):
                fields, _ = type(value).__tensor_flatten__(value)
                leaves = [getattr(value, n) for n in fields]
            else:
                _error("unproved selected state owner for FP8 alias closure")
            if any(_overlap(_storage_span(v), span) for v in leaves for span in spans):
                _error("finite FP8 fields alias unrelated selected state")
