"""Opt-in source-bound lineage for default TorchAO v2 dynamic-int8 Linear calls.

Opaque tokens carry actual module execution context. Public export module-call
signatures, exact typed caller DAG/order, unchanged state payloads and independently
repacked original weights validate the relationship before source IDs are stamped.
This is source correspondence and one-sample export correctness, never model
accuracy, target execution or a blanket quantization permission.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import torch
from torch.fx.traceback import annotate

from m2m.capture.subclass_trace import observe
from m2m.capture.trace import snapshot_exported_program
from m2m.capture.view_trace import SourceStorageAliases


class DynamicLinearLineageError(ValueError):
    """An original/selected program relationship could not be proved."""


def _exact(a, b):
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and a.stride() == b.stride()
        and torch.equal(
            a.detach().contiguous().view(torch.uint8),
            b.detach().contiguous().view(torch.uint8),
        )
    )


def _types(row):
    return [{k: v for k, v in r.items() if k != "id"} for r in row["results"]]


def _frame(tree, nodes, mapping):
    if isinstance(tree, list):
        return [_frame(v, nodes, mapping) for v in tree]
    if isinstance(tree, dict):
        if set(tree) == {"node_id", "value_id"}:
            row = nodes[tree["node_id"]]
            indices = [
                i for i, r in enumerate(row["results"]) if r["id"] == tree["value_id"]
            ]
            if len(indices) != 1 or row["id"] not in mapping:
                raise KeyError("unbound exact operand")
            return {"source": mapping[row["id"]], "result": indices[0]}
        return {k: _frame(v, nodes, mapping) for k, v in tree.items()}
    return tree


def _key(row, nodes, mapping):
    return json.dumps(
        {
            "op": row["op"],
            "target": row["target"],
            "args": _frame(row["args"], nodes, mapping),
            "kwargs": _frame(row["kwargs"], nodes, mapping),
            "types": _types(row),
        },
        sort_keys=True,
    )


def _readonly(snapshot):
    aliases = SourceStorageAliases(snapshot)
    roots = {
        aliases.root(n)
        for n in snapshot["nodes"]
        if n["op"] in ["placeholder", "get_attr"]
    }
    for row in snapshot["nodes"]:
        if row["op"] in ["placeholder", "get_attr", "output"]:
            continue
        if row["op"] != "call_function":
            raise DynamicLinearLineageError("non-schema call")
        schema = aliases.schema(row)
        if schema is None:
            if aliases.tuple_projection_root(row) is None:
                raise DynamicLinearLineageError("unknown source effects")
            continue
        for i, arg in enumerate(schema.arguments):
            if arg.alias_info and arg.alias_info.is_write:
                parent = aliases.tensor_ref(aliases.argument(row, i, arg.name))
                root = aliases.root(parent) if parent else None
                if root is None or root in roots:
                    raise DynamicLinearLineageError("external input/state mutation")


def _instrument(model):
    candidates = []
    seen = set()
    for path, module in model.named_modules(remove_duplicate=False):
        if not isinstance(module, torch.nn.Linear):
            continue
        if id(module) in seen:
            raise DynamicLinearLineageError("aliased module paths")
        seen.add(id(module))
        if (
            type(module) is not torch.nn.Linear
            or getattr(module.forward, "__func__", None) is not torch.nn.Linear.forward
            or module._forward_hooks
            or module._forward_pre_hooks
            or module.training
            or type(module.weight) is not torch.nn.Parameter
            or module._parameters.get("weight") is not module.weight
            or module.weight.dtype is not torch.float32
            or (
                module.bias is not None
                and (
                    type(module.bias) is not torch.nn.Parameter
                    or module._parameters.get("bias") is not module.bias
                    or module.bias.dtype is not torch.float32
                )
            )
        ):
            raise DynamicLinearLineageError(
                "only ordinary inference FP32 Linear behavior is supported"
            )
        candidates.append((path, module))
    # Admission above is atomic: a rejected later module leaves earlier modules untouched.
    sites = {}
    for path, module in candidates:
        token = "site" + str(len(sites))
        tag = {"m2m_dynamic_site": token}
        original = module.forward

        def wrap(original, tag):
            def forward(input):
                with annotate(tag):
                    return original(input)

            return forward

        wrapped = wrap(original, tag)
        sites[path] = {
            "module": module,
            "wrapper": wrapped,
            "original_forward": original,
            "had_instance_forward": "forward" in module.__dict__,
            "instance_forward": module.__dict__.get("forward"),
            "token": token,
            "weight": module.weight,
            "bias": module.bias,
        }
        module.forward = wrapped
    return sites


def _boundaries(exported, sites, snapshot, *, source):
    byname = {n.name: n for n in exported.graph.nodes}
    rows = {
        n.name: r for n, r in zip(exported.graph.nodes, snapshot["nodes"], strict=True)
    }
    calls = {
        entry.fqn: entry.signature
        for entry in exported.module_call_graph
        if entry.fqn in sites
    }
    if set(calls) != set(sites):
        raise DynamicLinearLineageError("missing exact module signature")
    result = {}
    for path, site in sites.items():
        sig = calls[path]
        if sig is None or len(sig.inputs) != 1 or len(sig.outputs) != 1:
            raise DynamicLinearLineageError("non-unary module signature")
        inside = [
            n
            for n in exported.graph.nodes
            if n.op == "call_function"
            and (n.meta.get("custom") or {}).get("m2m_dynamic_site") == site["token"]
        ]
        if source and (
            len(inside) != 1 or inside[0].target is not torch.ops.aten.linear.default
        ):
            raise DynamicLinearLineageError(
                "source is not one actual plain Linear call"
            )
        output = byname[sig.outputs[0].name]
        incoming = byname[sig.inputs[0].name]
        if output not in inside:
            raise DynamicLinearLineageError("module output lacks owned token")
        result[path] = {
            "input": rows[incoming.name]["id"],
            "output": rows[output.name]["id"],
            "inside": [rows[n.name]["id"] for n in inside],
        }
        if source:
            node = inside[0]
            schema = torch.ops.aten.linear.default._schema
            if (
                len(node.args) not in (2, 3)
                or node.args[0] is not incoming
                or set(node.kwargs) - {"bias"}
                or (len(node.args) == 3 and "bias" in node.kwargs)
            ):
                raise DynamicLinearLineageError("source linear schema/caller mismatch")
            if len(node.args) == 3:
                bias = node.args[2]
            elif "bias" in node.kwargs:
                bias = node.kwargs["bias"]
            elif len(schema.arguments) == 3 and schema.arguments[2].has_default_value():
                bias = schema.arguments[2].default_value
            else:
                raise DynamicLinearLineageError(
                    "source linear bias default is unobserved"
                )
            if bias is None and site["bias"] is not None:
                raise DynamicLinearLineageError("source linear dropped an owned bias")
            specs = {s.arg.name: s for s in exported.graph_signature.input_specs}
            weight = node.args[1]
            if (
                weight.op != "placeholder"
                or exported.state_dict[specs[weight.name].target] is not site["weight"]
            ):
                raise DynamicLinearLineageError(
                    "source weight not exact module parameter"
                )
            if bias is not None and (
                bias.op != "placeholder"
                or exported.state_dict[specs[bias.name].target] is not site["bias"]
            ):
                raise DynamicLinearLineageError(
                    "source bias not exact module parameter"
                )
            result[path]["weight"] = rows[weight.name]["id"]
    return result


def _bind_pair(source, selected, sites):
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, Int8Tensor
    from torchao.quantization.granularity import PerRow
    from torchao.quantization.quant_primitives import MappingType
    from torchao.quantization.quantize_.workflows.int8.int8_tensor import (
        QuantizeTensorToInt8Kwargs,
    )

    old = snapshot_exported_program(source, stage="original")
    new = snapshot_exported_program(selected, stage="quantized")
    if len(old["nodes"]) != len(list(source.graph.nodes)) or len(new["nodes"]) != len(
        list(selected.graph.nodes)
    ):
        raise DynamicLinearLineageError("nested scope unsupported")
    _readonly(old)
    _readonly(new)
    sb = _boundaries(source, sites, old, source=True)
    db = _boundaries(selected, sites, new, source=False)
    if (
        source.range_constraints != selected.range_constraints
        or source.call_spec != selected.call_spec
    ):
        raise DynamicLinearLineageError("public input/output call contract changed")
    oldrows = {n["id"]: n for n in old["nodes"]}
    newrows = {n["id"]: n for n in new["nodes"]}
    newnodes = {
        r["id"]: n for r, n in zip(new["nodes"], selected.graph.nodes, strict=True)
    }
    qspec = {s.arg.name: s for s in selected.graph_signature.input_specs}
    mapping = {}
    claims = {}
    kind = {}
    weight_rows = {}
    site_records = []
    for path, site in sites.items():
        if site["module"].forward is not site["wrapper"]:
            raise DynamicLinearLineageError("source module callable replaced")
        weight = site["module"].weight
        config = Int8DynamicActivationInt8WeightConfig()
        if (
            type(weight) is not Int8Tensor
            or config.version != 2
            or config.granularity != PerRow(-1)
        ):
            raise DynamicLinearLineageError("unsupported actual AO recipe")
        expected = Int8Tensor.from_hp(
            site["weight"],
            PerRow(-1),
            mapping_type=MappingType.SYMMETRIC,
            act_quant_kwargs=QuantizeTensorToInt8Kwargs(
                PerRow(-1), config.act_mapping_type, config.reduce_range
            ),
            reduce_range=config.reduce_range,
        )
        names, meta = weight.__tensor_flatten__()
        en, em = expected.__tensor_flatten__()
        if (
            names != en
            or meta != em
            or any(not _exact(getattr(weight, n), getattr(expected, n)) for n in names)
        ):
            raise DynamicLinearLineageError(
                "weight encoding differs from original FP32 recipe"
            )
        for node in selected.graph.nodes:
            sp = qspec.get(node.name)
            if (
                node.op == "placeholder"
                and sp is not None
                and sp.target in selected.state_dict
                and selected.state_dict[sp.target] is weight
            ):
                weight_rows[node.meta["_m2m_node_id"]] = sb[path]["weight"]
        so, qo = sb[path]["output"], db[path]["output"]
        if _types(oldrows[so]) != _types(newrows[qo]):
            raise DynamicLinearLineageError("linear output ABI changed")
        mapping[qo] = so
        site_records.append(
            {
                "module_path": path,
                "token": site["token"],
                "source_input_id": sb[path]["input"],
                "source_output_id": so,
                "source_weight_parameter_id": sb[path]["weight"],
                "selected_input_id": db[path]["input"],
                "selected_output_id": qo,
                "selected_call_ids": list(db[path]["inside"]),
                "source_weight_sha256": _sha_tensor(site["weight"]),
                "encoded_weight_fields": {
                    name: {
                        "sha256": _sha_tensor(getattr(weight, name)),
                        "dtype": str(getattr(weight, name).dtype),
                        "shape": list(getattr(weight, name).shape),
                    }
                    for name in names
                },
                "default_v2_recipe_verified": True,
                "exact_module_boundary": True,
            }
        )
        for identity in db[path]["inside"]:
            if identity in claims:
                raise DynamicLinearLineageError("overlapping module region")
            claims[identity] = so
            kind[identity] = "declared_dynamic_linear_quantization"
    # Bind source-owned global parameters/buffers and positional public inputs.
    op = [r for r in old["nodes"] if r["op"] == "placeholder"]
    qp = [r for r in new["nodes"] if r["op"] == "placeholder"]
    ous = [r for r in op if r["input_kind"] == "USER_INPUT"]
    qus = [r for r in qp if r["input_kind"] == "USER_INPUT"]
    if len(ous) != len(qus):
        raise DynamicLinearLineageError("user input count changed")
    for position, row in enumerate(qp):
        if row["id"] in weight_rows:
            matches = [oldrows[weight_rows[row["id"]]]]
        elif row["input_kind"] == "USER_INPUT":
            matches = [ous[qus.index(row)]]
        else:
            matches = [
                r
                for r in op
                if r["input_kind"] == row["input_kind"]
                and r["input_target"] == row["input_target"]
            ]
        if len(matches) != 1 or _types(matches[0]) != _types(row):
            raise DynamicLinearLineageError("state/input ownership not unique")
        prior = matches[0]
        mapping[row["id"]] = prior["id"]
        claims[row["id"]] = prior["id"]
        kind[row["id"]] = "source_input_binding"
        if row["id"] not in weight_rows and row["input_kind"] != "USER_INPUT":
            a = {**source.state_dict, **source.constants}[prior["input_target"]]
            b = {**selected.state_dict, **selected.constants}[row["input_target"]]
            if (
                type(a) not in [torch.Tensor, torch.nn.Parameter]
                or type(b) not in [torch.Tensor, torch.nn.Parameter]
                or not _exact(a, b)
            ):
                raise DynamicLinearLineageError("unchanged state payload differs")
    # Explicit field operations are verified by the independent lifting observer.
    fields = observe(selected)
    if fields is None:
        raise DynamicLinearLineageError("field/effect observation unavailable")
    for field in fields.fields:
        node = newnodes[field.source_id]
        receiver = node.args[0].meta["_m2m_node_id"]
        if receiver not in weight_rows:
            raise DynamicLinearLineageError(
                "field receiver not an exact selected site weight"
            )
        claims[field.source_id] = weight_rows[receiver]
        kind[field.source_id] = "declared_weight_field_reencoding"
    source_sites = {b["output"] for b in sb.values()}
    pending = [
        r
        for r in old["nodes"]
        if r["op"] == "call_function" and r["id"] not in source_sites
    ]
    identity = {identity: identity for identity in oldrows}
    while pending:
        progress = False
        for row in list(pending):
            try:
                k = _key(row, oldrows, identity)
            except KeyError:
                raise DynamicLinearLineageError("original has unbound operands")
            matches = []
            for other in new["nodes"]:
                if other["op"] != "call_function" or other["id"] in claims:
                    continue
                if old["operator_schemas"].get(row["target"]) != new[
                    "operator_schemas"
                ].get(other["target"]):
                    continue
                try:
                    equal = _key(other, newrows, mapping) == k
                except KeyError:
                    equal = False
                if equal:
                    matches.append(other)
            if len(matches) == 1:
                other = matches[0]
                mapping[other["id"]] = row["id"]
                claims[other["id"]] = row["id"]
                kind[other["id"]] = "exact_unchanged_call"
                pending.remove(row)
                progress = True
        if not progress:
            raise DynamicLinearLineageError(
                "unchanged source calls not uniquely anchored"
            )
    for path in sites:
        if mapping.get(db[path]["input"]) != sb[path]["input"]:
            raise DynamicLinearLineageError("linear caller changed")
    output_old = next(r for r in old["nodes"] if r["op"] == "output")
    output_new = next(r for r in new["nodes"] if r["op"] == "output")
    if _frame(output_old["args"], oldrows, identity) != _frame(
        output_new["args"], newrows, mapping
    ):
        raise DynamicLinearLineageError("output mapping changed")
    if any(r["op"] == "call_function" and r["id"] not in claims for r in new["nodes"]):
        raise DynamicLinearLineageError("unaccounted introduced call")
    expected_order = [r["id"] for r in old["nodes"] if r["op"] == "call_function"]
    observed_order = []
    outputs = {b["output"] for b in db.values()}
    for row in new["nodes"]:
        if row["op"] == "call_function" and (
            kind.get(row["id"]) == "exact_unchanged_call" or row["id"] in outputs
        ):
            observed_order.append(claims[row["id"]])
    if observed_order != expected_order:
        raise DynamicLinearLineageError("source effects/call order changed")
    # Atomic publication: failed checks above do not stamp a single source identity.
    for node_id, source_id in claims.items():
        node = newnodes[node_id]
        custom = dict(node.meta.get("custom") or {})
        origins = list(custom.get("m2m_lineage") or ())
        if source_id not in origins:
            origins.append(source_id)
        custom.update(m2m_lineage=origins, m2m_transform=kind[node_id])
        node.meta["custom"] = custom
    return old, {
        "source_calls": len(expected_order),
        "actual_quantized_calls": sum(r["op"] == "call_function" for r in new["nodes"]),
        "sites": len(sites),
        "caller_graph_and_order_verified": True,
        "all_original_weight_encodings_verified": True,
        "site_bindings": site_records,
    }


@dataclass(frozen=True)
class _Seal:
    target: str
    value: Any
    description: tuple
    sha256: str


def _description(value):
    if (
        type(value) not in (torch.Tensor, torch.nn.Parameter)
        or value.layout != torch.strided
    ):
        raise DynamicLinearLineageError(
            "source state is not an ordinary strided tensor"
        )
    shape, strides, offset = (
        tuple(value.shape),
        tuple(value.stride()),
        value.storage_offset(),
    )
    if any(type(v) is not int for v in (*shape, *strides, offset)):
        raise DynamicLinearLineageError("source state has dynamic geometry")
    return str(value.dtype), shape, strides, offset, str(value.device)


def _sha_tensor(value):
    return hashlib.sha256(
        memoryview(value.detach().contiguous().view(torch.uint8).cpu().numpy())
    ).hexdigest()


def _outputs(model, inputs):
    with torch.no_grad():
        leaves, spec = torch.utils._pytree.tree_flatten(model(*inputs))
    if not leaves or any(not isinstance(value, torch.Tensor) for value in leaves):
        raise DynamicLinearLineageError(
            "only complete tensor-output programs are supported"
        )
    return tuple(value.detach().clone() for value in leaves), spec


@dataclass(frozen=True)
class LinearSource:
    """An observed original program; source state and graph must remain unchanged."""

    exported: Any
    snapshot: dict
    sites: dict
    seals: tuple[_Seal, ...]
    inputs: tuple


class LinearInstrumentation:
    """Own temporary metadata wrappers on one source-declared shared model.

    It never quantizes a module. All source stages must be captured before the
    caller applies its one actual transform. Close the owner after all selected
    stages have been exported; ordinary forwards remain numerically unchanged.
    """

    def __init__(self, model):
        self._sites = _instrument(model)
        if not self._sites:
            raise DynamicLinearLineageError("no source plain FP32 Linear sites")
        self._closed = False

    def _program_sites(self, model):
        if self._closed:
            raise DynamicLinearLineageError("instrumentation owner is closed")
        owned = {id(site["module"]): site for site in self._sites.values()}
        sites = {}
        seen = set()
        for path, module in model.named_modules(remove_duplicate=False):
            site = owned.get(id(module))
            if site is None:
                continue
            if id(module) in seen:
                raise DynamicLinearLineageError("aliased module paths in program")
            if module.forward is not site["wrapper"]:
                raise DynamicLinearLineageError("owned module callable changed")
            seen.add(id(module))
            sites[path] = site
        if not sites:
            raise DynamicLinearLineageError("program does not reference owned sites")
        return sites

    def capture_source(self, model, inputs) -> LinearSource:
        """Strictly export an actual FP32 source stage before quantization."""
        if not isinstance(inputs, tuple) or any(
            not isinstance(x, torch.Tensor) for x in inputs
        ):
            raise DynamicLinearLineageError("explicit tensor tuple inputs required")
        sites = self._program_sites(model)
        if any(
            site["module"].weight is not site["weight"]
            or site["module"].bias is not site["bias"]
            for site in sites.values()
        ):
            raise DynamicLinearLineageError(
                "source weights already transformed or replaced"
            )
        exported = torch.export.export(
            model, inputs, strict=True, preserve_module_call_signature=tuple(sites)
        )
        snapshot = snapshot_exported_program(exported, stage="original")
        _readonly(snapshot)
        _boundaries(exported, sites, snapshot, source=True)
        states = {**exported.state_dict, **exported.constants}
        seals = tuple(
            _Seal(target, value, _description(value), _sha_tensor(value))
            for target, value in states.items()
        )
        return LinearSource(
            exported, copy.deepcopy(snapshot), dict(sites), seals, inputs
        )

    def export_selected(self, source: LinearSource, model, inputs, *, quantization):
        """Prove an actual selected stage and return its exact exported program.

        Input values may reflect a separately bound session's quantized prefix;
        the public tensor ABI/caller graph remains exact. This method does not
        assert equal FP32/quantized input payloads or application accuracy.
        """
        if (
            quantization.scheme != "int8_dyn_act_int8_weight"
            or quantization.per_module
            or quantization.group_size is not None
            or quantization.extra_args
        ):
            raise DynamicLinearLineageError(
                "only the default explicit dynamic W8A8 recipe is supported"
            )
        if not isinstance(source, LinearSource):
            raise TypeError("an owned original LinearSource is required")
        if not isinstance(inputs, tuple) or any(
            not isinstance(x, torch.Tensor) for x in inputs
        ):
            raise DynamicLinearLineageError("explicit tensor tuple inputs required")
        sites = self._program_sites(model)
        if set(sites) != set(source.sites) or any(
            sites[path] is not source.sites[path] for path in sites
        ):
            raise DynamicLinearLineageError(
                "selected stage does not reference the same source modules"
            )
        fresh = snapshot_exported_program(source.exported, stage="original")
        if fresh != source.snapshot:
            raise DynamicLinearLineageError(
                "original source graph changed after capture"
            )
        states = {**source.exported.state_dict, **source.exported.constants}
        if set(states) != {seal.target for seal in source.seals}:
            raise DynamicLinearLineageError("original source state roster changed")
        for seal in source.seals:
            value = states[seal.target]
            if (
                value is not seal.value
                or _description(value) != seal.description
                or _sha_tensor(value) != seal.sha256
            ):
                raise DynamicLinearLineageError(
                    "original source state changed after capture"
                )
        expected, expected_tree = _outputs(model, inputs)
        selected = torch.export.export(
            model, inputs, strict=True, preserve_module_call_signature=tuple(sites)
        )
        actual, actual_tree = _outputs(selected.module(), inputs)
        if (
            actual_tree != expected_tree
            or len(expected) != len(actual)
            or any(
                not _exact(left, right)
                for left, right in zip(expected, actual, strict=True)
            )
        ):
            raise DynamicLinearLineageError(
                "selected export differs from complete eager outputs"
            )
        _old, proof = _bind_pair(source.exported, selected, sites)
        proof.update(
            source_state_sha256={seal.target: seal.sha256 for seal in source.seals},
            selected_output_sha256=[_sha_tensor(value) for value in actual],
            source_graph_sha256=source.snapshot["sha256"],
            selected_export_exact_eager_output=True,
            scope="declared v2 dynamic Linear transform; source correspondence only",
            target_execution=False,
            application_accuracy=False,
        )
        selected._m2m_trace_dynamic_linear_proof = proof
        return selected, proof

    def close(self):
        if self._closed:
            return
        foreign = []
        for site in self._sites.values():
            if site["module"].forward is not site["wrapper"]:
                foreign.append(site["token"])
                continue
            if site["had_instance_forward"]:
                site["module"].forward = site["instance_forward"]
            else:
                del site["module"].forward
        self._closed = True
        if foreign:
            raise DynamicLinearLineageError(
                "left foreign callables unchanged: " + ", ".join(foreign)
            )

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
