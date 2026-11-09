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

from m2m.capture.context_trace import context_projection, live_nodes, observe_contexts
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
            _byte_view(a),
            _byte_view(b),
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


def _readonly(snapshot, *, contexts=None, live=None):
    contexts = contexts or {}
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
        if row["id"] in contexts:
            continue
        schema = aliases.schema(row)
        if schema is None:
            if aliases.tuple_projection_root(row) is None and not (
                live is not None and context_projection(row, snapshot, contexts, live)
            ):
                raise DynamicLinearLineageError("unknown source effects")
            continue
        for i, arg in enumerate(schema.arguments):
            if arg.alias_info and arg.alias_info.is_write:
                # Context tuple selection grants no storage-freshness fact.
                # Its alias root remains unknown, so writes through it (or its
                # views) decline below. Independent proven-fresh AO temporaries
                # retain their ordinary schema-owned storage lifetimes.
                parent = aliases.tensor_ref(aliases.argument(row, i, arg.name))
                root = aliases.root(parent) if parent else None
                if root is None or root in roots:
                    raise DynamicLinearLineageError("external input/state mutation")


def _instrument(model, embedding_policy=None):
    try:
        from torch.fx.traceback import annotate
    except ImportError as exc:
        raise DynamicLinearLineageError(
            "this runtime lacks explicit Linear source annotation support"
        ) from exc
    candidates = []
    seen = set()
    for path, module in model.named_modules(remove_duplicate=False):
        embedding = embedding_policy is not None and isinstance(
            module, torch.nn.Embedding
        )
        if not isinstance(module, torch.nn.Linear) and not embedding:
            continue
        if id(module) in seen:
            raise DynamicLinearLineageError("aliased module paths")
        seen.add(id(module))
        if embedding:
            from m2m.capture import embedding_trace

            embedding_trace.admit(module, embedding_policy)
        elif (
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
            "bias": getattr(module, "bias", None),
        }
        if embedding_policy is not None and isinstance(module, torch.nn.Embedding):
            from m2m.capture import embedding_trace
            from m2m.capture.embedding_forward import observe

            sites[path].update(
                embedding_policy=embedding_policy,
                embedding_options=embedding_trace.options(module),
                embedding_forward_contract=observe(module),
            )
        module.forward = wrapped
    return sites


def _boundaries(exported, sites, snapshot, *, source):
    byname = {n.name: n for n in exported.graph.nodes}
    root_rows = [
        row
        for row in snapshot["nodes"]
        if row["graph_id"] == f"g:{snapshot['stage']}:root"
    ]
    rows = {n.name: r for n, r in zip(exported.graph.nodes, root_rows, strict=True)}
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
        if "embedding_policy" in site:
            from m2m.capture import embedding_trace

            result[path] = embedding_trace.boundary(
                exported, site, inside, sig, byname, rows, source=source
            )
            continue
        if source and (
            len(inside) != 1 or inside[0].target is not torch.ops.aten.linear.default
        ):
            raise DynamicLinearLineageError(
                "source is not one actual plain Linear call"
            )
        if sig.outputs[0].name not in byname or sig.inputs[0].name not in byname:
            raise DynamicLinearLineageError(
                "Linear invocation inside a context is unsupported"
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
    old_live, new_live = live_nodes(source, old), live_nodes(selected, new)
    old_contexts, new_contexts = (
        observe_contexts(source, old),
        observe_contexts(selected, new),
    )
    for snapshot, contexts in ((old, old_contexts), (new, new_contexts)):
        nested = {
            row["graph_id"]
            for row in snapshot["nodes"]
            if row["graph_id"] != f"g:{snapshot['stage']}:root"
        }
        if nested != {context.scope for context in contexts.values()}:
            raise DynamicLinearLineageError(
                "nested scope lacks an exact readonly context binding"
            )
    _readonly(old, contexts=old_contexts, live=old_live)
    _readonly(new, contexts=new_contexts, live=new_live)
    sb = _boundaries(source, sites, old, source=True)
    db = _boundaries(selected, sites, new, source=False)
    if (
        source.range_constraints != selected.range_constraints
        or source.call_spec != selected.call_spec
    ):
        raise DynamicLinearLineageError("public input/output call contract changed")
    oldrows = {n["id"]: n for n in old["nodes"]}
    newrows = {n["id"]: n for n in new["nodes"]}
    newnodes = new_live
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
        if "embedding_policy" in site:
            from m2m.capture import embedding_trace

            expected = embedding_trace.expected_weight(site)
        else:
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
        if "embedding_policy" in site:
            if embedding_trace.frozen_metadata(meta) != embedding_trace.frozen_metadata(
                em
            ):
                raise DynamicLinearLineageError(
                    "embedding recipe metadata kinds or bits differ"
                )
            for name in names:
                embedding_trace.verify_field_geometry(
                    getattr(weight, name), getattr(expected, name)
                )
        for node in selected.graph.nodes:
            sp = qspec.get(node.name)
            if (
                node.op == "placeholder"
                and sp is not None
                and sp.target in selected.state_dict
                and selected.state_dict[sp.target] is weight
            ):
                identity = node.meta["_m2m_node_id"]
                if (
                    identity in weight_rows
                    and weight_rows[identity] != sb[path]["weight"]
                ):
                    raise DynamicLinearLineageError(
                        "selected weight has ambiguous original owners"
                    )
                weight_rows[identity] = sb[path]["weight"]
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
            kind[identity] = (
                "declared_weight_only_embedding_quantization"
                if "embedding_policy" in site
                else "declared_dynamic_linear_quantization"
            )
        if "embedding_policy" in site:
            from m2m.capture.embedding_forward import summary

            site_records[-1].update(
                source_operator="aten.embedding.default",
                selected_recipe="default_v2_per_row_weight_only_INT8",
                original_module_post_arithmetic_collapsed=False,
                module_output_source_id=sb[path]["module_output"],
                module_output_selected_id=db[path]["module_output"],
                forward_contract=summary(site["embedding_forward_contract"]),
            )
    if any("embedding_policy" in site for site in sites.values()):
        from m2m.capture import embedding_trace

        embedding_trace.verify_weight_uses(source, selected, sites, sb, db)
    # Bind source-owned global parameters/buffers and positional public inputs.
    op = [
        r
        for r in old["nodes"]
        if r["op"] == "placeholder" and r["graph_id"] == "g:original:root"
    ]
    qp = [
        r
        for r in new["nodes"]
        if r["op"] == "placeholder" and r["graph_id"] == "g:quantized:root"
    ]
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
    # Source reencoding is distinct from post-decomposition field lifting.
    # The selected weight bytes/metadata were independently repacked above.
    for identity, node in new_live.items():
        if (
            node.op != "call_function"
            or node.target is not torch.ops.export.access_subclass_inner_tensor.default
        ):
            continue
        if (
            len(node.args) != 2
            or node.kwargs
            or not isinstance(node.args[0], torch.fx.Node)
        ):
            raise DynamicLinearLineageError("malformed actual field accessor")
        receiver = node.args[0].meta.get("_m2m_node_id")
        if receiver not in weight_rows:
            raise DynamicLinearLineageError(
                "field receiver not an exact selected site weight"
            )
        spec = qspec.get(node.args[0].name)
        wrapper = selected.state_dict.get(spec.target) if spec else None
        names, _metadata = wrapper.__tensor_flatten__()
        if node.args[1] not in names:
            raise DynamicLinearLineageError(
                "field accessor is outside the verified weight fields"
            )
        value = torch.ops.export.access_subclass_inner_tensor.default(
            wrapper, node.args[1]
        )
        observed = node.meta.get("val")
        if (
            not isinstance(observed, torch.Tensor)
            or value.dtype != observed.dtype
            or value.shape != observed.shape
            or value.stride() != observed.stride()
            or value.storage_offset() != observed.storage_offset()
            or value.device != observed.device
        ):
            raise DynamicLinearLineageError(
                "field accessor type contradicts the live value"
            )
        claims[identity] = weight_rows[receiver]
        kind[identity] = "declared_weight_field_reencoding"
    source_sites = {b["output"] for b in sb.values()}
    source_order = [
        row
        for row in old["nodes"]
        if row["op"] == "call_function" and row["graph_id"] == "g:original:root"
    ]
    selected_outputs = {boundary["output"] for boundary in db.values()}
    selected_order = [
        row
        for row in new["nodes"]
        if row["op"] == "call_function"
        and row["graph_id"] == "g:quantized:root"
        and (row["id"] not in claims or row["id"] in selected_outputs)
    ]
    if len(source_order) != len(selected_order):
        raise DynamicLinearLineageError("source caller occurrence roster changed")
    identity = {identity: identity for identity in oldrows}
    # The contract requires the complete global call order to be unchanged.
    # Collapse ONLY independently verified actual Linear regions, then bind
    # each occurrence by this total order AND its exact typed DAG operands.
    # Repeated identical calls still have separate, ordered occurrences; no
    # arbitrary candidate, node name or shape-only correspondence is selected.
    for row, other in zip(source_order, selected_order, strict=True):
        if row["id"] in source_sites:
            if (
                other["id"] not in selected_outputs
                or claims.get(other["id"]) != row["id"]
            ):
                raise DynamicLinearLineageError("Linear occurrence/call order changed")
            continue
        if other["id"] in claims:
            raise DynamicLinearLineageError("selected caller occurrence already owned")
        if row["id"] in old_contexts:
            context = old_contexts[row["id"]]
            candidate = new_contexts.get(other["id"])
            if candidate is None or (
                candidate.control != context.control
                or candidate.canonical_body != context.canonical_body
                or _types(row) != _types(other)
            ):
                raise DynamicLinearLineageError("context occurrence/body changed")
            try:
                equal = _frame(context.arguments, oldrows, identity) == _frame(
                    candidate.arguments, newrows, mapping
                )
            except KeyError:
                equal = False
            if not equal:
                raise DynamicLinearLineageError(
                    "context ordered caller operands changed"
                )
            mapping[candidate.function_id] = context.function_id
            claims[candidate.function_id] = context.function_id
            kind[candidate.function_id] = "exact_context_function_binding"
            mapping[candidate.identity] = context.identity
            claims[candidate.identity] = context.identity
            kind[candidate.identity] = "exact_readonly_context_call"
            for before, after in zip(context.body, candidate.body, strict=True):
                mapping[after["id"]] = before["id"]
                claims[after["id"]] = before["id"]
                kind[after["id"]] = "exact_readonly_context_body"
            continue
        try:
            equal = (
                other["id"] not in new_contexts
                and old["operator_schemas"].get(row["target"])
                == new["operator_schemas"].get(other["target"])
                and _key(row, oldrows, identity) == _key(other, newrows, mapping)
            )
        except KeyError:
            equal = False
        if not equal:
            raise DynamicLinearLineageError("typed caller occurrence/operands changed")
        mapping[other["id"]] = row["id"]
        claims[other["id"]] = row["id"]
        kind[other["id"]] = "exact_unchanged_call"
    for path in sites:
        if mapping.get(db[path]["input"]) != sb[path]["input"]:
            raise DynamicLinearLineageError("linear caller changed")
        if "embedding_policy" in sites[path]:
            before, after = oldrows[sb[path]["output"]], newrows[db[path]["output"]]
            if old["operator_schemas"].get(before["target"]) != new[
                "operator_schemas"
            ].get(after["target"]) or _key(before, oldrows, identity) != _key(
                after, newrows, mapping
            ):
                raise DynamicLinearLineageError(
                    "embedding schema, controls or typed index caller changed"
                )
    output_old = next(
        r
        for r in old["nodes"]
        if r["op"] == "output" and r["graph_id"] == "g:original:root"
    )
    output_new = next(
        r
        for r in new["nodes"]
        if r["op"] == "output" and r["graph_id"] == "g:quantized:root"
    )
    if _frame(output_old["args"], oldrows, identity) != _frame(
        output_new["args"], newrows, mapping
    ):
        raise DynamicLinearLineageError("output mapping changed")
    if any(r["op"] == "call_function" and r["id"] not in claims for r in new["nodes"]):
        raise DynamicLinearLineageError("unaccounted introduced call")
    expected_order = [
        r["id"]
        for r in old["nodes"]
        if r["op"] == "call_function" and r["graph_id"] == "g:original:root"
    ]
    observed_order = []
    outputs = {b["output"] for b in db.values()}
    for row in new["nodes"]:
        if (
            row["op"] == "call_function"
            and row["graph_id"] == "g:quantized:root"
            and (
                kind.get(row["id"])
                in {"exact_unchanged_call", "exact_readonly_context_call"}
                or row["id"] in outputs
            )
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
        "source_calls": sum(r["op"] == "call_function" for r in old["nodes"]),
        "verified_readonly_contexts": len(old_contexts),
        "actual_quantized_calls": sum(r["op"] == "call_function" for r in new["nodes"]),
        "sites": len(sites),
        "caller_graph_and_order_verified": True,
        "complete_root_call_occurrences_verified": len(source_order),
        "all_original_weight_encodings_verified": True,
        "site_bindings": site_records,
    }


@dataclass(frozen=True)
class _Seal:
    target: str
    value: Any
    description: tuple
    sha256: str
    storage: tuple[int, int, int]


def _storage(value):
    storage = value.untyped_storage()
    return storage._cdata, storage.data_ptr(), storage.nbytes()


def _seal(target, value):
    return _Seal(
        target, value, _description(value), _sha_tensor(value), _storage(value)
    )


def _check_seal(value, seal):
    if (
        value is not seal.value
        or _description(value) != seal.description
        or _storage(value) != seal.storage
        or _sha_tensor(value) != seal.sha256
    ):
        raise DynamicLinearLineageError(
            "sealed tensor owner/storage/metadata/bytes changed"
        )


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


def _byte_view(value):
    raw = value.detach().contiguous().reshape(-1)
    # A one-element view can be logically contiguous with a non-unit stride;
    # dtype reinterpretation still requires unit byte-addressable last stride.
    if raw.numel() and raw.stride(0) != 1:
        raw = raw.clone(memory_format=torch.contiguous_format)
    return raw.view(torch.uint8)


def _sha_tensor(value):
    return hashlib.sha256(memoryview(_byte_view(value).cpu().numpy())).hexdigest()


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
    input_seals: tuple[_Seal, ...]
    call_contract: tuple
    capabilities: tuple = ()


@dataclass(frozen=True)
class _StateBinding:
    target: str
    owner: Any
    fields: tuple[str, ...]
    metadata: Any
    leaves: tuple[_Seal, ...]
    recipe_kind: str | None = None
    recipe_metadata: tuple | None = None


@dataclass(frozen=True)
class _SelectedBinding:
    selected: Any
    source: LinearSource
    snapshot: dict
    states: tuple[_StateBinding, ...]
    input_seals: tuple[_Seal, ...]
    call_contract: tuple
    capabilities: tuple = ()


def _capabilities(sites):
    return tuple(
        (path, "weight_only_embedding_v2", site["embedding_policy"])
        for path, site in sites.items()
        if "embedding_policy" in site
    )


def _check_capabilities(source, expected):
    """Recheck the original typed capability references, never serialized permission."""
    actual = _capabilities(source.sites)
    if len(source.capabilities) != len(expected) or len(actual) != len(expected):
        raise DynamicLinearLineageError("sealed source capability roster changed")
    for rows in zip(source.capabilities, actual, expected, strict=True):
        path, kind, policy = rows[2]
        if any(row[:2] != (path, kind) or row[2] is not policy for row in rows):
            raise DynamicLinearLineageError(
                "sealed source capability reference changed"
            )
        if kind == "weight_only_embedding_v2":
            from m2m.capture.embedding_trace import (
                Int8WeightOnlyEmbeddingPolicy,
                validate_site,
            )

            if type(policy) is not Int8WeightOnlyEmbeddingPolicy:
                raise DynamicLinearLineageError(
                    "sealed embedding capability type changed"
                )
            validate_site(source.sites[path])
        else:
            raise DynamicLinearLineageError("unknown sealed source capability")


def _recipe_metadata(kind, value):
    # Fixed source-owned capabilities, never caller-supplied validation callbacks.
    if kind == "weight_only_embedding_v2":
        from m2m.capture.embedding_trace import frozen_metadata

        return frozen_metadata(value)
    raise DynamicLinearLineageError("unknown sealed source recipe capability")


def _state_bindings(exported, *, source_recipes=()):
    from torchao.quantization import Int8Tensor

    result = []
    for target, value in {**exported.state_dict, **exported.constants}.items():
        if type(value) in (torch.Tensor, torch.nn.Parameter):
            result.append(
                _StateBinding(target, value, (), None, (_seal(target, value),))
            )
        elif type(value) is Int8Tensor:
            fields, metadata = value.__tensor_flatten__()
            if not fields or any(type(name) is not str for name in fields):
                raise DynamicLinearLineageError("unknown selected state fields")
            result.append(
                _StateBinding(
                    target,
                    value,
                    tuple(fields),
                    copy.deepcopy(metadata),
                    tuple(_seal(name, getattr(value, name)) for name in fields),
                    next(
                        (kind for owner, kind in source_recipes if value is owner), None
                    ),
                    next(
                        (
                            _recipe_metadata(kind, metadata)
                            for owner, kind in source_recipes
                            if value is owner
                        ),
                        None,
                    ),
                )
            )
        else:
            raise DynamicLinearLineageError("unsupported selected state owner")
    return tuple(result)


def _call_contract(exported):
    from m2m.capture.call_contract import freeze_call_contract

    try:
        return freeze_call_contract(exported)
    except ValueError as exc:
        raise DynamicLinearLineageError(
            "unproved sealed call contract: " + str(exc)
        ) from exc


def _check_call_contract(exported, expected):
    if _call_contract(exported) != expected:
        raise DynamicLinearLineageError(
            "sealed public/callee/range call contract changed"
        )


def validate_selected_source(exported, inputs):
    """Revalidate the actual owned source/selected export at conversion entry.

    Serialized origins or a copied proof record cannot replace the original
    live owner, immutable source graph, tensor storage and exact selected input
    example. No quantization is rerun and no changed selected state is repaired.
    """
    proof = getattr(exported, "_m2m_trace_dynamic_linear_proof", None)
    binding = getattr(exported, "_m2m_dynamic_linear_binding", None)
    if proof is None and binding is None:
        return False
    if not isinstance(binding, _SelectedBinding) or binding.selected is not exported:
        raise DynamicLinearLineageError(
            "source correspondence has no owned live binding"
        )
    source = binding.source
    _check_capabilities(source, binding.capabilities)
    _check_call_contract(source.exported, source.call_contract)
    _check_call_contract(exported, binding.call_contract)
    for program, snapshot in (
        (source.exported, source.snapshot),
        (exported, binding.snapshot),
    ):
        live_nodes(
            program, snapshot
        )  # Refuse changed ordinals before any snapshot can restamp IDs.
        if snapshot_exported_program(program, stage=snapshot["stage"]) != snapshot:
            raise DynamicLinearLineageError(
                "sealed source/selected graph or proof changed"
            )
    original_states = {**source.exported.state_dict, **source.exported.constants}
    if set(original_states) != {seal.target for seal in source.seals}:
        raise DynamicLinearLineageError("sealed original state roster changed")
    for seal in source.seals:
        _check_seal(original_states[seal.target], seal)
    for value, seal in zip(source.inputs, source.input_seals, strict=True):
        _check_seal(value, seal)
    states = {**exported.state_dict, **exported.constants}
    if set(states) != {state.target for state in binding.states}:
        raise DynamicLinearLineageError("sealed selected state roster changed")
    for state in binding.states:
        owner = states[state.target]
        if owner is not state.owner:
            raise DynamicLinearLineageError("sealed selected state owner changed")
        if state.fields:
            fields, metadata = owner.__tensor_flatten__()
            if (
                tuple(fields) != state.fields
                or metadata != state.metadata
                or (
                    state.recipe_metadata is not None
                    and _recipe_metadata(state.recipe_kind, metadata)
                    != state.recipe_metadata
                )
            ):
                raise DynamicLinearLineageError(
                    "sealed selected recipe metadata changed"
                )
            values = [getattr(owner, name) for name in fields]
        else:
            values = [owner]
        for value, seal in zip(values, state.leaves, strict=True):
            _check_seal(value, seal)
    if not isinstance(inputs, tuple) or len(inputs) != len(binding.input_seals):
        raise DynamicLinearLineageError("sealed selected input roster changed")
    for value, seal in zip(inputs, binding.input_seals, strict=True):
        _check_seal(value, seal)
    _check_capabilities(source, binding.capabilities)
    return True


class LinearInstrumentation:
    """Own temporary metadata wrappers on one source-declared shared model.

    It never quantizes a module. All source stages must be captured before the
    caller applies its one actual transform. Close the owner after all selected
    stages have been exported; ordinary forwards remain numerically unchanged.
    """

    def __init__(self, model, *, embedding_policy=None):
        if embedding_policy is not None:
            from m2m.capture.embedding_trace import Int8WeightOnlyEmbeddingPolicy

            if type(embedding_policy) is not Int8WeightOnlyEmbeddingPolicy:
                raise DynamicLinearLineageError(
                    "explicit Int8WeightOnlyEmbeddingPolicy required"
                )
        self._embedding_policy = embedding_policy
        self._sites = _instrument(model, embedding_policy)
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
            if "embedding_policy" in site:
                from m2m.capture import embedding_trace

                if site["embedding_policy"] is not self._embedding_policy:
                    raise DynamicLinearLineageError(
                        "owned embedding capability reference changed"
                    )
                embedding_trace.validate_site(site)
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
        capabilities = _capabilities(sites)
        if any(
            site["module"].weight is not site["weight"]
            or getattr(site["module"], "bias", None) is not site["bias"]
            for site in sites.values()
        ):
            raise DynamicLinearLineageError(
                "source weights already transformed or replaced"
            )
        exported = torch.export.export(
            model, inputs, strict=True, preserve_module_call_signature=tuple(sites)
        )
        snapshot = snapshot_exported_program(exported, stage="original")
        contexts = observe_contexts(exported, snapshot)
        nested = {
            row["graph_id"]
            for row in snapshot["nodes"]
            if row["graph_id"] != "g:original:root"
        }
        if nested != {context.scope for context in contexts.values()}:
            raise DynamicLinearLineageError(
                "source nested scope lacks a readonly context proof"
            )
        _readonly(snapshot, contexts=contexts, live=live_nodes(exported, snapshot))
        boundaries = _boundaries(exported, sites, snapshot, source=True)
        if any("embedding_policy" in site for site in sites.values()):
            from m2m.capture import embedding_trace

            embedding_trace.verify_source_weight_uses(exported, sites, boundaries)
        states = {**exported.state_dict, **exported.constants}
        seals = tuple(_seal(target, value) for target, value in states.items())
        source = LinearSource(
            exported,
            copy.deepcopy(snapshot),
            dict(sites),
            seals,
            inputs,
            tuple(_seal(str(index), value) for index, value in enumerate(inputs)),
            _call_contract(exported),
            capabilities,
        )
        _check_capabilities(source, capabilities)
        return source

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
        capabilities = source.capabilities
        _check_capabilities(source, capabilities)
        if not isinstance(inputs, tuple) or any(
            not isinstance(x, torch.Tensor) for x in inputs
        ):
            raise DynamicLinearLineageError("explicit tensor tuple inputs required")
        _check_call_contract(source.exported, source.call_contract)
        sites = self._program_sites(model)
        if set(sites) != set(source.sites) or any(
            sites[path] is not source.sites[path] for path in sites
        ):
            raise DynamicLinearLineageError(
                "selected stage does not reference the same source modules"
            )
        try:
            live_nodes(source.exported, source.snapshot)
        except ValueError as exc:
            raise DynamicLinearLineageError(
                "original source graph changed after capture"
            ) from exc
        fresh = snapshot_exported_program(source.exported, stage="original")
        if fresh != source.snapshot:
            raise DynamicLinearLineageError(
                "original source graph changed after capture"
            )
        states = {**source.exported.state_dict, **source.exported.constants}
        if set(states) != {seal.target for seal in source.seals}:
            raise DynamicLinearLineageError("original source state roster changed")
        for seal in source.seals:
            _check_seal(states[seal.target], seal)
        for value, seal in zip(source.inputs, source.input_seals, strict=True):
            _check_seal(value, seal)
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
            selected_input_sha256=[_sha_tensor(value) for value in inputs],
            source_graph_sha256=source.snapshot["sha256"],
            selected_export_exact_eager_output=True,
            scope="declared v2 dynamic Linear transform; source correspondence only",
            target_execution=False,
            application_accuracy=False,
        )
        embedding_count = sum("embedding_policy" in site for site in sites.values())
        if embedding_count:
            proof.update(
                embedding_weight_only_sites=embedding_count,
                scope="declared default-v2 dynamic Linear and per-row weight-only embedding source correspondence only",
                embedding_module_post_arithmetic_unchanged=True,
            )
        selected._m2m_trace_dynamic_linear_proof = proof
        selected._m2m_dynamic_linear_binding = _SelectedBinding(
            selected,
            source,
            copy.deepcopy(snapshot_exported_program(selected, stage="quantized")),
            _state_bindings(
                selected,
                source_recipes=tuple(
                    (site["module"].weight, "weight_only_embedding_v2")
                    for site in sites.values()
                    if "embedding_policy" in site
                ),
            ),
            tuple(_seal(str(index), value) for index, value in enumerate(inputs)),
            _call_contract(selected),
            capabilities,
        )
        validate_selected_source(selected, inputs)
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
