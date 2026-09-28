"""Serializable static frontend graphs and exact final-MLIR correspondence.

IDs describe captured call sites, never runtime invocation counts. Correspondence
uses metadata carried by the transform, not names, module paths, or op-family guesses.
Tensor contents deliberately remain outside this structural record.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _literal(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        import math
        return value if math.isfinite(value) else {"float": str(value)}
    if isinstance(value, (tuple, list)):
        return [_literal(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _literal(v) for k, v in value.items()}
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        return {"kind": "tensor", "shape": [str(s) for s in value.shape],
                "dtype": str(value.dtype).removeprefix("torch."), "data": "external"}
    return {"kind": type(value).__name__, "value": str(value)}


def _types(value: Any, node_id: str) -> list[dict[str, Any]]:
    values = list(value) if isinstance(value, (tuple, list)) else [value]
    result = []
    for index, val in enumerate(values):
        dtype = getattr(val, "dtype", None)
        shape = getattr(val, "shape", None)
        result.append({"id": f"{node_id}:v{index}", "kind": "tensor" if dtype else
                       "unknown" if val is None else type(val).__name__,
                       "shape": [int(s) if isinstance(s, int) else str(s) for s in shape]
                       if shape is not None else None,
                       "dtype": str(dtype).removeprefix("torch.") if dtype else None,
                       "storage_dtype": str(dtype).removeprefix("torch.") if dtype else None,
                       "compute_dtype": None,
                       "device": str(getattr(val, "device", "")) or None,
                       "layout": str(getattr(val, "layout", "")) or None,
                       "stride": [int(s) if isinstance(s, int) else str(s) for s in val.stride()]
                       if dtype is not None and getattr(val, "layout", None) is not None
                       and str(val.layout) == "torch.strided" else None})
    return result


def _graph_modules(gm: Any):
    """Include GraphModule bodies in graph-scoped identities, without double counting."""
    yield "root", gm
    for name, sub in gm.named_modules():
        if name and hasattr(sub, "graph"):
            yield name, sub


def snapshot_exported_program(exported: Any, *, stage: str) -> dict[str, Any]:
    """Snapshot and stamp exactly this program, for the subsequent lowering."""
    from torch.fx import Node
    gm = exported.graph_module
    sig = getattr(exported, "graph_signature", None)
    input_specs = {getattr(s.arg, "name", None): s for s in getattr(sig, "input_specs", ())}
    records, edges = [], []
    for scope, module in _graph_modules(gm):
        graph_id = f"g:{stage}:{scope}"
        ids = {n: f"{graph_id}:n{i}" for i, n in enumerate(module.graph.nodes)}
        for ordinal, node in enumerate(module.graph.nodes):
            node_id = ids[node]
            custom = dict(node.meta.get("custom") or {})
            prior = list(custom.get("m2m_lineage") or ())
            if node_id not in prior:
                prior.append(node_id)
            custom.update(m2m_node_id=node_id, m2m_lineage=prior)
            node.meta["custom"] = custom
            node.meta["_m2m_node_id"] = node_id
            node.meta["_m2m_origin_ids"] = [v for v in prior if v != node_id]

            def encode(arg: Any, path: str):
                if isinstance(arg, Node):
                    pid = ids.get(arg)
                    if pid is None:
                        raise ValueError("frontend edge crosses an unrecorded graph scope")
                    results = _types(arg.meta.get("val"), pid)
                    # Tuple producers are not flattened into a fabricated tensor edge.
                    typ = results[0] if len(results) == 1 else {}
                    if len(results) > 1 and path == "args/0":
                        import operator
                        if (node.target is operator.getitem and len(node.args) > 1
                                and isinstance(node.args[1], int)
                                and -len(results) <= node.args[1] < len(results)):
                            typ = results[node.args[1]]
                    edges.append({"producer_node_id": pid,
                                  "producer_value_id": typ.get("id"),
                                  "consumer_node_id": node_id, "argument_path": path,
                                  "shape": typ.get("shape"), "dtype": typ.get("dtype"),
                                  "value_kind": "tuple" if not typ and len(results) > 1 else typ.get("kind")})
                    return {"node_id": pid, "value_id": typ.get("id")}
                if isinstance(arg, (tuple, list)):
                    return [encode(a, f"{path}/{i}") for i, a in enumerate(arg)]
                if isinstance(arg, dict):
                    return {str(k): encode(v, f"{path}/{k}") for k, v in arg.items()}
                return _literal(arg)

            target = str(node.target)
            classification = ("aten" if target.startswith("aten.") else
                              "custom" if node.op == "call_function" else
                              "call" if node.op in {"call_method", "call_module"} else "structural")
            spec = input_specs.get(node.name) if scope == "root" else None
            records.append({"id": node_id, "graph_id": graph_id, "ordinal": ordinal,
                            "op": node.op, "target": target, "classification": classification,
                            "args": encode(node.args, "args"),
                            "kwargs": encode(node.kwargs, "kwargs"),
                            "results": _types(node.meta.get("val"), node_id),
                            "module_stack": _literal(node.meta.get("nn_module_stack") or {}),
                            "input_kind": str(getattr(spec, "kind", "")).removeprefix("InputKind.") or None,
                            "input_target": getattr(spec, "target", None),
                            "transformation": custom.get("m2m_transform"),
                            "origin_node_ids": [v for v in prior if v != node_id]})
    body = {"schema": "m2m.frontend_graph.v1", "stage": stage, "status": "complete",
            "graph_id": f"g:{stage}:root", "nodes": records, "edges": edges,
            "call_count": sum(n["op"] in {"call_function", "call_method", "call_module"}
                              for n in records),
            "counting_unit": "static_captured_call_sites",
            "by_target": dict(sorted(Counter(n["target"] for n in records
                              if n["op"] in {"call_function", "call_method", "call_module"}).items())),
            "graph_signature": str(sig),
            "range_constraints": {str(k): str(v) for k, v in
                                  (getattr(exported, "range_constraints", {}) or {}).items()},
            "runtime_versions": {"torch": __import__("torch").__version__}}
    body["eliminations"] = list(getattr(exported, "_m2m_trace_eliminations", ()))
    body["sha256"] = _digest(body)
    return body


def capture_frontend_snapshot(model: Any, inputs: Any = (), *, stage: str = "original") -> dict[str, Any]:
    """Record a pre-mutation frontend graph. Capture failures are explicit diagnostics.

    Pass an ExportedProgram to snapshot an existing capture without exporting again.
    This helper never applies quantization or decompositions.
    """
    try:
        from m2m.capture.torch_export import capture_model
        exported = model if hasattr(model, "graph_signature") and hasattr(model, "graph_module") \
            else capture_model(model, tuple(inputs))
        return snapshot_exported_program(exported, stage=stage)
    except Exception as exc:
        return {"schema": "m2m.frontend_graph.v1", "stage": stage,
                "status": "unavailable", "reason": str(exc), "call_count": None,
                "counting_unit": "static_captured_call_sites"}


def materialize_frontend_precision(model: Any, inputs: Any = (), *, dtype: Any,
                                  original_frontend_snapshot: dict[str, Any] | None = None):
    """Cast an owned materialization of a captured static program with exact origins.

    Returns ``(graph_module, inputs_tuple, original_snapshot, receipt)``. This is
    precision conversion of the already-captured program, not a claim that
    dtype-dependent Python branches would recapture identically. No graph-name
    matching, source-model mutation, decomposition or quantization takes place.
    """
    import torch
    from torch.utils._pytree import tree_flatten, tree_map

    if dtype not in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
        raise ValueError("precision materialization supports only native f16/bf16/f32/f64")
    exported = model if hasattr(model, "graph_signature") and hasattr(model, "graph_module") \
        else torch.export.export(model, tuple(inputs))
    captured = snapshot_exported_program(exported, stage="original")
    original = original_frontend_snapshot if original_frontend_snapshot is not None else captured
    if not attach_original_identity(exported, original, captured):
        raise ValueError("precision conversion requires the exact original captured graph")
    gm = exported.module()
    source_ids = {n["id"] for n in original["nodes"]}
    for _, module in _graph_modules(gm):
        for node in module.graph.nodes:
            custom = dict(node.meta.get("custom") or {})
            if source_ids.intersection(custom.get("m2m_lineage") or ()):
                custom["m2m_transform"] = "captured_precision_conversion"
                node.meta["custom"] = custom
    conversions = []
    # Export's materialization can share Parameter objects with the source model.
    # Register fresh owned Parameters/buffers instead of .to(), which may update
    # an aliased Parameter's data. Convert each distinct object exactly once.
    memo = {}
    for parameter, values in ((True, list(gm.named_parameters(remove_duplicate=False))),
                              (False, list(gm.named_buffers(remove_duplicate=False)))):
        for name, value in values:
            converted = memo.get(id(value))
            if converted is None:
                converted = value.detach().to(dtype=dtype if value.is_floating_point() else value.dtype, copy=True)
                if parameter:
                    converted = torch.nn.Parameter(converted, requires_grad=value.requires_grad)
                memo[id(value)] = converted
            parent_name, _, local_name = name.rpartition(".")
            owner = gm.get_submodule(parent_name) if parent_name else gm
            if parameter:
                owner.register_parameter(local_name, converted)
            else:
                owner.register_buffer(local_name, converted)
            if value.is_floating_point():
                conversions.append({"state_target": name, "from_dtype": str(value.dtype),
                                    "to_dtype": str(dtype), "shape": list(value.shape)})
    input_conversions = []
    for ordinal, value in enumerate(tree_flatten(tuple(inputs))[0]):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            input_conversions.append({"leaf_ordinal": ordinal, "from_dtype": str(value.dtype),
                                      "to_dtype": str(dtype), "shape": list(value.shape)})
    cast_inputs = tree_map(lambda v: v.to(dtype=dtype) if isinstance(v, torch.Tensor)
                          and v.is_floating_point() else v, tuple(inputs))
    from torch.ao.quantization import allow_exported_model_train_eval
    allow_exported_model_train_eval(gm)
    receipt = {"schema": "m2m.frontend-precision-conversion.v1",
               "scope": "precision conversion of captured static program",
               "original_graph_sha256": original["sha256"], "target_dtype": str(dtype),
               "state_conversions": conversions, "input_conversions": input_conversions}
    receipt["sha256"] = _digest(receipt)
    return gm, cast_inputs, original, receipt


def _structural_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Stage-independent exact graph equivalence, not node-name/FQN matching."""
    rows = []
    ids = {n["id"]: i for i, n in enumerate(snapshot.get("nodes", ()))}
    def remap(value):
        if isinstance(value, list):
            return [remap(v) for v in value]
        if isinstance(value, dict):
            if "node_id" in value:
                return {"ordinal": ids[value["node_id"]]}
            return {k: remap(v) for k, v in value.items() if k != "id"}
        return value
    for n in snapshot.get("nodes", ()):
        rows.append({k: remap(n[k]) for k in
                     ("op", "target", "args", "kwargs", "results", "input_kind", "input_target")})
    return rows


def attach_original_identity(exported: Any, original: dict[str, Any], actual: dict[str, Any]) -> bool:
    """Attach identities only after whole-graph ordered structural equality."""
    if original.get("status") != "complete":
        return False
    supplied = dict(original)
    claimed_hash = supplied.pop("sha256", None)
    if claimed_hash != _digest(supplied):
        raise ValueError("original frontend snapshot digest does not match its graph")
    if (_structural_rows(original) != _structural_rows(actual)
            or original.get("range_constraints") != actual.get("range_constraints")
            or original.get("runtime_versions") != actual.get("runtime_versions")):
        return False
    # A second exact export is an observation of the same source graph, not a
    # fourth public trace stage. Normalize only its proven one-to-one identities;
    # never discard arbitrary lineage or match nodes by names/operators alone.
    identities = {observed["id"]: source["id"]
                  for source, observed in zip(original["nodes"], actual["nodes"], strict=True)}
    original_ids = iter(n["id"] for n in original["nodes"])
    for _, module in _graph_modules(exported.graph_module):
        for node in module.graph.nodes:
            custom = dict(node.meta.get("custom") or {})
            lineage = list(dict.fromkeys(identities.get(value, value)
                                        for value in custom.get("m2m_lineage") or ()))
            prior_id = next(original_ids)
            if prior_id not in lineage:
                lineage.append(prior_id)
            custom["m2m_lineage"] = lineage
            if custom.get("m2m_node_id") in identities:
                custom["m2m_node_id"] = identities[custom["m2m_node_id"]]
            node.meta["custom"] = custom
            if node.meta.get("_m2m_node_id") in identities:
                node.meta["_m2m_node_id"] = identities[node.meta["_m2m_node_id"]]
            node.meta["_m2m_origin_ids"] = [value for value in lineage
                                            if value != node.meta.get("_m2m_node_id")]
    return True


def attach_quantization_boundaries(quantized: Any, contraction_targets: Any) -> int:
    """Attribute inserted Q/DQ input chains to their exact annotated consumers.

    The caller supplies the operator objects it actually annotated. Existing
    preserved source metadata is mandatory; this helper invents no source match.
    Call after PT2E conversion and before re-export. Shared boundaries retain all
    consumers' origins, so no one-to-one assumption loses a source operation.
    """
    from torch.fx import Node
    targets = set(contraction_targets)
    attributed = set()
    for _, gm in _graph_modules(quantized):
        for contraction in gm.graph.nodes:
            custom = contraction.meta.get("custom") or {}
            lineage = list(custom.get("m2m_lineage") or ())
            if not lineage or contraction.target not in targets:
                continue
            pending = [arg for arg in contraction.args[:2] if isinstance(arg, Node)]
            seen = set()
            while pending:
                boundary = pending.pop()
                if boundary in seen or not str(boundary.target).startswith("quantized_decomposed."):
                    continue
                seen.add(boundary)
                metadata = dict(boundary.meta.get("custom") or {})
                metadata["m2m_lineage"] = sorted(set([*metadata.get("m2m_lineage", ()), *lineage]))
                metadata["m2m_transform"] = "quantization_input_boundary"
                boundary.meta["custom"] = metadata
                attributed.add(boundary)
                pending.extend(arg for arg in boundary.args if isinstance(arg, Node))
    return len(attributed)


def tuple_selection_trace_program(exported: Any) -> Any:
    """Instrument exact tuple-result aliases on a detached captured graph.

    Export's decomposition interpreter can reuse an already-created result proxy
    for getitem, discarding the selecting call site's metadata. Observe that actual
    proxy object instead of reconstructing a relationship from graph names or order.
    The captured program's input/state/signature is unchanged; no new export runs.
    """
    import copy
    import operator
    import torch
    from torch.fx.experimental.proxy_tensor import get_proxy_mode, get_proxy_slot
    from torch.fx.traceback import get_current_meta

    if not any(n.target is operator.getitem for _, gm in _graph_modules(exported.graph_module)
               for n in gm.graph.nodes):
        return exported
    # GraphModule holds lifted structure; the ExportedProgram state_dict is not
    # deep-copied. _update validates the unchanged graph before instrumentation.
    detached = exported._update(graph_module=copy.deepcopy(exported.graph_module),
                                graph_signature=copy.deepcopy(exported.graph_signature))

    def traced_getitem(value, index):
        result = operator.getitem(value, index)
        mode = get_proxy_mode()
        if mode is not None and isinstance(result, torch.Tensor):
            underlying = result.from_functional() if hasattr(result, "from_functional") else result
            slot = get_proxy_slot(underlying, mode.tracer, default=None)
            if slot is not None:
                proxy_node = slot.proxy.node
                origins = list((get_current_meta().get("custom") or {}).get("m2m_lineage") or ())
                custom = dict(proxy_node.meta.get("custom") or {})
                custom["m2m_lineage"] = sorted(set([*custom.get("m2m_lineage", ()), *origins]))
                proxy_node.meta["custom"] = custom
        return result

    for _, gm in _graph_modules(detached.graph_module):
        changed = False
        for node in gm.graph.nodes:
            if node.target is operator.getitem:
                node.target = traced_getitem
                changed = True
        if changed:
            gm.recompile()
    return detached


def _identity_conversion_bypass(source: dict[str, Any], dest: dict[str, Any],
                                node: dict[str, Any]) -> dict[str, Any] | None:
    """Prove a vanished forward-value identity at each exact typed use.

    PyTorch export can remove ``aten.to`` before a registered decomposition
    observes it. Matching operation names or output dtypes alone is insufficient:
    the destination graph must route the cast's original producer to every
    corresponding consumer at the same typed argument path. Any missing lineage,
    changed dtype/shape, requested copy, or layout request leaves it unresolved.
    ``detach_`` affects autograd metadata, which this forward-value trace does not
    certify; its captured tensor value may be bypassed under the same edge proof.
    """
    target = node.get("target")
    if node.get("op") != "call_function" or target not in {
            "aten.to.dtype", "aten.to.dtype_layout", "aten.to.device", "aten.detach_.default"}:
        return None
    args, kwargs = node.get("args"), node.get("kwargs") or {}
    if not isinstance(args, list) or not args or not isinstance(kwargs, dict):
        return None
    input_ref = args[0]
    if not isinstance(input_ref, dict) or not isinstance(input_ref.get("node_id"), str):
        return None
    sources = {item["id"]: item for item in source["nodes"]}
    producer = sources.get(input_ref["node_id"])
    outputs = node.get("results") or []
    inputs = producer.get("results") or [] if producer is not None else []
    if len(outputs) != 1 or len(inputs) != 1 or input_ref.get("value_id") != inputs[0].get("id"):
        return None
    output, original = outputs[0], inputs[0]

    def same_spec(raw: Any, kind: str, expected: str) -> bool:
        return raw is None or (isinstance(raw, dict)
                               and raw.get("kind") == kind and raw.get("value") == expected)

    if target == "aten.detach_.default":
        if len(args) != 1 or kwargs:
            return None
    elif target == "aten.to.dtype":
        if (len(args) < 2 or len(args) > 5
                or set(kwargs) - {"non_blocking", "copy", "memory_format"}
                or args[1] is None
                or not same_spec(args[1], "dtype", f"torch.{original.get('dtype')}")
                or (args[2] if len(args) > 2 else kwargs.get("non_blocking", False)) is not False
                or (args[3] if len(args) > 3 else kwargs.get("copy", False)) is not False
                or (args[4] if len(args) > 4 else kwargs.get("memory_format")) is not None):
            return None
    elif target == "aten.to.dtype_layout":
        if (len(args) != 1
                or set(kwargs) - {"dtype", "layout", "device", "pin_memory",
                                  "non_blocking", "copy", "memory_format"}
                or not same_spec(kwargs.get("dtype"), "dtype", f"torch.{original.get('dtype')}")
                or not same_spec(kwargs.get("layout"), "layout", original.get("layout"))
                or not same_spec(kwargs.get("device"), "device", original.get("device"))
                or kwargs.get("pin_memory") is not None
                or kwargs.get("non_blocking", False) is not False
                or kwargs.get("copy", False) is not False
                or kwargs.get("memory_format") is not None):
            return None
    else:
        if (len(args) < 3 or len(args) > 6
                or set(kwargs) - {"non_blocking", "copy", "memory_format"}
                or args[1] is None or args[2] is None
                or not same_spec(args[1], "device", original.get("device"))
                or not same_spec(args[2], "dtype", f"torch.{original.get('dtype')}")
                or (args[3] if len(args) > 3 else kwargs.get("non_blocking", False)) is not False
                or (args[4] if len(args) > 4 else kwargs.get("copy", False)) is not False
                or (args[5] if len(args) > 5 else kwargs.get("memory_format")) is not None):
            return None
    if original.get("kind") != "tensor" or original.get("dtype") is None:
        return None
    if any(output.get(field) != original.get(field) for field in
           ("kind", "dtype", "storage_dtype", "compute_dtype", "shape",
            "device", "layout", "stride")):
        return None
    descendants: dict[str, set[str]] = {}
    for item in dest["nodes"]:
        for ancestor in item.get("origin_node_ids") or ():
            descendants.setdefault(ancestor, set()).add(item["id"])
    corresponding_producers = descendants.get(producer["id"], set())
    if not corresponding_producers:
        return None
    dest_edges = dest.get("edges") or []
    dest_values = {value["id"]: value for item in dest["nodes"]
                   for value in item.get("results") or [] if isinstance(value.get("id"), str)}
    uses = [edge for edge in source.get("edges") or []
            if edge.get("producer_node_id") == node["id"]]
    if not uses or any(edge.get("producer_value_id") != output.get("id") for edge in uses):
        return None
    matched = []
    for use in uses:
        consumers = descendants.get(use.get("consumer_node_id"), set())
        if not consumers:
            return None
        # One source consumer may decompose into several prepared calls. Demand
        # exactly one direct edge from the mapped producer into that descendant
        # set at the original argument path; an internal edge from a different
        # producer cannot establish that the vanished cast was bypassed.
        matching_edges = [edge for edge in dest_edges
                          if edge.get("consumer_node_id") in consumers
                          and edge.get("argument_path") == use.get("argument_path")
                          and edge.get("producer_node_id") in corresponding_producers]
        dest_value = dest_values.get(matching_edges[0].get("producer_value_id")) if len(matching_edges) == 1 else None
        if (len(matching_edges) != 1
                or matching_edges[0].get("dtype") != use.get("dtype")
                or matching_edges[0].get("shape") != use.get("shape")
                or dest_value is None
                or any(dest_value.get(field) != original.get(field) for field in
                       ("kind", "dtype", "storage_dtype", "compute_dtype", "shape",
                        "device", "layout", "stride"))):
            return None
        matched.extend((edge["producer_node_id"], edge["consumer_node_id"],
                        edge["argument_path"]) for edge in matching_edges)
    return {"source_ids": [node["id"]], "destination_ids": [], "kind": "eliminated",
            "reason": ("forward tensor-value-preserving detach_ with exact typed consumer-edge bypass"
                       if target == "aten.detach_.default" else
                       "type/device/layout-preserving aten.to with exact typed consumer-edge bypass"),
            "proof": {"input_node_id": producer["id"], "typed_bypass_edges": sorted(set(matched)),
                      "scope": "forward tensor values; autograd metadata is not certified"}}


def _unused_tuple_selection(source: dict[str, Any], node: dict[str, Any]) -> dict[str, Any] | None:
    """Prove an unobserved, in-bounds tuple getitem cannot affect program output."""
    if (node.get("op") != "call_function" or node.get("target") != "<built-in function getitem>"
            or node.get("kwargs") or len(node.get("args") or ()) != 2
            or any(edge.get("producer_node_id") == node.get("id") for edge in source.get("edges") or ())):
        return None
    ref, index = node["args"]
    if not isinstance(ref, dict) or type(index) is not int:
        return None
    producer = next((item for item in source["nodes"] if item["id"] == ref.get("node_id")), None)
    results = producer.get("results") or [] if producer is not None else []
    if len(results) < 2 or not -len(results) <= index < len(results):
        return None
    selected = results[index]
    output = node.get("results") or []
    if (len(output) != 1 or ref.get("value_id") != selected.get("id")
            or any(output[0].get(field) != selected.get(field) for field in
                   ("kind", "dtype", "storage_dtype", "compute_dtype", "shape", "device", "layout", "stride"))):
        return None
    incoming = [edge for edge in source.get("edges") or [] if edge.get("consumer_node_id") == node["id"]]
    if (len(incoming) != 1 or incoming[0].get("producer_node_id") != producer["id"]
            or incoming[0].get("producer_value_id") != selected.get("id")
            or incoming[0].get("argument_path") != "args/0"):
        return None
    return {"source_ids": [node["id"]], "destination_ids": [], "kind": "eliminated",
            "reason": "unused in-bounds selection of an exactly typed tuple result",
            "proof": {"tuple_node_id": producer["id"], "result_index": index}}


def graph_relation(source: dict[str, Any] | None, dest: dict[str, Any] | None) -> dict[str, Any]:
    if not source or not dest or source.get("status") != "complete" or dest.get("status") != "complete":
        return {"from_stage": (source or {}).get("stage"), "to_stage": (dest or {}).get("stage"),
                "status": "diagnostic", "relations": [], "reason": "graph unavailable"}
    src_ids = {n["id"] for n in source["nodes"]}
    relations, consumed, unknown = [], set(), []
    for node in dest["nodes"]:
        origins = sorted(src_ids.intersection(node.get("origin_node_ids", ())))
        if origins:
            consumed.update(origins)
            relations.append({"source_ids": origins, "destination_ids": [node["id"]],
                              "kind": "introduced" if node.get("transformation") else "transformation"})
        elif node["op"] not in {"placeholder", "output", "get_attr"}:
            unknown.append(node["id"])
    for elimination in dest.get("eliminations", ()):
        origins = sorted(src_ids.intersection(elimination.get("source_ids", ())))
        if origins:
            consumed.update(origins)
            relations.append({**elimination, "source_ids": origins, "destination_ids": []})
    for node in source["nodes"]:
        if node["id"] not in consumed:
            proved = (_identity_conversion_bypass(source, dest, node)
                      or _unused_tuple_selection(source, node))
            if proved is not None:
                consumed.add(node["id"])
                relations.append(proved)
    fanout = Counter(s for rel in relations for s in rel["source_ids"] if rel["destination_ids"])
    for rel in relations:
        if rel["kind"] == "transformation":
            rel["kind"] = "fusion" if len(rel["source_ids"]) > 1 else (
                "decomposition" if any(fanout[s] > 1 for s in rel["source_ids"]) else "identity")
    unresolved = sorted(n["id"] for n in source["nodes"] if n["id"] not in consumed
                        and n["op"] in {"call_function", "call_module", "call_method"})
    return {"from_stage": source["stage"], "to_stage": dest["stage"],
            "status": "complete" if not unresolved and not unknown else "diagnostic",
            "relations": relations, "unresolved_source_ids": unresolved,
            "unresolved_destination_ids": unknown}


def stamp_mlir_sources(op: Any, node_id: str | None, origins: Any = (), *, role: str = "lowering") -> None:
    if not node_id:
        return
    from xdsl.dialects.builtin import ArrayAttr, StringAttr
    for child in op.walk():
        existing = child.attributes.get("prov.source_node_ids")
        ids = [v.data for v in existing] if isinstance(existing, ArrayAttr) else []
        child.attributes["prov.source_node_ids"] = ArrayAttr([StringAttr(v) for v in
                                                              sorted(set([*ids, node_id]))])
        existing_origins = child.attributes.get("prov.origin_node_ids")
        prior = [v.data for v in existing_origins] if isinstance(existing_origins, ArrayAttr) else []
        child.attributes["prov.origin_node_ids"] = ArrayAttr([StringAttr(v) for v in
                                                              sorted(set([*prior, *origins]))])
        child.attributes["prov.trace_role"] = StringAttr(role)


def copy_mlir_sources(dest: Any, *sources: Any, role: str = "support") -> None:
    """Carry exact ancestry through owned MLIR transforms, including fusion."""
    from xdsl.dialects.builtin import ArrayAttr
    ids, origins = set(), set()
    for source in sources:
        for key, bucket in (("prov.source_node_ids", ids), ("prov.origin_node_ids", origins)):
            attr = source.attributes.get(key)
            if isinstance(attr, ArrayAttr):
                bucket.update(v.data for v in attr)
    for node_id in sorted(ids):
        stamp_mlir_sources(dest, node_id, sorted(origins), role=role)


def finalize_trace(graphs: dict[str, Any], mlir_text: str, *, path: str,
                   dispositions: dict[str, Any] | None = None) -> dict[str, Any]:
    from m2m.capture.torch_mlir_bridge import _parse_mlir_text_to_xdsl
    from xdsl.dialects.builtin import ArrayAttr, StringAttr, UnregisteredOp
    blockers, operations = [], []
    module = _parse_mlir_text_to_xdsl(mlir_text) if mlir_text else None
    prepared = graphs.get("prepared") or {}
    projections = []
    if path == "fx_importer":
        for node in prepared.get("nodes", ()):
            for value in node.get("results", ()):
                dtype = value.get("storage_dtype") or ""
                if dtype.startswith("float8_"):
                    projections.append({"node_id": node["id"], "value_id": value["id"],
                                        "frontend_dtype": dtype, "mlir_storage_dtype": "f32",
                                        "reason": "FXImporter/xDSL has no native FP8 arithmetic type support"})
        if projections:
            blockers.append("FP8 storage is projected to f32; exact storage precision is not qualified")
    calls = {n["id"] for n in prepared.get("nodes", ())
             if n["op"] in {"call_function", "call_method", "call_module"}}
    mapped: dict[str, list[int]] = {}
    if module is None:
        blockers.append("final MLIR cannot be parsed for exact correspondence")
    else:
        for ordinal, op in enumerate(module.walk()):
            def ids(key):
                attr = op.attributes.get(key)
                return [v.data for v in attr] if isinstance(attr, ArrayAttr) else []
            source_ids, origins = ids("prov.source_node_ids"), ids("prov.origin_node_ids")
            role = op.attributes.get("prov.trace_role")
            role = role.data if isinstance(role, StringAttr) else (
                "structural" if op.name in {"builtin.module", "func.func", "func.return", "linalg.yield", "scf.yield"}
                else "unresolved")
            if role == "unresolved":
                blockers.append(f"MLIR operation {ordinal} has no transform source")
            for sid in source_ids:
                mapped.setdefault(sid, []).append(ordinal)
            actual_name = op.op_name.data if isinstance(op, UnregisteredOp) else op.name
            operations.append({"ordinal": ordinal, "operation": actual_name,
                               "xdsl_operation": op.name,
                               "source_node_ids": source_ids, "origin_node_ids": origins,
                               "role": role, "operand_types": [str(v.type) for v in op.operands],
                               "result_types": [str(v.type) for v in op.results],
                               "location": str(op.location)})
    correspondence = []
    for sid in sorted(calls):
        ordinals = mapped.get(sid, [])
        disposition = (dispositions or {}).get(sid, {})
        state = disposition.get("status", "lowered" if ordinals else "unresolved")
        if not ordinals and state not in {"alias", "eliminated"}:
            state = "unresolved"
            blockers.append(f"prepared operation {sid} has no final lowering correspondence")
        correspondence.append({"node_id": sid, "mlir_ordinals": ordinals, "status": state,
                               **{k: v for k, v in disposition.items() if k != "status"}})
    transforms = [graph_relation(graphs.get("original"), graphs.get("quantized")),
                  graph_relation(graphs.get("quantized"), graphs.get("prepared"))]
    for relation in transforms:
        if relation["status"] != "complete":
            blockers.append(f"{relation['from_stage']} -> {relation['to_stage']} correspondence incomplete")
    if path != "fx_importer":
        blockers.append(f"exact frontend correspondence is not qualified for {path}")
    return {"schema": "m2m.frontend_trace.v1", "status": "diagnostic" if blockers else "complete",
            "counting_unit": "static_captured_call_sites", "graphs": graphs,
            "transformations": transforms,
            "precision": {"scope": "FXImporter FP8 storage projection audit",
                          "status": "projected" if projections else "no_known_projection",
                          "projections": projections},
            "mlir": {"sha256": hashlib.sha256(mlir_text.encode()).hexdigest(),
                     "bytes": len(mlir_text.encode()), "serialization": "generic" if path == "fx_importer" else "backend",
                     "operations": operations, "source_correspondence": correspondence},
            "blockers": blockers}
