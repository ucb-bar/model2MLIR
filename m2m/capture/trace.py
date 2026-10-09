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


def _result_metadata(
    metadata: dict[str, Any], results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Record actual metadata presence without changing historical value slots.

    A missing ``val`` and an explicitly observed Python None formerly had the
    same unknown result type. This additional record distinguishes them and
    binds every observed value to its existing ordinal. Metadata is not native
    execution, numerical equivalence or an operator-effect proof.
    """
    record: dict[str, Any] = {"schema": "m2m.frontend_result_metadata.v1"}
    if "val" not in metadata:
        return {**record, "status": "unobserved"}
    value = metadata["val"]
    values = list(value) if isinstance(value, (tuple, list)) else [value]
    return {
        **record,
        "status": "observed",
        "container": (
            type(value).__name__ if isinstance(value, (tuple, list)) else "single"
        ),
        "values": [
            {"result_id": result["id"], "kind": "none" if val is None else result["kind"]}
            for val, result in zip(values, results, strict=True)
        ],
    }


def _graph_modules(gm: Any):
    """Include GraphModule bodies in graph-scoped identities, without double counting."""
    yield "root", gm
    for name, sub in gm.named_modules():
        if name and hasattr(sub, "graph"):
            yield name, sub


def _constant_input_paths(node: Any, constant: Any) -> list[list[Any]]:
    """The exact FX operand positions using a lifted constant."""
    paths = []

    def visit(value: Any, path: list[Any]) -> None:
        if value is constant:
            paths.append(path)
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, [*path, index])
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, [*path, str(key)])

    visit(node.args, ["args"])
    visit(node.kwargs, ["kwargs"])
    return paths


def defer_lifted_constant_user_lineage(graph_module: Any, exported: Any) -> int:
    """Make single-user constants safe for Torch's placeholder metadata copy.

    Torch can copy a direct consumer's custom metadata onto a lifted constant
    placeholder, then reject the different metadata of the source ``get_attr``.
    Defer only that consumer's ancestry across export. The snapshotter restores
    it after checking the exact constant bytes, target, and FX input binding.
    """
    from torch.export.graph_signature import InputKind

    targets = {str(spec.target) for spec in exported.graph_signature.input_specs
               if spec.kind == InputKind.CONSTANT_TENSOR}
    deferred = 0
    for scope, module in _graph_modules(graph_module):
        if scope != "root":
            continue
        for constant in module.graph.nodes:
            if constant.op != "get_attr" or str(constant.target) not in targets:
                continue
            if len(constant.users) != 1:
                continue
            consumer = next(iter(constant.users))
            if consumer.op != "call_function":
                continue
            left = dict(constant.meta.get("custom") or {})
            right = dict(consumer.meta.get("custom") or {})
            rows = list(right.get("m2m_deferred_constant_users") or ())
            lineage = list(right.get("m2m_lineage") or (rows[0]["lineage"] if rows else ()))
            consumer_id = right.get("m2m_node_id") or (lineage[-1] if lineage else None)
            if not left.get("m2m_lineage") or not lineage:
                continue
            if left.get("m2m_node_id") == consumer_id:
                continue
            paths = _constant_input_paths(consumer, constant)
            if not paths:
                raise ValueError("lifted constant consumer has no direct FX input binding")
            parent, _, name = str(constant.target).rpartition(".")
            owner = module.get_submodule(parent) if parent else module
            value = getattr(owner, name)
            if rows and any(row["lineage"] != lineage for row in rows):
                raise ValueError("lifted constant consumer has inconsistent deferred ancestry")
            rows.append({"constant_target": str(constant.target),
                         "constant_sha256": _tensor_bytes_sha256(value),
                         "consumer_target": str(consumer.target),
                         "input_paths": paths, "lineage": lineage})
            right.pop("m2m_node_id", None)
            right.pop("m2m_lineage", None)
            right["m2m_deferred_constant_users"] = rows
            consumer.meta["custom"] = right
            deferred += 1
    return deferred


def _restore_deferred_constant_user(exported: Any, node: Any, custom: dict[str, Any]) -> None:
    """Admit deferred ancestry only for an unchanged constant-to-call edge."""
    from torch.export.graph_signature import InputKind

    rows = custom.pop("m2m_deferred_constant_users", None)
    if rows is None or node.op == "placeholder":
        return
    if node.op != "call_function" or not isinstance(rows, list) or not rows:
        raise ValueError("deferred constant ancestry is not on a call")
    specs = {str(spec.target): spec for spec in exported.graph_signature.input_specs
             if spec.kind == InputKind.CONSTANT_TENSOR}
    graph_nodes = {candidate.name: candidate for candidate in exported.graph_module.graph.nodes}
    for row in rows:
        if not isinstance(row, dict) or str(node.target) != row.get("consumer_target"):
            raise ValueError("deferred constant consumer target changed")
        spec = specs.get(row.get("constant_target"))
        constant = graph_nodes.get(spec.arg.name) if spec is not None else None
        value = exported.constants.get(row.get("constant_target"))
        if (constant is None or constant.op != "placeholder" or len(constant.users) != 1
                or node not in constant.users or not isinstance(row.get("input_paths"), list)
                or _constant_input_paths(node, constant) != row["input_paths"]
                or value is None or _tensor_bytes_sha256(value) != row.get("constant_sha256")):
            raise ValueError("deferred constant ancestry lost its exact binding or bytes")
        lineage = row.get("lineage")
        if not isinstance(lineage, list) or not lineage or not all(
                isinstance(identity, str) and identity for identity in lineage):
            raise ValueError("deferred constant ancestry is incomplete")
        custom["m2m_lineage"] = list(dict.fromkeys([*custom.get("m2m_lineage", ()), *lineage]))


def snapshot_exported_program(exported: Any, *, stage: str) -> dict[str, Any]:
    """Snapshot and stamp exactly this program, for the subsequent lowering."""
    from torch.fx import Node
    gm = exported.graph_module
    sig = getattr(exported, "graph_signature", None)
    input_specs = {getattr(s.arg, "name", None): s for s in getattr(sig, "input_specs", ())}
    records, edges = [], []
    operator_schemas = {}
    for scope, module in _graph_modules(gm):
        graph_id = f"g:{stage}:{scope}"
        ids = {n: f"{graph_id}:n{i}" for i, n in enumerate(module.graph.nodes)}
        for ordinal, node in enumerate(module.graph.nodes):
            node_id = ids[node]
            # Export can share the mutable metadata dictionary of an outer
            # getitem with the inner node it selects. Stamp node-local metadata
            # before writing an identity, or the later inner snapshot silently
            # replaces the outer call's provenance.
            node.meta = dict(node.meta)
            custom = dict(node.meta.get("custom") or {})
            _restore_deferred_constant_user(exported, node, custom)
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
            schema = getattr(node.target, "_schema", None) if node.op == "call_function" else None
            if schema is not None:
                text = str(schema)
                if target in operator_schemas and operator_schemas[target] != text:
                    raise ValueError("frontend operator schema changed during snapshot")
                operator_schemas[target] = text
            classification = ("aten" if target.startswith("aten.") else
                              "custom" if node.op == "call_function" else
                              "call" if node.op in {"call_method", "call_module"} else "structural")
            spec = input_specs.get(node.name) if scope == "root" else None
            results = _types(node.meta.get("val"), node_id)
            records.append({"id": node_id, "graph_id": graph_id, "ordinal": ordinal,
                            "op": node.op, "target": target, "classification": classification,
                            "args": encode(node.args, "args"),
                            "kwargs": encode(node.kwargs, "kwargs"),
                            "results": results,
                            "result_metadata": _result_metadata(node.meta, results),
                            "module_stack": _literal(node.meta.get("nn_module_stack") or {}),
                            "input_kind": str(getattr(spec, "kind", "")).removeprefix("InputKind.") or None,
                            "input_target": getattr(spec, "target", None),
                            "transformation": custom.get("m2m_transform"),
                            "origin_node_ids": [v for v in prior if v != node_id]})
    body = {"schema": "m2m.frontend_graph.v1", "stage": stage, "status": "complete",
            "graph_id": f"g:{stage}:root", "nodes": records, "edges": edges,
            "operator_schemas": dict(sorted(operator_schemas.items())),
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


def _retarget_captured_float_dtypes(graph_module: Any, dtype: Any,
                                    original: dict[str, Any]) -> list[dict[str, Any]]:
    """Retarget only explicit floating ``dtype`` operands of schema-known FX calls."""
    import torch

    def dtype_literals(value: Any) -> list[Any]:
        if isinstance(value, torch.dtype):
            return [value]
        if isinstance(value, (tuple, list)):
            return [item for child in value for item in dtype_literals(child)]
        if isinstance(value, dict):
            return [item for child in value.values() for item in dtype_literals(child)]
        return []

    decisions = []
    seen_ids = set()
    original_nodes = {node["id"]: node for node in original["nodes"]}
    for _, module in _graph_modules(graph_module):
        module_changed = False
        for node in module.graph.nodes:
            if node.op != "call_function":
                if dtype_literals(node.args) or dtype_literals(node.kwargs):
                    raise ValueError("captured dtype literal lacks a schema-owned call")
                continue
            schema = getattr(node.target, "_schema", None)
            args = list(node.args)
            kwargs = dict(node.kwargs)
            node_changed = False
            node_decisions = 0
            dtype_positions = ([(i, arg.name) for i, arg in enumerate(schema.arguments)
                                if arg.name == "dtype"] if schema is not None else [])
            for index, name in dtype_positions:
                if name in kwargs:
                    location, value = "kwargs", kwargs[name]
                elif index < len(args):
                    location, value = "args", args[index]
                else:
                    continue
                if not isinstance(value, torch.dtype):
                    if value is not None:
                        raise ValueError("captured dtype operand is not a static torch.dtype")
                    continue
                if value.is_complex:
                    raise ValueError("captured complex dtype is not FP32-staged")
                replacement = dtype if value.is_floating_point else value
                source_id = node.meta.get("_m2m_node_id")
                if not isinstance(source_id, str) or not source_id:
                    raise ValueError("captured dtype operand has no source node identity")
                if source_id in seen_ids:
                    raise ValueError("captured dtype operand has duplicate source node identity")
                seen_ids.add(source_id)
                source = original_nodes.get(source_id)
                source_value = (source["kwargs"].get(name) if location == "kwargs"
                                else source["args"][index]) if source else None
                if (source is None or source["op"] != "call_function"
                        or source["target"] != str(node.target)
                        or source_value != {"kind": "dtype", "value": str(value)}):
                    raise ValueError("captured dtype operand differs from exact original typed source")
                decisions.append({"source_node_id": source_id,
                                  "target": str(node.target), "argument": name,
                                  "location": location, "from_dtype": str(value),
                                  "to_dtype": str(replacement)})
                node_decisions += 1
                if replacement != value:
                    if location == "kwargs":
                        kwargs[name] = replacement
                    else:
                        args[index] = replacement
                    node_changed = True
            # A floating dtype literal anywhere other than a schema's dtype
            # operand would otherwise evade both the rewrite and its audit.
            all_literals = dtype_literals(node.args) + dtype_literals(node.kwargs)
            if len(all_literals) != node_decisions:
                raise ValueError("captured dtype literal lacks a schema-owned dtype operand")
            if node_changed:
                node.args = tuple(args)
                node.kwargs = kwargs
                module_changed = True
        if module_changed:
            module.graph.lint()
            module.recompile()
    return decisions


def _tensor_bytes_sha256(value: Any) -> str:
    """Hash exact dense tensor storage bytes, including BF16 and integer values."""
    import torch

    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise ValueError("captured constant has no dense tensor storage")
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(memoryview(raw.numpy()).cast("B")).hexdigest()


def _owned_lifted_constant_conversions(exported: Any, graph_module: Any,
                                        dtype: Any) -> list[dict[str, Any]]:
    """Clone only floating exported CONSTANT_TENSOR values into the owned module."""
    import torch
    from torch.export.graph_signature import InputKind

    rows = []
    for ordinal, spec in enumerate(exported.graph_signature.input_specs):
        if spec.kind != InputKind.CONSTANT_TENSOR:
            continue
        name = str(spec.target)
        source = exported.constants.get(name)
        parent_name, _, local_name = name.rpartition(".")
        owner = graph_module.get_submodule(parent_name) if parent_name else graph_module
        materialized = getattr(owner, local_name, None)
        if not isinstance(source, torch.Tensor) or not isinstance(materialized, torch.Tensor):
            raise ValueError("captured constant has no owned tensor attribute")
        if source.is_complex():
            raise ValueError("captured complex constant is not FP32-staged")
        source_hash = _tensor_bytes_sha256(source)
        if (source.dtype != materialized.dtype or source.shape != materialized.shape
                or source_hash != _tensor_bytes_sha256(materialized)):
            raise ValueError("captured constant differs from its owned module attribute")
        converted = (source.detach().to(dtype=dtype, copy=True)
                     if source.is_floating_point() else materialized)
        if source.is_floating_point():
            setattr(owner, local_name, converted)
        rows.append({"source_input_ordinal": ordinal, "source_target": name,
                     "from_dtype": str(source.dtype), "to_dtype": str(converted.dtype),
                     "shape": list(source.shape), "source_sha256": source_hash,
                     "staged_sha256": _tensor_bytes_sha256(converted)})
    return rows


def _audit_staged_float_precision(exported: Any, graph_module: Any, inputs: Any,
                                  snapshot: dict[str, Any], dtype: Any) -> dict:
    """Check actual re-exported tensor types, not the stale pre-rewrite FX metadata."""
    import torch
    from torch.utils._pytree import tree_flatten

    state = list(graph_module.named_parameters()) + list(graph_module.named_buffers())
    captured_state = list((getattr(exported, "state_dict", {}) or {}).values())
    captured_constants = list((getattr(exported, "constants", {}) or {}).values())
    input_values = tree_flatten(tuple(inputs))[0]
    wrong = []
    checked = 0
    if snapshot.get("status") != "complete":
        raise ValueError("staged graph has no complete typed frontend snapshot")
    for kind, values in (("state", [value for _, value in state]),
                         ("exported_state", captured_state),
                         ("exported_constant", captured_constants),
                         ("input", input_values)):
        for value in values:
            if isinstance(value, torch.Tensor) and value.is_complex():
                raise ValueError(f"{kind} has complex precision outside FP32 staging")
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                checked += 1
                if value.dtype != dtype:
                    wrong.append(f"{kind}:{value.dtype}")
    rows = {node["id"]: node for node in snapshot["nodes"]}
    for _, module in _graph_modules(exported.graph_module):
        for node in module.graph.nodes:
            if node.op not in {"call_function", "call_method", "call_module"}:
                continue
            schema = getattr(node.target, "_schema", None)
            tensor_return = bool(schema and any("Tensor" in str(value.type)
                                                 for value in schema.returns))
            row = rows.get(node.meta.get("_m2m_node_id"))
            if row is None:
                raise ValueError("staged call lacks a typed snapshot identity")
            if (tensor_return or schema is None) and any(
                    result.get("kind") == "unknown" for result in row["results"]):
                raise ValueError("staged call has an unknown tensor result dtype")
    for node in snapshot["nodes"]:
        for result in node["results"]:
            spelling = result.get("dtype")
            if spelling is None and result.get("kind") == "tensor":
                raise ValueError("staged tensor result lacks a dtype")
            if spelling is None:
                continue
            value_dtype = getattr(torch, spelling, None)
            if not isinstance(value_dtype, torch.dtype):
                raise ValueError(f"staged result has an unrecognized dtype: {spelling}")
            if value_dtype.is_complex:
                raise ValueError(f"staged result has complex precision: {node['id']}")
            if value_dtype.is_floating_point:
                checked += 1
                if value_dtype != dtype:
                    wrong.append(f"{node['id']}:{spelling}")
    if wrong:
        raise ValueError(f"staged graph retains non-target floating precision: {wrong[:8]}")
    return {"status": "complete", "target_dtype": str(dtype),
            "checked_floating_values": checked, "non_target_floating_values": 0}


def materialize_frontend_precision(model: Any, inputs: Any = (), *, dtype: Any,
                                  original_frontend_snapshot: dict[str, Any] | None = None,
                                  retarget_float_dtype_arguments: bool = False):
    """Cast an owned materialization of a captured static program with exact origins.

    Returns ``(graph_module, inputs_tuple, original_snapshot, receipt)``. This is
    precision conversion of the already-captured program, not a claim that
    dtype-dependent Python branches would recapture identically. No graph-name
    matching, source-model mutation, decomposition or quantization takes place.
    Opt-in retargeting changes only schema-identified floating dtype operands
    and owned floating lifted constants in that captured graph, then re-exports
    to audit the resulting real types. Integer/bool values remain unchanged;
    no numerical-equivalence claim follows from a successful type audit.
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
    constant_targets = set()
    if retarget_float_dtype_arguments:
        from torch.export.graph_signature import InputKind

        constant_targets = {str(spec.target) for spec in exported.graph_signature.input_specs
                            if spec.kind == InputKind.CONSTANT_TENSOR}
    memo = {}
    for parameter, values in ((True, list(gm.named_parameters(remove_duplicate=False))),
                              (False, list(gm.named_buffers(remove_duplicate=False)))):
        for name, value in values:
            # Some Torch versions register lifted constants as buffers in the
            # owned module; the exact constant pass below binds those bytes.
            if name in constant_targets:
                continue
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
    decisions = None
    constants = None
    staged_snapshot = None
    staged_audit = None
    if retarget_float_dtype_arguments:
        constants = _owned_lifted_constant_conversions(exported, gm, dtype)
        decisions = _retarget_captured_float_dtypes(gm, dtype, original)
        prepare_lifted_constant_lineage(gm)
        defer_lifted_constant_user_lineage(gm, exported)
        staged = torch.export.export(gm, cast_inputs)
        from torch.export.graph_signature import InputKind

        def verify_constants(program):
            specs = [spec for spec in program.graph_signature.input_specs
                     if spec.kind == InputKind.CONSTANT_TENSOR]
            if len(specs) != len(constants):
                raise ValueError("staged constant roster differs from captured source")
            for row, spec in zip(constants, specs, strict=True):
                value = program.constants.get(str(spec.target))
                if (not isinstance(value, torch.Tensor) or str(value.dtype) != row["to_dtype"]
                        or list(value.shape) != row["shape"]
                        or _tensor_bytes_sha256(value) != row["staged_sha256"]):
                    raise ValueError("staged constant differs from owned converted source")
                row["staged_target"] = str(spec.target)

        verify_constants(staged)
        snapshot_exported_program(staged, stage="staged")
        gm = staged.module()
        allow_exported_model_train_eval(gm)
        prepare_lifted_constant_lineage(gm)
        defer_lifted_constant_user_lineage(gm, staged)
        # The returned module has its own final lineage/constant metadata.
        # Bind the receipt to a fresh export of *that* module, rather than an
        # intermediate program whose graph can differ after materialization.
        returned = torch.export.export(gm, cast_inputs)
        if ([spec.kind for spec in returned.graph_signature.input_specs]
                != [spec.kind for spec in staged.graph_signature.input_specs]
                or [spec.kind for spec in returned.graph_signature.output_specs]
                != [spec.kind for spec in staged.graph_signature.output_specs]
                or returned.call_spec.in_spec != staged.call_spec.in_spec
                or returned.call_spec.out_spec != staged.call_spec.out_spec):
            raise ValueError("returned staged module changed input/output structure")
        verify_constants(returned)
        staged_snapshot = snapshot_exported_program(returned, stage="staged")
        staged_audit = _audit_staged_float_precision(returned, gm, cast_inputs,
                                                      staged_snapshot, dtype)
        # Return an export-safe module; the next traced conversion restores
        # these call origins from exact constant signatures and bytes.
        defer_lifted_constant_user_lineage(gm, returned)
    receipt = {"schema": "m2m.frontend-precision-conversion.v1",
               "scope": "precision conversion of captured static program",
               "original_graph_sha256": original["sha256"], "target_dtype": str(dtype),
               "state_conversions": conversions, "input_conversions": input_conversions}
    if retarget_float_dtype_arguments:
        receipt.update(staged_graph_sha256=staged_snapshot["sha256"],
                       dtype_decisions=decisions, constant_conversions=constants,
                       staged_precision_audit=staged_audit,
                       graph_dtype_retargeting="schema_float_dtype_operands")
    receipt["sha256"] = _digest(receipt)
    return gm, cast_inputs, original, receipt


def _structural_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Stage-independent exact graph equivalence, not node-name/FQN matching."""
    rows = []
    ids = {n["id"]: i for i, n in enumerate(snapshot.get("nodes", ()))}
    nodes = {n["id"]: n for n in snapshot.get("nodes", ())}
    def remap(value):
        if isinstance(value, list):
            return [remap(v) for v in value]
        if isinstance(value, dict):
            if "node_id" in value:
                producer = nodes[value["node_id"]]
                result_id = value.get("value_id")
                matches = [i for i, result in enumerate(producer.get("results") or ())
                           if result["id"] == result_id]
                if result_id is not None and len(matches) != 1:
                    raise ValueError("frontend edge has no unique typed producer result")
                return {"ordinal": ids[value["node_id"]],
                        "result_index": matches[0] if matches else None}
            return {k: remap(v) for k, v in value.items() if k != "id"}
        return value
    for n in snapshot.get("nodes", ()):
        rows.append({k: remap(n[k]) for k in
                     ("op", "target", "args", "kwargs", "results", "input_kind", "input_target")})
    return rows


def _reordered_graph_identity(original: dict[str, Any], actual: dict[str, Any]) -> dict[str, str] | None:
    """Prove a unique typed DAG isomorphism when independent FX calls reorder.

    Only exact operators, ordered operands/results and source-owned inputs may
    match. Repeated indistinguishable owners are ambiguous and fail closed;
    graph names, module stacks, tensor shapes alone, and node ordinals are not
    used as evidence for a reordered call. Input targets establish structural
    ownership, not equality of external parameter/buffer/constant payloads.
    """
    source = original.get("nodes") or []
    observed = actual.get("nodes") or []
    if len(source) != len(observed):
        return None
    source_by_id = {node["id"]: node for node in source}
    observed_by_id = {node["id"]: node for node in observed}
    if len(source_by_id) != len(source) or len(observed_by_id) != len(observed):
        return None

    def placeholder_positions(nodes: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        positions = {}
        for node in nodes:
            if node["op"] == "placeholder" and node.get("input_target") is None:
                scope = node["graph_id"].split(":", 2)[-1]
                positions[node["id"]] = counts.get(scope, 0)
                counts[scope] = positions[node["id"]] + 1
        return positions

    source_positions = placeholder_positions(source)
    observed_positions = placeholder_positions(observed)

    def key(node: dict[str, Any], nodes: dict[str, dict[str, Any]],
            mapped: dict[str, str], positions: dict[str, int]) -> str:
        def remap(value: Any) -> Any:
            if isinstance(value, list):
                return [remap(item) for item in value]
            if isinstance(value, dict):
                if "node_id" in value:
                    if set(value) != {"node_id", "value_id"}:
                        raise ValueError("frontend edge has unrecognized fields")
                    producer_id = value["node_id"]
                    producer = nodes[producer_id]
                    result_id = value["value_id"]
                    matches = [i for i, result in enumerate(producer.get("results") or ())
                               if result["id"] == result_id]
                    if result_id is not None and len(matches) != 1:
                        raise ValueError("frontend edge has no unique typed producer result")
                    return {"owner": mapped[producer_id],
                            "result_index": matches[0] if matches else None}
                return {field: remap(item) for field, item in value.items()}
            return value

        body = {
            "scope": node["graph_id"].split(":", 2)[-1],
            "op": node["op"], "target": node["target"],
            "args": remap(node["args"]), "kwargs": remap(node["kwargs"]),
            "results": [{field: value for field, value in result.items() if field != "id"}
                        for result in node.get("results") or ()],
            "input_kind": node.get("input_kind"),
            "input_target": node.get("input_target"),
            # Positional user inputs are owned by the public graph ABI. Lifted
            # state is instead owned by its explicit input_target.
            "input_position": positions.get(node["id"]),
        }
        return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)

    try:
        source_identity = {node_id: node_id for node_id in source_by_id}
        candidates: dict[str, list[str]] = {}
        for node in source:
            candidates.setdefault(key(node, source_by_id, source_identity, source_positions), []).append(node["id"])
        mapping: dict[str, str] = {}
        used = set()
        for node in observed:
            matches = candidates.get(key(node, observed_by_id, mapping, observed_positions), ())
            if len(matches) != 1 or matches[0] in used:
                return None
            mapping[node["id"]] = matches[0]
            used.add(matches[0])
    except (KeyError, TypeError, ValueError):
        return None
    return mapping if len(used) == len(source) else None


def attach_original_identity(exported: Any, original: dict[str, Any], actual: dict[str, Any]) -> bool:
    """Attach structural source identities, never a tensor-payload equality claim.

    Precision and capture receipts independently bind owned lifted constants
    and materialized weights. This correspondence compares operators, typed
    edges and input ownership; it does not inspect tensor contents.
    """
    if original.get("status") != "complete" or actual.get("status") != "complete":
        return False
    supplied = dict(original)
    claimed_hash = supplied.pop("sha256", None)
    if claimed_hash != _digest(supplied):
        raise ValueError("original frontend snapshot digest does not match its graph")
    if (original.get("range_constraints") != actual.get("range_constraints")
            or original.get("runtime_versions") != actual.get("runtime_versions")):
        return False
    # Preserve the established ordered identity path, including duplicate
    # identical calls. A reordered graph needs unique typed dataflow ownership.
    try:
        ordered = _structural_rows(original) == _structural_rows(actual)
    except (KeyError, TypeError, ValueError):
        return False
    if ordered:
        identities = {observed["id"]: source["id"]
                      for source, observed in zip(original["nodes"], actual["nodes"], strict=True)}
    else:
        identities = _reordered_graph_identity(original, actual)
    if identities is None:
        return False
    graph_nodes = [node for _, module in _graph_modules(exported.graph_module)
                   for node in module.graph.nodes]
    if (len(graph_nodes) != len(actual["nodes"])
            or any(node.meta.get("_m2m_node_id") != row["id"]
                   for node, row in zip(graph_nodes, actual["nodes"], strict=True))):
        return False
    # A second exact export is an observation of the same source graph, not a
    # fourth public trace stage. Normalize only proven one-to-one identities.
    for node, observed in zip(graph_nodes, actual["nodes"], strict=True):
        custom = dict(node.meta.get("custom") or {})
        lineage = list(dict.fromkeys(identities.get(value, value)
                                    for value in custom.get("m2m_lineage") or ()))
        prior_id = identities[observed["id"]]
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


def prepare_lifted_constant_lineage(graph_module: Any) -> int:
    """Keep both source IDs when re-export lifts a copied tensor constant.

    Export may replace a single-use ``get_attr -> copy`` pair with
    one placeholder; decomposition can turn ``lift_fresh_copy`` into ``clone``.
    Its placeholder naming pass requires identical custom
    metadata on both nodes. The merged lineage records that exact value path;
    other constant uses and non-lineage custom metadata are left alone.
    """
    import torch

    count = 0
    for _, module in _graph_modules(graph_module):
        for constant in module.graph.nodes:
            if constant.op != "get_attr" or len(constant.users) != 1:
                continue
            copy = next(iter(constant.users))
            if (copy.target not in {torch.ops.aten.lift_fresh_copy.default,
                                    torch.ops.aten.clone.default}
                    or copy.args != (constant,) or copy.kwargs):
                continue
            left = dict(constant.meta.get("custom") or {})
            right = dict(copy.meta.get("custom") or {})
            if not left.get("m2m_lineage") or not right.get("m2m_lineage"):
                continue
            other_left = {k: v for k, v in left.items()
                          if k not in {"m2m_node_id", "m2m_lineage"}}
            other_right = {k: v for k, v in right.items()
                           if k not in {"m2m_node_id", "m2m_lineage"}}
            if other_left != other_right:
                raise ValueError("lifted constant has conflicting custom metadata")
            merged = {**other_left, "m2m_lineage": list(dict.fromkeys(
                [*left["m2m_lineage"], *right["m2m_lineage"]]))}
            constant.meta["custom"] = dict(merged)
            copy.meta["custom"] = dict(merged)
            count += 1
    return count


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


def pt2e_conv_bn_fold_candidates(
    graph_module: Any, original: dict[str, Any]
) -> list[tuple[Any, Any, tuple[Any, ...], str]]:
    """Remember exact live Conv -> BatchNorm edges before PT2E preparation.

    The returned FX nodes are object identities, not graph names. PT2E's fold
    mutates the Conv and erases the BatchNorm; the candidates are only attributed
    after that rewrite has actually happened.
    """
    import torch

    original_ids = {node["id"] for node in original.get("nodes", ())}
    candidates = []
    for bn in graph_module.graph.nodes:
        if bn.op != "call_function" or bn.target != torch.ops.aten.batch_norm.default:
            continue
        if not bn.args or not isinstance(bn.args[0], torch.fx.Node):
            continue
        conv = bn.args[0]
        if conv.op != "call_function" or conv.target != torch.ops.aten.conv2d.default:
            continue
        origins = original_ids.intersection((bn.meta.get("custom") or {}).get("m2m_lineage") or ())
        if len(origins) != 1 or not bn.users:
            continue
        conv_val, bn_val = conv.meta.get("val"), bn.meta.get("val")
        if (getattr(conv_val, "shape", None) != getattr(bn_val, "shape", None)
                or getattr(conv_val, "dtype", None) != getattr(bn_val, "dtype", None)):
            continue
        candidates.append((conv, bn, tuple(bn.users), next(iter(origins))))
    return candidates


def attach_pt2e_conv_bn_folds(
    graph_module: Any, candidates: list[tuple[Any, Any, tuple[Any, ...], str]]
) -> int:
    """Carry BN ancestry onto a Conv only after PT2E's exact fold is observed.

    Called by Quantizer.transform_for_annotation, immediately after torchao's
    `_fuse_conv_bn_` and before observer insertion. A dead BN, a different
    replacement, or an incomplete rewrite cannot certify a fusion.
    """
    live = set(graph_module.graph.nodes)
    attributed = 0
    for conv, bn, users, origin in candidates:
        if conv not in live or bn in live or not bn._erased:
            continue
        surviving_users = [user for user in users if user in live]
        if not surviving_users or any(
            bn in user.all_input_nodes or conv not in user.all_input_nodes
            for user in surviving_users
        ):
            continue
        custom = dict(conv.meta.get("custom") or {})
        custom["m2m_lineage"] = list(dict.fromkeys([*custom.get("m2m_lineage", ()), origin]))
        conv.meta["custom"] = custom
        attributed += 1
    return attributed


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


def _forward_identity_value(source: dict[str, Any], node: dict[str, Any]
                            ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Return an exact input/output tensor identity for a supported no-op call."""
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
    return producer, original, output


def _identity_conversion_bypass(source: dict[str, Any], dest: dict[str, Any],
                                node: dict[str, Any]) -> dict[str, Any] | None:
    """Prove an eliminated identity chain by exact typed terminal use edges.

    PyTorch can remove both ``detach_`` and an adjacent no-op ``to`` during
    decomposition. Every skipped link must separately preserve the exact tensor
    value and metadata, and every terminal use must bypass to the same mapped
    producer at the same typed consumer argument. Autograd is not certified.
    """
    checked = _forward_identity_value(source, node)
    if checked is None:
        return None
    producer, original, output = checked
    from m2m.capture.view_trace import SourceStorageAliases

    aliases = SourceStorageAliases(source)
    if not aliases.valid or source.get("runtime_versions") != dest.get("runtime_versions"):
        return None

    def unmodified_input(identity: dict[str, Any], input_node: dict[str, Any]) -> bool:
        if identity.get("graph_id") != input_node.get("graph_id"):
            return False
        state = aliases.last_writer(identity)
        return (state is not None and
                (state[1] is None or state[1]["ordinal"] <= input_node["ordinal"]))

    if not unmodified_input(node, producer):
        return None
    sources = {item["id"]: item for item in source["nodes"]}
    descendants: dict[str, set[str]] = {}
    for item in dest["nodes"]:
        for ancestor in item.get("origin_node_ids") or ():
            descendants.setdefault(ancestor, set()).add(item["id"])
    bypassed = {node["id"]}
    while producer["id"] not in descendants:
        upstream = _forward_identity_value(source, producer)
        if upstream is None:
            return None
        parent, parent_value, parent_output = upstream
        if (parent_output != original or producer["id"] in bypassed
                or not unmodified_input(producer, parent)):
            return None
        bypassed.add(producer["id"])
        producer, original = parent, parent_value
    corresponding_producers = descendants.get(producer["id"], set())
    if not corresponding_producers:
        return None
    dest_edges = dest.get("edges") or []
    dest_values = {value["id"]: value for item in dest["nodes"]
                   for value in item.get("results") or [] if isinstance(value.get("id"), str)}
    source_edges = source.get("edges") or []

    def terminal_uses(value_node: dict[str, Any], value: dict[str, Any],
                      visited: set[str]) -> list[dict[str, Any]] | None:
        if value_node["id"] in visited:
            return None
        visited = {*visited, value_node["id"]}
        uses = [edge for edge in source_edges
                if edge.get("producer_node_id") == value_node["id"]]
        if not uses or any(edge.get("producer_value_id") != value.get("id") for edge in uses):
            return None
        terminal = []
        for use in uses:
            consumer_id = use.get("consumer_node_id")
            if consumer_id in descendants:
                terminal.append(use)
                continue
            consumer = sources.get(consumer_id)
            next_link = _forward_identity_value(source, consumer) if consumer else None
            if (next_link is None or next_link[0]["id"] != value_node["id"]
                    or next_link[1] != value or use.get("argument_path") != "args/0"
                    or not unmodified_input(consumer, value_node)):
                return None
            child = terminal_uses(consumer, next_link[2], visited)
            if child is None:
                return None
            terminal.extend(child)
        return terminal

    uses = terminal_uses(node, output, set())
    if not uses:
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
                       if node["target"] == "aten.detach_.default" else
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


def _tuple_selector_relations(source: dict[str, Any], dest: dict[str, Any],
                              relations: list[dict[str, Any]], consumed: set[str]) -> list[dict[str, Any]]:
    """Recover a selector whose exporter lineage was copied from its tuple parent.

    Require an exact typed tuple value, a uniquely mapped producer, and one
    selector at the same index carrying that producer's lineage. This covers
    split/min tuple selectors reconstructed by export without matching names.
    """
    target = "<built-in function getitem>"
    source_nodes = {node["id"]: node for node in source["nodes"]}
    dest_nodes = {node["id"]: node for node in dest["nodes"]}
    mapped: dict[str, set[str]] = {}
    for relation in relations:
        for identity in relation["source_ids"]:
            mapped.setdefault(identity, set()).update(relation["destination_ids"])

    def spec(value):
        return {key: field for key, field in value.items() if key != "id"}

    source_keys = Counter()
    for node in source["nodes"]:
        args = node.get("args") or []
        if (node.get("target") == target and len(args) == 2
                and isinstance(args[0], dict) and type(args[1]) is int):
            source_keys[(args[0].get("node_id"), args[1])] += 1
    dest_selectors: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for node in dest["nodes"]:
        args = node.get("args") or []
        if (node.get("target") == target and len(args) == 2
                and isinstance(args[0], dict) and type(args[1]) is int):
            dest_selectors.setdefault((args[0].get("node_id"), args[1]), []).append(node)

    result = []
    for selector in source["nodes"]:
        if selector["id"] in consumed or selector.get("target") != target:
            continue
        args = selector.get("args") or []
        if (len(args) != 2 or not isinstance(args[0], dict) or type(args[1]) is not int
                or selector.get("kwargs")):
            continue
        parent_id, index = args[0].get("node_id"), args[1]
        parent = source_nodes.get(parent_id)
        values = parent.get("results") or [] if parent else []
        if (len(values) < 2 or not -len(values) <= index < len(values)
                or selector.get("graph_id") != parent.get("graph_id")
                or source_keys[(parent_id, index)] != 1
                or args[0].get("value_id") != values[index].get("id")
                or len(selector.get("results") or []) != 1
                or spec(selector["results"][0]) != spec(values[index])):
            continue
        candidates = []
        for dest_id in mapped.get(parent_id, ()):
            candidate_parent = dest_nodes.get(dest_id)
            dest_values = candidate_parent.get("results") or [] if candidate_parent else []
            if (candidate_parent is None or candidate_parent.get("target") != parent.get("target")
                    or candidate_parent.get("graph_id", "").split(":", 2)[-1]
                    != parent.get("graph_id", "").split(":", 2)[-1]
                    or len(dest_values) != len(values)
                    or [spec(value) for value in dest_values] != [spec(value) for value in values]):
                continue
            for candidate in dest_selectors.get((dest_id, index), ()):
                ref = candidate["args"][0]
                if (candidate.get("graph_id") == candidate_parent.get("graph_id")
                        and set(candidate.get("origin_node_ids") or ()).intersection(source_nodes)
                        <= {parent_id}
                        and ref.get("value_id") == dest_values[index].get("id")
                        and len(candidate.get("results") or []) == 1
                        and spec(candidate["results"][0]) == spec(selector["results"][0])):
                    candidates.append(candidate)
        if len(candidates) == 1:
            result.append({"source_ids": [selector["id"]],
                           "destination_ids": [candidates[0]["id"]],
                           "kind": "tuple_selection",
                           "proof": {"source_tuple_node_id": parent_id,
                                     "destination_tuple_node_id": candidates[0]["args"][0]["node_id"],
                                     "result_index": index, "exact_typed_value": True}})
    return result


_ANCHORED_TARGETS = {
    "aten.sym_size.int", "aten.sym_constrain_range_for_size.default",
    "<built-in function ge>", "<built-in function le>", "aten._assert_scalar.default",
    "aten._assert_tensor_metadata.default", "aten.unsqueeze.default", "aten.expand.default",
    "aten.to.dtype", "aten.to.dtype_layout", "aten.matmul.default", "aten.transpose.int",
    "aten.cat.default", "aten.cos.default", "aten.sin.default", "aten.mul.Tensor",
    "aten.dropout.default",
}

# Exact ATen forward-value operations whose outputs can be discarded without
# changing any observable tensor result. Do not infer purity from an absent
# mutation marker: random and externally observable calls need separate proof.
_DEAD_VALUE_TARGETS = {
    "aten.arange.default", "aten.unsqueeze.default", "aten.to.dtype", "aten.mul.Tensor",
}


def _dead_forward_value_relations(source: dict[str, Any], consumed: set[str]) -> list[dict[str, Any]]:
    users: dict[str, set[str]] = {}
    for edge in source.get("edges") or []:
        users.setdefault(edge["producer_node_id"], set()).add(edge["consumer_node_id"])
    dead: set[str] = set()
    while True:
        newly_dead = {
            node["id"] for node in source["nodes"]
            if node["id"] not in consumed and node["id"] not in dead
            and node.get("op") == "call_function" and node.get("target") in _DEAD_VALUE_TARGETS
            and users.get(node["id"], set()) <= dead
        }
        if not newly_dead:
            break
        dead.update(newly_dead)
    return [{"source_ids": [node["id"]], "destination_ids": [], "kind": "eliminated",
             "reason": "dead forward tensor value from an explicitly pure ATen operation",
             "proof": {"operator": node["target"],
                       "source_users": sorted(users.get(node["id"], set())),
                       "all_users_dead": True}}
            for node in source["nodes"] if node["id"] in dead]


def _inlined_parameter_relations(source: dict[str, Any], dest: dict[str, Any],
                                 relations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[str]]:
    """Bind a uniquely invoked grad-disabled body's inputs to its caller values."""
    src_nodes = {node["id"]: node for node in source["nodes"]}
    dst_nodes = {node["id"]: node for node in dest["nodes"]}
    mapped: dict[str, set[str]] = {}
    for row in relations:
        for identity in row["source_ids"]:
            mapped.setdefault(identity, set()).update(row["destination_ids"])

    def specs(node: dict[str, Any]) -> list[dict[str, Any]]:
        return [{key: value for key, value in result.items() if key != "id"}
                for result in node.get("results") or []]

    result = []
    verified_scopes = set()
    scopes = {node["graph_id"].split(":", 2)[-1] for node in source["nodes"]
              if isinstance(node.get("graph_id"), str)
              and node["graph_id"] != f"g:{source['stage']}:root"}
    for scope in sorted(scopes):
        attrs = [node for node in source["nodes"] if node.get("op") == "get_attr"
                 and node.get("graph_id") == f"g:{source['stage']}:root" and node.get("target") == scope]
        if len(attrs) != 1:
            continue
        wrappers = [node for node in source["nodes"] if node.get("target") == "wrap_with_set_grad_enabled"
                    and node.get("graph_id") == f"g:{source['stage']}:root"
                    and isinstance(node.get("args"), list) and len(node["args"]) >= 2
                    and node["args"][0] is False
                    and isinstance(node["args"][1], dict)
                    and node["args"][1].get("node_id") == attrs[0]["id"]]
        if len(wrappers) != 1:
            continue
        placeholders = [node for node in source["nodes"] if node.get("op") == "placeholder"
                        and node.get("graph_id") == f"g:{source['stage']}:{scope}"]
        arguments = wrappers[0]["args"][2:]
        if len(placeholders) != len(arguments):
            continue
        bindings = []
        for placeholder, argument in zip(placeholders, arguments, strict=True):
            if not isinstance(argument, dict):
                break
            parent = src_nodes.get(argument.get("node_id"))
            selected = next((index for index, value in enumerate(parent.get("results") or [])
                             if value.get("id") == argument.get("value_id")), None) if parent else None
            if selected is None or specs(placeholder) != [specs(parent)[selected]]:
                break
            candidates = [dst_nodes[identity] for identity in mapped.get(parent["id"], set())
                          if identity in dst_nodes and len(specs(dst_nodes[identity])) > selected
                          and specs(dst_nodes[identity])[selected] == specs(parent)[selected]]
            if len(candidates) != 1:
                break
            bindings.append({"source_ids": [placeholder["id"]], "destination_ids": [candidates[0]["id"]],
                             "kind": "inlined_parameter_binding",
                             "proof": {"wrapper_node_id": wrappers[0]["id"],
                                       "submodule": scope, "argument_result_index": selected}})
        if len(bindings) == len(placeholders):
            result.extend(bindings)
            verified_scopes.add(scope)
    return result, verified_scopes


def _no_op_dtype_aliases(source: dict[str, Any], dest: dict[str, Any],
                         relations: list[dict[str, Any]], consumed: set[str]) -> list[dict[str, Any]]:
    """Map a no-copy, same-dtype ``aten.to`` to the current storage value."""
    from m2m.capture.view_trace import SourceStorageAliases

    aliases_of_source = SourceStorageAliases(source)
    if (not aliases_of_source.valid
            or source.get("runtime_versions") != dest.get("runtime_versions")):
        return []
    src_nodes = {node["id"]: node for node in source["nodes"]}
    dst_nodes = {node["id"]: node for node in dest["nodes"]}
    mapped: dict[str, set[str]] = {}
    for row in relations:
        for identity in row["source_ids"]:
            mapped.setdefault(identity, set()).update(row["destination_ids"])

    def specs(node: dict[str, Any]) -> list[dict[str, Any]]:
        return [{key: value for key, value in result.items() if key != "id"}
                for result in node.get("results") or []]

    aliases = []
    for node in source["nodes"]:
        args, kwargs = node.get("args"), node.get("kwargs") or {}
        target = node.get("target")
        if (node["id"] in consumed or target not in {"aten.to.dtype", "aten.to.dtype_layout"}
                or not isinstance(args, list) or not args or not isinstance(args[0], dict)):
            continue
        producer = src_nodes.get(args[0].get("node_id"))
        if producer is None or len(producer.get("results") or []) != 1 or len(node.get("results") or []) != 1:
            continue
        if producer.get("graph_id") != node.get("graph_id"):
            continue
        original = producer["results"][0]
        if (args[0].get("value_id") != original.get("id")
                or original.get("kind") != "tensor" or specs(node) != specs(producer)):
            continue
        if target == "aten.to.dtype":
            if (len(args) != 2 or args[1] != {"kind": "dtype", "value": f"torch.{original.get('dtype')}"}
                    or set(kwargs) - {"non_blocking", "copy", "memory_format"}
                    or kwargs.get("non_blocking", False) is not False
                    or kwargs.get("copy", False) is not False
                    or kwargs.get("memory_format") is not None):
                continue
        elif (len(args) != 1
              or set(kwargs) - {"dtype", "layout", "device", "pin_memory",
                                "non_blocking", "copy", "memory_format"}
              or kwargs.get("dtype") != {"kind": "dtype", "value": f"torch.{original.get('dtype')}"}
              or kwargs.get("layout") not in (None, {"kind": "layout", "value": original.get("layout")})
              or kwargs.get("device") not in (None, {"kind": "device", "value": original.get("device")})
              or kwargs.get("pin_memory") is not None
              or kwargs.get("non_blocking", False) is not False
              or kwargs.get("copy", False) is not False
              or kwargs.get("memory_format") is not None):
            continue
        state = aliases_of_source.last_writer(node)
        if state is None:
            continue
        root, writer = state
        updated_after_producer = writer is not None and writer["ordinal"] > producer["ordinal"]
        current = writer if updated_after_producer else producer
        candidates = [dst_nodes[identity] for identity in mapped.get(current["id"], set())
                      if identity in dst_nodes and specs(dst_nodes[identity]) == specs(producer)]
        if len(candidates) == 1 and all(
            edge.get("producer_value_id") == candidates[0]["results"][0]["id"]
            and edge.get("dtype") == original.get("dtype")
            and edge.get("shape") == original.get("shape")
            for edge in dest.get("edges") or [] if edge.get("producer_node_id") == candidates[0]["id"]
        ) and all(
            edge.get("producer_value_id") == node["results"][0]["id"]
            and edge.get("dtype") == original.get("dtype")
            and edge.get("shape") == original.get("shape")
            for edge in source.get("edges") or [] if edge.get("producer_node_id") == node["id"]
        ):
            aliases.append({"source_ids": [node["id"]], "destination_ids": [candidates[0]["id"]],
                            "kind": "value_alias",
                            "proof": {"no_copy_same_dtype": True, "input_node_id": producer["id"],
                                      "storage_root_node_id": root,
                                      "last_writer_node_id": writer["id"] if writer else None,
                                      "updated_after_input": updated_after_producer,
                                      "exact_typed_value": True}})
    return aliases


def _anchored_guard_relations(source: dict[str, Any], dest: dict[str, Any],
                              relations: list[dict[str, Any]], consumed: set[str]) -> list[dict[str, Any]]:
    """Match identical calls through uniquely mapped typed input values.

    Decomposition can retain a guard unchanged while dropping its FX lineage.
    Names, ordinals, operator spelling and output type alone are never enough:
    every referenced argument must already be mapped to the exact corresponding
    producer result, and there must be exactly one matching destination call.
    """
    src_nodes = {node["id"]: node for node in source["nodes"]}
    dst_nodes = {node["id"]: node for node in dest["nodes"]}
    from m2m.capture.view_trace import SourceStorageAliases

    aliases = SourceStorageAliases(source)
    inferred = []

    def value_index(nodes: dict[str, Any], ref: dict[str, Any]) -> int | None:
        owner = nodes.get(ref.get("node_id"))
        return next((index for index, value in enumerate(owner.get("results") or [])
                     if value.get("id") == ref.get("value_id")), None) if owner else None

    def same_args(left: Any, right: Any, mapped: dict[str, set[str]]) -> tuple[bool, int]:
        if isinstance(left, dict) and "node_id" in left:
            if not isinstance(right, dict) or "node_id" not in right:
                return False, 0
            origin, target = left["node_id"], right["node_id"]
            src_index, dst_index = value_index(src_nodes, left), value_index(dst_nodes, right)
            return (target in mapped.get(origin, set()) and src_index is not None
                    and src_index == dst_index
                    and specs(src_nodes[origin])[src_index] == specs(dst_nodes[target])[dst_index]), 1
        if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
            checked = [same_args(a, b, mapped) for a, b in zip(left, right, strict=True)]
            return all(ok for ok, _ in checked), sum(count for _, count in checked)
        if isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys():
            checked = [same_args(left[key], right[key], mapped) for key in left]
            return all(ok for ok, _ in checked), sum(count for _, count in checked)
        return left == right, 0

    def specs(node: dict[str, Any]) -> list[dict[str, Any]]:
        return [{key: value for key, value in result.items() if key != "id"}
                for result in node.get("results") or []]

    def call_arguments(node: dict[str, Any]) -> tuple[Any, Any] | None:
        args, kwargs = node.get("args"), node.get("kwargs") or {}
        if node["target"] == "aten.dropout.default":
            if (not isinstance(args, list) or len(args) != 3 or kwargs
                    or not isinstance(args[0], dict)
                    or set(args[0]) != {"node_id", "value_id"}
                    or type(args[1]) not in (float, int) or not 0 <= args[1] <= 1
                    or args[2] is not False):
                return None
            owner = src_nodes.get(args[0]["node_id"]) or dst_nodes.get(args[0]["node_id"])
            values = [value for value in owner.get("results") or []
                      if value["id"] == args[0]["value_id"]] if owner else []
            if len(values) != 1 or specs(node) != specs({"results": values}):
                return None
            # In eval, every legal probability is the same exact tensor value.
            return [args[0], "inactive_dropout"], {}
        if node["target"] != "aten._assert_tensor_metadata.default":
            return args, kwargs
        fields = ("a", "size", "stride", "dtype", "device", "layout")
        if (not isinstance(args, list) or len(args) > len(fields)
                or not isinstance(kwargs, dict) or set(kwargs) - set(fields)
                or any(field in kwargs for field in fields[:len(args)])):
            return None
        return [args[index] if index < len(args) else kwargs.get(field)
                for index, field in enumerate(fields)], {}

    while True:
        mapped: dict[str, set[str]] = {}
        for row in [*relations, *inferred]:
            for origin in row["source_ids"]:
                mapped.setdefault(origin, set()).update(row["destination_ids"])
        candidates = []
        for node in source["nodes"]:
            if node["id"] in consumed or node["target"] not in _ANCHORED_TARGETS:
                continue
            identity = _forward_identity_value(source, node)
            if identity is not None:
                producer = identity[0]
                state = aliases.last_writer(node)
                if (not aliases.valid
                        or source.get("runtime_versions") != dest.get("runtime_versions")
                        or producer.get("graph_id") != node.get("graph_id")
                        or state is None
                        or (state[1] is not None
                            and state[1]["ordinal"] > producer["ordinal"])):
                    continue
            matches = []
            for other in dest["nodes"]:
                if (other["id"] in {row["destination_ids"][0] for row in inferred}
                        or node["op"] != other["op"] or node["target"] != other["target"]
                        or specs(node) != specs(other)):
                    continue
                left_call, right_call = call_arguments(node), call_arguments(other)
                if left_call is None or right_call is None:
                    continue
                left_args, left_kwargs = left_call
                right_args, right_kwargs = right_call
                if node["target"] == "aten._assert_scalar.default":
                    if (not isinstance(left_args, list) or not isinstance(right_args, list)
                            or len(left_args) != 2 or len(right_args) != 2
                            or not isinstance(left_args[1], str) or not isinstance(right_args[1], str)):
                        continue
                    # Dynamo rewrites diagnostic FX node names in the text;
                    # the predicate and runtime rejection, not that prose,
                    # are the functional guard contract.
                    left_args, right_args = left_args[:1], right_args[:1]
                args_ok, anchors = same_args(left_args, right_args, mapped)
                kwargs_ok, kw_anchors = same_args(left_kwargs, right_kwargs, mapped)
                if args_ok and kwargs_ok and anchors + kw_anchors > 0:
                    matches.append(other)
            if len(matches) == 1:
                candidates.append((node, matches[0]))
        if not candidates:
            break
        # Two source calls competing for one destination call are ambiguous.
        counts = Counter(other["id"] for _, other in candidates)
        accepted = [(node, other) for node, other in candidates if counts[other["id"]] == 1]
        if not accepted:
            break
        for node, other in accepted:
            inferred.append({"source_ids": [node["id"]], "destination_ids": [other["id"]],
                             "kind": "structural_call_equivalence",
                             "proof": {"exact_typed_anchored_arguments": True,
                                       "unique_destination_call": True,
                                       **({"inactive_dropout": True}
                                          if node["target"] == "aten.dropout.default" else {}),
                                       "diagnostic_text_equated": node["target"] !=
                                       "aten._assert_scalar.default" or node["args"][1] == other["args"][1]}})
            consumed.add(node["id"])
    return inferred


def _introduced_metadata_guard_relations(dest: dict[str, Any],
                                         relations: list[dict[str, Any]],
                                         unknown: list[str]) -> list[dict[str, Any]]:
    """Account for export guards on values with proven source lineage."""
    nodes = {node["id"]: node for node in dest["nodes"]}
    origins: dict[str, set[str]] = {}
    for relation in relations:
        for target in relation["destination_ids"]:
            origins.setdefault(target, set()).update(relation["source_ids"])
    fields = ("a", "size", "stride", "dtype", "device", "layout")
    introduced = []
    for identity in unknown:
        guard = nodes.get(identity)
        if not guard or guard["op"] != "call_function" or guard["target"] != "aten._assert_tensor_metadata.default":
            continue
        results = guard.get("results") or []
        if results and not (len(results) == 1 and results[0].get("kind") == "unknown"
                            and results[0].get("shape") is None and results[0].get("dtype") is None):
            continue
        args, kwargs = guard.get("args"), guard.get("kwargs") or {}
        if (not isinstance(args, list) or not 1 <= len(args) <= len(fields)
                or not isinstance(kwargs, dict) or set(kwargs) - set(fields)
                or any(field in kwargs for field in fields[:len(args)])):
            continue
        values = {field: args[index] if index < len(args) else kwargs.get(field)
                  for index, field in enumerate(fields)}
        ref = values.pop("a")
        if not isinstance(ref, dict) or set(ref) != {"node_id", "value_id"}:
            continue
        producer = nodes.get(ref["node_id"])
        source_ids = sorted(origins.get(ref["node_id"], ()))
        if producer is None or not source_ids:
            continue
        typed = [row for row in producer.get("results") or [] if row.get("id") == ref["value_id"]]
        if len(typed) != 1 or typed[0].get("kind") != "tensor":
            continue
        result = typed[0]
        expected = {
            "size": result.get("shape"), "stride": result.get("stride"),
            "dtype": {"kind": "dtype", "value": f"torch.{result['dtype']}"},
            "device": {"kind": "device", "value": result.get("device")},
            "layout": {"kind": "layout", "value": result.get("layout")},
        }
        asserted = {field: value for field, value in values.items() if value is not None}
        if not asserted or any(expected[field] is None or value != expected[field]
                               for field, value in asserted.items()):
            continue
        introduced.append({
            "source_ids": source_ids, "destination_ids": [identity],
            "kind": "introduced_metadata_guard",
            "proof": {"producer_value_id": ref["value_id"],
                      "asserted_fields": sorted(asserted), "exact_typed_value": True},
        })
    return introduced


def _unchanged_nested_graph_relations(source: dict[str, Any], dest: dict[str, Any],
                                      relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover a nested graph's identity only when its body and caller are exact.

    PT2E may reconstruct a grad-disabled GraphModule without carrying inner FX
    lineage. Matching its operator names or result types alone is insufficient:
    the entire typed body, ordered arguments, and uniquely mapped caller values
    must agree. A changed or ambiguously invoked body remains unresolved.
    """
    def specs(node: dict[str, Any]) -> list[dict[str, Any]]:
        return [{key: value for key, value in result.items() if key != "id"}
                for result in node.get("results") or []]

    def scopes(snapshot: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        root = f"g:{snapshot['stage']}:root"
        for node in snapshot["nodes"]:
            graph_id = node.get("graph_id")
            if isinstance(graph_id, str) and graph_id != root:
                grouped.setdefault(graph_id.split(":", 2)[-1], []).append(node)
        return grouped

    def body(nodes: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        ids = {node["id"]: index for index, node in enumerate(nodes)}
        if len(ids) != len(nodes):
            return None

        def remap(value: Any) -> Any:
            if isinstance(value, list):
                return [remap(item) for item in value]
            if isinstance(value, dict):
                if "node_id" in value:
                    owner = nodes[ids[value["node_id"]]]
                    indices = [index for index, result in enumerate(owner.get("results") or [])
                               if result.get("id") == value.get("value_id")]
                    if len(indices) != 1:
                        raise ValueError("unbound nested result")
                    return {"ordinal": ids[value["node_id"]], "result_index": indices[0]}
                return {key: remap(item) for key, item in value.items()}
            return value

        rows = []
        try:
            for node in nodes:
                args, kwargs = node.get("args"), node.get("kwargs") or {}
                if node.get("target") == "aten._assert_tensor_metadata.default":
                    fields = ("a", "size", "stride", "dtype", "device", "layout")
                    if (not isinstance(args, list) or len(args) > len(fields)
                            or not isinstance(kwargs, dict) or set(kwargs) - set(fields)
                            or any(field in kwargs for field in fields[:len(args)])):
                        return None
                    args = [args[index] if index < len(args) else kwargs.get(field)
                            for index, field in enumerate(fields)]
                    kwargs = {}
                rows.append({"op": node.get("op"), "target": node.get("target"),
                             "args": remap(args), "kwargs": remap(kwargs),
                             "results": specs(node), "input_kind": node.get("input_kind"),
                             "input_target": node.get("input_target")})
        except (KeyError, ValueError):
            return None
        return rows

    def caller(snapshot: dict[str, Any], scope: str) -> dict[str, Any] | None:
        root = f"g:{snapshot['stage']}:root"
        attrs = [node for node in snapshot["nodes"] if node.get("graph_id") == root
                 and node.get("op") == "get_attr" and node.get("target") == scope]
        if len(attrs) != 1:
            return None
        wrappers = [node for node in snapshot["nodes"] if node.get("graph_id") == root
                    and node.get("target") == "wrap_with_set_grad_enabled"
                    and isinstance(node.get("args"), list) and len(node["args"]) >= 2
                    and node["args"][0] is False and isinstance(node["args"][1], dict)
                    and node["args"][1].get("node_id") == attrs[0]["id"]]
        if len(wrappers) != 1:
            return None
        attr_results = attrs[0].get("results") or []
        if (len(attr_results) != 1
                or wrappers[0]["args"][1].get("value_id") != attr_results[0].get("id")):
            return None
        return wrappers[0]

    mapped: dict[str, set[str]] = {}
    reverse: dict[str, set[str]] = {}
    for relation in relations:
        for source_id in relation["source_ids"]:
            mapped.setdefault(source_id, set()).update(relation["destination_ids"])
            for dest_id in relation["destination_ids"]:
                reverse.setdefault(dest_id, set()).add(source_id)
    src_nodes = {node["id"]: node for node in source["nodes"]}
    dst_nodes = {node["id"]: node for node in dest["nodes"]}

    def same_caller(source_wrapper: dict[str, Any], dest_wrapper: dict[str, Any],
                    src_body: list[dict[str, Any]], dst_body: list[dict[str, Any]]) -> bool:
        source_args, dest_args = source_wrapper["args"][2:], dest_wrapper["args"][2:]
        source_inputs = [node for node in src_body if node.get("op") == "placeholder"]
        dest_inputs = [node for node in dst_body if node.get("op") == "placeholder"]
        if (len(source_args) != len(dest_args) or len(source_args) != len(source_inputs)
                or len(dest_args) != len(dest_inputs)
                or source_wrapper.get("kwargs") != dest_wrapper.get("kwargs")
                or specs(source_wrapper) != specs(dest_wrapper)):
            return False
        for source_arg, dest_arg, source_input, dest_input in zip(
                source_args, dest_args, source_inputs, dest_inputs, strict=True):
            if not isinstance(source_arg, dict) or not isinstance(dest_arg, dict):
                return False
            source_parent = src_nodes.get(source_arg.get("node_id"))
            dest_parent = dst_nodes.get(dest_arg.get("node_id"))
            if (source_parent is None or dest_parent is None
                    or mapped.get(source_parent["id"]) != {dest_parent["id"]}):
                return False
            source_index = [index for index, value in enumerate(source_parent.get("results") or [])
                            if value.get("id") == source_arg.get("value_id")]
            dest_index = [index for index, value in enumerate(dest_parent.get("results") or [])
                          if value.get("id") == dest_arg.get("value_id")]
            if (len(source_index) != 1 or source_index != dest_index
                    or specs(source_parent)[source_index[0]] != specs(dest_parent)[dest_index[0]]
                    or specs(source_input) != [specs(source_parent)[source_index[0]]]
                    or specs(dest_input) != [specs(dest_parent)[dest_index[0]]]):
                return False
        return True

    source_scopes, dest_scopes = scopes(source), scopes(dest)
    candidates = []
    for source_scope, source_body in source_scopes.items():
        source_wrapper = caller(source, source_scope)
        source_rows = body(source_body)
        if source_wrapper is None or source_rows is None:
            continue
        matches = []
        for dest_scope, dest_body in dest_scopes.items():
            dest_wrapper = caller(dest, dest_scope)
            pairs = (list(zip(source_body, dest_body, strict=True))
                     if dest_wrapper is not None and len(source_body) == len(dest_body) else [])
            if (dest_wrapper is not None and source_rows == body(dest_body)
                    and all(mapped.get(left["id"], {right["id"]}) == {right["id"]}
                            and reverse.get(right["id"], {left["id"]}) == {left["id"]}
                            for left, right in pairs)
                    and same_caller(source_wrapper, dest_wrapper, source_body, dest_body)):
                matches.append((dest_scope, dest_body, dest_wrapper))
        if len(matches) == 1:
            candidates.append((source_scope, source_body, source_wrapper, *matches[0]))
    counts = Counter(dest_scope for _, _, _, dest_scope, _, _ in candidates)
    inferred = []
    for source_scope, source_body, source_wrapper, dest_scope, dest_body, dest_wrapper in candidates:
        if counts[dest_scope] != 1:
            continue
        for source_node, dest_node in zip(source_body, dest_body, strict=True):
            inferred.append({"source_ids": [source_node["id"]],
                             "destination_ids": [dest_node["id"]],
                             "kind": "nested_structural_identity",
                             "proof": {"exact_typed_body": True, "exact_caller_bindings": True,
                                       "source_scope": source_scope, "destination_scope": dest_scope}})
        inferred.append({"source_ids": [source_wrapper["id"]],
                         "destination_ids": [dest_wrapper["id"]],
                         "kind": "nested_call_identity",
                         "proof": {"exact_typed_body": True, "exact_caller_bindings": True,
                                   "source_scope": source_scope, "destination_scope": dest_scope}})
    return inferred


def graph_relation(source: dict[str, Any] | None, dest: dict[str, Any] | None) -> dict[str, Any]:
    if not source or not dest or source.get("status") != "complete" or dest.get("status") != "complete":
        return {"from_stage": (source or {}).get("stage"), "to_stage": (dest or {}).get("stage"),
                "status": "diagnostic", "relations": [], "reason": "graph unavailable"}
    src_ids = {n["id"] for n in source["nodes"]}
    src_nodes = {n["id"]: n for n in source["nodes"]}
    def scope(node: dict[str, Any], stage: str) -> str:
        return str(node.get("graph_id") or f"g:{stage}:root").split(":", 2)[-1]

    relations, consumed, unknown = [], set(), []
    pending_inlined = []
    for node in dest["nodes"]:
        origins = sorted(src_ids.intersection(node.get("origin_node_ids", ())))
        # A nested origin can be copied into a flat graph only after the
        # enclosing HOP's actual operands have been bound to its placeholders.
        # Otherwise a changed caller could inherit a stale inner-node tag.
        direct = [identity for identity in origins
                  if scope(src_nodes[identity], source["stage"]) == scope(node, dest["stage"])]
        inlined = [identity for identity in origins if identity not in direct]
        if inlined:
            pending_inlined.append((node, inlined))
        if direct:
            consumed.update(direct)
            relations.append({"source_ids": direct, "destination_ids": [node["id"]],
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
    nested = _unchanged_nested_graph_relations(source, dest, relations)
    relations.extend(nested)
    consumed.update(identity for relation in nested for identity in relation["source_ids"])
    unknown = [identity for identity in unknown if identity not in {
        target for relation in nested for target in relation["destination_ids"]}]
    bindings, verified_scopes = _inlined_parameter_relations(source, dest, relations)
    relations.extend(bindings)
    for node, identities in pending_inlined:
        accepted = [identity for identity in identities
                    if scope(src_nodes[identity], source["stage"]) in verified_scopes]
        if accepted:
            consumed.update(accepted)
            relations.append({"source_ids": accepted, "destination_ids": [node["id"]],
                              "kind": "inlined_origin",
                              "proof": {"exact_caller_bindings": True,
                                        "scopes": sorted({scope(src_nodes[identity], source["stage"])
                                                          for identity in accepted})}})
            unknown = [identity for identity in unknown if identity != node["id"]]
    while True:
        inferred = _anchored_guard_relations(source, dest, relations, consumed)
        aliases = _no_op_dtype_aliases(source, dest, [*relations, *inferred], consumed)
        new = [*inferred, *aliases]
        if not new:
            break
        relations.extend(new)
        consumed.update(identity for relation in new for identity in relation["source_ids"])
        unknown = [identity for identity in unknown if identity not in {
            target for relation in new for target in relation["destination_ids"]}]
    introduced_guards = _introduced_metadata_guard_relations(dest, relations, unknown)
    relations.extend(introduced_guards)
    unknown = [identity for identity in unknown if identity not in {
        target for relation in introduced_guards for target in relation["destination_ids"]}]
    selectors = _tuple_selector_relations(source, dest, relations, consumed)
    relations.extend(selectors)
    consumed.update(identity for relation in selectors for identity in relation["source_ids"])
    unknown = [identity for identity in unknown if identity not in {
        target for relation in selectors for target in relation["destination_ids"]}]
    from m2m.capture.view_trace import replayed_view_relations
    views = replayed_view_relations(source, dest, relations, consumed)
    relations.extend(views)
    consumed.update(identity for relation in views for identity in relation["source_ids"])
    unknown = [identity for identity in unknown if identity not in {
        target for relation in views for target in relation["destination_ids"]}]
    dead = _dead_forward_value_relations(source, consumed)
    relations.extend(dead)
    consumed.update(origin for relation in dead for origin in relation["source_ids"])
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
                     "bytes": len(mlir_text.encode()),
                     "serialization": "generic" if path == "fx_importer" else "backend",
                     "operations": operations, "source_correspondence": correspondence},
            "blockers": blockers}
