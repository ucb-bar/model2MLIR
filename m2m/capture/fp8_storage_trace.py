"""Observe exact FP8 storage types, separately from arithmetic qualification."""

from __future__ import annotations

from xdsl.dialects.builtin import ArrayAttr, StringAttr, TensorType

FORMATS = {"float8_e4m3fn": "f8E4M3FN", "float8_e5m2": "f8E5M2"}


def observe_storage(prepared, module, dispositions=None):
    observed = {}
    externs = set()
    opaque = []
    if module is not None:
        for op in module.walk():
            if op.name == "func.func":
                if not list(op.body.blocks):
                    externs.add(op.sym_name.data)
                ids = op.attributes.get("prov.argument_source_node_ids")
                if isinstance(ids, ArrayAttr) and len(ids) == len(
                    op.function_type.inputs
                ):
                    for sid, type_ in zip(ids, op.function_type.inputs):
                        if isinstance(sid, StringAttr):
                            observed.setdefault(sid.data, []).append(type_)
            ids = op.attributes.get("prov.source_node_ids")
            if isinstance(ids, ArrayAttr):
                for sid in ids:
                    if isinstance(sid, StringAttr):
                        observed.setdefault(sid.data, []).extend(
                            v.type for v in op.results
                        )
        for op in module.walk():
            if op.name != "func.call" or op.callee.string_value() not in externs:
                continue
            types = [value.type for value in (*op.operands, *op.results)]
            if any(
                isinstance(t, TensorType) and str(t.element_type) in FORMATS.values()
                for t in types
            ):
                opaque.append(op.callee.string_value())
    values, projections, unknown = [], [], []
    dispositions = dispositions or {}
    result_counts = {
        node["id"]: len(node.get("results", ())) for node in prepared.get("nodes", ())
    }

    def actual_types(sid):
        visited = []
        while True:
            # Node ancestry does not identify an individual result slot. Until
            # an exact value/ordinal map exists, neither a multi-result node nor
            # an alias into one can establish a per-value storage observation.
            if result_counts.get(sid) != 1:
                return observed.get(sid, []), visited, True
            if sid in observed:
                return observed[sid], visited, False
            if sid in visited:
                return [], visited, True
            visited.append(sid)
            alias = dispositions.get(sid, {})
            if (
                alias.get("status") != "alias"
                or type(alias.get("result_index")) is not int
                or alias["result_index"] != 0
            ):
                return [], visited, True
            sid = alias.get("alias_of_node_id")
            if not isinstance(sid, str):
                return [], visited, True

    for node in prepared.get("nodes", ()):
        for value in node.get("results", ()):
            dtype = value.get("storage_dtype") or ""
            if not dtype.startswith("float8_"):
                continue
            expected = FORMATS.get(dtype)
            shape = tuple(value.get("shape") or ())
            mapped_types, aliases, ambiguous = actual_types(node["id"])
            types = [
                t
                for t in mapped_types
                if isinstance(t, TensorType) and t.get_shape() == shape
            ]
            formats = sorted({str(t.element_type) for t in types})
            row = {
                "node_id": node["id"],
                "value_id": value["id"],
                "frontend_dtype": dtype,
                "shape": list(shape),
                "expected_mlir_storage_dtype": expected,
                "observed_mlir_storage_dtypes": formats,
                "storage_type_alias_chain": aliases,
                "source_result_count": result_counts.get(node["id"]),
            }
            if ambiguous or len(formats) > 1:
                unknown.append(
                    row
                    | {
                        "reason": "source result ordinal or competing same-shape storage is unproved"
                    }
                )
            elif expected is not None and formats == [expected]:
                values.append(row | {"status": "preserved"})
            elif formats and expected is not None and "f32" in formats:
                projections.append(
                    row
                    | {
                        "mlir_storage_dtype": "f32",
                        "reason": "actual mapped MLIR value has projected FP32 storage",
                    }
                )
            else:
                unknown.append(
                    row | {"reason": "exact storage format correspondence is unproved"}
                )
    status = (
        "projected"
        if projections
        else "unknown"
        if unknown
        else "preserved"
        if values
        else "no_known_projection"
    )
    return {
        "scope": "actual mapped frontend/MLIR FP8 storage audit",
        "status": status,
        "projections": projections,
        "values": values,
        "unknown_values": unknown,
        "opaque_fp8_calls": sorted(set(opaque)),
        "arithmetic_qualified": False,
    }
