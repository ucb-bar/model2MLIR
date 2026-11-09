"""Read-only static aliases of an exactly bound functionalized tensor write.

A decomposition may replay a pure view independently for different readers.
The proof below follows actual schema/argument/edge paths from one bound write;
matching shape or an FX name never establishes an input/storage version.
"""

from __future__ import annotations

import math
import struct

from m2m.capture.static_view_trace import _input_edge, _static_view, _typed
from m2m.capture.trace import _forward_identity_value
from m2m.capture.view_trace import (
    SourceStorageAliases,
    _tensor_value,
    unchanged_through_uses,
)

_VIEWS = {
    "aten.to.dtype",
    "aten.view.default",
    "aten.reshape.default",
    "aten.permute.default",
    "aten.transpose.int",
    "aten.select.int",
    "aten.unsqueeze.default",
    "aten.slice.Tensor",
    "aten.expand.default",
}
_FIELDS = ("kind", "dtype", "storage_dtype", "compute_dtype", "device", "layout")


def _shape(value):
    shape, stride = value.get("shape"), value.get("stride")
    return (
        value.get("kind") == "tensor"
        and value.get("layout") == "torch.strided"
        and isinstance(shape, list)
        and isinstance(stride, list)
        and len(shape) == len(stride)
        and bool(shape)
        and all(type(dim) is int and dim > 0 for dim in shape)
        and all(type(dim) is int and dim >= 0 for dim in stride)
        and all(field in value for field in _FIELDS)
        and all(
            isinstance(value[field], str) and value[field]
            for field in ("dtype", "storage_dtype", "device")
        )
        and value["dtype"] == value["storage_dtype"]
        and (value["compute_dtype"] is None or isinstance(value["compute_dtype"], str))
    )


def _runs(shape, strides):
    """Canonical lexicographic address runs, excluding singleton dimensions."""
    runs = []
    for dim, stride in zip(shape, strides, strict=True):
        if dim == 1:
            continue
        if runs and runs[-1][1] == dim * stride:
            runs[-1] = (runs[-1][0] * dim, stride)
        else:
            runs.append((dim, stride))
    return runs


def _projection(graph, aliases, node, parent, offset, incoming):
    """Derive every result address relative to a supplied exact tensor owner."""
    original, result = _tensor_value(parent), _tensor_value(node)
    if (
        original is None
        or result is None
        or not _shape(original)
        or not _shape(result)
        or node.get("op") != "call_function"
        or node.get("target") not in _VIEWS
        or node.get("graph_id") != parent.get("graph_id")
        or any(original[field] != result[field] for field in _FIELDS)
        or not _input_edge(graph, node, parent, incoming)
    ):
        return None
    schema = aliases.schema(node)
    if (
        schema is None
        or len(schema.returns) != 1
        or schema.returns[0].alias_info is None
        or any(
            arg.alias_info is not None and arg.alias_info.is_write
            for arg in schema.arguments
        )
        or aliases.root(node) != aliases.root(parent)
    ):
        return None
    shape, stride = original["shape"], original["stride"]
    args, target = node.get("args") or [], node["target"]
    relative = 0
    if target == "aten.to.dtype":
        if _forward_identity_value(graph, node) is None:
            return None
    elif target in {"aten.view.default", "aten.reshape.default"}:
        requested = args[1] if len(args) == 2 and not node.get("kwargs") else None
        if (
            not isinstance(requested, list)
            or not requested
            or any(type(dim) is not int or dim <= 0 for dim in requested)
            or result["shape"] != requested
            or math.prod(requested) != math.prod(shape)
            or _runs(shape, stride) != _runs(requested, result["stride"])
        ):
            return None
    elif target == "aten.expand.default":
        # Only an explicitly identical shape, not broadcasting or a -1 hint.
        if (
            len(args) != 2
            or node.get("kwargs")
            or args[1] != shape
            or result["shape"] != shape
            or result["stride"] != stride
        ):
            return None
    elif target == "aten.select.int":
        if len(args) != 3 or node.get("kwargs"):
            return None
        axis, index = args[1:]
        if type(axis) is not int or not -len(shape) <= axis < len(shape):
            return None
        axis %= len(shape)
        if type(index) is not int or not -shape[axis] <= index < shape[axis]:
            return None
        index %= shape[axis]
        if result["shape"] != [*shape[:axis], *shape[axis + 1 :]] or result[
            "stride"
        ] != [*stride[:axis], *stride[axis + 1 :]]:
            return None
        relative = index * stride[axis]
    else:
        key = _static_view(node, parent)
        if key is None:
            return None
        if key[0] == "slice":
            relative = key[2] * stride[key[1]]
    if ("storage_offset" in original or "storage_offset" in result) and (
        type(original.get("storage_offset")) is not int
        or type(result.get("storage_offset")) is not int
        or original["storage_offset"] < 0
        or result["storage_offset"] != original["storage_offset"] + relative
    ):
        return None
    return offset + relative


def _scalar_control(value):
    """Retain the exact Torch Scalar kind and bits without evaluating objects."""
    if type(value) is int and -(1 << 63) <= value < (1 << 63):
        return ("int64", value.to_bytes(8, "little", signed=True))
    if type(value) is float and math.isfinite(value):
        return ("float64", struct.pack("<d", value))
    return None


def _bound_add(source, dest, writer, candidate, mapped, source_aliases, dest_aliases):
    """Exact ordinary add operands and schema, with one fresh target result."""
    import torch

    if (
        writer.get("target") != "aten.add_.Tensor"
        or candidate.get("target") != "aten.add.Tensor"
        or writer.get("op") != "call_function"
        or candidate.get("op") != "call_function"
        or writer.get("graph_id", "").split(":", 2)[-1] != "root"
        or candidate.get("graph_id", "").split(":", 2)[-1] != "root"
        or source_aliases.schema(writer) != torch.ops.aten.add_.Tensor._schema
        or dest_aliases.schema(candidate) != torch.ops.aten.add.Tensor._schema
        or _tensor_value(writer) is None
        or _tensor_value(candidate) is None
        or _typed(_tensor_value(writer)) != _typed(_tensor_value(candidate))
        or not _shape(_tensor_value(writer))
        or _runs(_tensor_value(writer)["shape"], _tensor_value(writer)["stride"])
        != (
            [(math.prod(_tensor_value(writer)["shape"]), 1)]
            if math.prod(_tensor_value(writer)["shape"]) > 1
            else []
        )
        or dest_aliases.root(candidate) != candidate["id"]
    ):
        return False
    left, right = writer.get("args"), candidate.get("args")
    if (
        not isinstance(left, list)
        or not isinstance(right, list)
        or len(left) != 2
        or len(right) != 2
    ):
        return False
    for node in (writer, candidate):
        options = node.get("kwargs") or {}
        if set(options) - {"alpha"} or type(options.get("alpha", 1)) not in {
            int,
            float,
        }:
            return False
    before_alpha = _scalar_control((writer.get("kwargs") or {}).get("alpha", 1))
    after_alpha = _scalar_control((candidate.get("kwargs") or {}).get("alpha", 1))
    if before_alpha is None or after_alpha is None or before_alpha != after_alpha:
        return False
    for before, after in zip(left, right, strict=True):
        original = source_aliases.tensor_ref(before)
        observed = dest_aliases.tensor_ref(after)
        if (
            original is None
            or observed is None
            or observed["id"] not in mapped.get(original["id"], ())
            or _tensor_value(original) is None
            or _tensor_value(observed) is None
            or _typed(_tensor_value(original)) != _typed(_tensor_value(observed))
        ):
            return False
        for owners, operand, boundary in (
            (source_aliases, original, writer),
            (dest_aliases, observed, candidate),
        ):
            initial = owners.last_writer(operand)
            current = owners.last_writer_at(operand, boundary)
            version = lambda state: (
                (state[0], state[1]["id"] if state[1] else None) if state else None
            )
            schema = owners.schema(operand)
            if (
                initial is None
                or current is None
                or version(initial) != version(current)
                or (
                    schema is not None
                    and any(
                        arg.alias_info is not None and arg.alias_info.is_write
                        for arg in schema.arguments
                    )
                )
            ):
                return False
    # The exact operand edges are observations separate from encoded FX args.
    for graph, node in ((source, writer), (dest, candidate)):
        edges = [
            edge
            for edge in graph.get("edges", ())
            if edge.get("consumer_node_id") == node["id"]
        ]
        if len(edges) != 2:
            return False
        for index, ref in enumerate(node["args"]):
            owners = source_aliases if graph is source else dest_aliases
            value = _tensor_value(owners.tensor_ref(ref))
            expected = {
                "producer_node_id": ref["node_id"],
                "producer_value_id": ref["value_id"],
                "consumer_node_id": node["id"],
                "argument_path": f"args/{index}",
                "value_kind": "tensor",
                "dtype": value["dtype"],
                "shape": value["shape"],
            }
            if (
                sum(
                    all(edge.get(key) == item for key, item in expected.items())
                    for edge in edges
                )
                != 1
            ):
                return False
    return True


def _alias_paths(graph, aliases, root, incoming, outgoing):
    """Every admitted path has one unchanged storage version and exact edges."""
    state = aliases.last_writer(root)
    schema = aliases.schema(root)
    if schema is not None and any(
        arg.alias_info is not None and arg.alias_info.is_write
        for arg in schema.arguments
    ):
        return {}
    if state is None or not unchanged_through_uses(
        aliases, root, outgoing.get(root["id"], ())
    ):
        return {}
    version = (state[0], state[1]["id"] if state[1] else None)
    paths = {root["id"]: (0, [])}
    for node in graph["nodes"]:
        args = node.get("args") or []
        parent = aliases.tensor_ref(args[0]) if args else None
        if parent is None or parent["id"] not in paths:
            continue
        prior_offset, prior_path = paths[parent["id"]]
        offset = _projection(graph, aliases, node, parent, prior_offset, incoming)
        current = aliases.last_writer(node)
        if (
            offset is not None
            and current is not None
            and (current[0], current[1]["id"] if current[1] else None) == version
            and unchanged_through_uses(aliases, node, outgoing.get(node["id"], ()))
        ):
            paths[node["id"]] = (offset, [*prior_path, node["id"]])
    return paths


def _matmul_use(
    source,
    dest,
    source_aliases,
    dest_aliases,
    use,
    candidate,
    matches,
    mapped,
    incoming,
    outgoing,
):
    """Exact equal-batch matmul -> flatten-batch BMM view decomposition.

    The input slot is proved at the actual BMM, never inferred from a prefix
    view's args/0. Broadcasting, copying, dtype changes and arbitrary consumer
    decompositions are unsupported by this deliberately narrow rule.
    """
    import torch

    consumer = source_aliases.nodes.get(use.get("consumer_node_id"))
    if (
        consumer is None
        or consumer.get("op") != "call_function"
        or consumer.get("target") != "aten.matmul.default"
        or source_aliases.schema(consumer) != torch.ops.aten.matmul.default._schema
        or len(consumer.get("args") or []) != 2
        or consumer.get("kwargs")
        or use.get("argument_path") not in {"args/0", "args/1"}
    ):
        return None
    operands = [source_aliases.tensor_ref(ref) for ref in consumer["args"]]
    values = [_tensor_value(node) if node else None for node in operands]
    result = _tensor_value(consumer)
    if any(value is None or not _shape(value) for value in [*values, result]):
        return None
    a, b = values
    if (
        len(a["shape"]) < 3
        or len(b["shape"]) != len(a["shape"])
        or a["shape"][:-2] != b["shape"][:-2]
        or a["shape"][-1] != b["shape"][-2]
        or result["shape"] != [*a["shape"][:-2], a["shape"][-2], b["shape"][-1]]
        or any(
            any(value[field] != a[field] for field in _FIELDS) for value in [b, result]
        )
    ):
        return None
    batch = math.prod(a["shape"][:-2])
    expected_inputs = [[batch, *a["shape"][-2:]], [batch, *b["shape"][-2:]]]
    expected_output = [batch, a["shape"][-2], b["shape"][-1]]
    bmms = [
        dest_aliases.nodes[identity]
        for identity in mapped.get(consumer["id"], ())
        if identity in dest_aliases.nodes
        and dest_aliases.nodes[identity].get("target") == "aten.bmm.default"
    ]
    if len(bmms) != 1:
        return None
    bmm = bmms[0]
    if (
        bmm.get("op") != "call_function"
        or dest_aliases.schema(bmm) != torch.ops.aten.bmm.default._schema
        or len(bmm.get("args") or []) != 2
        or bmm.get("kwargs")
        or _tensor_value(bmm) is None
        or not _shape(_tensor_value(bmm))
        or _tensor_value(bmm).get("shape") != expected_output
        or any(_tensor_value(bmm).get(field) != result[field] for field in _FIELDS)
        or _runs(_tensor_value(bmm)["shape"], _tensor_value(bmm)["stride"])
        != _runs(result["shape"], result["stride"])
    ):
        return None
    slot = int(use["argument_path"][-1])
    input_paths = []
    for index, original in enumerate(operands):
        endpoint = dest_aliases.tensor_ref(bmm["args"][index])
        if (
            endpoint is None
            or _tensor_value(endpoint) is None
            or not _shape(_tensor_value(endpoint))
            or _tensor_value(endpoint).get("shape") != expected_inputs[index]
            or _runs(values[index]["shape"], values[index]["stride"])
            != _runs(
                _tensor_value(endpoint)["shape"], _tensor_value(endpoint)["stride"]
            )
        ):
            return None
        owned = original
        source_identity_path = []
        starts = (
            {candidate}
            if index == slot
            else matches.get(owned["id"], mapped.get(owned["id"], set()))
        )
        while not starts and owned["id"] not in source_identity_path:
            identity = _forward_identity_value(source, owned)
            source_uses = [
                edge
                for edge in source.get("edges", ())
                if edge.get("producer_node_id") == owned["id"]
            ]
            version = source_aliases.last_writer(owned)
            if (
                identity is None
                or version is None
                or (
                    version[1] is not None
                    and version[1]["ordinal"] > identity[0]["ordinal"]
                )
                or not unchanged_through_uses(source_aliases, owned, source_uses)
            ):
                break
            source_identity_path.append(owned["id"])
            owned = identity[0]
            starts = matches.get(owned["id"], mapped.get(owned["id"], set()))
        paths = []
        for identity in sorted(starts):
            start = dest_aliases.nodes.get(identity)
            if (
                start is None
                or _tensor_value(start) is None
                or _typed(_tensor_value(start)) != _typed(values[index])
            ):
                continue
            projected = _alias_paths(dest, dest_aliases, start, incoming, outgoing)
            witness = projected.get(endpoint["id"])
            if witness is None or witness[0] != 0:
                continue
            edge = {
                "producer_node_id": endpoint["id"],
                "producer_value_id": _tensor_value(endpoint)["id"],
                "consumer_node_id": bmm["id"],
                "argument_path": f"args/{index}",
                "value_kind": "tensor",
                "dtype": values[index]["dtype"],
                "shape": expected_inputs[index],
            }
            actual = [
                item
                for item in outgoing.get(endpoint["id"], ())
                if all(item.get(key) == value for key, value in edge.items())
            ]
            if len(actual) != 1:
                continue
            paths.append(
                {
                    "prepared_start": identity,
                    "source_identity_path": source_identity_path,
                    "alias_path": witness[1],
                    "BMM_edge": actual[0],
                }
            )
        if not paths:
            return None
        input_paths.append(paths)
    projected_outputs = _alias_paths(dest, dest_aliases, bmm, incoming, outgoing)
    outputs = [
        {"prepared_output": identity, "alias_path": path}
        for identity, (offset, path) in projected_outputs.items()
        if offset == 0
        and identity in mapped.get(consumer["id"], ())
        and _typed(_tensor_value(dest_aliases.nodes[identity])) == _typed(result)
    ]
    if not outputs:
        return None
    return {
        "source_consumer": consumer["id"],
        "prepared_BMM": bmm["id"],
        "source_argument_slot": slot,
        "equal_batch_shape": a["shape"][:-2],
        "input_alias_paths": input_paths,
        "output_alias_paths": outputs,
        "exact_schema_ordered_BMM_operands_and_batch_reshape": True,
    }


def functional_view_relations(source, dest, relations, consumed):
    """Prove duplicated read-only view values from a bound functionalized add.

    Source call positions remain distinct. A single pure view value may have
    multiple prepared uses only through explicit, bounded alias paths from the
    exact functionalized write and typed consumer edges. No numerical operation
    or arbitrary mutation equivalence is inferred by this view proof.
    """
    if source.get("runtime_versions") != dest.get("runtime_versions"):
        return []
    source_aliases, dest_aliases = (
        SourceStorageAliases(source),
        SourceStorageAliases(dest),
    )
    if not source_aliases.valid or not dest_aliases.valid:
        return []
    mapped, outgoing, incoming = {}, {}, {}
    for row in relations:
        for identity in row["source_ids"]:
            mapped.setdefault(identity, set()).update(row["destination_ids"])
    for graph in (source, dest):
        outgoing[id(graph)], incoming[id(graph)] = {}, {}
        for edge in graph.get("edges", ()):
            outgoing[id(graph)].setdefault(edge["producer_node_id"], []).append(edge)
            incoming[id(graph)].setdefault(edge["consumer_node_id"], []).append(edge)
    inferred = []
    for writer in source["nodes"]:
        if writer.get("target") != "aten.add_.Tensor":
            continue
        anchors = [
            dest_aliases.nodes[identity]
            for identity in mapped.get(writer["id"], ())
            if identity in dest_aliases.nodes
            and _bound_add(
                source,
                dest,
                writer,
                dest_aliases.nodes[identity],
                mapped,
                source_aliases,
                dest_aliases,
            )
        ]
        if len(anchors) != 1:
            continue
        anchor = anchors[0]
        source_paths = {writer["id"]: (0, [])}
        dest_paths = {anchor["id"]: (0, [])}
        for graph, aliases, paths, root, version in (
            (source, source_aliases, source_paths, writer, writer["id"]),
            (dest, dest_aliases, dest_paths, anchor, None),
        ):
            for node in graph["nodes"]:
                args = node.get("args") or []
                parent = aliases.tensor_ref(args[0]) if args else None
                if parent is None or parent["id"] not in paths:
                    continue
                prior_offset, prior_path = paths[parent["id"]]
                offset = _projection(
                    graph, aliases, node, parent, prior_offset, incoming[id(graph)]
                )
                state = aliases.last_writer(node)
                uses = outgoing[id(graph)].get(node["id"], ())
                if (
                    offset is None
                    or state is None
                    or state[0] != aliases.root(root)
                    or (state[1]["id"] if state[1] else None) != version
                    or not unchanged_through_uses(aliases, node, uses)
                ):
                    continue
                # All accessed element positions remain inside this anchor's
                # contiguous tensor span, including nonzero select/slice offsets.
                value = _tensor_value(node)
                highest = offset + sum(
                    (dim - 1) * stride
                    for dim, stride in zip(value["shape"], value["stride"], strict=True)
                )
                if highest >= math.prod(_tensor_value(root)["shape"]):
                    continue
                paths[node["id"]] = (offset, [*prior_path, node["id"]])
        source_ids = [
            identity
            for identity in source_paths
            if identity not in consumed and identity != writer["id"]
        ]
        if not source_ids:
            continue
        matches = {}
        for identity in source_ids:
            node = source_aliases.nodes[identity]
            matches[identity] = {
                candidate
                for candidate, (offset, _) in dest_paths.items()
                if source_paths[identity][0] == offset
                and _typed(_tensor_value(node))
                == _typed(_tensor_value(dest_aliases.nodes[candidate]))
            }
        if any(not candidates for candidates in matches.values()):
            continue
        # Keep only actual observed reader edges. Every source use must have an
        # exact argument-path witness; each prepared value must participate in
        # at least one such edge. Repeat until internal chain pruning is stable.
        witnesses = {}
        while True:
            retained, witnesses, complete = {}, {}, True
            for identity, candidates in matches.items():
                uses = outgoing[id(source)].get(identity, ())
                values, found = set(), []
                for use in uses:
                    targets = matches.get(
                        use["consumer_node_id"],
                        mapped.get(use["consumer_node_id"], set()),
                    )
                    edges = [
                        edge
                        for candidate in sorted(candidates)
                        for edge in outgoing[id(dest)].get(candidate, ())
                        if edge.get("consumer_node_id") in targets
                        and edge.get("argument_path") == use.get("argument_path")
                        and edge.get("value_kind") == use.get("value_kind") == "tensor"
                        and edge.get("dtype") == use.get("dtype")
                        and edge.get("shape") == use.get("shape")
                        and edge.get("producer_value_id")
                        == _tensor_value(dest_aliases.nodes[candidate])["id"]
                    ]
                    consumer = source_aliases.nodes.get(use["consumer_node_id"])
                    if (
                        consumer is not None
                        and consumer.get("target") == "aten.matmul.default"
                    ):
                        # A generated prefix view's args/0 is not the original
                        # matmul input slot. Only the complete BMM proof grants it.
                        edges = []
                    decomposed = []
                    for candidate in sorted(candidates):
                        proof = _matmul_use(
                            source,
                            dest,
                            source_aliases,
                            dest_aliases,
                            use,
                            candidate,
                            matches,
                            mapped,
                            incoming[id(dest)],
                            outgoing[id(dest)],
                        )
                        if proof is not None:
                            decomposed.append(
                                {"prepared_start": candidate, "proof": proof}
                            )
                    if not edges and not decomposed:
                        complete = False
                        break
                    values.update(edge["producer_node_id"] for edge in edges)
                    values.update(item["prepared_start"] for item in decomposed)
                    found.append(
                        {
                            "source_edge": use,
                            "prepared_edges": edges,
                            "matmul_decomposition_edges": decomposed,
                        }
                    )
                retained[identity] = values
                witnesses[identity] = found
                complete &= bool(uses) and bool(values)
            if not complete:
                matches = {}
                break
            if retained == matches:
                break
            matches = retained
        if not matches:
            continue
        for identity, candidates in matches.items():
            inferred.append(
                {
                    "source_ids": [identity],
                    "destination_ids": sorted(candidates),
                    "kind": "functionalized_static_view_value",
                    "proof": {
                        "source_write_id": writer["id"],
                        "prepared_functional_add_id": anchor["id"],
                        "exact_ordered_write_operands": True,
                        "source_view_path": source_paths[identity][1],
                        "prepared_view_paths": {
                            candidate: dest_paths[candidate][1]
                            for candidate in sorted(candidates)
                        },
                        "relative_element_offset": source_paths[identity][0],
                        "exact_schema_shape_stride_and_bounds": True,
                        "unchanged_source_and_prepared_consumer_lifetimes": True,
                        "typed_consumer_edges": witnesses[identity],
                        "scope": "same read-only tensor view value of an exactly bound functionalized add; not arbitrary mutation or numerical equivalence",
                    },
                }
            )
    return inferred
