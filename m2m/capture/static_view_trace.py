"""Exact static view correspondence, anchored by already-proved tensor values.

No operator names, tensor shapes or scope names establish a source binding.
The selected ATen alias schema, exact producer result, static view arguments,
metadata, typed edges and unmodified storage version must all agree.  A scope
change additionally requires the existing complete HOP caller-input proof.
"""

from __future__ import annotations

from m2m.capture.view_trace import SourceStorageAliases, _tensor_value

_TARGETS = {
    "aten.transpose.int",
    "aten.permute.default",
    "aten.unsqueeze.default",
    "aten.slice.Tensor",
}
_FIELDS = ("kind", "dtype", "storage_dtype", "compute_dtype", "device", "layout")


def _typed(value):
    return {key: item for key, item in value.items() if key != "id"}


def _static_view(node, producer):
    """Validate the exact static shape/stride/relative-offset view semantics."""
    original, result = _tensor_value(producer), _tensor_value(node)
    if original is None or result is None or node.get("op") != "call_function":
        return None
    args, kwargs = node.get("args"), node.get("kwargs") or {}
    if not isinstance(args, list) or not args or kwargs:
        return None
    if args[0] != {"node_id": producer["id"], "value_id": original.get("id")}:
        return None
    if any(
        field not in original or field not in result or original[field] != result[field]
        for field in _FIELDS
    ):
        return None
    if original["kind"] != "tensor" or original["layout"] != "torch.strided":
        return None
    if any(
        not isinstance(original[field], str) or not original[field]
        for field in ("dtype", "storage_dtype", "device")
    ):
        return None
    shape, stride = original.get("shape"), original.get("stride")
    if (
        not isinstance(shape, list)
        or not shape
        or not isinstance(stride, list)
        or len(shape) != len(stride)
        or any(type(x) is not int or x < 0 for x in [*shape, *stride])
    ):
        return None
    rank = len(shape)
    target, offset = node.get("target"), 0
    if target == "aten.transpose.int" and len(args) == 3:
        if any(type(axis) is not int or not -rank <= axis < rank for axis in args[1:]):
            return None
        order = list(range(rank))
        first, second = (axis % rank for axis in args[1:])
        order[first], order[second] = order[second], order[first]
        key = ("permute", tuple(order))
        expected_shape, expected_stride = (
            [shape[i] for i in order],
            [stride[i] for i in order],
        )
    elif target == "aten.permute.default" and len(args) == 2:
        order = args[1]
        if (
            not isinstance(order, list)
            or len(order) != rank
            or any(type(axis) is not int or not -rank <= axis < rank for axis in order)
            or {axis % rank for axis in order} != set(range(rank))
        ):
            return None
        order = [axis % rank for axis in order]
        key = ("permute", tuple(order))
        expected_shape, expected_stride = (
            [shape[i] for i in order],
            [stride[i] for i in order],
        )
    elif target == "aten.unsqueeze.default" and len(args) == 2:
        axis = args[1]
        if type(axis) is not int or not -rank - 1 <= axis <= rank:
            return None
        axis %= rank + 1
        key = ("insert_axis", axis)
        expected_shape = [*shape[:axis], 1, *shape[axis:]]
        inserted_stride = stride[axis] * shape[axis] if axis < rank else 1
        expected_stride = [*stride[:axis], inserted_stride, *stride[axis:]]
    elif target == "aten.slice.Tensor" and 2 <= len(args) <= 5:
        axis = args[1]
        if type(axis) is not int or not -rank <= axis < rank:
            return None
        axis %= rank
        start = args[2] if len(args) > 2 else None
        end = args[3] if len(args) > 3 else None
        step = args[4] if len(args) > 4 else 1
        if (
            any(x is not None and type(x) is not int for x in (start, end))
            or type(step) is not int
            or step <= 0
        ):
            return None
        start, end, step = slice(start, end, step).indices(shape[axis])
        key = ("slice", axis, start, end, step)
        offset = start * stride[axis]
        expected_shape, expected_stride = list(shape), list(stride)
        expected_shape[axis] = max(0, (end - start + step - 1) // step)
        expected_stride[axis] *= step
    else:
        return None
    if result.get("shape") != expected_shape or result.get("stride") != expected_stride:
        return None
    # Current snapshots do not require absolute offsets. If observed, they must
    # agree with the offset derived from the exact selected operation arguments.
    if ("storage_offset" in original or "storage_offset" in result) and (
        type(original.get("storage_offset")) is not int
        or type(result.get("storage_offset")) is not int
        or original["storage_offset"] < 0
        or result["storage_offset"] != original["storage_offset"] + offset
    ):
        return None
    return key


def _input_edge(graph, node, producer, incoming):
    value = _tensor_value(producer)
    edges = incoming.get(node["id"], ())
    return (
        len(edges) == 1
        and edges[0].get("producer_node_id") == producer["id"]
        and edges[0].get("producer_value_id") == value.get("id")
        and edges[0].get("argument_path") == "args/0"
        and edges[0].get("value_kind") == "tensor"
        and edges[0].get("dtype") == value.get("dtype")
        and edges[0].get("shape") == value.get("shape")
    )


def _selected_alias(aliases, node, producer):
    spec = aliases.schema(node)
    if spec is None or len(spec.returns) != 1 or spec.returns[0].alias_info is None:
        return False
    if any(
        arg.alias_info is not None and arg.alias_info.is_write for arg in spec.arguments
    ):
        return False
    root, parent_root = aliases.root(node), aliases.root(producer)
    return root is not None and root == parent_root


def static_view_relations(source, dest, relations, consumed):
    """Bind live views by exact producer and all observed consumer edges."""
    from m2m.capture.trace import _inlined_parameter_relations

    if not any(
        node["id"] not in consumed and node.get("target") in _TARGETS
        for node in source["nodes"]
    ):
        return []
    aliases, destination_aliases = (
        SourceStorageAliases(source),
        SourceStorageAliases(dest),
    )
    if (
        not aliases.valid
        or not destination_aliases.valid
        or source.get("runtime_versions") != dest.get("runtime_versions")
    ):
        return []
    mapped, incoming, outgoing = {}, {}, {}
    for relation in relations:
        for identity in relation["source_ids"]:
            mapped.setdefault(identity, set()).update(relation["destination_ids"])
    for graph in (source, dest):
        incoming[id(graph)], outgoing[id(graph)] = {}, {}
        for edge in graph.get("edges") or []:
            incoming[id(graph)].setdefault(edge["consumer_node_id"], []).append(edge)
            outgoing[id(graph)].setdefault(edge["producer_node_id"], []).append(edge)
    destinations = {node["id"]: node for node in dest["nodes"]}
    _, verified_scopes = _inlined_parameter_relations(source, dest, relations)
    inferred = []
    for node in source["nodes"]:
        if node["id"] in consumed or node.get("target") not in _TARGETS:
            continue
        args = node.get("args") or []
        producer = aliases.tensor_ref(args[0]) if args else None
        if producer is None or producer.get("graph_id") != node.get("graph_id"):
            continue
        expected = _static_view(node, producer)
        state = aliases.last_writer(node)
        if (
            expected is None
            or not _selected_alias(aliases, node, producer)
            or state is None
            or (state[1] is not None and state[1]["ordinal"] > producer["ordinal"])
            or not _input_edge(source, node, producer, incoming[id(source)])
        ):
            continue
        uses = outgoing[id(source)].get(node["id"], ())
        if not uses:
            continue
        candidates = []
        for owner_id in mapped.get(producer["id"], ()):
            owner = destinations.get(owner_id)
            if (
                owner is None
                or _tensor_value(owner) is None
                or _typed(_tensor_value(owner)) != _typed(_tensor_value(producer))
            ):
                continue
            for input_edge in outgoing[id(dest)].get(owner_id, ()):
                candidate = destinations[input_edge["consumer_node_id"]]
                origin_scope = node["graph_id"].split(":", 2)[-1]
                target_scope = candidate["graph_id"].split(":", 2)[-1]
                inlined = origin_scope != target_scope
                if inlined and not (
                    origin_scope in verified_scopes and target_scope == "root"
                ):
                    continue
                observed = _tensor_value(candidate)
                destination_state = destination_aliases.last_writer(candidate)
                if (
                    candidate.get("graph_id") != owner.get("graph_id")
                    or observed is None
                    or _typed(observed) != _typed(_tensor_value(node))
                    or _static_view(candidate, owner) != expected
                    or not _selected_alias(destination_aliases, candidate, owner)
                    or destination_state is None
                    or (
                        destination_state[1] is not None
                        and destination_state[1]["ordinal"] > owner["ordinal"]
                    )
                    or not _input_edge(dest, candidate, owner, incoming[id(dest)])
                ):
                    continue
                matched = []
                for use in uses:
                    value = _tensor_value(node)
                    if (
                        use.get("producer_value_id") != value["id"]
                        or use.get("value_kind") != "tensor"
                        or use.get("dtype") != value["dtype"]
                        or use.get("shape") != value["shape"]
                    ):
                        break
                    edges = [
                        edge
                        for edge in outgoing[id(dest)].get(candidate["id"], ())
                        if edge.get("consumer_node_id")
                        in mapped.get(use["consumer_node_id"], ())
                        and edge.get("argument_path") == use.get("argument_path")
                        and edge.get("producer_value_id") == observed["id"]
                        and edge.get("value_kind") == "tensor"
                        and edge.get("dtype") == use.get("dtype")
                        and edge.get("shape") == use.get("shape")
                    ]
                    if len(edges) != 1:
                        break
                    matched.append(edges[0])
                if len(matched) == len(uses):
                    candidates.append((candidate, matched, inlined))
        if len(candidates) == 1:
            candidate, uses, inlined = candidates[0]
            inferred.append(
                {
                    "source_ids": [node["id"]],
                    "destination_ids": [candidate["id"]],
                    "kind": "anchored_static_view",
                    "proof": {
                        "input_node_id": producer["id"],
                        "destination_input_node_id": candidate["args"][0]["node_id"],
                        "source_storage_root": state[0],
                        "last_writer_node_id": state[1]["id"] if state[1] else None,
                        "no_write_after_input": True,
                        "exact_static_view": list(expected),
                        "exact_tensor_metadata": True,
                        "typed_consumer_edges": uses,
                        "exact_caller_bindings": inlined,
                        "scope": "forward tensor view; not arbitrary mutation equivalence",
                    },
                }
            )
    # A destination view must not establish two otherwise indistinguishable source calls.
    counts = {}
    for row in inferred:
        identity = row["destination_ids"][0]
        counts[identity] = counts.get(identity, 0) + 1
    return [row for row in inferred if counts[row["destination_ids"][0]] == 1]


def dead_static_view_relations(source, consumed):
    """Discard only valid, statically bounded, unused, schema-pure views."""
    if not any(
        node["id"] not in consumed and node.get("target") in _TARGETS
        for node in source["nodes"]
    ):
        return []
    aliases = SourceStorageAliases(source)
    if not aliases.valid:
        return []
    incoming, outgoing = {}, {}
    for edge in source.get("edges") or []:
        incoming.setdefault(edge["consumer_node_id"], []).append(edge)
        outgoing.setdefault(edge["producer_node_id"], []).append(edge)
    result = []
    for node in source["nodes"]:
        if (
            node["id"] in consumed
            or node.get("target") not in _TARGETS
            or outgoing.get(node["id"])
        ):
            continue
        args = node.get("args") or []
        producer = aliases.tensor_ref(args[0]) if args else None
        if producer is None or producer.get("graph_id") != node.get("graph_id"):
            continue
        key = _static_view(node, producer)
        if (
            key is None
            or not _selected_alias(aliases, node, producer)
            or aliases.last_writer(node) is None
            or not _input_edge(source, node, producer, incoming)
        ):
            continue
        result.append(
            {
                "source_ids": [node["id"]],
                "destination_ids": [],
                "kind": "eliminated",
                "reason": "unused valid static ATen view with exact alias schema and no observable consumers",
                "proof": {
                    "operator": node["target"],
                    "exact_static_view": list(key),
                    "source_storage_root": aliases.root(node),
                    "source_users": [],
                    "valid_static_shape_stride": True,
                    "no_write_effects": True,
                },
            }
        )
    return result
