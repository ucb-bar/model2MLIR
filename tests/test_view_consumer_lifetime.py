"""Views alias storage through every read, not only at creation."""

import copy

import m2m
import pytest
import torch
from m2m.capture.static_view_trace import static_view_relations
from m2m.capture.trace import graph_relation
from m2m.capture.view_trace import SourceStorageAliases, replayed_view_relations


class View(torch.nn.Module):
    def forward(self, x):
        with torch.no_grad():
            value = x * 2
            return value.transpose(1, 2).contiguous() + 1


@pytest.fixture(scope="module")
def static_capture():
    result = m2m.convert(
        View(),
        (torch.arange(36, dtype=torch.float32).reshape(1, 6, 6),),
        backend="fx_importer",
        capture_trace=True,
    )
    assert result.ok
    source, dest = (
        result.capture_trace["graphs"]["quantized"],
        result.capture_trace["graphs"]["prepared"],
    )
    view = next(
        node for node in source["nodes"] if node["target"] == "aten.transpose.int"
    )
    relation = graph_relation(source, dest)
    premises = [
        row for row in relation["relations"] if view["id"] not in row["source_ids"]
    ]
    consumed = {identity for row in premises for identity in row["source_ids"]}
    rows = static_view_relations(source, dest, premises, consumed)
    assert len(rows) == 1
    return source, dest, premises, consumed, view["id"], rows[0]["destination_ids"][0]


def _node(graph, identity):
    return next(node for node in graph["nodes"] if node["id"] == identity)


def _uses(graph, identity):
    return [edge for edge in graph["edges"] if edge["producer_node_id"] == identity]


def _duplicate_reader(graph, view_id):
    edge = _uses(graph, view_id)[0]
    reader = _node(graph, edge["consumer_node_id"])
    duplicate = copy.deepcopy(reader)
    duplicate["id"] += "_second_read"
    duplicate["ordinal"] = reader["ordinal"] + 1
    duplicate["results"][0]["id"] = duplicate["id"] + ":v0"
    for node in graph["nodes"]:
        if (
            node["ordinal"] > reader["ordinal"]
            and node["graph_id"] == reader["graph_id"]
        ):
            node["ordinal"] += 2
    graph["nodes"].insert(graph["nodes"].index(reader) + 1, duplicate)
    new_edge = copy.deepcopy(edge)
    new_edge["consumer_node_id"] = duplicate["id"]
    graph["edges"].append(new_edge)
    return duplicate


def _write_before(graph, view_id, boundary, *, through_view=False, unknown=False):
    view = _node(graph, view_id)
    owner = view if through_view else _node(graph, view["args"][0]["node_id"])
    for node in graph["nodes"]:
        node["ordinal"] *= 2
    writer = copy.deepcopy(owner)
    writer.update(
        id=owner["id"] + "_late_write",
        ordinal=boundary["ordinal"] - 1,
        op="call_function",
        target="unknown_storage_effect.write.default"
        if unknown
        else "aten.copy_.default",
        kwargs={},
        origin_node_ids=[],
    )
    value = owner["results"][0]
    ref = {"node_id": owner["id"], "value_id": value["id"]}
    writer["args"] = [ref] if unknown else [ref, copy.deepcopy(ref)]
    writer["results"][0]["id"] = writer["id"] + ":v0"
    if not unknown:
        graph["operator_schemas"][writer["target"]] = str(
            torch.ops.aten.copy_.default._schema
        )
    graph["nodes"].insert(graph["nodes"].index(boundary), writer)
    for index in range(len(writer["args"])):
        graph["edges"].append(
            {
                "producer_node_id": owner["id"],
                "producer_value_id": value["id"],
                "consumer_node_id": writer["id"],
                "argument_path": f"args/{index}",
                "dtype": value["dtype"],
                "shape": value["shape"],
                "value_kind": "tensor",
            }
        )
    assert view["ordinal"] < writer["ordinal"] < boundary["ordinal"]
    return writer


@pytest.mark.parametrize("side", ["source", "destination"])
@pytest.mark.parametrize("kind", ["owner", "alias", "unknown"])
def test_write_after_view_before_first_reader_refuses(side, kind, static_capture):
    source, dest, premises, consumed, source_id, dest_id = static_capture
    source, dest = copy.deepcopy(source), copy.deepcopy(dest)
    graph, identity = (source, source_id) if side == "source" else (dest, dest_id)
    boundary = _node(graph, _uses(graph, identity)[0]["consumer_node_id"])
    _write_before(
        graph,
        identity,
        boundary,
        through_view=kind == "alias",
        unknown=kind == "unknown",
    )
    assert static_view_relations(source, dest, premises, consumed) == []


def test_multiple_readers_without_intervening_write_remain_proved(static_capture):
    source, dest, premises, consumed, source_id, dest_id = static_capture
    source, dest, premises = (
        copy.deepcopy(source),
        copy.deepcopy(dest),
        copy.deepcopy(premises),
    )
    a, b = _duplicate_reader(source, source_id), _duplicate_reader(dest, dest_id)
    premises.append(
        {"source_ids": [a["id"]], "destination_ids": [b["id"]], "kind": "identity"}
    )
    rows = static_view_relations(source, dest, premises, consumed)
    assert len(rows) == 1
    assert len(rows[0]["proof"]["typed_consumer_edges"]) == 2


@pytest.mark.parametrize("side", ["source", "destination"])
def test_write_between_multiple_readers_refuses_even_after_first_read(
    side, static_capture
):
    source, dest, premises, consumed, source_id, dest_id = static_capture
    source, dest, premises = (
        copy.deepcopy(source),
        copy.deepcopy(dest),
        copy.deepcopy(premises),
    )
    a, b = _duplicate_reader(source, source_id), _duplicate_reader(dest, dest_id)
    premises.append(
        {"source_ids": [a["id"]], "destination_ids": [b["id"]], "kind": "identity"}
    )
    graph, identity, boundary = (
        (source, source_id, a) if side == "source" else (dest, dest_id, b)
    )
    _write_before(graph, identity, boundary)
    assert static_view_relations(source, dest, premises, consumed) == []


def test_mapping_a_late_write_does_not_make_old_view_version_valid(static_capture):
    source, dest, premises, consumed, source_id, dest_id = static_capture
    source, dest, premises = (
        copy.deepcopy(source),
        copy.deepcopy(dest),
        copy.deepcopy(premises),
    )
    source_boundary = _node(source, _uses(source, source_id)[0]["consumer_node_id"])
    destination_boundary = _node(dest, _uses(dest, dest_id)[0]["consumer_node_id"])
    a = _write_before(source, source_id, source_boundary)
    b = _write_before(dest, dest_id, destination_boundary)
    premises.append(
        {"source_ids": [a["id"]], "destination_ids": [b["id"]], "kind": "identity"}
    )
    consumed = {*consumed, a["id"]}
    assert static_view_relations(source, dest, premises, consumed) == []


def test_storage_version_query_cannot_cross_capture_objects(static_capture):
    source, _, _, _, identity, _ = static_capture
    view = _node(source, identity)
    boundary = _node(source, _uses(source, identity)[0]["consumer_node_id"])
    aliases = SourceStorageAliases(source)
    assert aliases.last_writer_at(view, boundary) is not None
    assert aliases.last_writer_at(copy.deepcopy(view), boundary) is None
    assert aliases.last_writer_at(view, copy.deepcopy(boundary)) is None


class UpdatedBufferViews(torch.nn.Module):
    def forward(self, x):
        value = torch.empty_like(x)
        value[..., :4] = x[..., :4] * 2
        value[..., 4:] = x[..., 4:] * 3
        expanded = value.to(torch.float32).unsqueeze(3).expand(1, 7, 3, 2, 8)
        lhs = expanded.reshape(1, 7, 6, 8).transpose(1, 2)
        return torch.matmul(lhs, lhs.transpose(2, 3))


@pytest.mark.parametrize("side", ["source", "destination"])
def test_existing_replayed_writer_path_also_rejects_later_mutation(side):
    converted = m2m.convert(
        UpdatedBufferViews(),
        (torch.randn(1, 7, 3, 8),),
        backend="fx_importer",
        capture_trace=True,
    )
    assert converted.ok
    source, dest = (
        converted.capture_trace["graphs"]["quantized"],
        converted.capture_trace["graphs"]["prepared"],
    )
    view = next(
        node for node in source["nodes"] if node["target"] == "aten.unsqueeze.default"
    )
    relation = graph_relation(source, dest)
    premises = [
        row for row in relation["relations"] if view["id"] not in row["source_ids"]
    ]
    consumed = {identity for row in premises for identity in row["source_ids"]}
    positive = replayed_view_relations(source, dest, premises, consumed)
    assert len(positive) == 1
    source, dest = copy.deepcopy(source), copy.deepcopy(dest)
    graph, identity = (
        (source, view["id"])
        if side == "source"
        else (dest, positive[0]["destination_ids"][0])
    )
    reader = _node(graph, _uses(graph, identity)[0]["consumer_node_id"])
    _write_before(graph, identity, reader)
    assert replayed_view_relations(source, dest, premises, consumed) == []


@pytest.mark.parametrize("side", ["source", "destination"])
@pytest.mark.parametrize(
    "kind",
    ["call_module", "call_method", "unknown", "unknown_schema", "mutating_schema"],
)
def test_unproved_or_mutating_consumer_kind_refuses(side, kind, static_capture):
    source, dest, premises, consumed, source_id, dest_id = static_capture
    source, dest = copy.deepcopy(source), copy.deepcopy(dest)
    graph, identity = (source, source_id) if side == "source" else (dest, dest_id)
    reader = _node(graph, _uses(graph, identity)[0]["consumer_node_id"])
    if kind == "unknown_schema":
        reader.update(
            op="call_function", target="unregistered_consumer.consume.default"
        )
    elif kind == "mutating_schema":
        reader.update(op="call_function", target="aten.copy_.default")
        graph["operator_schemas"][reader["target"]] = str(
            torch.ops.aten.copy_.default._schema
        )
    else:
        reader.update(op=kind, target="unproved_consumer")
    assert static_view_relations(source, dest, premises, consumed) == []


def test_only_canonical_output_is_a_structural_consumer():
    from m2m.capture.view_trace import unchanged_through_uses

    class ReturnView(torch.nn.Module):
        def forward(self, x):
            return x.transpose(1, 2)

    converted = m2m.convert(
        ReturnView(), (torch.zeros(1, 2, 3),), backend="fx_importer", capture_trace=True
    )
    assert converted.ok
    source = converted.capture_trace["graphs"]["quantized"]
    view = next(
        node for node in source["nodes"] if node["target"] == "aten.transpose.int"
    )
    uses = _uses(source, view["id"])
    assert len(uses) == 1
    assert unchanged_through_uses(SourceStorageAliases(source), view, uses)
    output = _node(source, uses[0]["consumer_node_id"])
    assert output["op"] == output["target"] == "output"
    output["target"] = "unproved_side_effect"
    assert not unchanged_through_uses(SourceStorageAliases(source), view, uses)
