"""Tuple tensor storage is owned by an exact schema return, never by shape."""

import copy

import pytest
import torch

from m2m.capture.trace import snapshot_exported_program
from m2m.capture.view_trace import SourceStorageAliases


class Sort(torch.nn.Module):
    def forward(self, x):
        values, indices = torch.sort(x)
        return values.to(torch.float32), indices


@pytest.fixture
def source():
    exported = torch.export.export(Sort(), (torch.tensor([[3.0, 1.0, 2.0]]),))
    source = snapshot_exported_program(exported, stage="quantized")
    node = next(
        n for n in source["nodes"] if n["target"] == "<built-in function getitem>"
    )
    return source, node["id"]


def test_actual_schema_owned_tuple_return_has_a_storage_root(source):
    graph, identity = source
    aliases = SourceStorageAliases(graph)
    node = aliases.nodes[identity]
    assert aliases.root(node) == node["args"][0]["node_id"] + ":tuple_slot:0"
    assert aliases.last_writer(node) == (aliases.root(node), None)


@pytest.mark.parametrize(
    "change",
    [
        "target",
        "slot_bool",
        "slot_float",
        "slot_range",
        "source",
        "value",
        "scope",
        "ordinal",
        "dtype",
        "shape",
        "edge",
        "schema",
        "kwargs",
    ],
)
def test_missing_or_contradictory_tuple_premises_decline(source, change):
    graph, identity = source
    graph = copy.deepcopy(graph)
    aliases = SourceStorageAliases(graph)
    node = aliases.nodes[identity]
    parent = aliases.nodes[node["args"][0]["node_id"]]
    if change == "target":
        node["target"] = "other_getitem"
    elif change == "slot_bool":
        node["args"][1] = True
    elif change == "slot_float":
        node["args"][1] = 0.0
    elif change == "slot_range":
        node["args"][1] = 42
    elif change == "source":
        node["args"][0]["node_id"] = graph["nodes"][0]["id"]
    elif change == "value":
        node["args"][0]["value_id"] = parent["results"][1]["id"]
    elif change == "scope":
        parent["graph_id"] = "g:quantized:other"
    elif change == "ordinal":
        parent["ordinal"] = node["ordinal"]
    elif change == "dtype":
        node["results"][0]["dtype"] = "int64"
    elif change == "shape":
        node["results"][0]["shape"] = [3, 1]
    elif change == "edge":
        next(e for e in graph["edges"] if e["consumer_node_id"] == identity)[
            "argument_path"
        ] = "args/1"
    elif change == "schema":
        graph["operator_schemas"][parent["target"]] = (
            "aten::sort(Tensor(a!) self, int dim=-1, bool descending=False) -> (Tensor, Tensor)"
        )
    elif change == "kwargs":
        node["kwargs"] = {"unknown": 1}
    assert aliases.root(node) is None


def test_repeated_selection_is_the_same_storage_class(source):
    graph, identity = source
    graph = copy.deepcopy(graph)
    node = next(n for n in graph["nodes"] if n["id"] == identity)
    repeated = copy.deepcopy(node)
    repeated["id"] += "_second"
    repeated["ordinal"] += 10
    repeated["results"][0]["id"] = repeated["id"] + ":v0"
    graph["nodes"].append(repeated)
    edge = copy.deepcopy(
        next(e for e in graph["edges"] if e["consumer_node_id"] == identity)
    )
    edge["consumer_node_id"] = repeated["id"]
    graph["edges"].append(edge)
    aliases = SourceStorageAliases(graph)
    assert aliases.root(node) == aliases.root(repeated)


def test_repeated_slot_preserves_an_intervening_write(source):
    graph, identity = source
    graph = copy.deepcopy(graph)
    original = next(n for n in graph["nodes"] if n["id"] == identity)
    start = original["ordinal"]
    for node in graph["nodes"]:
        if node["ordinal"] > start:
            node["ordinal"] += 2
    mutation = copy.deepcopy(original)
    mutation.update(
        id=identity + "_write",
        ordinal=start + 1,
        target="aten.add_.Tensor",
        args=[{"node_id": identity, "value_id": original["results"][0]["id"]}, 1],
    )
    mutation["results"][0]["id"] = mutation["id"] + ":v0"
    repeated = copy.deepcopy(original)
    repeated.update(id=identity + "_again", ordinal=start + 2)
    repeated["results"][0]["id"] = repeated["id"] + ":v0"
    graph["nodes"].extend([mutation, repeated])
    graph["nodes"].sort(key=lambda row: row["ordinal"])
    graph["operator_schemas"]["aten.add_.Tensor"] = str(
        torch.ops.aten.add_.Tensor._schema
    )
    edge = copy.deepcopy(
        next(e for e in graph["edges"] if e["consumer_node_id"] == identity)
    )
    edge["consumer_node_id"] = repeated["id"]
    graph["edges"].append(edge)
    write_edge = {
        **edge,
        "consumer_node_id": mutation["id"],
        "producer_node_id": original["id"],
        "producer_value_id": original["results"][0]["id"],
    }
    graph["edges"].append(write_edge)
    aliases = SourceStorageAliases(graph)
    assert aliases.last_writer(repeated) == (aliases.root(original), mutation)


@pytest.mark.parametrize(
    "schema",
    [
        "unknown_pair::call(Tensor(a) input) -> (Tensor(a), Tensor)",
        "unknown_pair::call(Tensor(a!) input) -> (Tensor, Tensor)",
        "unknown_pair::call(Tensor input) -> (int, Tensor)",
    ],
)
def test_aliasing_mutating_or_nontensor_producer_return_is_not_fresh(source, schema):
    graph, identity = copy.deepcopy(source)
    node = next(n for n in graph["nodes"] if n["id"] == identity)
    parent = next(n for n in graph["nodes"] if n["id"] == node["args"][0]["node_id"])
    parent["target"] = "unknown_pair.call.default"
    graph["operator_schemas"][parent["target"]] = schema
    assert SourceStorageAliases(graph).root(node) is None
