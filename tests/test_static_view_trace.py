"""Static views require exact producers, storage versions and typed uses."""

import copy

import m2m
import pytest
import torch
from m2m.capture.static_view_trace import (
    dead_static_view_relations,
    static_view_relations,
)
from m2m.capture.trace import _no_op_dtype_aliases, graph_relation


class View(torch.nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target

    def forward(self, x):
        with torch.no_grad():
            value = x * 2
            if self.target == "transpose":
                view = value.transpose(1, 2)
            elif self.target == "slice":
                view = value[:, 1:5:2]
            elif self.target == "permute":
                view = value.permute(2, 0, 1)
            else:
                view = value.unsqueeze(1)
            return view.contiguous() + 1


@pytest.fixture(scope="module")
def snapshots():
    result = {}
    for name in ("transpose", "slice", "permute", "unsqueeze"):
        converted = m2m.convert(
            View(name),
            (torch.arange(36, dtype=torch.float32).reshape(1, 6, 6),),
            backend="fx_importer",
            capture_trace=True,
        )
        assert converted.ok
        trace = converted.capture_trace
        source, dest = trace["graphs"]["quantized"], trace["graphs"]["prepared"]
        target = "aten." + (
            "transpose.int"
            if name == "transpose"
            else "slice.Tensor"
            if name == "slice"
            else name + ".default"
        )
        node = next(n for n in source["nodes"] if n["target"] == target)
        if name == "slice":
            node = next(
                n
                for n in source["nodes"]
                if n["target"] == target and len(n["args"]) > 4 and n["args"][4] == 2
            )
        original = graph_relation(source, dest)
        premises = [
            row for row in original["relations"] if node["id"] not in row["source_ids"]
        ]
        consumed = {identity for row in premises for identity in row["source_ids"]}
        views = static_view_relations(source, dest, premises, consumed)
        assert len(views) == 1, (name, views)
        result[name] = (
            source,
            dest,
            premises,
            consumed,
            node["id"],
            views[0]["destination_ids"][0],
        )
    return result


@pytest.mark.parametrize("name", ["transpose", "slice", "permute", "unsqueeze"])
def test_actual_static_views_bind_through_verified_hop_input_and_consumer(
    name, snapshots
):
    source, dest, premises, consumed, identity, _ = snapshots[name]
    proof = static_view_relations(source, dest, premises, consumed)[0]
    assert proof["source_ids"] == [identity]
    assert proof["kind"] == "anchored_static_view"
    assert proof["proof"]["exact_caller_bindings"] is True
    assert proof["proof"]["no_write_after_input"] is True
    assert proof["proof"]["typed_consumer_edges"]


@pytest.mark.parametrize(
    "failure",
    [
        "source",
        "offset",
        "stride",
        "dtype",
        "alias",
        "consumer",
        "duplicate",
        "runtime",
        "scope",
        "caller",
    ],
)
def test_mismatched_view_premises_are_refused(failure, snapshots):
    original_source, original_dest, premises, consumed, identity, destination = (
        snapshots["slice"]
    )
    source, dest = copy.deepcopy(original_source), copy.deepcopy(original_dest)
    node = next(n for n in source["nodes"] if n["id"] == identity)
    candidate = next(n for n in dest["nodes"] if n["id"] == destination)
    if failure == "source":
        owner = next(n for n in dest["nodes"] if n["op"] == "placeholder")
        candidate["args"][0] = {
            "node_id": owner["id"],
            "value_id": owner["results"][0]["id"],
        }
    elif failure == "offset":
        # Same shape and stride: selecting 0:4:2 does not select 1:5:2.
        candidate["args"][2:4] = [0, 4]
    elif failure == "stride":
        candidate["results"][0]["stride"][1] += 1
    elif failure == "dtype":
        candidate["results"][0]["storage_dtype"] = "float64"
    elif failure == "alias":
        source["operator_schemas"][node["target"]] = str(
            torch.ops.aten.clone.default._schema
        )
    elif failure == "consumer":
        edge = next(e for e in dest["edges"] if e["producer_node_id"] == destination)
        edge["argument_path"] = "args/99"
    elif failure == "duplicate":
        duplicate = copy.deepcopy(candidate)
        duplicate["id"] += "_duplicate"
        duplicate["results"][0]["id"] = duplicate["id"] + ":v0"
        dest["nodes"].append(duplicate)
        for edge in list(dest["edges"]):
            if edge["consumer_node_id"] == destination:
                changed = copy.deepcopy(edge)
                changed["consumer_node_id"] = duplicate["id"]
                dest["edges"].append(changed)
            if edge["producer_node_id"] == destination:
                changed = copy.deepcopy(edge)
                changed["producer_node_id"] = duplicate["id"]
                changed["producer_value_id"] = duplicate["results"][0]["id"]
                dest["edges"].append(changed)
    elif failure == "runtime":
        dest["runtime_versions"]["torch"] = "unobserved"
    elif failure == "scope":
        candidate["graph_id"] = "g:prepared:another_unverified_body"
    else:
        wrapper = next(
            n for n in source["nodes"] if n["target"] == "wrap_with_set_grad_enabled"
        )
        wrapper["args"][2] = wrapper["args"][1]
    assert static_view_relations(source, dest, premises, consumed) == [], failure


def _insert_mutation(graph, before, producer):
    for n in graph["nodes"]:
        n["ordinal"] *= 2
    writer = copy.deepcopy(producer)
    writer.update(
        id=producer["id"] + "_write",
        ordinal=producer["ordinal"] + 1,
        op="call_function",
        target="aten.copy_.default",
        kwargs={},
    )
    ref = {"node_id": producer["id"], "value_id": producer["results"][0]["id"]}
    writer["args"] = [ref, copy.deepcopy(ref)]
    writer["results"][0]["id"] = writer["id"] + ":v0"
    graph["operator_schemas"]["aten.copy_.default"] = str(
        torch.ops.aten.copy_.default._schema
    )
    graph["nodes"].insert(graph["nodes"].index(before), writer)
    value = producer["results"][0]
    for index in range(2):
        graph["edges"].append(
            {
                "producer_node_id": producer["id"],
                "producer_value_id": value["id"],
                "consumer_node_id": writer["id"],
                "argument_path": f"args/{index}",
                "shape": value["shape"],
                "dtype": value["dtype"],
                "value_kind": "tensor",
            }
        )


@pytest.mark.parametrize("which", ["source", "destination"])
def test_intervening_write_cannot_reuse_old_mapped_storage(which, snapshots):
    source, dest, premises, consumed, identity, destination = snapshots["transpose"]
    source, dest = copy.deepcopy(source), copy.deepcopy(dest)
    graph, name = (source, identity) if which == "source" else (dest, destination)
    view = next(n for n in graph["nodes"] if n["id"] == name)
    producer = next(n for n in graph["nodes"] if n["id"] == view["args"][0]["node_id"])
    _insert_mutation(graph, view, producer)
    assert static_view_relations(source, dest, premises, consumed) == []


@pytest.mark.parametrize(
    "failure", [None, "use", "offset", "stride", "step", "alias", "input_edge"]
)
def test_only_valid_unused_static_slices_are_discarded(failure, snapshots):
    source = copy.deepcopy(snapshots["slice"][0])
    identity = snapshots["slice"][4]
    node = next(n for n in source["nodes"] if n["id"] == identity)
    source["edges"] = [
        edge for edge in source["edges"] if edge["producer_node_id"] != identity
    ]
    consumed = {n["id"] for n in source["nodes"] if n["id"] != identity}
    if failure == "use":
        value = node["results"][0]
        source["edges"].append(
            {
                "producer_node_id": identity,
                "producer_value_id": value["id"],
                "consumer_node_id": "escaped",
                "argument_path": "args/0",
                "shape": value["shape"],
                "dtype": value["dtype"],
                "value_kind": "tensor",
            }
        )
    elif failure == "offset":
        node["args"][2] = 3  # Result shape now disagrees with the selected bounds.
    elif failure == "stride":
        node["results"][0]["stride"][1] += 1
    elif failure == "step":
        node["args"][4] = 0  # A throwing operation must not be declared dead.
    elif failure == "alias":
        source["operator_schemas"][node["target"]] = str(
            torch.ops.aten.clone.default._schema
        )
    elif failure == "input_edge":
        edge = next(e for e in source["edges"] if e["consumer_node_id"] == identity)
        edge["producer_value_id"] = "wrong_value"
    rows = dead_static_view_relations(source, consumed)
    assert bool(rows) is (failure is None)


def test_device_noop_alias_keeps_exact_device_copy_and_storage_requirements(snapshots):
    source, dest, premises, _, _, _ = snapshots["transpose"]
    source = copy.deepcopy(source)
    producer = next(n for n in source["nodes"] if n["target"] == "aten.mul.Tensor")
    alias = copy.deepcopy(producer)
    alias.update(
        id="device_alias",
        ordinal=producer["ordinal"] + 0.5,
        target="aten.to.device",
        kwargs={},
    )
    alias["args"] = [
        {"node_id": producer["id"], "value_id": producer["results"][0]["id"]},
        {"kind": "device", "value": "cpu"},
        {"kind": "dtype", "value": "torch.float32"},
    ]
    alias["results"][0]["id"] = "device_alias:v0"
    source["nodes"].append(alias)
    source["operator_schemas"][alias["target"]] = str(torch.ops.aten.to.device._schema)
    value = producer["results"][0]
    source["edges"].append(
        {
            "producer_node_id": producer["id"],
            "producer_value_id": value["id"],
            "consumer_node_id": alias["id"],
            "argument_path": "args/0",
            "shape": value["shape"],
            "dtype": value["dtype"],
            "value_kind": "tensor",
        }
    )
    consumed = {n["id"] for n in source["nodes"] if n["id"] != alias["id"]}
    assert len(_no_op_dtype_aliases(source, dest, premises, consumed)) == 1
    for name in ("copy", "device", "dtype"):
        changed = copy.deepcopy(source)
        node = next(n for n in changed["nodes"] if n["id"] == alias["id"])
        if name == "copy":
            node["kwargs"]["copy"] = True
        elif name == "device":
            node["args"][1]["value"] = "cuda:0"
        else:
            node["args"][2]["value"] = "torch.float64"
        assert _no_op_dtype_aliases(changed, dest, premises, consumed) == []


def test_existing_dead_view_proof_is_preserved(snapshots):
    from m2m.capture.trace import _dead_forward_value_relations

    class Base(torch.nn.Module):
        def forward(self, x):
            return x + 1

    converted = m2m.convert(
        Base(), (torch.zeros(1, 6, 6),), backend="fx_importer", capture_trace=True
    )
    source = copy.deepcopy(converted.capture_trace["graphs"]["quantized"])
    dest = converted.capture_trace["graphs"]["prepared"]
    # Retain a valid, unused view in the source snapshot. Export normally DCEs
    # it before capture; the destination is the unchanged actual Base graph.
    unused = copy.deepcopy(
        next(
            n
            for n in snapshots["unsqueeze"][0]["nodes"]
            if n["id"] == snapshots["unsqueeze"][4]
        )
    )
    producer = next(n for n in source["nodes"] if n["op"] == "placeholder")
    for node in source["nodes"]:
        node["ordinal"] *= 2
    unused.update(
        id="g:quantized:root:unused_static_view",
        graph_id=producer["graph_id"],
        ordinal=producer["ordinal"] + 1,
        origin_node_ids=[],
    )
    unused["args"][0] = {
        "node_id": producer["id"],
        "value_id": producer["results"][0]["id"],
    }
    unused["results"][0]["id"] = unused["id"] + ":v0"
    unused["result_metadata"]["values"][0]["result_id"] = unused["results"][0]["id"]
    source["operator_schemas"][unused["target"]] = str(
        torch.ops.aten.unsqueeze.default._schema
    )
    source["nodes"].insert(1, unused)
    source["edges"].append(
        {
            "producer_node_id": producer["id"],
            "producer_value_id": producer["results"][0]["id"],
            "consumer_node_id": unused["id"],
            "argument_path": "args/0",
            "value_kind": "tensor",
            "dtype": "float32",
            "shape": [1, 6, 6],
        }
    )
    consumed = {node["id"] for node in source["nodes"] if node["id"] != unused["id"]}
    legacy = _dead_forward_value_relations(source, consumed)
    assert len(legacy) == 1
    relation = graph_relation(source, dest)
    assert legacy[0] in relation["relations"]
