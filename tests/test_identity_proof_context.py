"""Capture-local indices preserve exact identity proof results and refusals."""
import copy

import pytest
import torch

from m2m.capture.trace import (
    _identity_conversion_bypass,
    _IdentityProofContext,
    _unused_tuple_selection,
    graph_relation,
)


def graphs(count=8):
    def value(name):
        return {"id": name + ":v0", "kind": "tensor", "dtype": "float32", "storage_dtype": "float32",
                    "compute_dtype": "float32", "shape": [2, 4], "device": "cpu", "layout": "torch.strided", "stride": [4, 1]}

    def node(name, ordinal, target=None, argument=None):
        return {"id": name, "ordinal": ordinal, "graph_id": "g:quantized:root", "results": [value(name)],
                    "op": "call_function" if target else "placeholder", "target": target,
                    "args": argument or [], "kwargs": {}}

    source = {"stage": "quantized", "status": "complete", "runtime_versions": {"torch": torch.__version__},
                  "nodes": [node("q:x", 0)], "edges": []}
    previous = "q:x"
    identities = []
    for ordinal in range(1, count + 1):
        name = f"q:to{ordinal}"
        identity = node(name, ordinal, "aten.to.dtype",
                        [{"node_id": previous, "value_id": previous + ":v0"},
                         {"kind": "dtype", "value": "torch.float32"}])
        source["nodes"].append(identity)
        source["edges"].append({"producer_node_id": previous, "producer_value_id": previous + ":v0",
                                    "consumer_node_id": name, "argument_path": "args/0", "dtype": "float32", "shape": [2, 4]})
        identities.append(identity)
        previous = name
    source["nodes"].append(node("q:relu", count + 1, "aten.relu.default",
                                [{"node_id": previous, "value_id": previous + ":v0"}]))
    source["edges"].append({"producer_node_id": previous, "producer_value_id": previous + ":v0",
                                "consumer_node_id": "q:relu", "argument_path": "args/0", "dtype": "float32", "shape": [2, 4]})
    dest = {"stage": "prepared", "status": "complete", "runtime_versions": source["runtime_versions"].copy(),
                "nodes": [{**node("p:x", 0), "origin_node_ids": ["q:x"]},
                       {**node("p:relu", 1, "aten.relu.default"), "origin_node_ids": ["q:relu"]}],
                "edges": [{"producer_node_id": "p:x", "producer_value_id": "p:x:v0", "consumer_node_id": "p:relu",
                            "argument_path": "args/0", "dtype": "float32", "shape": [2, 4]}]}
    return source, dest, identities


def test_shared_context_matches_fresh_alias_and_edge_proofs():
    source, dest, nodes = graphs(16)
    context = _IdentityProofContext(source, dest)
    for node in nodes:
        expected = _identity_conversion_bypass(source, dest, node)
        assert expected is not None
        assert _identity_conversion_bypass(source, dest, node, context=context) == expected


@pytest.mark.parametrize("failure", ["dtype", "shape", "argument", "duplicate_edge", "runtime"])
def test_cached_context_does_not_admit_failed_predicates(failure):
    source, dest, nodes = graphs()
    if failure == "dtype":
        dest["edges"][0]["dtype"] = "float16"
    elif failure == "shape":
        dest["edges"][0]["shape"] = [4, 2]
    elif failure == "argument":
        dest["edges"][0]["argument_path"] = "args/1"
    elif failure == "duplicate_edge":
        dest["edges"].append(copy.deepcopy(dest["edges"][0]))
    else:
        dest["runtime_versions"]["torch"] = "unknown"
    context = _IdentityProofContext(source, dest)
    for node in nodes:
        assert _identity_conversion_bypass(source, dest, node) is None
        assert _identity_conversion_bypass(source, dest, node, context=context) is None


def test_capture_context_cannot_be_reused_for_another_graph_object():
    source, dest, nodes = graphs()
    context = _IdentityProofContext(source, dest)
    other = copy.deepcopy(source)
    with pytest.raises(ValueError, match="different capture"):
        _identity_conversion_bypass(other, dest, other["nodes"][1], context=context)
    with pytest.raises(ValueError, match="different capture"):
        _unused_tuple_selection(other, nodes[0], context=context)


def test_graph_relation_rebuilds_context_for_changed_snapshots():
    source, dest, _ = graphs()
    good = graph_relation(source, dest)
    assert good["status"] == "complete"
    dest["edges"][0]["dtype"] = "float16"
    bad = graph_relation(source, dest)
    assert bad["status"] != "complete"
    assert bad["unresolved_source_ids"]
