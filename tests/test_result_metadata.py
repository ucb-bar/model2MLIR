"""Original metadata presence is distinct from an observed Python None result."""

import copy
import json

import pytest
import torch

from m2m.capture.trace import snapshot_exported_program


class Metadata(torch.nn.Module):
    def forward(self, value):
        observed_none = torch.ops.aten._assert_tensor_metadata.default(
            value, dtype=torch.int8
        )
        return value.clone(), observed_none


def _export():
    return torch.export.export(
        Metadata(), (torch.arange(6, dtype=torch.int8).reshape(2, 3),)
    )


def _assertion(snapshot):
    return next(
        node
        for node in snapshot["nodes"]
        if node["target"] == "aten._assert_tensor_metadata.default"
    )


def test_actual_export_distinguishes_observed_none_from_absent_metadata():
    exported = _export()
    native_node = next(
        node
        for node in exported.graph_module.graph.nodes
        if node.target == torch.ops.aten._assert_tensor_metadata.default
    )
    assert "val" in native_node.meta and native_node.meta["val"] is None
    observed = snapshot_exported_program(exported, stage="original")
    record = _assertion(observed)
    assert record["result_metadata"] == {
        "schema": "m2m.frontend_result_metadata.v1",
        "status": "observed",
        "container": "single",
        "values": [{"result_id": record["results"][0]["id"], "kind": "none"}],
    }
    # Preserve the historical value slots; metadata is additional provenance.
    assert len(record["results"]) == 1 and record["results"][0]["kind"] == "unknown"
    assert observed["schema"] == "m2m.frontend_graph.v1"
    native_node.meta.pop("val")
    absent = snapshot_exported_program(exported, stage="original")
    missing = _assertion(absent)
    assert missing["results"] == record["results"]
    assert missing["result_metadata"] == {
        "schema": "m2m.frontend_result_metadata.v1",
        "status": "unobserved",
    }
    assert observed["sha256"] != absent["sha256"]
    # The old serialization cannot distinguish these actual producer states.
    legacy_left, legacy_right = copy.deepcopy(observed), copy.deepcopy(absent)
    for graph in (legacy_left, legacy_right):
        graph.pop("sha256")
        for node in graph["nodes"]:
            node.pop("result_metadata")
    assert legacy_left == legacy_right


def test_actual_native_none_does_not_make_metadata_assertion_pure():
    value = torch.arange(6, dtype=torch.int8).reshape(2, 3)
    operation = torch.ops.aten._assert_tensor_metadata.default
    assert list(operation._schema.returns) == []
    assert operation(value, dtype=torch.int8) is None
    with pytest.raises(RuntimeError, match="dtype"):
        operation(value, dtype=torch.float32)


def test_tensor_and_missing_output_metadata_keep_complete_original_slots():
    exported = _export()
    snapshot = json.loads(
        json.dumps(
            snapshot_exported_program(exported, stage="original"), allow_nan=False
        )
    )
    clone = next(
        node for node in snapshot["nodes"] if node["target"] == "aten.clone.default"
    )
    assert clone["result_metadata"]["values"] == [
        {"result_id": clone["results"][0]["id"], "kind": "tensor"}
    ]
    output = next(node for node in snapshot["nodes"] if node["op"] == "output")
    assert output["result_metadata"]["status"] == "unobserved"
    assert output["args"][0][1] is None


def test_actual_sequence_metadata_preserves_each_result_ordinal():
    class Split(torch.nn.Module):
        def forward(self, value):
            return value.split(2)

    exported = torch.export.export(Split(), (torch.arange(4, dtype=torch.int8),))
    native = next(
        node
        for node in exported.graph_module.graph.nodes
        if node.target == torch.ops.aten.split.Tensor
    )
    assert isinstance(native.meta["val"], (list, tuple))
    original_container = type(native.meta["val"]).__name__
    snapshot = snapshot_exported_program(exported, stage="original")
    split = next(
        node for node in snapshot["nodes"] if node["target"] == "aten.split.Tensor"
    )
    assert split["result_metadata"]["container"] == original_container
    assert len(split["results"]) == 2
    assert split["result_metadata"]["values"] == [
        {"result_id": result["id"], "kind": "tensor"} for result in split["results"]
    ]
