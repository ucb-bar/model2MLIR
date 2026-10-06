"""Source views regenerated at their consumers need exact alias ownership."""

import copy

import torch

import m2m
from m2m.capture.trace import graph_relation
from m2m.capture.view_trace import replayed_view_relations


class UpdatedBufferViews(torch.nn.Module):
    def forward(self, x):
        value = torch.empty_like(x)
        value[..., :4] = x[..., :4] * 2
        value[..., 4:] = x[..., 4:] * 3
        expanded = value.to(torch.float32).unsqueeze(3).expand(1, 7, 3, 2, 8)
        lhs = expanded.reshape(1, 7, 6, 8).transpose(1, 2)
        return torch.matmul(lhs, lhs.transpose(2, 3))


def test_regenerated_view_has_exact_latest_writer_and_static_axis_proof():
    torch.manual_seed(0)
    result = m2m.convert(UpdatedBufferViews(), (torch.randn(1, 7, 3, 8),), backend="fx_importer", capture_trace=True)
    trace = result.capture_trace
    assert result.ok and trace["status"] == "complete", trace["blockers"]
    source, dest = trace["graphs"]["quantized"], trace["graphs"]["prepared"]
    relations = graph_relation(source, dest)
    source_nodes = {node["id"]: node for node in source["nodes"]}
    dest_nodes = {node["id"]: node for node in dest["nodes"]}
    cast = next(node for node in source["nodes"] if node["target"] == "aten.to.dtype")
    alias = next(
        row for row in relations["relations"] if row["kind"] == "value_alias" and row["source_ids"] == [cast["id"]]
    )
    assert alias["proof"]["updated_after_input"] is True
    writer = source_nodes[alias["proof"]["last_writer_node_id"]]
    assert writer["target"] == "aten.copy_.default"
    assert writer["ordinal"] == max(
        node["ordinal"] for node in source["nodes"] if node["target"] == "aten.copy_.default"
    )
    assert dest_nodes[alias["destination_ids"][0]]["target"] == "aten.slice_scatter.default"
    source_view = next(node for node in source["nodes"] if node["target"] == "aten.unsqueeze.default")
    # The corrected alias also enables ordinary structural matching. Exercise
    # the view fallback independently, retaining the same actual producer and
    # consumer correspondence from this real capture.
    premises = [row for row in relations["relations"] if source_view["id"] not in row["source_ids"]]
    consumed = {identity for row in premises for identity in row["source_ids"]}
    views = replayed_view_relations(source, dest, premises, consumed)
    assert len(views) == 1
    view = views[0]
    assert source_nodes[view["source_ids"][0]]["target"] == "aten.unsqueeze.default"
    writer = source_nodes[view["proof"]["last_writer_node_id"]]
    assert writer["target"] == "aten.copy_.default"
    assert writer["ordinal"] == max(
        node["ordinal"] for node in source["nodes"] if node["target"] == "aten.copy_.default"
    )
    destination_id = view["destination_ids"][0]

    for mutation in ("axis", "value", "type", "ambiguous", "runtime", "consumer"):
        changed = copy.deepcopy(dest)
        candidate = next(node for node in changed["nodes"] if node["id"] == destination_id)
        if mutation == "axis":
            candidate["args"][1] = 2
        elif mutation == "value":
            # Keep all result types but bind the view to an earlier allocation.
            allocation = next(node for node in changed["nodes"] if node["op"] == "placeholder")
            candidate["args"][0] = {"node_id": allocation["id"], "value_id": allocation["results"][0]["id"]}
        elif mutation == "type":
            candidate["results"][0]["dtype"] = "float64"
        elif mutation == "ambiguous":
            duplicate = copy.deepcopy(candidate)
            duplicate["id"] += "_duplicate"
            duplicate["results"][0]["id"] = duplicate["id"] + ":v0"
            changed["nodes"].append(duplicate)
            for edge in list(changed["edges"]):
                if edge["producer_node_id"] == destination_id:
                    copied = copy.deepcopy(edge)
                    copied["producer_node_id"] = duplicate["id"]
                    copied["producer_value_id"] = duplicate["results"][0]["id"]
                    changed["edges"].append(copied)
        elif mutation == "runtime":
            changed["runtime_versions"] = {"torch": "another selected runtime"}
        else:
            changed["edges"] = [edge for edge in changed["edges"] if edge["producer_node_id"] != destination_id]
        assert replayed_view_relations(source, changed, premises, consumed) == [], mutation
    assert dest_nodes[destination_id]["target"] == "aten.unsqueeze.default"
    invalid_source = copy.deepcopy(source)
    invalid_use = next(edge for edge in invalid_source["edges"] if edge["producer_node_id"] == source_view["id"])
    invalid_use["producer_value_id"] = "not_the_view_result"
    assert replayed_view_relations(invalid_source, dest, premises, consumed) == []
