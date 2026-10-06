"""No-op dtype casts must observe intervening writes to aliased storage."""

import copy

import m2m
import torch
from m2m.capture.trace import _anchored_guard_relations, _identity_conversion_bypass, graph_relation
from m2m.capture.view_trace import SourceStorageAliases


class UpdatedBufferCast(torch.nn.Module):
    def forward(self, x):
        value = torch.empty_like(x)
        value[:, :2] = x[:, :2] * 2
        value[:, 2:] = x[:, 2:] * 3
        return value.to(torch.float32) + 1


def test_mutated_storage_cast_maps_final_writer_and_refuses_old_allocation_bypass():
    converted = m2m.convert(UpdatedBufferCast(), (torch.randn(2, 4),), backend="fx_importer", capture_trace=True)
    assert converted.ok and converted.capture_trace["status"] == "complete"
    source = converted.capture_trace["graphs"]["quantized"]
    prepared = converted.capture_trace["graphs"]["prepared"]
    cast = next(node for node in source["nodes"] if node["target"] == "aten.to.dtype")
    writes = [node for node in source["nodes"] if node["target"] == "aten.copy_.default"]
    assert len(writes) == 2
    relation = graph_relation(source, prepared)
    alias = next(row for row in relation["relations"] if row["source_ids"] == [cast["id"]])
    destination = next(node for node in prepared["nodes"] if node["id"] == alias["destination_ids"][0])
    assert alias["kind"] == "value_alias"
    assert alias["proof"]["last_writer_node_id"] == writes[-1]["id"]
    assert alias["proof"]["updated_after_input"] is True
    assert destination["target"] == "aten.slice_scatter.default"

    # A manipulated prepared graph can have an exactly typed consumer edge to
    # the old allocation, but that edge does not preserve the value after copy_.
    stale = copy.deepcopy(prepared)
    allocation = next(node for node in stale["nodes"] if node["target"] == "aten.empty.memory_format")
    consumer = next(node for node in stale["nodes"] if node["target"] == "aten.add.Tensor")
    consumer["args"][0] = {"node_id": allocation["id"], "value_id": allocation["results"][0]["id"]}
    edge = next(
        edge
        for edge in stale["edges"]
        if edge["consumer_node_id"] == consumer["id"] and edge["argument_path"] == "args/0"
    )
    edge["producer_node_id"] = allocation["id"]
    edge["producer_value_id"] = allocation["results"][0]["id"]
    assert _identity_conversion_bypass(source, stale, cast) is None

    # A retained same-dtype cast on the old allocation is equally unsafe: an
    # exact-looking call/argument match cannot override the intervening writes.
    stale_cast = copy.deepcopy(cast)
    stale_cast["id"] = "g:prepared:root:stale_cast"
    stale_cast["graph_id"] = "g:prepared:root"
    stale_cast["origin_node_ids"] = []
    stale_cast["args"][0] = {"node_id": allocation["id"], "value_id": allocation["results"][0]["id"]}
    stale_cast["results"][0]["id"] = stale_cast["id"] + ":v0"
    stale["nodes"].append(stale_cast)
    source_allocation = next(node for node in source["nodes"] if node["target"] == "aten.empty_like.default")
    mapped_allocation = [
        {"source_ids": [source_allocation["id"]], "destination_ids": [allocation["id"]], "kind": "identity"}
    ]
    inferred = _anchored_guard_relations(source, stale, mapped_allocation, set())
    assert not any(cast["id"] in row["source_ids"] for row in inferred)


def test_selected_operator_schema_is_required_and_matches_live_registration():
    converted = m2m.convert(UpdatedBufferCast(), (torch.randn(2, 4),), backend="fx_importer", capture_trace=True)
    assert converted.ok
    source = converted.capture_trace["graphs"]["quantized"]
    cast = next(node for node in source["nodes"] if node["target"] == "aten.to.dtype")
    copy_op = next(node for node in source["nodes"] if node["target"] == "aten.copy_.default")
    captured = source["operator_schemas"][copy_op["target"]]
    assert captured == str(torch.ops.aten.copy_.default._schema)
    for replacement in (
        "not a selected operator schema",
        str(torch.ops.aten.relu.default._schema),
        captured.replace("bool non_blocking=False", "bool non_blocking=True"),
    ):
        changed = copy.deepcopy(source)
        changed["operator_schemas"][copy_op["target"]] = replacement
        aliases = SourceStorageAliases(changed)
        assert aliases.schema(copy_op) is None
        assert aliases.last_writer(cast) is None

    # A namespace absent from the offline registry is uncertain without its
    # captured schema. A selected schema with no alias/write effects can be
    # inspected without registering that namespace again; the capture receipt
    # is responsible for authenticating this source evidence.
    changed = copy.deepcopy(source)
    guard = next(node for node in changed["nodes"] if node["target"] == "aten._assert_tensor_metadata.default")
    custom = "m2m_trace_unregistered.identity.default"
    guard["target"] = custom
    guard["args"] = guard["args"][:1]
    guard["kwargs"] = {}
    assert SourceStorageAliases(changed).last_writer(cast) is None
    changed["operator_schemas"][custom] = "m2m_trace_unregistered::identity(Tensor input) -> Tensor"
    aliases = SourceStorageAliases(changed)
    assert aliases.schema(guard) is not None
    state = aliases.last_writer(cast)
    assert state is not None and state[1] is not None
    assert state[1]["target"] == "aten.copy_.default"
