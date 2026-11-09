"""Actual subclass fields must retain source ownership, storage, payload and effects."""

import copy

import pytest
import torch

from m2m.capture.subclass_trace import bind, observe
from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization
from m2m.capture.trace import snapshot_exported_program


class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)

    def forward(self, value):
        return self.linear(value)


@pytest.fixture
def pair():
    pytest.importorskip("torchao")
    module = apply_quantization(
        Linear().eval(), QuantizationConfig(scheme="int8_dyn_act_int8_weight")
    )
    source = torch.export.export(module, (torch.ones(1, 32),))
    snapshot_exported_program(source, stage="quantized")
    observed = observe(source)
    assert observed is not None and len(observed.fields) == 2
    prepared = source.run_decompositions({})
    return source, observed, prepared


def test_actual_qdata_scale_fields_have_unique_same_storage_liftings(pair):
    _source, observed, prepared = pair
    records = bind(observed, prepared)
    assert {r["field"] for r in records} == {"qdata", "scale"}
    assert all(r["same_live_storage"] and r["source_state_unwritten"] for r in records)
    snapshot = snapshot_exported_program(prepared, stage="prepared")
    origins = {
        identity for row in snapshot["nodes"] for identity in row["origin_node_ids"]
    }
    assert {field.source_id for field in observed.fields} <= origins


def test_cloned_equal_payload_is_not_storage_correspondence(pair):
    _, observed, prepared = pair
    for name, value in tuple(prepared.state_dict.items()):
        prepared.state_dict[name] = value.detach().clone()
    assert bind(observed, prepared) == []


def test_changed_source_field_payload_is_refused(pair):
    _, observed, prepared = pair
    with torch.no_grad():
        observed.fields[0].value.view(-1)[0] += 1
    assert bind(observed, prepared) == []


def test_changed_source_state_owner_is_refused(pair):
    source, observed, prepared = pair
    field = observed.fields[0]
    source.state_dict[field.source_target] = copy.deepcopy(field.owner)
    assert bind(observed, prepared) == []


def test_changed_source_call_between_observation_and_binding_is_refused(pair):
    source, observed, prepared = pair
    node = next(
        node
        for node in source.graph.nodes
        if node.op == "call_function" and str(node.target) == "aten.t.default"
    )
    node.args = (next(node for node in source.graph.nodes if node.op == "placeholder"),)
    assert bind(observed, prepared) == []


def test_unknown_source_effects_are_refused_before_a_claim(pair):
    source, _, _ = pair
    node = next(
        node
        for node in source.graph.nodes
        if node.op == "call_function" and str(node.target) == "aten.t.default"
    )
    node.target = lambda value: value
    assert observe(source) is None


def test_write_to_source_field_is_refused(pair):
    source, _, _ = pair
    field = next(
        node
        for node in source.graph.nodes
        if node.op == "call_function"
        and str(node.target) == "export.access_subclass_inner_tensor.default"
    )
    with source.graph.inserting_after(field):
        mutation = source.graph.call_function(torch.ops.aten.add_.Tensor, (field, 1))
    mutation.meta = dict(field.meta)
    mutation.meta["custom"] = dict(field.meta.get("custom") or {})
    assert observe(source) is None


def test_missing_subclass_attribute_is_refused(pair):
    source, _, _ = pair
    field = next(
        node
        for node in source.graph.nodes
        if node.op == "call_function"
        and str(node.target) == "export.access_subclass_inner_tensor.default"
    )
    field.args = (field.args[0], "missing")
    assert observe(source) is None


def test_replaced_source_field_object_is_refused_even_if_old_leaf_still_lives(pair):
    source, observed, prepared = pair
    field = observed.fields[0]
    setattr(
        source.state_dict[field.source_target],
        field.field,
        field.value.detach().clone(),
    )
    assert bind(observed, prepared) == []


def test_same_shape_but_changed_prepared_stride_is_refused(pair):
    _, observed, prepared = pair
    field = next(f for f in observed.fields if f.field == "qdata")
    for name, value in tuple(prepared.state_dict.items()):
        if value.untyped_storage()._cdata == field.storage:
            prepared.state_dict[name] = value.t()
    assert bind(observed, prepared) == []


def test_complete_actual_quantized_prepared_trace_stays_distinct_from_original_lineage():
    import m2m
    from m2m.capture.trace import capture_frontend_snapshot

    pytest.importorskip("torchao")
    model = Linear().eval()
    value = torch.randn(1, 32)
    original = capture_frontend_snapshot(model, (value,))
    quant = QuantizationConfig(scheme="int8_dyn_act_int8_weight")
    selected = apply_quantization(model, quant)
    converted = m2m.convert(
        selected,
        (value,),
        quantization=quant,
        quantization_preapplied=True,
        capture_trace=True,
        original_frontend_snapshot=original,
        backend="fx_importer",
    )
    assert converted.ok
    first, second = converted.capture_trace["transformations"]
    assert first["status"] == "diagnostic"
    assert second["status"] == "complete"
    assert second["unresolved_source_ids"] == second["unresolved_destination_ids"] == []
    records = converted.capture_trace["graphs"]["prepared"]["tensor_subclass_fields"]
    assert {row["field"] for row in records} == {"qdata", "scale"}
    assert all(
        row["source_graph_sha256"]
        == converted.capture_trace["graphs"]["quantized"]["sha256"]
        for row in records
    )


def test_source_getter_type_must_match_the_actual_field(pair):
    source, _, _ = pair
    node = next(
        node
        for node in source.graph.nodes
        if node.op == "call_function"
        and str(node.target) == "export.access_subclass_inner_tensor.default"
    )
    node.meta["val"] = torch.ones((32, 32), dtype=torch.float32)
    assert observe(source) is None


def test_prepared_placeholder_type_must_match_its_actual_state(pair):
    _, observed, prepared = pair
    spec = next(
        spec
        for spec in prepared.graph_signature.input_specs
        if spec.target is not None and str(spec.target).endswith("original0")
    )
    node = next(node for node in prepared.graph.nodes if node.name == spec.arg.name)
    node.meta["val"] = torch.ones((32, 32), dtype=torch.float32)
    assert bind(observed, prepared) == []


def test_ambiguous_aliasing_prepared_parameters_refuse_atomically(pair):
    from torch.export.graph_signature import InputKind, InputSpec, TensorArgument

    _, observed, prepared = pair
    field = next(f for f in observed.fields if f.field == "qdata")
    target = next(
        name
        for name, value in prepared.state_dict.items()
        if value.untyped_storage()._cdata == field.storage
    )
    spec = next(s for s in prepared.graph_signature.input_specs if s.target == target)
    original = next(node for node in prepared.graph.nodes if node.name == spec.arg.name)
    first = next(iter(prepared.graph.nodes))
    with prepared.graph.inserting_before(first):
        alias = prepared.graph.placeholder("source_leaf_alias")
    alias.meta = dict(original.meta)
    prepared.state_dict["source_leaf_alias"] = prepared.state_dict[target].view_as(
        prepared.state_dict[target]
    )
    prepared.graph_signature.input_specs.insert(
        0,
        InputSpec(
            kind=InputKind.PARAMETER,
            arg=TensorArgument(name=alias.name),
            target="source_leaf_alias",
            persistent=None,
        ),
    )
    before = {
        node.name: dict(node.meta.get("custom") or {}) for node in prepared.graph.nodes
    }
    assert bind(observed, prepared) == []
    assert before == {
        node.name: dict(node.meta.get("custom") or {}) for node in prepared.graph.nodes
    }


def test_external_input_write_refuses_even_before_field_access(pair):
    source, _, _ = pair
    spec = next(
        spec
        for spec in source.graph_signature.input_specs
        if str(spec.kind) == "InputKind.USER_INPUT"
    )
    user_input = next(node for node in source.graph.nodes if node.name == spec.arg.name)
    with source.graph.inserting_after(user_input):
        mutation = source.graph.call_function(
            torch.ops.aten.add_.Tensor, (user_input, 1)
        )
    mutation.meta = dict(user_input.meta)
    mutation.meta["custom"] = dict(user_input.meta.get("custom") or {})
    assert observe(source) is None
