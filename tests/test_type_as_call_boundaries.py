"""Callee metadata follows actual identity values, never stale tensor names."""

import copy

import pytest
import torch

import m2m
from m2m.capture.call_boundary_trace import normalize_call_boundary_aliases
from m2m.capture.dynamic_linear_trace import LinearInstrumentation
from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization
from m2m.capture.trace import _forward_identity_value, snapshot_exported_program


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32, bias=False)
        self.register_buffer("reference", torch.zeros(1, dtype=torch.float32))

    def forward(self, value):
        hidden = value.sin().type_as(self.reference)
        return self.linear(hidden)


def source_and_selected():
    model = Model().eval().requires_grad_(False)
    inputs = (torch.randn(1, 32),)
    quant = QuantizationConfig(scheme="int8_dyn_act_int8_weight")
    with LinearInstrumentation(model) as owner:
        source = owner.capture_source(model, inputs)
        apply_quantization(model, quant)
        selected, _ = owner.export_selected(source, model, inputs, quantization=quant)
    return inputs, quant, source, selected


def test_real_source_ao_conversion_keeps_raw_call_contract_and_full_trace():
    inputs, quant, source, selected = source_and_selected()
    before = snapshot_exported_program(selected, stage="quantized")
    calls = copy.deepcopy(selected.module_call_graph)
    result = m2m.convert(
        selected,
        inputs,
        quantization=quant,
        quantization_preapplied=True,
        original_frontend_snapshot=source.snapshot,
        capture_trace=True,
        backend="fx_importer",
    )
    assert result.ok, result.diagnostics
    assert result.capture_trace["status"] == "complete", result.capture_trace[
        "blockers"
    ]
    assert snapshot_exported_program(selected, stage="quantized") == before
    assert selected.module_call_graph == calls
    records = result.capture_trace["graphs"]["prepared"]["module_call_boundary_aliases"]
    assert (
        len(records) == 1
        and records[0]["typed_identity_and_consumer_lifetime_verified"]
    )
    assert records[0]["graph_arithmetic_changed"] is False


@pytest.mark.parametrize(
    "change", ["dtype", "device", "layout", "output_stride", "edge", "schema"]
)
def test_type_as_identity_requires_exact_static_type_controls(change):
    _, _, _, selected = source_and_selected()
    snapshot = snapshot_exported_program(selected, stage="quantized")
    row = next(
        row for row in snapshot["nodes"] if row["target"] == "aten.type_as.default"
    )
    sources = {node["id"]: node for node in snapshot["nodes"]}
    reference = sources[row["args"][1]["node_id"]]["results"][0]
    if change == "dtype":
        reference["dtype"] = "float64"
    elif change == "device":
        reference["device"] = "cuda:0"
    elif change == "layout":
        reference["layout"] = "torch.sparse_coo"
    elif change == "output_stride":
        row["results"][0]["stride"] = [64, 2]
    elif change == "edge":
        row["args"][1]["value_id"] = "not-the-reference-result"
    else:
        snapshot["operator_schemas"][row["target"]] = str(
            torch.ops.aten.relu.default._schema
        )
    assert _forward_identity_value(snapshot, row) is None


def test_normalizer_changes_only_owned_signature_metadata_and_retains_all_graph_words():
    _, _, _, selected = source_and_selected()
    snapshot = snapshot_exported_program(selected, stage="quantized")
    old_calls = copy.deepcopy(selected.module_call_graph)
    owned, records = normalize_call_boundary_aliases(selected)
    assert owned is not selected and records
    assert snapshot_exported_program(owned, stage="quantized") == snapshot
    assert selected.module_call_graph == old_calls
    assert all(
        owned.state_dict[name] is value for name, value in selected.state_dict.items()
    )
    before = next(
        entry for entry in selected.module_call_graph if entry.fqn == "linear"
    )
    after = next(entry for entry in owned.module_call_graph if entry.fqn == "linear")
    assert before.signature.inputs[0].name == "type_as"
    assert after.signature.inputs[0].name == "sin"


def test_intervening_write_after_type_as_before_callee_refuses():
    model = Model().eval().requires_grad_(False)
    exported = torch.export.export(
        model,
        (torch.randn(1, 32),),
        preserve_module_call_signature=("linear",),
        strict=True,
    )
    alias = next(
        node
        for node in exported.graph.nodes
        if node.target is torch.ops.aten.type_as.default
    )
    # Add an actual same-storage writer after the recorded alias and before
    # its module consumer, retaining the original tensor-valued callee slot.
    with exported.graph.inserting_after(alias):
        writer = exported.graph.call_function(
            torch.ops.aten.add_.Tensor, (alias.args[0], 1)
        )
    writer.meta["val"] = alias.args[0].meta["val"]
    exported.graph_module.recompile()
    exported.validate()
    with pytest.raises(ValueError, match="alias/effect/consumer lifetime"):
        normalize_call_boundary_aliases(exported)


def test_duplicate_module_paths_keep_their_distinct_entry_occurrences():
    _, _, _, selected = source_and_selected()
    original = next(
        entry for entry in selected.module_call_graph if entry.fqn == "linear"
    )
    selected.module_call_graph.append(copy.deepcopy(original))
    selected.validate()
    owned, records = normalize_call_boundary_aliases(selected)
    assert len(records) == 2
    assert len({record["module_call_entry"] for record in records}) == 2
    assert all(
        entry.signature.inputs[0].name == "sin"
        for entry in owned.module_call_graph
        if entry.fqn == "linear"
    )
    assert all(
        entry.signature.inputs[0].name == "type_as"
        for entry in selected.module_call_graph
        if entry.fqn == "linear"
    )


def test_real_nonidentity_type_conversion_is_left_for_framework_decomposition():
    class RealCast(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.leaf = torch.nn.ReLU()
            self.register_buffer("reference", torch.zeros(1, dtype=torch.float64))

        def forward(self, value):
            return self.leaf(value.type_as(self.reference))

    model = RealCast().eval()
    exported = torch.export.export(
        model,
        (torch.randn(2, 4),),
        preserve_module_call_signature=("leaf",),
        strict=True,
    )
    owned, records = normalize_call_boundary_aliases(exported)
    assert owned is exported and records == []
    prepared = owned.run_decompositions({})
    prepared.validate()
    assert prepared.module()(torch.randn(2, 4)).dtype == torch.float64
