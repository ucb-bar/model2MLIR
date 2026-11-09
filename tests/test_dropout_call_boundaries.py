"""Disabled dropout aliases cannot leave erased module-call signature slots."""

import copy

import pytest
import torch

import m2m
from m2m.capture.call_boundary_trace import normalize_call_boundary_aliases
from m2m.capture.dynamic_linear_trace import LinearInstrumentation
from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization
from m2m.capture.trace import _forward_identity_value, snapshot_exported_program


class Model(torch.nn.Module):
    def __init__(self, probability=0.0, chain=False):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
        self.probability, self.chain = probability, chain
        self.register_buffer("reference", torch.zeros(1))

    def forward(self, value):
        hidden = value.sin()
        hidden = torch.nn.functional.dropout(hidden, self.probability, training=False)
        if self.chain:
            hidden = hidden.type_as(self.reference)
        return self.linear(hidden)


def source_model(probability=0.0, chain=False):
    model = Model(probability, chain).eval().requires_grad_(False)
    inputs = (torch.linspace(-1, 1, 32).reshape(2, 16),)
    return model, inputs


def test_actual_AO_source_boundary_keeps_raw_contract_and_complete_trace():
    model, inputs = source_model()
    quant = QuantizationConfig(scheme="int8_dyn_act_int8_weight")
    with LinearInstrumentation(model) as owner:
        source = owner.capture_source(model, inputs)
        apply_quantization(model, quant)
        selected, _ = owner.export_selected(source, model, inputs, quantization=quant)
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
    assert len(records) == 1 and records[0]["identity_chain"] == ["dropout"]
    assert records[0]["typed_identity_and_consumer_lifetime_verified"]
    assert records[0]["graph_arithmetic_changed"] is False


@pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("chain", [False, True])
def test_disabled_dropout_and_type_as_chain_change_only_owned_call_metadata(
    probability, chain
):
    model, inputs = source_model(probability, chain)
    exported = torch.export.export(
        model, inputs, preserve_module_call_signature=("linear",)
    )
    before = snapshot_exported_program(exported, stage="quantized")
    calls = copy.deepcopy(exported.module_call_graph)
    owned, records = normalize_call_boundary_aliases(exported)
    assert owned is not exported and len(records) == 1
    assert snapshot_exported_program(owned, stage="quantized") == before
    assert exported.module_call_graph == calls
    assert all(
        owned.state_dict[key] is value for key, value in exported.state_dict.items()
    )
    assert len(records[0]["identity_chain"]) == (2 if chain else 1)
    for entry in owned.module_call_graph:
        if entry.fqn == "linear":
            assert entry.signature.inputs[0].name == "sin"
    prepared = owned.run_decompositions({})
    prepared.validate()
    torch.testing.assert_close(
        prepared.module()(*inputs), model(*inputs), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "mutation", ["training", "probability", "edge", "dtype", "stride", "schema"]
)
def test_snapshot_identity_requires_source_controls_exact_edges_and_types(mutation):
    model, inputs = source_model()
    exported = torch.export.export(
        model, inputs, preserve_module_call_signature=("linear",)
    )
    snapshot = snapshot_exported_program(exported, stage="quantized")
    row = next(r for r in snapshot["nodes"] if r["target"] == "aten.dropout.default")
    if mutation == "training":
        row["args"][2] = True
    elif mutation == "probability":
        row["args"][1] = float("nan")
    elif mutation == "edge":
        row["args"][0]["value_id"] = "unknown-result"
    elif mutation == "dtype":
        row["results"][0]["dtype"] = "float16"
    elif mutation == "stride":
        row["results"][0]["stride"] = [32, 2]
    else:
        snapshot["operator_schemas"][row["target"]] = str(
            torch.ops.aten.relu.default._schema
        )
    assert _forward_identity_value(snapshot, row) is None


@pytest.mark.parametrize("writer_target", ["input", "dropout"])
def test_alias_write_before_callee_refuses_instead_of_rebinding_a_stale_value(
    writer_target,
):
    model, inputs = source_model()
    exported = torch.export.export(
        model, inputs, preserve_module_call_signature=("linear",)
    )
    alias = next(
        n for n in exported.graph.nodes if n.target is torch.ops.aten.dropout.default
    )
    with exported.graph.inserting_after(alias):
        writer = exported.graph.call_function(
            torch.ops.aten.add_.Tensor,
            (alias.args[0] if writer_target == "input" else alias, 1.0),
        )
    writer.meta["val"] = alias.meta["val"]
    exported.graph_module.recompile()
    exported.validate()
    with pytest.raises(ValueError, match="alias/effect/consumer lifetime"):
        normalize_call_boundary_aliases(exported)


def test_training_dropout_is_not_claimed_as_an_identity():
    class Training(Model):
        def forward(self, value):
            return self.linear(torch.nn.functional.dropout(value, 0.5, training=True))

    model = Training().eval().requires_grad_(False)
    exported = torch.export.export(
        model, (torch.ones(2, 16),), preserve_module_call_signature=("linear",)
    )
    owned, records = normalize_call_boundary_aliases(exported)
    assert owned is exported and records == []


def test_owned_output_signature_follows_the_same_proved_tensor_identity():
    class Leaf(torch.nn.Module):
        def forward(self, value):
            return torch.nn.functional.dropout(value.sin(), 0.25, training=False)

    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.leaf = Leaf()

        def forward(self, value):
            return self.leaf(value).cos()

    model = Parent().eval()
    inputs = (torch.ones(2, 4),)
    exported = torch.export.export(
        model, inputs, preserve_module_call_signature=("leaf",)
    )
    before = copy.deepcopy(exported.module_call_graph)
    snapshot = snapshot_exported_program(exported, stage="quantized")
    owned, records = normalize_call_boundary_aliases(exported)
    assert len(records) == 1 and records[0]["direction"] == "outputs"
    assert exported.module_call_graph == before
    assert snapshot_exported_program(owned, stage="quantized") == snapshot
    prepared = owned.run_decompositions({})
    prepared.validate()
    torch.testing.assert_close(
        prepared.module()(*inputs), model(*inputs), rtol=0, atol=0
    )
