"""A successful source proof cannot authorize later changed selected bytes."""

import copy

import pytest
import torch

import m2m
from m2m.capture.dynamic_linear_trace import LinearInstrumentation
from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization
from m2m.capture.trace_copy import copy_trace_program

pytest.importorskip("torchao")


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)
        self.register_buffer("offset", torch.full((1, 32), 0.25))

    def forward(self, value):
        return self.linear(value) + self.offset


def fixture():
    torch.manual_seed(39)
    model = Model().eval().requires_grad_(False)
    inputs = (torch.randn(1, 32),)
    quant = QuantizationConfig(scheme="int8_dyn_act_int8_weight")
    with LinearInstrumentation(model) as owner:
        source = owner.capture_source(model, inputs)
        apply_quantization(model, quant)
        selected, proof = owner.export_selected(
            source, model, inputs, quantization=quant
        )
    return model, inputs, quant, source, selected, proof


def convert(selected, inputs, quant, source):
    return m2m.convert(
        selected,
        inputs,
        quantization=quant,
        quantization_preapplied=True,
        original_frontend_snapshot=source.snapshot,
        capture_trace=True,
        backend="fx_importer",
    )


def test_owned_unchanged_selected_program_can_be_converted_after_owner_closes():
    _, inputs, quant, source, selected, proof = fixture()
    for _ in range(2):
        result = convert(selected, inputs, quant, source)
        assert result.ok and result.capture_trace["status"] == "complete"
    assert proof["selected_input_sha256"]


@pytest.mark.parametrize(
    "change",
    [
        "qdata",
        "scale",
        "zero_point",
        "bias",
        "buffer",
        "original_state",
        "owner",
        "leaf_owner",
        "leaf_storage",
        "recipe",
        "graph",
        "proof",
        "input",
        "input_owner",
        "removed_binding",
        "copied_record_only",
        "state_roster",
    ],
)
def test_later_selected_or_source_changes_refuse_without_repair_or_new_source_ids(
    change,
):
    model, inputs, quant, source, selected, proof = fixture()
    weight = model.linear.weight
    if change in ("qdata", "scale", "zero_point"):
        getattr(weight, change).reshape(-1)[0] += 1
    elif change == "bias":
        model.linear.bias.reshape(-1)[0] += 1
    elif change == "buffer":
        model.offset.reshape(-1)[0] += 1
    elif change == "original_state":
        source.seals[0].value.reshape(-1)[0] += 1
    elif change == "owner":
        selected.state_dict["linear.weight"] = copy.deepcopy(weight)
    elif change == "leaf_owner":
        weight.qdata = weight.qdata.clone()
    elif change == "leaf_storage":
        weight.qdata.data = weight.qdata.clone()
    elif change == "recipe":
        weight.reduce_range = not weight.reduce_range
    elif change == "graph":
        incoming = next(
            node
            for node in selected.graph.nodes
            if node.op == "placeholder" and node.name == "value"
        )
        with selected.graph.inserting_after(incoming):
            extra = selected.graph.call_function(
                torch.ops.aten.neg.default, (incoming,)
            )
        extra.meta["val"] = incoming.meta["val"]
        selected.graph_module.recompile()
    elif change == "proof":
        proof["site_bindings"][0]["source_weight_sha256"] = "0" * 64
    elif change == "input":
        inputs[0].reshape(-1)[0] += 1
    elif change == "input_owner":
        inputs = (inputs[0].clone(),)
    elif change == "removed_binding":
        del selected._m2m_dynamic_linear_binding
    elif change == "copied_record_only":
        selected = copy_trace_program(selected)
    else:
        selected.constants["unexpected"] = torch.ones(1)
    before = {
        node.name: copy.deepcopy(node.meta.get("custom"))
        for node in selected.graph.nodes
    }
    result = convert(selected, inputs, quant, source)
    assert not result.ok and result.capture_trace["status"] == "diagnostic"
    assert any(
        "sealed" in error
        or "binding" in error
        or "captured occurrence" in error
        or "roster differs" in error
        for error in result.diagnostics
    )
    assert before == {
        node.name: node.meta.get("custom") for node in selected.graph.nodes
    }


def test_source_storage_replacement_before_selected_export_refuses_even_with_equal_bytes():
    from m2m.capture.dynamic_linear_trace import DynamicLinearLineageError

    model = Model().eval().requires_grad_(False)
    inputs = (torch.randn(1, 32),)
    quant = QuantizationConfig(scheme="int8_dyn_act_int8_weight")
    with LinearInstrumentation(model) as owner:
        source = owner.capture_source(model, inputs)
        model.linear.weight.data = model.linear.weight.clone()
        apply_quantization(model, quant)
        with pytest.raises(DynamicLinearLineageError, match="sealed"):
            owner.export_selected(source, model, inputs, quantization=quant)


@pytest.mark.parametrize(
    "value", [torch.tensor(3, dtype=torch.int64), torch.arange(40)[39:40]]
)
def test_scalar_and_one_element_strided_input_bytes_hash_without_stride_assumptions(
    value,
):
    import hashlib

    from m2m.capture.dynamic_linear_trace import _sha_tensor

    if value.ndim:
        value = torch.arange(80, dtype=torch.int64)[::40].reshape(1, 2)[:, 1:2]
        assert value.numel() == 1 and value.stride(-1) == 40
    expected = hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
    assert _sha_tensor(value) == expected


def test_zero_rank_original_buffer_and_constant_keep_exact_original_ao_source_relation():
    class Scalar(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 32, bias=False)
            self.register_buffer("factor", torch.tensor(0.25))

        def forward(self, value):
            return self.linear(value) * self.factor + torch.tensor(0.125)

    model = Scalar().eval().requires_grad_(False)
    inputs = (torch.ones(1, 32),)
    quant = QuantizationConfig(scheme="int8_dyn_act_int8_weight")
    with LinearInstrumentation(model) as owner:
        source = owner.capture_source(model, inputs)
        apply_quantization(model, quant)
        selected, _ = owner.export_selected(source, model, inputs, quantization=quant)
        result = convert(selected, inputs, quant, source)
        assert result.ok and result.capture_trace["status"] == "complete"
