"""Read-only grad contexts retain their real bodies and ordered callers."""

import copy

import pytest
import torch

from m2m.capture.context_trace import observe_contexts
from m2m.capture.dynamic_linear_trace import (
    DynamicLinearLineageError,
    LinearInstrumentation,
    _bind_pair,
)
from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization
from m2m.capture.trace import snapshot_exported_program

pytest.importorskip("torchao")


class ContextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32, bias=False)

    def forward(self, value, other):
        with torch.no_grad():
            hidden = value.sin().cos() * 2 + other
            retained = value.view_as(other)
        return self.linear(hidden) + retained


def fixture():
    torch.manual_seed(17)
    model = ContextModel().eval().requires_grad_(False)
    inputs = (torch.randn(1, 32), torch.randn(1, 32))
    quant = QuantizationConfig(scheme="int8_dyn_act_int8_weight")
    return model, inputs, quant


def test_actual_ao_context_binds_raw_body_context_and_ordered_operands():
    model, inputs, quant = fixture()
    with LinearInstrumentation(model) as owner:
        source = owner.capture_source(model, inputs)
        raw = copy.deepcopy(source.snapshot)
        apply_quantization(model, quant)
        selected, proof = owner.export_selected(
            source, model, inputs, quantization=quant
        )
        assert proof["verified_readonly_contexts"] == 1
        assert proof["caller_graph_and_order_verified"]
        assert proof["selected_export_exact_eager_output"]
        assert source.snapshot == raw
        for ep, stage in ((source.exported, "original"), (selected, "quantized")):
            observed = observe_contexts(ep, snapshot_exported_program(ep, stage=stage))
            assert len(observed) == 1
            context = next(iter(observed.values()))
            assert context.control is False
            assert any(row["target"] == "aten.sin.default" for row in context.body)
            assert any(row["target"] == "aten.view_as.default" for row in context.body)
        contexts = [
            node
            for node in selected.graph.nodes
            if node.target is torch.ops.higher_order.wrap_with_set_grad_enabled
        ]
        assert len(contexts) == 1  # The proof did not inline or erase the context.


@pytest.mark.parametrize(
    "change", ["flag", "body", "ordered_inputs", "output_slot", "alias_write"]
)
def test_changed_actual_context_refuses_without_source_attribution(change):
    model, inputs, quant = fixture()
    with LinearInstrumentation(model) as owner:
        source = owner.capture_source(model, inputs)
        apply_quantization(model, quant)
        selected = torch.export.export(
            model,
            inputs,
            strict=True,
            preserve_module_call_signature=tuple(source.sites),
        )
        hop = next(
            node
            for node in selected.graph.nodes
            if node.target is torch.ops.higher_order.wrap_with_set_grad_enabled
        )
        body = selected.graph_module.get_submodule(str(hop.args[1].target))
        if change == "flag":
            hop.args = (True, *hop.args[1:])
        elif change == "body":
            sine = next(
                node
                for node in body.graph.nodes
                if node.target is torch.ops.aten.sin.default
            )
            sine.target = torch.ops.aten.neg.default
        elif change == "ordered_inputs":
            assert len(hop.args) == 4
            hop.args = (*hop.args[:2], hop.args[3], hop.args[2])
        elif change == "output_slot":
            output = next(node for node in body.graph.nodes if node.op == "output")
            output.args = (tuple(reversed(output.args[0])),)
        else:
            projection = next(
                node
                for node in selected.graph.nodes
                if node.op == "call_function" and node.args == (hop, 1)
            )
            with selected.graph.inserting_after(projection):
                write = selected.graph.call_function(
                    torch.ops.aten.add_.Tensor, (projection, 1)
                )
            write.meta["val"] = projection.meta["val"]
        body.recompile()
        selected.graph_module.recompile()
        selected.graph.lint()
        with pytest.raises((DynamicLinearLineageError, ValueError)):
            _bind_pair(source.exported, selected, source.sites)
        assert not any(
            str(origin).startswith("g:original:")
            for node in selected.graph.nodes
            for origin in (node.meta.get("custom") or {}).get("m2m_lineage", [])
        )


@pytest.mark.parametrize("change", ["writing_body", "unknown_body", "shared_body"])
def test_body_effect_or_occurrence_ambiguity_is_not_a_context_proof(change):
    model, inputs, _ = fixture()
    ep = torch.export.export(model, inputs, strict=True)
    hop = next(
        node
        for node in ep.graph.nodes
        if node.target is torch.ops.higher_order.wrap_with_set_grad_enabled
    )
    body = ep.graph_module.get_submodule(str(hop.args[1].target))
    if change == "shared_body":
        with ep.graph.inserting_after(hop):
            extra = ep.graph.call_function(hop.target, hop.args)
        extra.meta = copy.copy(hop.meta)
    else:
        node = next(
            node
            for node in body.graph.nodes
            if node.target is torch.ops.aten.sin.default
        )
        if change == "writing_body":
            node.target = torch.ops.aten.sin_.default
        else:
            node.target = torch.sin
    body.recompile()
    ep.graph_module.recompile()
    snapshot = snapshot_exported_program(ep, stage="original")
    assert not observe_contexts(ep, snapshot)


@pytest.mark.parametrize("change", ["dtype", "stride", "output_dtype"])
def test_context_formal_and_return_metadata_must_match_exact_tensor_edges(change):
    model, inputs, _ = fixture()
    ep = torch.export.export(model, inputs, strict=True)
    hop = next(
        node
        for node in ep.graph.nodes
        if node.target is torch.ops.higher_order.wrap_with_set_grad_enabled
    )
    body = ep.graph_module.get_submodule(str(hop.args[1].target))
    formal = next(node for node in body.graph.nodes if node.op == "placeholder")
    value = formal.meta["val"]
    if change == "dtype":
        formal.meta["val"] = value.to(torch.float64)
    elif change == "stride":
        formal.meta["val"] = torch.empty_strided(
            value.shape, (64, 2), dtype=value.dtype, device=value.device
        )
    else:
        hop.meta["val"] = tuple(value.to(torch.float64) for value in hop.meta["val"])
    snapshot = snapshot_exported_program(ep, stage="original")
    assert not observe_contexts(ep, snapshot)
