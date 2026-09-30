"""Exact integer-reference checks at the PT2E Q/DQ → integer-graph boundary."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference
from m2m.capture.pt2e_integerize import integerize_pt2e


def _per_tensor_pt2e(model: nn.Module, inputs: tuple[torch.Tensor, ...]):
    """Build a target-neutral static PT2E graph with tensor-scale W and A."""
    from torchao.quantization.pt2e import HistogramObserver, MinMaxObserver
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    from torchao.quantization.pt2e.quantizer import (
        QuantizationAnnotation,
        QuantizationSpec,
        Quantizer,
    )

    class QuantizerForTest(Quantizer):
        def __init__(self):
            self.activation = QuantizationSpec(
                dtype=torch.int8,
                observer_or_fake_quant_ctr=HistogramObserver.with_args(eps=2**-12),
                quant_min=-128, quant_max=127, qscheme=torch.per_tensor_symmetric,
            )
            self.weight = QuantizationSpec(
                dtype=torch.int8,
                observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=2**-12),
                quant_min=-127, quant_max=127, qscheme=torch.per_tensor_symmetric,
            )

        def annotate(self, graph_module):
            for node in graph_module.graph.nodes:
                if node.target in (torch.ops.aten.linear.default, torch.ops.aten.conv2d.default):
                    node.meta["quantization_annotation"] = QuantizationAnnotation(
                        input_qspec_map={
                            node.args[0]: self.activation, node.args[1]: self.weight,
                        },
                        _annotated=True,
                    )
            return graph_module

        def validate(self, graph_module):
            del graph_module

    exported = torch.export.export(model.eval(), inputs).module()
    prepared = prepare_pt2e(exported, QuantizerForTest())
    with torch.no_grad():
        prepared(*inputs)
    converted = convert_pt2e(prepared, use_reference_representation=False, fold_quantize=True)
    from torch.ao.quantization import allow_exported_model_train_eval

    allow_exported_model_train_eval(converted)
    return converted


@pytest.mark.parametrize("kind", ["linear", "conv2d"])
def test_per_tensor_pt2e_exact_integer_reference_and_export(kind):
    import m2m
    from m2m.capture.torchao_pipeline import QuantizationConfig

    torch.manual_seed(3)
    if kind == "linear":
        model = nn.Linear(8, 5)
        inputs = (torch.randn(2, 8),)
    else:
        model = nn.Conv2d(3, 5, 3, padding=1, stride=2)
        inputs = (torch.randn(1, 3, 7, 7),)
    converted = _per_tensor_pt2e(model, inputs)
    dequant = torch.ops.quantized_decomposed.dequantize_per_tensor.default
    assert sum(n.target == dequant for n in converted.graph.nodes) == 2
    independent = run_pt2e_integer_reference(converted, inputs, expected_contractions=1)
    portable = converted(*inputs)
    integerized, receipt = integerize_pt2e(converted, inputs)
    assert receipt["quantized_contractions_seen"] == 1
    assert receipt["quantized_contractions_remaining"] == 0
    assert receipt["integer_mm_emitted"] == 1
    assert receipt["accumulator_bound_checked"] is True
    assert receipt["refusals"] == []
    assert receipt["qparams"][0]["weight"]["granularity"] == "tensor"
    assert sum(n.target == torch.ops.aten._int_mm.default for n in integerized.graph.nodes) == 1
    assert torch.equal(independent.output, integerized(*inputs))
    assert torch.allclose(portable, independent.output, atol=1e-5, rtol=1e-5)

    lowered = m2m.convert(
        integerized, inputs, backend="fx_importer", level="linalg-on-tensors",
        quantization=QuantizationConfig("int8_static_act_int8_weight"),
        quantization_preapplied=True,
    )
    assert lowered.ok, lowered.diagnostics
    assert "opaque=0" in str(lowered.diagnostics) or "0 opaque" in str(lowered.diagnostics)
    assert "xi32" in lowered.mlir_text


def test_per_channel_reference_matches_integerized_conv_linear():
    from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 2, 3, padding=1)
            self.linear = nn.Linear(2, 3)

        def forward(self, x):
            return self.linear(self.conv(x).mean((2, 3)))

    inputs = (torch.randn(1, 2, 4, 4),)
    converted = apply_quantization(
        Model().eval(), QuantizationConfig("int8_static_act_int8_weight", calibration_samples=1),
        example_inputs=inputs, calibration_inputs=[inputs],
    )
    independent = run_pt2e_integer_reference(converted, inputs, expected_contractions=2)
    integerized, receipt = integerize_pt2e(converted, inputs)
    assert independent.conv2d_count == independent.linear_count == 1
    assert independent.matmul_count == 0
    assert receipt["quantized_contractions_integerized"] == 2
    assert torch.equal(independent.output, integerized(*inputs))


def test_integer_reference_preserves_unselected_attention_matmul():
    """Only Q/DQ-selected contractions use integer arithmetic; attention stays float."""
    torch.manual_seed(7)

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(8, 8, bias=False)
            self.key = nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.query(x) @ self.key(x).transpose(-2, -1)

    inputs = (torch.randn(1, 4, 8),)
    converted = _per_tensor_pt2e(Attention(), inputs)
    independent = run_pt2e_integer_reference(converted, inputs, expected_contractions=2)
    integerized, receipt = integerize_pt2e(converted, inputs)
    assert receipt["linear_integerized"] == 2
    assert receipt["matmul_seen"] == 1
    assert receipt["matmul_integerized"] == 0
    assert independent.linear_count == 2
    assert independent.matmul_count == 0
    assert torch.equal(independent.output, integerized(*inputs))


def test_explicit_per_tensor_matmul_qdq_exact_integer_reference():
    """A static non-broadcast matmul is supported when both Q/DQ operands exist."""
    from torch.fx import Graph, GraphModule

    class Parameters(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("wq", torch.arange(12, dtype=torch.int8).reshape(4, 3))

    graph = Graph()
    x = graph.placeholder("x")
    wq = graph.get_attr("wq")
    qx = graph.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default,
                             args=(x, 0.125, 0, -128, 127, torch.int8))
    dx = graph.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                             args=(qx, 0.125, 0, -128, 127, torch.int8))
    dw = graph.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                             args=(wq, 0.0625, 0, -127, 127, torch.int8))
    graph.output(graph.call_function(torch.ops.aten.mm.default, args=(dx, dw)))
    converted = GraphModule(Parameters().eval(), graph)
    inputs = (torch.randn(2, 4),)
    independent = run_pt2e_integer_reference(converted, inputs, expected_contractions=1)
    integerized, receipt = integerize_pt2e(converted, inputs)
    assert independent.matmul_count == receipt["matmul_integerized"] == 1
    assert receipt["quantized_contractions_remaining"] == 0
    assert torch.equal(independent.output, integerized(*inputs))


def test_unsupported_reduction_axis_fails_closed():
    """A scale along K cannot be factored out of an integer dot product."""
    from torch.fx import Graph, GraphModule

    class BadMatmul(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("wq", torch.arange(12, dtype=torch.int8).reshape(4, 3))
            self.register_buffer("scales", torch.ones(4, dtype=torch.float32) / 8)
            self.register_buffer("zeros", torch.zeros(4, dtype=torch.int64))

    inputs = (torch.randn(2, 4),)
    graph = Graph()
    x = graph.placeholder("x")
    wq = graph.get_attr("wq")
    scales = graph.get_attr("scales")
    zeros = graph.get_attr("zeros")
    qx = graph.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default,
                             args=(x, 0.125, 0, -128, 127, torch.int8))
    dx = graph.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                             args=(qx, 0.125, 0, -128, 127, torch.int8))
    dw = graph.call_function(torch.ops.quantized_decomposed.dequantize_per_channel.default,
                             args=(wq, scales, zeros, 0, -127, 127, torch.int8))
    graph.output(graph.call_function(torch.ops.aten.mm.default, args=(dx, dw)))
    converted = GraphModule(BadMatmul().eval(), graph)
    with pytest.raises(ValueError, match="qparam axes"):
        run_pt2e_integer_reference(converted, inputs)
    integerized, receipt = integerize_pt2e(converted, inputs)
    assert receipt["quantized_contractions_seen"] == 1
    assert receipt["quantized_contractions_integerized"] == 0
    assert receipt["quantized_contractions_remaining"] == 1
    assert receipt["refusals"] and "output dimension" in receipt["refusals"][0]["reason"]
    assert not any(n.target == torch.ops.aten._int_mm.default for n in integerized.graph.nodes)
