"""Coarse capture checks for factorable, frozen per-channel PT2E qparams."""

import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401 -- registers portable PT2E ops
from torch import nn

import m2m
from m2m.capture.pt2e_integerize import integerize_pt2e
from m2m.coverage import opaque_report


class ChannelContraction(nn.Module):
    def __init__(self, kind, *, axis=None, zero_point=0, scale_dtype=torch.float32):
        super().__init__()
        self.kind = kind
        shape = {"linear": (5, 8), "conv2d": (5, 3, 3, 3), "matmul": (2, 8, 5)}[kind]
        self.axis = (2 if kind == "matmul" else 0) if axis is None else axis
        self.register_buffer("weight", torch.arange(torch.tensor(shape).prod()).reshape(shape).remainder(15).sub(7).to(torch.int8))
        self.register_buffer("scales", torch.arange(1, shape[self.axis] + 1, dtype=scale_dtype) / 16)
        self.register_buffer("zeros", torch.full((shape[self.axis],), zero_point, dtype=torch.int64))
        self.register_buffer("bias", torch.arange(5, dtype=torch.float32) / 8)

    def forward(self, x):
        qx = torch.ops.quantized_decomposed.quantize_per_tensor.default(x, 0.125, 0, -128, 127, torch.int8)
        dx = torch.ops.quantized_decomposed.dequantize_per_tensor.default(qx, 0.125, 0, -128, 127, torch.int8)
        dw = torch.ops.quantized_decomposed.dequantize_per_channel.default(
            self.weight, self.scales, self.zeros, self.axis, -127, 127, torch.int8)
        if self.kind == "linear":
            return torch.ops.aten.linear.default(dx, dw, self.bias)
        if self.kind == "conv2d":
            return torch.ops.aten.conv2d.default(dx, dw, self.bias, [2, 2], [1, 1], [2, 2])
        return torch.ops.aten.matmul.default(dx, dw)


def _inputs(kind):
    shape = {"linear": (2, 3, 8), "conv2d": (2, 3, 8, 8), "matmul": (2, 4, 8)}[kind]
    return (torch.arange(torch.tensor(shape).prod()).reshape(shape).remainder(31).sub(15).float() / 8,)


def test_frozen_output_channel_linear_conv_and_batched_matmul_capture():
    """Each real export rewrites, executes, and lowers with exact source accounting."""
    from torch.ao.quantization import allow_exported_model_train_eval
    from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference

    for kind in ("linear", "conv2d", "matmul"):
        inputs = _inputs(kind)
        model = torch.export.export(ChannelContraction(kind).eval(), inputs).module()
        expected = model(*inputs)
        reference = run_pt2e_integer_reference(model, inputs, expected_contractions=1) if kind == "matmul" else None
        model, receipt = integerize_pt2e(model, inputs)
        assert receipt["quantized_contractions_integerized"] == 1, receipt
        assert receipt["remaining_dequant_count"] == 0 and receipt["refusals"] == []
        torch.testing.assert_close(model(*inputs), expected, atol=1e-5, rtol=1e-5)
        if reference is not None:
            torch.testing.assert_close(model(*inputs), reference.output, atol=0, rtol=0)
        allow_exported_model_train_eval(model)
        result = m2m.convert(model, inputs, backend="fx_importer", capture_trace=True)
        assert result.ok and opaque_report(result.mlir_text) == {}
        assert result.capture_trace["status"] == "complete"
        assert 'prov.aten = "aten._int_mm.default"' in result.mlir_text


def test_channel_reduction_axes_and_nonzero_zero_points_remain_diagnostic():
    for kind, axis, zp in (("linear", 1, 0), ("conv2d", 1, 0), ("matmul", 1, 0), ("linear", 0, 3)):
        inputs = _inputs(kind)
        model = torch.export.export(ChannelContraction(kind, axis=axis, zero_point=zp).eval(), inputs).module()
        expected = model(*inputs)
        model, receipt = integerize_pt2e(model, inputs)
        assert receipt["quantized_contractions_seen"] == receipt["quantized_contractions_remaining"] == 1
        assert receipt["quantized_contractions_integerized"] == 0 and receipt["refusals"]
        assert receipt["remaining_dequant_count"] == 2
        torch.testing.assert_close(model(*inputs), expected)


def test_channel_scale_dtype_and_qparams_are_not_substituted():
    from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference

    inputs = _inputs("linear")
    model = torch.export.export(ChannelContraction("linear", scale_dtype=torch.float64).eval(), inputs).module()
    expected = model(*inputs)
    reference = run_pt2e_integer_reference(model, inputs, expected_contractions=1)
    model, receipt = integerize_pt2e(model, inputs)
    assert receipt["quantized_contractions_integerized"] == 1, receipt
    assert model.scales.dtype == torch.float64
    assert model(*inputs).dtype == expected.dtype == torch.float32
    torch.testing.assert_close(reference.output, model(*inputs), atol=0, rtol=0)
    torch.testing.assert_close(model(*inputs), expected, atol=1e-5, rtol=1e-5)


def test_actual_torchao_pt2e_channel_pipeline_preserves_original_trace():
    from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization

    for model, inputs in ((nn.Sequential(nn.Linear(8, 5), nn.ReLU()).eval(), _inputs("linear")),
                          (nn.Sequential(nn.Conv2d(3, 5, 3, padding=1), nn.ReLU()).eval(), _inputs("conv2d"))):
        original = m2m.capture_frontend_snapshot(model, inputs)
        config = QuantizationConfig(scheme="int8_static_act_int8_weight")
        quantized = apply_quantization(model, config, example_inputs=inputs,
                                      original_frontend_snapshot=original)
        portable = quantized(*inputs)
        integer, receipt = integerize_pt2e(quantized, inputs)
        assert receipt["quantized_contractions_integerized"] == 1 and receipt["refusals"] == []
        assert receipt["qparams"][0]["weight"]["granularity"] == "channel"
        torch.testing.assert_close(integer(*inputs), portable, atol=1e-5, rtol=1e-5)
        result = m2m.convert(integer, inputs, backend="fx_importer", capture_trace=True,
                             quantization=config, quantization_preapplied=True,
                             original_frontend_snapshot=original)
        assert result.ok and result.capture_trace["status"] == "complete"
        assert result.capture_trace["graphs"]["original"]["sha256"] == original["sha256"]


def test_dynamic_or_nonfinite_channel_scales_cannot_qualify():
    class Dynamic(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.ones(5, 8, dtype=torch.int8))
            self.register_buffer("zeros", torch.zeros(5, dtype=torch.int64))

        def forward(self, x, scales):
            qx = torch.ops.quantized_decomposed.quantize_per_tensor.default(x, 0.125, 0, -128, 127, torch.int8)
            dx = torch.ops.quantized_decomposed.dequantize_per_tensor.default(qx, 0.125, 0, -128, 127, torch.int8)
            dw = torch.ops.quantized_decomposed.dequantize_per_channel.default(
                self.weight, scales, self.zeros, 0, -127, 127, torch.int8)
            return torch.ops.aten.linear.default(dx, dw)

    inputs = (*_inputs("linear"), torch.ones(5))
    model = torch.export.export(Dynamic(), inputs).module()
    _, receipt = integerize_pt2e(model, inputs)
    assert receipt["quantized_contractions_remaining"] == 1
    assert "frozen" in receipt["refusals"][0]["reason"]
    for scale in (0.0, -1.0, float("nan"), float("inf")):
        module = ChannelContraction("linear")
        module.scales.fill_(scale)
        model = torch.export.export(module, _inputs("linear")).module()
        _, receipt = integerize_pt2e(model, _inputs("linear"))
        assert receipt["quantized_contractions_remaining"] == 1
        assert "finite positive" in receipt["refusals"][0]["reason"]
