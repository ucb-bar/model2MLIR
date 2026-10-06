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


def test_integer_reference_preserves_float_attention_after_integer_linears():
    """An attention matmul over linear results is not itself a PT2E Q/DQ contraction."""
    from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference

    class Attention(nn.Module):
        def __init__(self, style):
            super().__init__()
            self.style = style
            self.register_buffer("query_weight", torch.arange(64).remainder(13).sub(6).reshape(8, 8).to(torch.int8))
            self.register_buffer("key_weight", torch.arange(64).remainder(11).sub(5).reshape(8, 8).to(torch.int8))

        def dequantize(self, value):
            return torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                value, 0.125, 0, -128, 127, torch.int8)

        def forward(self, x, context):
            query = torch.ops.aten.linear.default(
                self.dequantize(torch.ops.quantized_decomposed.quantize_per_tensor.default(
                    x, 0.125, 0, -128, 127, torch.int8)),
                self.dequantize(self.query_weight), None)
            key = torch.ops.aten.linear.default(
                self.dequantize(torch.ops.quantized_decomposed.quantize_per_tensor.default(
                    context, 0.125, 0, -128, 127, torch.int8)),
                self.dequantize(self.key_weight), None)
            if self.style == "causal_decoder":
                query = query.reshape(1, 2, 2, 4).transpose(1, 2)
                key = key.reshape(1, 2, 2, 4).transpose(1, 2)
            return query @ key.transpose(-2, -1)

    for style, context_length in (("causal_decoder", 2), ("multimodal_policy", 3)):
        inputs = (torch.arange(16, dtype=torch.float32).reshape(1, 2, 8) / 8,
                  torch.arange(context_length * 8, dtype=torch.float32).reshape(1, context_length, 8) / 8)
        model = torch.export.export(Attention(style).eval(), inputs).module()
        portable = model(*inputs)
        reference = run_pt2e_integer_reference(model, inputs, expected_contractions=2)
        rewritten, receipt = integerize_pt2e(model, inputs)
        assert receipt["linear_integerized"] == 2
        assert receipt["matmul_integerized"] == 0
        assert reference.linear_count == 2 and reference.matmul_count == 0
        torch.testing.assert_close(reference.output, rewritten(*inputs), atol=0, rtol=0)
        torch.testing.assert_close(reference.output, portable, atol=1e-5, rtol=1e-5)


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


def test_bf16_dequant_linear_requires_a_distinct_precision_contract():
    """BF16 Q/DQ rounds each operand before the contraction, unlike W8A8/i32."""
    from types import SimpleNamespace

    import pytest

    class Bf16QDQLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.tensor([[120, -110, 61, 90]], dtype=torch.int8))

        def forward(self, qx):
            dx = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                qx, 0.0137, 0, -128, 127, torch.int8, out_dtype=torch.bfloat16)
            dw = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                self.weight, 0.0193, 0, -127, 127, torch.int8, out_dtype=torch.bfloat16)
            return torch.ops.aten.linear.default(dx, dw)

    qx = torch.tensor([[123, 117, -90, 31]], dtype=torch.int8)
    exported = torch.export.export(Bf16QDQLinear(), (qx,)).module()
    portable = exported(qx)
    from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference

    reference = run_pt2e_integer_reference(exported, (qx,), expected_contractions=0)
    torch.testing.assert_close(reference.output, portable, atol=0, rtol=0)
    integer_result = ((qx.int() @ exported.weight.int().T).float() * 0.0137 * 0.0193).to(torch.bfloat16)
    assert abs(float(portable.item()) - float(integer_result.item())) > 0.005

    rewritten, receipt = integerize_pt2e(exported, (qx,))
    assert receipt["quantized_contractions_seen"] == receipt["quantized_contractions_remaining"] == 1
    assert receipt["linear_integerized"] == receipt["integer_mm_emitted"] == 0
    assert receipt["remaining_dequant_count"] == 2
    assert len(receipt["refusals"]) == 1
    assert "torch.bfloat16" in receipt["refusals"][0]["reason"]
    assert receipt["precision_decision_counts"] == {"integerized_i32": 0, "preserve_float_qdq": 1, "unresolved": 0}
    torch.testing.assert_close(rewritten(qx), portable, atol=0, rtol=0)
    from torch.ao.quantization import allow_exported_model_train_eval

    allow_exported_model_train_eval(rewritten)
    lowered = m2m.convert(rewritten, (qx,), backend="fx_importer")
    assert lowered.ok and opaque_report(lowered.mlir_text) == {}
    assert "quant_ext.dequantize_per_tensor" in lowered.mlir_text
    assert "tensor<1x4xbf16>" in lowered.mlir_text
    assert 'prov.aten = "aten._int_mm.default"' not in lowered.mlir_text
    dequant = torch.ops.quantized_decomposed.dequantize_per_tensor.default
    bf16_node = next(node for node in exported.graph.nodes
                     if node.target == dequant and node.kwargs.get("out_dtype") == torch.bfloat16)
    bf16_node.meta["tensor_meta"] = SimpleNamespace(dtype=torch.float32)
    with pytest.raises(ValueError, match="dequantize output dtype disagrees with its declaration"):
        run_pt2e_integer_reference(exported, (qx,), expected_contractions=0)


def test_mixed_qdq_precision_records_integer_and_float_lanes():
    """One supported integer rewrite must not relabel a neighbouring BF16 Q/DQ site."""
    class MixedQDQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("integer_weight", torch.tensor([[2, 3, -4, 5]], dtype=torch.int8))
            self.register_buffer("bf16_weight", torch.tensor([[120, -110, 61, 90]], dtype=torch.int8))

        def forward(self, qx):
            dx_f32 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                qx, 0.125, 0, -128, 127, torch.int8)
            dw_f32 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                self.integer_weight, 0.0625, 0, -127, 127, torch.int8)
            dx_bf16 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                qx, 0.0137, 0, -128, 127, torch.int8, out_dtype=torch.bfloat16)
            dw_bf16 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                self.bf16_weight, 0.0193, 0, -127, 127, torch.int8, out_dtype=torch.bfloat16)
            return torch.ops.aten.linear.default(dx_f32, dw_f32), torch.ops.aten.linear.default(dx_bf16, dw_bf16)

    qx = torch.tensor([[123, 117, -90, 31]], dtype=torch.int8)
    exported = torch.export.export(MixedQDQ(), (qx,)).module()
    portable = exported(qx)
    from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference

    reference = run_pt2e_integer_reference(exported, (qx,), expected_contractions=1)
    assert reference.linear_count == 1
    assert len(reference.partial_sum_evidence) == 1
    torch.testing.assert_close(reference.output[1], portable[1], atol=0, rtol=0)
    rewritten, receipt = integerize_pt2e(exported, (qx,))
    assert receipt["quantized_contractions_seen"] == 2
    assert receipt["quantized_contractions_integerized"] == 1
    assert receipt["quantized_contractions_remaining"] == 1
    assert receipt["precision_decision_counts"] == {"integerized_i32": 1, "preserve_float_qdq": 1, "unresolved": 0}
    sites = receipt["precision_decisions"]
    assert len(sites) == receipt["quantized_contractions_seen"]
    assert {row["decision"] for row in sites} == {"integerized_i32", "preserve_float_qdq"}
    assert next(row for row in sites if row["decision"] == "preserve_float_qdq")["source_dtype"] == "torch.bfloat16"
    actual = rewritten(qx)
    torch.testing.assert_close(reference.output[0], actual[0], atol=0, rtol=0)
    torch.testing.assert_close(reference.output[1], actual[1], atol=0, rtol=0)
    torch.testing.assert_close(actual[0], portable[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1], portable[1], atol=0, rtol=0)

    from torch.ao.quantization import allow_exported_model_train_eval

    allow_exported_model_train_eval(rewritten)
    lowered = m2m.convert(rewritten, (qx,), backend="fx_importer")
    assert lowered.ok and opaque_report(lowered.mlir_text) == {}
    assert lowered.mlir_text.count('prov.aten = "aten._int_mm.default"') >= 1
    assert "quant_ext.dequantize_per_tensor" in lowered.mlir_text
    assert "tensor<1x4xbf16>" in lowered.mlir_text
