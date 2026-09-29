"""MX operand boundaries and the TorchAO graph route."""

import pytest
import torch

from m2m.capture.mx_gemmini_quant import (
    MXGemminiLinear,
    _dequant_operand_for_graph,
    functional_contraction_operands,
    linear_contraction_operands,
    quantize_mx_gemmini,
)


def test_mx_operand_rounding_scale_floor_and_fp4_intermediate():
    values = torch.zeros(32, dtype=torch.bfloat16)
    values[0] = 1.0  # shared exponent zero
    values[1] = 0.1328125  # E4M3 tie: nearest even is 0.125
    values[2] = 0.28125  # E3M2 tie: nearest even is 0.25
    values[3] = 0.26  # direct E2M1 -> 0.5, E3M1 -> E2M1 -> 0
    values[4] = -0.5
    expected = {"mxfp8": (0.125, 0.28125), "mxfp6": (0.125, 0.25), "mxfp4": (0.0, 0.5)}
    for fmt in expected:
        dequant, codes, scales = quantize_mx_gemmini(values, fmt)
        assert scales.tolist() == [127]
        assert codes.dtype is torch.uint8
        assert dequant[4].item() < 0
        if fmt == "mxfp8":
            assert dequant[1].item() == expected[fmt][0]
        elif fmt == "mxfp6":
            assert dequant[2].item() == expected[fmt][1]
        else:
            assert dequant[3].item() == expected[fmt][0]
    _, _, zero_scale = quantize_mx_gemmini(torch.zeros(32), "mxfp8")
    assert zero_scale.tolist() == [104]  # 127 - 23
    with pytest.raises(ValueError, match="multiple of 32"):
        quantize_mx_gemmini(torch.zeros(31))
    with pytest.raises(ValueError, match="nonempty multiple of 32"):
        quantize_mx_gemmini(torch.zeros(0))
    bad = torch.zeros(32)
    bad[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        quantize_mx_gemmini(bad)


def test_rhs_matmul_groups_scales_on_k_axis():
    rhs = torch.zeros(64, 32)
    rhs[:32, :] = 1.0
    rhs[32:, :] = 4.0
    dequant, codes, scales = quantize_mx_gemmini(rhs, "mxfp8", axis=-2)
    assert dequant.shape == codes.shape == rhs.shape
    assert scales.shape == (2, 32)
    assert torch.all(scales[0] == 127)
    assert torch.all(scales[1] == 129)


@pytest.mark.parametrize("format", ["mxfp8", "mxfp6", "mxfp4"])
def test_linear_handoff_preserves_k_axis_and_weight_scale_orientation(format):
    source = torch.nn.Linear(64, 32, bias=False)
    with torch.no_grad():
        source.weight.zero_()
        source.weight[0, :] = 1.0
        source.weight[1, :] = 4.0
    module = MXGemminiLinear(source, format)
    activation = torch.cat((torch.ones(32, 32), 4.0 * torch.ones(32, 32)), dim=1)
    operand = linear_contraction_operands(module, activation)
    assert operand.format == format
    assert operand.activation_codes.shape == (32, 64)
    assert operand.weight_codes.shape == (64, 32)
    assert operand.activation_scales.shape == (32, 2)
    assert operand.weight_scales.shape == (32, 2)
    assert operand.activation_scales[0].tolist() == [127, 129]
    assert operand.weight_scales[0].tolist() == [127, 127]
    assert operand.weight_scales[1].tolist() == [129, 129]
    assert torch.equal(operand.weight_codes[:, 0], module.weight_codes[0, :])
    assert torch.equal(operand.weight_codes[:, 1], module.weight_codes[1, :])
    with pytest.raises(ValueError, match="activation K differs"):
        linear_contraction_operands(module, torch.zeros(32, 32))


@pytest.mark.parametrize("format", ["mxfp8", "mxfp6", "mxfp4"])
def test_functional_matmul_handoff_matches_linear_and_preserves_batch_axes(format):
    source = torch.nn.Linear(32, 32, bias=False)
    with torch.no_grad():
        source.weight[:16].fill_(1.0)
        source.weight[16:].fill_(2.0)
    activation = torch.ones(32, 32)
    activation[16:].fill_(2.0)
    expected = linear_contraction_operands(MXGemminiLinear(source, format), activation)
    actual = functional_contraction_operands(activation, source.weight.T, format)
    for name in ("activation_codes", "weight_codes", "activation_scales", "weight_scales"):
        assert torch.equal(getattr(actual, name), getattr(expected, name))

    batched = functional_contraction_operands(
        activation.expand(2, 3, 32, 32), source.weight.T.expand(2, 3, 32, 32), format
    )
    assert batched.activation_codes.shape == (2, 3, 32, 32)
    assert batched.weight_codes.shape == (2, 3, 32, 32)
    assert batched.activation_scales.shape == (2, 3, 32, 1)
    assert batched.weight_scales.shape == (2, 3, 32, 1)
    assert torch.equal(batched.weight_scales[1, 2], actual.weight_scales)

    with pytest.raises(ValueError, match="matching rank-2 to rank-4 batch axes"):
        functional_contraction_operands(activation.expand(2, 32, 32), source.weight.T, format)
    with pytest.raises(ValueError, match="matching K/32"):
        functional_contraction_operands(activation[:, :-1], source.weight.T, format)
    with pytest.raises(ValueError, match="matching K/32"):
        functional_contraction_operands(activation[:0], source.weight.T, format)


@pytest.mark.parametrize("format,expected_negative_zero", [
    ("mxfp8", True), ("mxfp6", True), ("mxfp4", False),
])
def test_signed_zero_matches_selected_rtl_conversion(format, expected_negative_zero):
    # BF16ToE4M3/BF16ToE3M2 retain the sign on underflow; E3M1Tofp4
    # explicitly emits code zero, discarding it.
    values = torch.zeros(32, dtype=torch.bfloat16)
    values[0] = 1.0
    values[1] = -0.0
    values[2] = -(2.0 ** -20)
    expected, codes, _ = quantize_mx_gemmini(values, format)
    exported = _dequant_operand_for_graph(values, format, -1)
    assert torch.equal(expected.view(torch.int16), exported.view(torch.int16))
    assert torch.signbit(expected[1]).item() is expected_negative_zero
    assert torch.signbit(expected[2]).item() is expected_negative_zero
    if format == "mxfp4":
        assert codes[1].item() == codes[2].item() == 0


@pytest.mark.parametrize("format", ["mxfp8", "mxfp6", "mxfp4"])
def test_export_arithmetic_matches_code_lookup_for_finite_bf16(format):
    generator = torch.Generator().manual_seed(91)
    rhs = torch.randn(64, 32, generator=generator, dtype=torch.float32).to(torch.bfloat16)
    rhs[0, 0] = 0.26  # FP4 double-rounding boundary
    rhs[1, 1] = 2.0 ** -23
    for axis in (-1, -2):
        expected = quantize_mx_gemmini(rhs, format, axis=axis)[0]
        observed = _dequant_operand_for_graph(rhs, format, axis)
        assert torch.equal(observed, expected)


@pytest.mark.parametrize("scheme", ["mx_gemmini_fp8", "mx_gemmini_fp6", "mx_gemmini_fp4"])
def test_torchao_handler_and_functional_matmul_capture(scheme):
    pytest.importorskip("torchao")
    from m2m.capture.mx_gemmini_quant import MXGemminiLinear
    from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.project = torch.nn.Linear(32, 32)

        def forward(self, x):
            projected = self.project(x)
            return torch.matmul(projected, projected.transpose(-1, -2))

    model = Tiny().eval()
    inputs = (torch.randn(32, 32),)
    transformed = apply_quantization(model, QuantizationConfig(scheme=scheme), example_inputs=inputs)
    assert isinstance(model.project, MXGemminiLinear)
    stats = transformed._m2m_quantization_stats
    assert stats["torchao_linear_modules"] == 1
    assert len(stats["functional_contractions_quantized"]) == 1
    assert not stats["functional_contractions_skipped"]
    assert torch.isfinite(transformed(*inputs)).all()
    torch.export.export(transformed, inputs)


def test_fused_attention_refused_when_contractions_are_hidden():
    pytest.importorskip("torchao")
    from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization

    class Fused(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.scaled_dot_product_attention(x, x, x)

    with pytest.raises(ValueError, match="fused SDPA"):
        apply_quantization(Fused().eval(), QuantizationConfig(scheme="mx_gemmini_fp8"),
                           example_inputs=(torch.randn(1, 2, 32, 32),))


def test_mx_capture_reports_linear_modules_left_unquantized():
    pytest.importorskip("torchao")
    from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization

    class Mixed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.supported = torch.nn.Linear(32, 32)
            self.unsupported = torch.nn.Linear(32, 31)

        def forward(self, x):
            return self.supported(x), self.unsupported(x)

    captured = apply_quantization(
        Mixed().eval(), QuantizationConfig(scheme="mx_gemmini_fp8"),
        example_inputs=(torch.randn(32, 32),),
    )
    stats = captured._m2m_quantization_stats
    assert stats["linear_modules_total"] == 2
    assert stats["torchao_linear_modules"] == 1
    assert stats["linear_modules_skipped"] == [
        {"module": "unsupported", "reason": "N outside MX tile 16"},
    ]
    assert len(stats["functional_contractions_skipped"]) == 1


@pytest.mark.parametrize("scheme", ["mx_gemmini_fp8", "mx_gemmini_fp6", "mx_gemmini_fp4"])
def test_mx_capture_import_has_explicit_contract_and_no_opaque_calls(scheme):
    pytest.importorskip("torchao")
    from m2m.api import convert
    from m2m.capture.torchao_pipeline import QuantizationConfig

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 32)

        def forward(self, x):
            y = self.linear(x)
            return torch.matmul(y, y.transpose(-1, -2))

    result = convert(Tiny().eval(), (torch.randn(32, 32),),
                     quantization=QuantizationConfig(scheme=scheme), backend="fx_importer")
    assert result.ok, result.diagnostics
    assert "prov.mx_capture_contract" in result.mlir_text
    assert "GemminiMxFPConfigs.standaloneMxFPConfig" in result.mlir_text
    assert 'operand_fake_quant_only' in result.mlir_text
    assert 'func.func private' not in result.mlir_text


def test_mx_capture_trace_reports_missing_external_quantizer_lineage():
    pytest.importorskip("torchao")
    from m2m.api import convert
    from m2m.capture.torchao_pipeline import QuantizationConfig

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 32, bias=False)

        def forward(self, x):
            return self.linear(x) @ x.transpose(-1, -2)

    result = convert(
        Tiny().eval(), (torch.ones(32, 32),),
        quantization=QuantizationConfig(scheme="mx_gemmini_fp8"),
        backend="fx_importer", capture_trace=True,
    )
    assert result.ok, result.diagnostics
    assert "prov.mx_capture_contract" in result.mlir_text
    assert result.capture_trace is not None
    assert result.capture_trace["status"] == "diagnostic"
    assert "original -> quantized correspondence incomplete" in result.capture_trace["blockers"]
    assert result.capture_trace["graphs"]["original"]["status"] == "complete"
    assert result.capture_trace["graphs"]["quantized"]["status"] == "complete"


def test_preapplied_mx_capture_is_not_quantized_twice(monkeypatch):
    pytest.importorskip("torchao")
    import m2m.api as api
    from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 32)

        def forward(self, x):
            return self.linear(x)

    inputs = (torch.randn(32, 32),)
    config = QuantizationConfig(scheme="mx_gemmini_fp8")
    captured = apply_quantization(Tiny().eval(), config, example_inputs=inputs)

    def fail_requantization(*_args, **_kwargs):
        raise AssertionError("preapplied quantization was applied again")

    monkeypatch.setattr(api, "apply_quantization", fail_requantization)
    result = api.convert(captured, inputs, quantization=config,
                         quantization_preapplied=True, backend="fx_importer")
    assert result.ok, result.diagnostics
    assert 'prov.mx_capture_contract' in result.mlir_text
    with pytest.raises(ValueError, match="requires a quantization config"):
        api.convert(captured, inputs, quantization_preapplied=True)
