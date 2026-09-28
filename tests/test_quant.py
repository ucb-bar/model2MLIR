"""Quantization lowering: int8/fp8 weight-only (QDQ) + true-quantized int8 (no QDQ) +
mixed precision all lower to standard dialects with 0 opaque ops."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import m2m
from m2m.capture.torchao_pipeline import QuantizationConfig
from m2m.coverage import opaque_report

torchao = pytest.importorskip("torchao")


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(64, 128)
        self.b = nn.Linear(128, 64)

    def forward(self, x):  # noqa: D102
        return self.b(torch.relu(self.a(x)))


X = (torch.randn(4, 64),)

# (scheme, expect_int8_storage) -- weight-only keeps QDQ (i8 + dequant), dyn-act is true int matmul
SCHEMES = [
    "int8_weight_only",
    "float8_weight_only_e4m3",
    "float8_weight_only_e5m2",
    "int8_dyn_act_int8_weight",
]


@pytest.mark.parametrize("scheme", SCHEMES)
def test_quant_scheme_lowers(scheme):
    r = m2m.convert(MLP().eval(), X, backend="fx_importer", quantization=QuantizationConfig(scheme=scheme))
    assert r.ok, f"{scheme} did not produce a valid module"
    opaque = opaque_report(r.mlir_text)
    assert sum(opaque.values()) == 0, f"{scheme} left opaque: {opaque}"
    assert "prov.quantization" in r.mlir_text


def test_int8_weight_only_preserves_qdq():
    """preserve_qdq (default) folds the implicit weight dequant into explicit
    quant_ext.dequantize ops so the quantization is first-class/matchable."""
    r = m2m.convert(MLP().eval(), X, backend="fx_importer",
                    quantization=QuantizationConfig(scheme="int8_weight_only"))
    assert "quant_ext.dequantize" in r.mlir_text
    assert sum(opaque_report(r.mlir_text).values()) == 0
    # opting out leaves the unfused (still 0-opaque) form
    r2 = m2m.convert(MLP().eval(), X, backend="fx_importer", preserve_qdq=False,
                     quantization=QuantizationConfig(scheme="int8_weight_only"))
    assert "quant_ext.dequantize" not in r2.mlir_text


def test_fully_standard_has_no_ext_dialects():
    """fully_standard=True lowers QDQ (and any *_ext op) to pure upstream MLIR -- no custom
    dialect remains; quant_ext.dequantize becomes linalg.generic + arith."""
    import re
    r = m2m.convert(MLP().eval(), X, backend="fx_importer", fully_standard=True,
                    quantization=QuantizationConfig(scheme="int8_weight_only"))
    assert "_ext." not in r.mlir_text                       # no quant_ext / linalg_ext / tensor_ext
    assert sum(opaque_report(r.mlir_text).values()) == 0    # still 0 opaque
    # only core dialects in the op stream
    op_dialects = {d.split(".")[0] for d in re.findall(r"%\w+ = (\w+)\.", r.mlir_text)}
    assert op_dialects <= {"arith", "linalg", "tensor", "scf", "math", "builtin", "func", "cf"}, op_dialects


def test_true_int_matmul_present():
    """Dynamic-activation int8 lowers to a real i8->i32 integer matmul (no dequant roundtrip)."""
    r = m2m.convert(MLP().eval(), X, backend="fx_importer",
                    quantization=QuantizationConfig(scheme="int8_dyn_act_int8_weight"))
    assert "i32" in r.mlir_text  # int32 accumulation


def test_static_w8a8_pt2e_quantizes_conv_and_uses_calibration():
    """Static W8A8 is graph quantization, not a Linear-only quantize_ label."""
    from m2m.capture.torchao_pipeline import apply_quantization

    class ConvNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3)
            self.fc = nn.Linear(8, 4)

        def forward(self, x):
            return self.fc(torch.relu(self.conv(x)).mean((2, 3)))

    x = (torch.randn(1, 3, 16, 16),)
    cfg = QuantizationConfig(
        scheme="int8_static_act_int8_weight", calibration_samples=2
    )
    quantized = apply_quantization(
        ConvNet().eval(), cfg, example_inputs=x, calibration_inputs=[x, x, x]
    )
    stats = quantized._m2m_quantization_stats
    assert stats["annotated_contractions"] == 2
    assert stats["calibration_samples"] == 2
    assert stats["pruned_dead_state_tensors"] == 2
    frozen = [v for k, v in quantized.state_dict().items() if k.startswith("_frozen_param")]
    assert len(frozen) == 2 and all(v.dtype is torch.int8 for v in frozen)
    assert not any(k.endswith(".weight") for k in quantized.state_dict())

    r = m2m.convert(
        quantized,
        x,
        backend="fx_importer",
        decompose=False,
        quantization=cfg,
        quantization_preapplied=True,
    )
    assert r.ok
    assert "0 opaque" in " ".join(r.diagnostics)
    assert r.mlir_text.count("quant_ext.dequantize_per_channel") == 2
    assert r.mlir_text.count("quant_ext.quantize_per_tensor") == 2
    assert "tensor<8x3x3x3xi8>" in r.mlir_text


def test_pt2e_integer_reference_executes_conv_and_linear():
    """The integer oracle uses frozen PT2E qparams and accounts for every contraction."""
    from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference
    from m2m.capture.torchao_pipeline import apply_quantization

    class ConvNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(4, 4, 3, padding=1, groups=2)
            self.fc = nn.Linear(4, 3)

        def forward(self, x):
            return self.fc(torch.relu(self.conv(x)).mean((2, 3)))

    inputs = (torch.randn(1, 4, 8, 8),)
    quantized = apply_quantization(
        ConvNet().eval(),
        QuantizationConfig(scheme="int8_static_act_int8_weight", calibration_samples=1),
        example_inputs=inputs,
        calibration_inputs=[inputs],
    )
    result = run_pt2e_integer_reference(quantized, inputs, expected_contractions=2)
    assert result.conv2d_count == 1
    assert result.linear_count == 1
    assert result.output.shape == (1, 3)
    assert torch.isfinite(result.output).all()


def test_pt2e_scalar_qparams_are_materialized_before_import():
    """PT2E calibration stores per-tensor scale and zero point as FX scalars."""

    class QDQ(nn.Module):
        def forward(self, x):
            q = torch.ops.quantized_decomposed.quantize_per_tensor.default(
                x, 0.125, 0, -128, 127, torch.int8
            )
            return torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                q, 0.125, 0, -128, 127, torch.int8
            )

    result = m2m.convert(QDQ().eval(), (torch.randn(2, 8),), backend="fx_importer", decompose=False)
    assert result.ok
    assert opaque_report(result.mlir_text) == {}
    assert "quant_ext.quantize_per_tensor" in result.mlir_text
    assert "quant_ext.dequantize_per_tensor" in result.mlir_text


def test_pt2e_linear_integerization_preserves_frozen_qparams_and_bias():
    from m2m.capture.pt2e_integerize import integerize_pt2e

    class QDQLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("qweight", torch.randint(-8, 8, (16, 32), dtype=torch.int8))
            self.register_buffer("bias", torch.randn(16))

        def forward(self, x):
            qx = torch.ops.quantized_decomposed.quantize_per_tensor.default(
                x, 0.125, 0, -128, 127, torch.int8
            )
            dx = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                qx, 0.125, 0, -128, 127, torch.int8
            )
            dw = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                self.qweight, 0.25, 0, -127, 127, torch.int8
            )
            return torch.ops.aten.linear.default(dx, dw, self.bias)

    inputs = (torch.randn(2, 3, 32),)
    original = QDQLinear().eval()
    captured = torch.export.export(original, inputs).module()
    expected = captured(*inputs)
    rewritten, receipt = integerize_pt2e(captured, inputs)
    assert receipt["linear_integerized"] == receipt["linear_seen"] == 1
    assert receipt["refusals"] == []
    torch.testing.assert_close(rewritten(*inputs), expected, atol=1e-6, rtol=1e-6)
    from torch.ao.quantization import allow_exported_model_train_eval

    allow_exported_model_train_eval(rewritten)
    lowered = m2m.convert(rewritten, inputs, backend="fx_importer")
    assert lowered.ok and opaque_report(lowered.mlir_text) == {}
    assert 'prov.aten = "aten._int_mm.default"' in lowered.mlir_text


def test_pt2e_linear_integerization_refuses_nonzero_zero_point():
    from m2m.capture.pt2e_integerize import integerize_pt2e

    class Asymmetric(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("qweight", torch.ones((8, 8), dtype=torch.int8))

        def forward(self, x):
            qx = torch.ops.quantized_decomposed.quantize_per_tensor.default(
                x, 0.125, 1, -128, 127, torch.int8
            )
            dx = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                qx, 0.125, 1, -128, 127, torch.int8
            )
            dw = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                self.qweight, 0.25, 0, -127, 127, torch.int8
            )
            return torch.ops.aten.linear.default(dx, dw, None)

    inputs = (torch.randn(2, 8),)
    captured = torch.export.export(Asymmetric().eval(), inputs).module()
    _, receipt = integerize_pt2e(captured, inputs)
    assert receipt["linear_seen"] == 1
    assert receipt["linear_integerized"] == 0
    assert "zero point 0" in receipt["refusals"][0]["reason"]


def test_pt2e_conv_integerization_preserves_window_scales_and_bias():
    from m2m.capture.pt2e_integerize import integerize_pt2e

    class QDQConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("qweight", torch.randint(-8, 8, (5, 3, 3, 3), dtype=torch.int8))
            self.register_buffer("bias", torch.randn(5))

        def forward(self, x):
            qx = torch.ops.quantized_decomposed.quantize_per_tensor.default(
                x, 0.125, 0, -128, 127, torch.int8
            )
            dx = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                qx, 0.125, 0, -128, 127, torch.int8
            )
            dw = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                self.qweight, 0.25, 0, -127, 127, torch.int8
            )
            return torch.ops.aten.conv2d.default(dx, dw, self.bias, [2, 2], [1, 1], [2, 2])

    inputs = (torch.randn(2, 3, 8, 8),)
    captured = torch.export.export(QDQConv().eval(), inputs).module()
    expected = captured(*inputs)
    rewritten, receipt = integerize_pt2e(captured, inputs)
    assert receipt["conv2d_seen"] == receipt["conv2d_integerized"] == 1
    assert receipt["refusals"] == []
    torch.testing.assert_close(rewritten(*inputs), expected, atol=1e-6, rtol=1e-6)

    from torch.ao.quantization import allow_exported_model_train_eval

    allow_exported_model_train_eval(rewritten)
    lowered = m2m.convert(rewritten, inputs, backend="fx_importer")
    assert lowered.ok and opaque_report(lowered.mlir_text) == {}
    assert 'prov.aten = "aten._int_mm.default"' in lowered.mlir_text


def test_pt2e_grouped_conv_is_explicitly_refused():
    from m2m.capture.pt2e_integerize import integerize_pt2e

    class Grouped(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("qweight", torch.ones((4, 2, 3, 3), dtype=torch.int8))

        def forward(self, x):
            qx = torch.ops.quantized_decomposed.quantize_per_tensor.default(
                x, 0.125, 0, -128, 127, torch.int8
            )
            dx = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                qx, 0.125, 0, -128, 127, torch.int8
            )
            dw = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                self.qweight, 0.25, 0, -127, 127, torch.int8
            )
            return torch.ops.aten.conv2d.default(dx, dw, None, [1, 1], [1, 1], [1, 1], 2)

    inputs = (torch.randn(1, 4, 8, 8),)
    model = torch.export.export(Grouped().eval(), inputs).module()
    _, receipt = integerize_pt2e(model, inputs)
    assert receipt["quantized_by_kind"]["conv2d"] == {"seen": 1, "integerized": 0, "remaining": 1}
    assert receipt["quantized_contractions_remaining"] == 1
    assert "grouped convolution" in receipt["refusals"][0]["reason"]


def test_pt2e_census_excludes_unquantized_float_island():
    from m2m.capture.pt2e_integerize import integerize_pt2e

    class Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("qweight", torch.ones((8, 8), dtype=torch.int8))
            self.float_weight = nn.Parameter(torch.randn(8, 8))

        def forward(self, x):
            qx = torch.ops.quantized_decomposed.quantize_per_tensor.default(
                x, 0.125, 0, -128, 127, torch.int8
            )
            dx = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                qx, 0.125, 0, -128, 127, torch.int8
            )
            dw = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                self.qweight, 0.25, 0, -127, 127, torch.int8
            )
            return torch.ops.aten.linear.default(dx, dw, None) + torch.ops.aten.linear.default(
                x, self.float_weight, None
            )

    inputs = (torch.randn(2, 8),)
    model = torch.export.export(Mixed().eval(), inputs).module()
    expected = model(*inputs)
    rewritten, receipt = integerize_pt2e(model, inputs)
    assert receipt["linear_seen"] == 2
    assert receipt["quantized_by_kind"]["linear"] == {"seen": 1, "integerized": 1, "remaining": 0}
    assert receipt["quantized_contractions_remaining"] == 0
    torch.testing.assert_close(rewritten(*inputs), expected, atol=1e-6, rtol=1e-6)


def test_pt2e_batched_matmul_integerizes_each_static_batch():
    from m2m.capture.pt2e_integerize import integerize_pt2e

    class QDQBatch(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("qweight", torch.randint(-8, 8, (1, 2, 8, 6), dtype=torch.int8))

        def forward(self, x):
            qx = torch.ops.quantized_decomposed.quantize_per_tensor.default(
                x, 0.125, 0, -128, 127, torch.int8
            )
            dx = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                qx, 0.125, 0, -128, 127, torch.int8
            )
            dw = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                self.qweight, 0.25, 0, -127, 127, torch.int8
            )
            return torch.ops.aten.matmul.default(dx, dw)

    inputs = (torch.randn(1, 2, 4, 8),)
    captured = torch.export.export(QDQBatch().eval(), inputs).module()
    expected = captured(*inputs)
    rewritten, receipt = integerize_pt2e(captured, inputs)
    assert receipt["matmul_seen"] == receipt["matmul_integerized"] == 1
    assert receipt["integer_mm_emitted"] == 2
    assert receipt["quantized_contractions_remaining"] == 0
    assert receipt["remaining_dequant_count"] == 0
    torch.testing.assert_close(rewritten(*inputs), expected, atol=1e-6, rtol=1e-6)

    from torch.ao.quantization import allow_exported_model_train_eval

    allow_exported_model_train_eval(rewritten)
    lowered = m2m.convert(rewritten, inputs, backend="fx_importer")
    assert lowered.ok and opaque_report(lowered.mlir_text) == {}
    assert lowered.mlir_text.count('prov.aten = "aten._int_mm.default"') >= 2


def test_fp8_type_renders_native_spelling():
    """The shim fp8 type prints with the MLIR-native spelling (f8E4M3FN) on text emission,
    so an artifact carrying f8 storage parses in a standard MLIR toolchain."""
    from xdsl.dialects.builtin import ModuleOp, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region

    from m2m.capture.torch_mlir_bridge import module_to_text
    from m2m.ir.types import Float8E4M3FNType

    t = TensorType(Float8E4M3FNType(), [4, 8])
    blk = Block(arg_types=[t])
    blk.add_op(ReturnOp(blk.args[0]))
    txt = module_to_text(ModuleOp([FuncOp("f", ((t,), (t,)), region=Region(blk))]))
    assert "f8E4M3FN" in txt and "builtin_ext" not in txt


def test_mixed_precision_lowers():
    """One network mixing int8 + fp8 + full-precision submodules lowers to 0 opaque."""
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc_a = nn.Linear(64, 64)
            self.fc_b = nn.Linear(64, 64)
            self.fc_c = nn.Linear(64, 64)

        def forward(self, x):
            return self.fc_c(torch.relu(self.fc_b(torch.relu(self.fc_a(x)))))

    cfg = QuantizationConfig(scheme="none", per_module={
        r"fc_a": "int8_weight_only",
        r"fc_b": "float8_weight_only_e4m3",
    })
    r = m2m.convert(Net().eval(), X, backend="fx_importer", quantization=cfg)
    assert r.ok
    assert sum(opaque_report(r.mlir_text).values()) == 0
    assert "prov.quantization_mixed" in r.mlir_text


class _EmbedAndNorm(nn.Module):
    """An embedding, a Linear, and a LayerNorm — the shape that broke mixed precision.

    The LayerNorm matters: it is the module with no quantizable `weight` that a name-only filter
    offered to a Linear-only transform.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(512, 32)
        self.norm = nn.LayerNorm(32)
        self.proj = nn.Linear(32, 16)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(self.embed_tokens(idx)))


def test_per_module_default_rule_keeps_torchaos_linear_only_filter() -> None:
    """Passing any ``filter_fn`` REPLACES torchao's own (``_is_linear``), so a name-based default rule
    used to offer every module in the tree to a Linear-only transform — and the first one without a
    ``weight`` asserted ("applying int8 weight only quant requires module to have weight attribute").
    Mixed precision therefore only worked when the rules happened to cover every non-Linear module.
    """
    from m2m.capture.torchao_pipeline import apply_quantization

    model = _EmbedAndNorm().eval()
    cfg = QuantizationConfig(scheme="int8_weight_only",
                             per_module={"embed_tokens": "int8_embedding_weight_only"})
    quantized = apply_quantization(model, cfg)          # used to raise AssertionError

    # The named rule reached the embedding: int8 storage, a quarter of the fp32 bytes.
    emb = quantized.embed_tokens.weight
    inner = getattr(emb, "qdata", getattr(emb, "int_data", None))
    assert inner is not None and inner.dtype is torch.int8
    # The default rule still reached the Linear.
    assert type(quantized.proj.weight).__name__ != "Parameter"
    # And the LayerNorm was left alone rather than asserting.
    assert quantized.norm.weight.dtype is torch.float32
    # The gather still returns floats, so downstream ops are unchanged.
    out = quantized(torch.tensor([[1, 2, 3]]))
    assert out.dtype is torch.float32 and out.shape == (1, 3, 16)


def test_an_embedding_is_not_quantized_without_an_explicit_rule() -> None:
    """The reason a large-vocabulary "int8" bundle can be dominated by one fp32 table: torchao's
    default filter matches nn.Linear only. Measured on whisper-tiny, 76.0 MB of a 116.8 MB int8
    bundle was the fp32 token-embedding table."""
    from m2m.capture.torchao_pipeline import apply_quantization

    model = _EmbedAndNorm().eval()
    quantized = apply_quantization(model, QuantizationConfig(scheme="int8_weight_only"))
    assert quantized.embed_tokens.weight.dtype is torch.float32
