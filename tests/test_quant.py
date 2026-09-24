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
