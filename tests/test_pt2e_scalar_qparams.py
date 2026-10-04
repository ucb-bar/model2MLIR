"""PT2E's scalar activation qparams must not become opaque calls."""

import torch

import m2m


def test_fx_importer_materializes_literal_per_tensor_qparams():
    import torch.ao.quantization.fx._decomposed  # noqa: F401 -- registers the ops

    class Quantize(torch.nn.Module):
        def forward(self, x):
            q = torch.ops.quantized_decomposed.quantize_per_tensor.default(
                x, 0.125, 0, -128, 127, torch.int8
            )
            return torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                q, 0.125, 0, -128, 127, torch.int8
            )

    result = m2m.convert(
        Quantize().eval(),
        (torch.arange(8, dtype=torch.float32).reshape(2, 4),),
        backend="fx_importer",
    )
    assert "quant_ext.quantize_per_tensor" in result.mlir_text
    assert "quant_ext.dequantize_per_tensor" in result.mlir_text
    assert "func.call" not in result.mlir_text
