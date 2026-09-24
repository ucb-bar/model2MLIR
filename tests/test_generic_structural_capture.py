"""Raw torch.export → FX importer must lower common structural aten overloads."""

import torch

import m2m
from m2m.coverage import opaque_report


class _Split(torch.nn.Module):
    def forward(self, x):
        a, b, c = torch.ops.aten.split.Tensor(x, 2, -1)
        return a + b + c


class _WhereScalarOther(torch.nn.Module):
    def forward(self, x):
        return torch.ops.aten.where.ScalarOther(x > 0, x, -0.5)


class _NewOnes(torch.nn.Module):
    def forward(self, x):
        return torch.ops.aten.new_ones.default(x, [2, 3])


def test_raw_export_structural_overloads_have_no_opaque_calls():
    # decompose=False is essential: the default torch-export decomposition pass rewrites these
    # overloads before the FX importer sees them, hiding its missing handlers.
    cases = (
        (_Split(), torch.randn(2, 5), "aten.split.Tensor", "tensor.extract_slice"),
        (_WhereScalarOther(), torch.randn(2, 3), "aten.where.ScalarOther", "arith.select"),
        (_WhereScalarOther(), torch.tensor([[-1, 2, 3], [1, 0, -5]], dtype=torch.int32),
         "aten.where.ScalarOther", "arith.select"),
        (_NewOnes(), torch.randn(2, 3), "aten.new_ones.default", "tensor.splat"),
        (_NewOnes(), torch.zeros(2, 3, dtype=torch.int32), "aten.new_ones.default", "tensor.splat"),
    )
    for model, example, source_op, mlir_op in cases:
        exported = torch.export.export(model.eval(), (example,))
        assert source_op in {str(node.target) for node in exported.graph.nodes if node.op == "call_function"}
        result = m2m.convert(model, (example,), backend="fx_importer", level="linalg-on-tensors", decompose=False)
        assert result.ok, (source_op, result.diagnostics)
        assert not opaque_report(result.mlir_text), (source_op, opaque_report(result.mlir_text))
        assert mlir_op in result.mlir_text, source_op
        if source_op == "aten.split.Tensor":
            assert result.mlir_text.count("tensor.extract_slice") == 3
            assert "tensor<2x1xf32>" in result.mlir_text  # ragged last chunk, not dropped
