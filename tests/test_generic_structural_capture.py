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


class _IdentityAsStrided(torch.nn.Module):
    def forward(self, x):
        # Singleton strides are irrelevant to element mapping.  A ResNet
        # global-average-pool export emits exactly this shape of view.
        return torch.ops.aten.as_strided.default(x, [1, 2, 1, 1], [2, 1, 2, 2], 0)


class _ReorderedAsStrided(torch.nn.Module):
    def forward(self, x):
        return torch.ops.aten.as_strided.default(x, [2, 3], [1, 2], 0)


def test_raw_export_identity_as_strided_has_no_opaque_call():
    example = torch.randn(1, 2, 1, 1)
    exported = torch.export.export(_IdentityAsStrided(), (example,))
    assert "aten.as_strided.default" in {
        str(node.target) for node in exported.graph.nodes if node.op == "call_function"
    }
    result = m2m.convert(
        _IdentityAsStrided(), (example,), backend="fx_importer",
        level="linalg-on-tensors", decompose=False,
    )
    assert result.ok, result.diagnostics
    assert not opaque_report(result.mlir_text), opaque_report(result.mlir_text)


def test_nonidentity_as_strided_remains_opaque():
    result = m2m.convert(
        _ReorderedAsStrided(), (torch.randn(2, 3),), backend="fx_importer",
        level="linalg-on-tensors", decompose=False,
    )
    assert result.ok, result.diagnostics
    assert opaque_report(result.mlir_text) == {"aten_as_strided_default": 1}


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


def test_mixed_dtype_bucketize_preserves_operands_and_promotes_comparisons():
    from torch_mlir import ir

    from m2m.coverage import differential_op

    class Search(torch.nn.Module):
        def __init__(self, right):
            super().__init__()
            self.right = right
        def forward(self, x, boundaries):
            return torch.bucketize(x, boundaries, right=self.right)

    for dtype in (torch.bfloat16, torch.float16, torch.int32):
        inputs = (torch.tensor([-2, -1, 0, 1, 2], dtype=dtype),
                  torch.tensor([-1.5, -1, 0.5, 1], dtype=torch.float32))
        for right in (False, True):
            model = Search(right)
            result = m2m.convert(model, inputs, backend="fx_importer", capture_trace=True)
            assert result.ok and not opaque_report(result.mlir_text)
            result.module.verify()
            with ir.Context() as context:
                context.allow_unregistered_dialects = True
                assert ir.Module.parse(result.mlir_text).operation.verify()
            oracle = differential_op(model, inputs, name="mixed_dtype_bucketize")
            assert oracle.ours_lowered, oracle.error
            assert oracle.verdict in {"ok", "ok (non-canonical ops)", "no-oracle"}
