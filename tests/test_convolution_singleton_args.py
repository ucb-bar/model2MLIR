"""ATen singleton spatial arguments broadcast through both convolution paths."""

import pytest
import torch
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region

import m2m
from m2m.coverage import opaque_report
from m2m.ir.decompositions import decompose_convolution


def _assert_lowered_convolution(module, expected, lowering):
    module.verify()
    assert not any(isinstance(op, func.CallOp) for op in module.walk())
    returns = [op for op in module.walk() if isinstance(op, func.ReturnOp)]
    assert len(returns) == 1
    assert [value.type for value in returns[0].operands] == [
        builtin.TensorType(builtin.f32, expected.shape)
    ]
    path = "direct_contraction" if lowering == "direct" else "im2col_matmul"
    paths = {
        op.attributes["prov.conv_path"].data
        for op in module.walk()
        if "prov.conv_path" in op.attributes
    }
    assert paths == {path}


@pytest.mark.parametrize(
    "rank,lowering",
    [(3, "im2col"), (4, "im2col"), (4, "direct")],
    ids=["conv1d_im2col", "conv2d_im2col", "conv2d_direct"],
)
@pytest.mark.parametrize(
    "stride,padding,dilation",
    [(1, 0, 1), (2, 1, 1), (1, 2, 2)],
    ids=["valid", "strided_padded", "dilated"],
)
def test_singleton_convolution_args_lower(
    monkeypatch, rank, lowering, stride, padding, dilation
):
    """Use ATen's accepted int[] spelling without export normalizing it away."""
    monkeypatch.setenv("M2M_CONV_LOWERING", lowering)
    monkeypatch.setenv("M2M_IM2COL_MAX_ELEMS", "1000000")
    spatial_shape = (11,) if rank == 3 else (11, 13)
    x = torch.randn(1, 2, *spatial_shape)
    weight = torch.randn(3, 2, *([3] * (rank - 2)))
    args = ([stride], [padding], [dilation], False, [0], 1)
    expected = torch.ops.aten.convolution.default(x, weight, None, *args)
    types = [builtin.TensorType(builtin.f32, value.shape) for value in (x, weight)]
    block = Block(arg_types=types)
    result = decompose_convolution(
        list(block.args),
        {
            "val": expected,
            "_aten_target": "aten.convolution.default",
            "_fx_args": (None, None, None, *args),
        },
        "singleton_conv",
    )
    block.add_ops(result.ops)
    block.add_op(func.ReturnOp(result.result))
    module = builtin.ModuleOp([
        func.FuncOp(
            "forward",
            (types, [builtin.TensorType(builtin.f32, expected.shape)]),
            region=Region(block),
        )
    ])
    _assert_lowered_convolution(module, expected, lowering)


@pytest.mark.parametrize("padding", ["valid", "same"])
@pytest.mark.parametrize("lowering", ["im2col", "direct"])
def test_string_padding_capture_lowers_convolution(monkeypatch, padding, lowering):
    """Cover the public capture that can produce singleton padding=[0]."""
    monkeypatch.setenv("M2M_CONV_LOWERING", lowering)
    monkeypatch.setenv("M2M_IM2COL_MAX_ELEMS", "1000000")
    model = torch.nn.Conv2d(2, 3, 3, padding=padding).eval()
    x = torch.randn(1, 2, 11, 13)
    with torch.no_grad():
        expected = model(x)
    converted = m2m.convert(model, (x,), backend="fx_importer")
    assert converted.ok, converted.diagnostics
    assert converted.module is not None
    assert opaque_report(converted.mlir_text) == {}
    _assert_lowered_convolution(converted.module, expected, lowering)
