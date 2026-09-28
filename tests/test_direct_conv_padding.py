"""Direct mode must handle padding without falling back to an opaque convolution."""
import pytest
import torch

import m2m
from m2m.coverage import validate_op


@pytest.mark.parametrize("kernel,padding,stride", [(3, 1, 1), (3, 1, 2), (7, 3, 2)])
def test_direct_convolution_with_padding(monkeypatch, kernel, padding, stride):
    monkeypatch.setenv("M2M_IM2COL_MAX_ELEMS", "0")
    result = validate_op(
        lambda x, w: torch.nn.functional.conv2d(x, w, padding=padding, stride=stride),
        (torch.randn(1, 3, 16, 16), torch.randn(4, 3, kernel, kernel)), name="direct_padded_conv")
    assert result.ok, result


@pytest.mark.parametrize(
    "channels,groups",
    [
        (8, 8),   # depthwise: one input/output channel per group
        (8, 4),   # grouped: two input/output channels per group
    ],
)
@pytest.mark.parametrize("lowering", ["direct", "grouped_direct"])
def test_direct_grouped_convolution_eliminates_im2col(
    monkeypatch, channels, groups, lowering
):
    """The memory-safe direct path must cover grouped and depthwise convolution.

    These are the exact convolution classes whose rank-7 im2col gathers dominate
    LSTMNetVIT on K1.  A numerically correct opaque fallback is not sufficient:
    the emitted IR must contain one direct contraction and no im2col tensor.
    """
    monkeypatch.setenv("M2M_CONV_LOWERING", lowering)
    class GroupedConv(torch.nn.Module):
        def forward(self, x, w):
            return torch.nn.functional.conv2d(
                x, w, padding=1, stride=1, groups=groups
            )

    inputs = (
        torch.randn(1, channels, 8, 8),
        torch.randn(channels, channels // groups, 3, 3),
    )
    result = validate_op(
        GroupedConv().eval().forward,
        inputs,
        name="direct_grouped_conv",
    )
    assert result.ok, result
    converted = m2m.convert(GroupedConv().eval(), inputs, backend="fx_importer")
    assert converted.ok, converted.diagnostics
    assert 'prov.conv_path = "direct_contraction"' in converted.mlir_text
    assert "tensor<8x1x3x3x1x8x8x" not in converted.mlir_text


def test_grouped_direct_mode_leaves_ordinary_conv_on_im2col(monkeypatch):
    """The targeted mode changes only grouped convs, keeping the existing fast GEMM arm elsewhere."""
    monkeypatch.setenv("M2M_CONV_LOWERING", "grouped_direct")

    class Conv(torch.nn.Module):
        def forward(self, x, w):
            return torch.nn.functional.conv2d(x, w, padding=1)

    inputs = (torch.randn(1, 8, 8, 8), torch.randn(8, 8, 3, 3))
    converted = m2m.convert(Conv().eval(), inputs, backend="fx_importer")
    assert converted.ok, converted.diagnostics
    assert 'prov.conv_path = "im2col_matmul"' in converted.mlir_text
    assert 'prov.conv_path = "direct_contraction"' not in converted.mlir_text
