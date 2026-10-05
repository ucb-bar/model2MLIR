"""RMSNorm's scalar square must not require a per-element libm power call."""
import pytest
import torch

import m2m
from m2m.coverage import validate_op


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('exponent', [2, 2.0])
def test_scalar_square_lowering(dtype, exponent):
    class Square(torch.nn.Module):
        def forward(self, x):
            return torch.pow(x, exponent)

    x = torch.tensor([-3.5, -0.0, 0.0, 2.0, float('inf'), -float('inf'), float('nan')], dtype=dtype)
    expected, multiplied = x.pow(exponent), x * x
    torch.testing.assert_close(expected, multiplied, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(torch.signbit(expected[:-1]), torch.signbit(multiplied[:-1]))
    result = m2m.convert(Square(), (x,), backend='fx_importer')
    assert result.ok, result.diagnostics
    assert 'arith.mulf' in result.mlir_text
    assert 'math.powf' not in result.mlir_text
    report = validate_op(Square().forward, (torch.randn(4, 8, dtype=dtype),), name='scalar_square')
    assert report.ok, report


def test_other_scalar_exponent_keeps_power():
    class Cube(torch.nn.Module):
        def forward(self, x):
            return torch.pow(x, 3)

    result = m2m.convert(Cube(), (torch.randn(4),), backend='fx_importer')
    assert result.ok, result.diagnostics
    assert 'math.powf' in result.mlir_text
