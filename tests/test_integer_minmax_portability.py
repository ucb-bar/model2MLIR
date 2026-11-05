"""Signed tensor extrema retain complete endpoint values in portable scalar IR."""

import numpy as np
import pytest
import torch

import m2m
from test_activation_boundary_semantics import native_activation


class Saturation(torch.nn.Module):
    def forward(self, x):
        lower = torch.maximum(x, torch.full_like(x, -128))
        return torch.minimum(lower, torch.full_like(lower, 127)).to(torch.int8)


@pytest.mark.parametrize('dtype', [np.int8, np.int32, np.int64])
@pytest.mark.parametrize('optimization', ['-O0', '-O2'])
def test_complete_native_signed_saturation_endpoints(tmp_path, dtype, optimization):
    info = np.iinfo(dtype)
    palette = np.array([v for v in [info.min, info.min + 1, -129, -128, -127,
                       -1, 0, 1, 126, 127, 128, info.max - 1, info.max]
                       if info.min <= v <= info.max], dtype=dtype)
    values = np.resize(palette, (7, 13))
    actual = native_activation(tmp_path, Saturation(), values, optimization)
    expected = np.clip(values, -128, 127).astype(np.int8)
    np.testing.assert_array_equal(actual, expected)


def test_signed_minmax_scalar_vocabulary_and_float_path():
    converted = m2m.convert(Saturation(), (torch.zeros(2, 3, dtype=torch.int32),),
                            backend='fx_importer')
    converted.module.verify()
    names = [op.name for op in converted.module.walk()]
    assert names.count('arith.cmpi') == 2
    assert names.count('arith.select') == 2
    assert 'arith.maxsi' not in names and 'arith.minsi' not in names
    assert 'arith.sitofp' not in names and 'arith.fptosi' not in names

    class Floating(torch.nn.Module):
        def forward(self, a, b):
            return torch.minimum(torch.maximum(a, b), b)

    floating = m2m.convert(Floating(), (torch.zeros(2, 3), torch.ones(2, 3)),
                           backend='fx_importer')
    floating.module.verify()
    names = [op.name for op in floating.module.walk()]
    assert 'arith.maximumf' in names and 'arith.minimumf' in names
    assert 'arith.cmpi' not in names
