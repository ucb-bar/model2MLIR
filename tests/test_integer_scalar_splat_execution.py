"""Original integer coefficients survive FX export and native scalar lowering."""

import numpy as np
import pytest
import torch

from test_activation_boundary_semantics import native_activation


class ScalarBinary(torch.nn.Module):
    def __init__(self, coefficient, operation):
        super().__init__()
        self.coefficient = coefficient
        self.operation = operation

    def forward(self, x):
        if self.operation == "mul":
            return x * self.coefficient
        return x / self.coefficient


@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
@pytest.mark.parametrize("operation", ["mul", "div"])
@pytest.mark.parametrize("exponent,sign", [(53, 1), (53, -1), (60, 1), (60, -1)])
def test_original_integer_coefficient_native_result_bits(tmp_path, optimization, operation, exponent, sign):
    coefficient = sign * ((1 << exponent) + (1 << (exponent - 24)) + 1)
    model = ScalarBinary(coefficient, operation)
    palette = np.array([-np.inf, -3.5, -1.0, -0.0, 0.0, 1.0, 3.5, np.inf, np.nan], dtype=np.float32)
    array = np.resize(palette, (5, 7))
    actual = native_activation(tmp_path, model, array, optimization)
    expected = model(torch.from_numpy(array)).numpy()
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    observed = ~np.isnan(expected)
    np.testing.assert_array_equal(actual.view(np.uint32)[observed], expected.view(np.uint32)[observed])
