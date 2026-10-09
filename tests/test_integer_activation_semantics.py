"""Integer activation values through ordinary FX and native MLIR compilation."""

import numpy as np
import pytest
import torch

from test_activation_boundary_semantics import Activation, native_activation


def integer_values(dtype, shape=(269,)):
    info = np.iinfo(dtype)
    if dtype in (np.int8, np.uint8):
        palette = np.arange(info.min, info.max + 1, dtype=dtype)
    else:
        palette = np.array([
            info.min, info.min + 1, -3, -2, -1, 0, 1, 2, 3,
            info.max - 1, info.max,
        ], dtype=dtype)
    return np.resize(palette, shape)


@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.int32, np.int64])
@pytest.mark.parametrize("operation", ["relu", "round", "clamp"])
def test_native_integer_activation_full_outputs(tmp_path, optimization, dtype, operation):
    lower = 2 if dtype == np.uint8 else -2
    model = Activation(operation, lower, 3)
    array = integer_values(dtype)
    actual = native_activation(tmp_path, model, array, optimization)
    expected = model(torch.from_numpy(array)).numpy()
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int32, np.int64])
@pytest.mark.parametrize("bounds", [(-1.5, 200.5), (0.0, None), (None, 127.5)])
def test_native_integer_clamp_floating_promotion(tmp_path, dtype, bounds):
    array = integer_values(dtype, shape=(7, 37))
    model = Activation("clamp", *bounds)
    actual = native_activation(tmp_path, model, array, "-O2")
    expected = model(torch.from_numpy(array)).numpy()
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("bounds", [
    (None, -(2**53 + 1)),
    (2**53 + 1, None),
    (2**63 - 2, 2**63 - 1),
    (3, -2),
])
def test_native_integer_clamp_exact_large_and_inverted_bounds(tmp_path, bounds):
    array = np.array([
        -(2**63), -(2**53 + 2), -(2**53 + 1), -3, 0, 3,
        2**53, 2**53 + 1, 2**53 + 2, 2**63 - 2, 2**63 - 1,
    ], dtype=np.int64)
    model = Activation("clamp", *bounds)
    actual = native_activation(tmp_path, model, array, "-O2")
    np.testing.assert_array_equal(actual, model(torch.from_numpy(array)).numpy())


@pytest.mark.parametrize("operation", ["relu", "round", "clamp"])
def test_native_integer_scalar_tensor(tmp_path, operation):
    array = np.array(-3, dtype=np.int32)
    model = Activation(operation, -2, 3)
    actual = native_activation(tmp_path, model, array, "-O2")
    np.testing.assert_array_equal(actual, model(torch.from_numpy(array)).numpy())


@pytest.mark.parametrize("bounds", [(-255, None), (None, -1), (-2, 3), (2, -254)])
def test_native_unsigned_clamp_checked_negative_scalar_conversion(tmp_path, bounds):
    array = integer_values(np.uint8)
    model = Activation("clamp", *bounds)
    actual = native_activation(tmp_path, model, array, "-O2")
    np.testing.assert_array_equal(actual, model(torch.from_numpy(array)).numpy())


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_native_float_round_even_remains_float(tmp_path, dtype):
    array = np.array([-np.inf, -2.5, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, 2.5, np.inf, np.nan], dtype=dtype)
    model = Activation("round")
    actual = native_activation(tmp_path, model, array, "-O2")
    expected = model(torch.from_numpy(array)).numpy()
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    observed = ~np.isnan(expected)
    unsigned = np.dtype("u" + str(array.itemsize))
    np.testing.assert_array_equal(actual.view(unsigned)[observed], expected.view(unsigned)[observed])
