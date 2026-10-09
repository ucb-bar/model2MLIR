"""Integer scalar bounds convert directly to original f32 storage, once."""

import numpy as np
import pytest
import torch

from test_activation_boundary_semantics import Activation, native_activation


def scalar_bounds():
    cases = []
    for exponent in (53, 60):
        midpoint = (1 << exponent) + (1 << (exponent - 24))
        for sign in (1, -1):
            for delta, name in ((-1, "below"), (0, "tie"), (1, "above")):
                cases.append(pytest.param(
                    sign * (midpoint + delta),
                    id=("p" if sign > 0 else "n") + str(exponent) + "_" + name,
                ))
        # The next midpoint rounds toward the other even significand.
        odd_midpoint = midpoint + (1 << (exponent - 23))
        cases.extend((
            pytest.param(odd_midpoint, id="p" + str(exponent) + "_odd_tie"),
            pytest.param(-odd_midpoint, id="n" + str(exponent) + "_odd_tie"),
        ))
    cases.extend(pytest.param(value, id="int64_" + str(index)) for index, value in enumerate((
        -(1 << 63), -(1 << 63) + 1, (1 << 63) - 2, (1 << 63) - 1,
    )))
    return cases


@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
@pytest.mark.parametrize("which", ["lower", "upper"])
@pytest.mark.parametrize("bound", scalar_bounds())
def test_native_f32_integer_scalar_rounds_without_binary64_intermediate(tmp_path, optimization, which, bound):
    array = np.array([
        -np.inf, -float(1 << 64), -float(1 << 60), -0.0,
        0.0, float(1 << 53), float(1 << 60), float(1 << 64), np.inf, np.nan,
    ], dtype=np.float32)
    model = Activation("clamp", bound if which == "lower" else None, bound if which == "upper" else None)
    actual = native_activation(tmp_path, model, array, optimization)
    expected = model(torch.from_numpy(array)).numpy()
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    selected = ~np.isnan(expected)
    np.testing.assert_array_equal(actual.view(np.uint32)[selected], expected.view(np.uint32)[selected])
