"""Signed integer coefficients retain a single original f32 rounding step.

The oracle uses exact integer distances, without converting the original scalar
through Python binary64. These controls inspect emitted typed scalar IR only;
ordinary framework and compiled execution controls are separate prerequisites.
"""

import struct
from fractions import Fraction

import pytest
from xdsl.dialects.arith import ConstantOp, SIToFPOp
from xdsl.dialects.builtin import (
    FloatAttr,
    IntegerAttr,
    TensorType,
    bf16,
    f16,
    f32,
    f64,
    i8,
    i64,
)
from xdsl.dialects.tensor import SplatOp

from m2m.ir.decompositions import _splat_scalar


def direct_f32_bits(original):
    assert type(original) is int and -(1 << 63) <= original < (1 << 63)
    if original == 0:
        return 0
    magnitude = abs(original)
    spacing = 1 << max(0, magnitude.bit_length() - 24)
    lower = magnitude // spacing * spacing
    upper = lower + spacing
    dl, du = magnitude - lower, upper - magnitude
    rounded = lower if dl < du else upper
    if dl == du:
        rounded = lower if lower // spacing % 2 == 0 else upper
    assert Fraction.from_float(float(rounded)) == rounded
    exact_float = float(-rounded if original < 0 else rounded)
    return int.from_bytes(struct.pack("<f", exact_float), "little")


def emitted_coefficient_bits(operations):
    """Read only a constant or exact signed-i64-to-f32 scalar conversion."""
    assert isinstance(operations[-1], SplatOp)
    scalar = operations[-1].input
    if isinstance(scalar.owner, ConstantOp):
        constant = scalar.owner
        assert isinstance(constant.value, FloatAttr)
        assert constant.result.type == f32
        return int.from_bytes(struct.pack("<f", constant.value.value.data), "little")
    assert isinstance(scalar.owner, SIToFPOp)
    assert scalar.type == f32
    source = scalar.owner.input
    assert source.type == i64 and isinstance(source.owner, ConstantOp)
    assert isinstance(source.owner.value, IntegerAttr)
    return direct_f32_bits(source.owner.value.value.data)


def signed64_coefficients():
    cases = [0, 1, -(1 << 63), (1 << 63) - 1]
    for exponent in (53, 60):
        midpoint = (1 << exponent) + (1 << (exponent - 24))
        for sign in (-1, 1):
            cases.extend(sign * (midpoint + delta) for delta in (-1, 0, 1))
            cases.append(sign * (midpoint + (1 << (exponent - 23))))
    return cases


@pytest.mark.parametrize("original", signed64_coefficients())
def test_signed64_coefficient_splat_rounds_once(original):
    result_type = TensorType(f32, [2, 3])
    operations, result = _splat_scalar(original, result_type)
    for operation in operations:
        operation.verify()
    assert result.type == result_type
    assert emitted_coefficient_bits(operations) == direct_f32_bits(original)


@pytest.mark.parametrize("exponent", [53, 60])
@pytest.mark.parametrize("sign", [-1, 1])
def test_exact_midpoint_neighbor_exposes_binary64_double_rounding(exponent, sign):
    original = sign * ((1 << exponent) + (1 << (exponent - 24)) + 1)
    direct = direct_f32_bits(original)
    via_binary64 = int.from_bytes(struct.pack("<f", float(original)), "little")
    assert direct != via_binary64
    assert (direct & 0x7FFFFFFF) == ((via_binary64 & 0x7FFFFFFF) + 1)


@pytest.mark.parametrize(
    "original", [-(1 << 63), (1 << 60) + (1 << 36) + 1, (1 << 63) - 1]
)
def test_f32_splat_retains_original_signed64_constant(original):
    result_type = TensorType(f32, [3, 5])
    operations, result = _splat_scalar(original, result_type)
    assert [operation.name for operation in operations] == [
        "arith.constant",
        "arith.sitofp",
        "tensor.splat",
    ]
    constant, conversion, splat = operations
    assert constant.result.type == i64
    assert constant.value.value.data == original
    assert conversion.input is constant.result and conversion.result.type == f32
    assert splat.input is conversion.result and result is splat.result
    assert result.type == result_type


@pytest.mark.parametrize(
    "original,element",
    [
        (1.5, f32),
        (-0.0, f32),
        (True, f32),
        (1.5, f16),
        (1.5, bf16),
        ((1 << 53) + 1, f64),
        (3, i8),
        (1 << 63, f32),
    ],
)
def test_other_scalar_splat_paths_keep_existing_constant_roster(original, element):
    result_type = TensorType(element, [2, 3])
    operations, result = _splat_scalar(original, result_type)
    assert [operation.name for operation in operations] == [
        "arith.constant",
        "tensor.splat",
    ]
    constant, splat = operations
    assert splat.input is constant.result
    assert constant.result.type == element and result.type == result_type
    for operation in operations:
        operation.verify()
    if type(original) is float and original == 0.0:
        assert struct.pack("<f", constant.value.value.data) == struct.pack(
            "<f", original
        )


def test_dynamic_scalar_splat_keeps_existing_refusal():
    assert _splat_scalar(1, TensorType(f32, [-1, 3])) is None
