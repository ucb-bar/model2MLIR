"""Frontend casts must implement PyTorch truth values, not signed i1 or parity."""

import math

import m2m
import pytest
import torch
from m2m.coverage import opaque_report


class Cast(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, value):
        return value.to(self.dtype)


def evaluate(block, value):
    """Evaluate every emitted scalar instruction with its actual IR semantics."""
    env = {block.args[0]: value}
    for op in block.ops:
        args = [env[arg] for arg in op.operands]
        if op.name == "linalg.yield":
            return args[0]
        if op.name == "arith.constant":
            result = op.value.value.data
        elif op.name == "arith.uitofp":
            result = float(int(args[0]) & ((1 << op.operands[0].type.width.data) - 1))
        elif op.name == "arith.sitofp":
            width = op.operands[0].type.width.data
            bits = int(args[0]) & ((1 << width) - 1)
            result = float(bits - (1 << width) if bits >> (width - 1) else bits)
        elif op.name == "arith.cmpf":
            assert op.predicate.value.data == 13  # unordered not-equal (une)
            result = math.isnan(args[0]) or math.isnan(args[1]) or args[0] != args[1]
        elif op.name == "arith.cmpi":
            assert op.predicate.value.data == 1  # not-equal
            result = args[0] != args[1]
        elif op.name == "arith.trunci":
            result = int(args[0]) & ((1 << op.results[0].type.width.data) - 1)
        elif op.name == "arith.fptosi":
            width = op.results[0].type.width.data
            assert math.isfinite(args[0]) and -(1 << (width - 1)) <= math.trunc(args[0]) < (1 << (width - 1))
            result = math.trunc(args[0])
        else:
            raise AssertionError(f"uninterpreted cast instruction: {op.name}")
        env[op.results[0]] = result
    raise AssertionError("cast has no yield")


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_boolean_to_float_exhaustive(dtype):
    value = torch.tensor([False, True])
    result = m2m.convert(Cast(dtype), (value,), backend="fx_importer")
    result.module.verify()
    assert not opaque_report(result.mlir_text)
    generic = next(op for op in result.module.walk() if op.name == "linalg.generic")
    observed = [evaluate(generic.body.block, element) for element in value.tolist()]
    assert observed == value.to(dtype).tolist()
    assert "arith.uitofp" in result.mlir_text and "arith.sitofp" not in result.mlir_text


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.float64, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_numeric_to_boolean_nonzero_not_truncation(dtype):
    values = [-3, -2, -1, 0, 1, 2, 3]
    if dtype.is_floating_point:
        values += [-0.0, 0.5, -0.5, float("inf"), float("-inf"), float("nan")]
    value = torch.tensor(values, dtype=dtype)
    result = m2m.convert(Cast(torch.bool), (value,), backend="fx_importer")
    result.module.verify()
    assert not opaque_report(result.mlir_text)
    generic = next(op for op in result.module.walk() if op.name == "linalg.generic")
    observed = [bool(evaluate(generic.body.block, element)) for element in value.tolist()]
    assert observed == value.to(torch.bool).tolist()
    assert "arith.fptosi" not in result.mlir_text and "arith.trunci" not in result.mlir_text
