"""Bucketize's emitted search body must agree with the selected source operator."""

import math

import m2m
import pytest
import torch
from m2m.coverage import opaque_report


class Search(torch.nn.Module):
    def __init__(self, right: bool):
        super().__init__()
        self.right = right

    def forward(self, values, boundaries):
        return torch.bucketize(values, boundaries, right=self.right)


def _constant_or_body_value(value, env):
    if value in env:
        return env[value]
    owner = value.owner
    assert owner is not None and owner.name == "arith.constant"
    return owner.value.value.data


def _search_body(block, value, boundary, accumulator):
    env = dict(zip(block.args, (value, boundary, accumulator), strict=True))
    predicates = set()
    for op in block.ops:
        args = [_constant_or_body_value(arg, env) for arg in op.operands]
        if op.name == "linalg.yield":
            return args[0], predicates
        if op.name == "arith.cmpf":
            predicate = op.predicate.value.data
            predicates.add(predicate)
            assert predicate in {4, 5, 11, 12}
            unordered = math.isnan(args[0]) or math.isnan(args[1])
            observed = (unordered and predicate in {11, 12}) or (
                not unordered and (args[0] <= args[1] if predicate in {5, 12} else args[0] < args[1])
            )
        elif op.name == "arith.cmpi":
            predicate = op.predicate.value.data
            assert predicate in {2, 3}  # signed less-than / less-or-equal
            observed = args[0] <= args[1] if predicate == 3 else args[0] < args[1]
        elif op.name == "arith.select":
            observed = args[1] if args[0] else args[2]
        elif op.name == "arith.addi":
            observed = args[0] + args[1]
        else:
            raise AssertionError(f"unreviewed search-body operation: {op.name}")
        env[op.results[0]] = observed
    raise AssertionError("bucketize search body has no yield")


@pytest.mark.parametrize("right", [False, True])
def test_float_bucketize_nan_in_last_bucket_and_finite_boundaries(right):
    # Non-decreasing, non-NaN boundaries: duplicate zeros exercise left/right insertion.
    boundaries = torch.tensor([-1.0, -0.0, 0.0, 1.0], dtype=torch.float32)
    values = torch.tensor([float("nan"), -float("inf"), float("inf"), -1.0, -0.0, 0.0, 1.0, 2.0])
    result = m2m.convert(Search(right).eval(), (values, boundaries), backend="fx_importer", capture_trace=True)
    assert result.ok, result.diagnostics
    result.module.verify()
    assert not opaque_report(result.mlir_text)
    assert result.capture_trace["status"] == "complete"
    generic = next(op for op in result.module.walk() if op.name == "linalg.generic")
    observed = []
    predicates = set()
    for value in values.numpy():
        accumulator = 0
        for boundary in boundaries.numpy():
            accumulator, used = _search_body(generic.body.block, value, boundary, accumulator)
            predicates.update(used)
        observed.append(accumulator)
    assert observed == torch.bucketize(values, boundaries, right=right).tolist()
    assert predicates == ({12} if right else {11})  # unordered comparison makes input NaN count every boundary


@pytest.mark.parametrize("right", [False, True])
def test_signed_integer_bucketize_preserves_existing_comparison(right):
    boundaries = torch.tensor([-4, -1, -1, 3], dtype=torch.int64)
    values = torch.tensor([-(2**63), -4, -1, 0, 3, 2**63 - 1], dtype=torch.int64)
    result = m2m.convert(Search(right).eval(), (values, boundaries), backend="fx_importer")
    assert result.ok and not opaque_report(result.mlir_text)
    generic = next(op for op in result.module.walk() if op.name == "linalg.generic")
    observed = []
    for value in values.tolist():
        accumulator = 0
        for boundary in boundaries.tolist():
            accumulator, _ = _search_body(generic.body.block, value, boundary, accumulator)
        observed.append(accumulator)
    assert observed == torch.bucketize(values, boundaries, right=right).tolist()


@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8])
def test_unproved_unsigned_bucketize_refuses_signed_comparison(dtype):
    values = torch.tensor([0, 1], dtype=dtype)
    boundaries = torch.tensor([0, 1], dtype=dtype)
    result = m2m.convert(Search(False).eval(), (values, boundaries), backend="fx_importer")
    assert result.ok
    assert opaque_report(result.mlir_text) == {"aten_bucketize": 1}
