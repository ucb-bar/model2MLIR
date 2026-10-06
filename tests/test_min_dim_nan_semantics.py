"""A dim extremum keeps PyTorch's first NaN and first equal-value index."""

import math
from itertools import permutations

import m2m
import numpy as np
import pytest
import torch
from m2m.coverage import opaque_report


class DimExtremum(torch.nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind

    def forward(self, value):
        return torch.min(value, dim=1) if self.kind == "min" else torch.max(value, dim=1)


def _evaluate_body(block, value, best_value, best_index, lane_index):
    """Interpret only the emitted scalar body, preserving selected float bits."""
    env = dict(zip(block.args, (value, best_value, best_index), strict=True))
    predicates = set()
    for op in block.ops:
        args = [env[arg] for arg in op.operands]
        if op.name == "linalg.yield":
            return args, predicates
        if op.name == "linalg.index":
            observed = lane_index
        elif op.name == "arith.index_cast":
            observed = int(args[0])
        elif op.name == "arith.cmpf":
            predicate = op.predicate.value.data
            predicates.add(predicate)
            if predicate == 4:  # ordered less-than
                observed = bool(args[0] < args[1])
            elif predicate == 2:  # ordered greater-than
                observed = bool(args[0] > args[1])
            elif predicate == 1:  # ordered equal, including both signs of zero
                observed = bool(args[0] == args[1])
            elif predicate == 7:  # ordered: neither input is NaN
                observed = not (math.isnan(args[0]) or math.isnan(args[1]))
            elif predicate == 14:  # unordered: at least one input is NaN
                observed = math.isnan(args[0]) or math.isnan(args[1])
            else:
                raise AssertionError(f"unreviewed float comparison predicate: {predicate}")
        elif op.name == "arith.cmpi":
            predicate = op.predicate.value.data
            if predicate == 6:  # unsigned less-than for nonnegative source indices
                observed = int(args[0]) < int(args[1])
            elif predicate == 0:  # integer equality
                observed = int(args[0]) == int(args[1])
            else:
                raise AssertionError(f"unreviewed index comparison predicate: {predicate}")
        elif op.name == "arith.andi":
            observed = bool(args[0]) and bool(args[1])
        elif op.name == "arith.ori":
            observed = bool(args[0]) or bool(args[1])
        elif op.name == "arith.select":
            observed = args[1] if args[0] else args[2]
        else:
            raise AssertionError(f"unreviewed extremum body instruction: {op.name}")
        env[op.results[0]] = observed
    raise AssertionError("extremum body has no yield")


@pytest.mark.parametrize("kind", ["min", "max"])
def test_dim_extremum_first_nan_tie_infinity_and_signed_zero(kind):
    values = torch.tensor(
        [
            [1.0, float("nan"), -2.0],
            [float("nan"), 1.0, float("nan")],
            [float("inf"), float("inf"), float("inf")],
            [-float("inf"), -float("inf"), 1.0],
            [0.0, -0.0, 0.0],
            [-0.0, 0.0, -0.0],
            [2.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    result = m2m.convert(DimExtremum(kind).eval(), (values,), backend="fx_importer", capture_trace=True)
    assert result.ok, result.diagnostics
    result.module.verify()
    assert not opaque_report(result.mlir_text)
    assert result.capture_trace["status"] == "complete"
    assert "9223372036854775807 : i64" in result.mlir_text
    generic = next(op for op in result.module.walk() if op.name == "linalg.generic")
    expected = torch.min(values, dim=1) if kind == "min" else torch.max(values, dim=1)
    all_predicates = set()
    for row_number, row in enumerate(values.numpy()):
        for order in permutations(range(row.size)):
            best_value, best_index = np.float32(np.inf if kind == "min" else -np.inf), (1 << 63) - 1
            for lane_index in order:
                (best_value, best_index), predicates = _evaluate_body(
                    generic.body.block, row[lane_index], best_value, best_index, lane_index
                )
                all_predicates.update(predicates)
            assert np.asarray([best_value], dtype=np.float32).view(np.uint32).item() == (
                expected.values[row_number : row_number + 1].numpy().view(np.uint32).item()
            )
            assert best_index == expected.indices[row_number].item()
    assert 14 in all_predicates  # emitted NaN test, not an ordered-only reduction


@pytest.mark.parametrize("kind", ["min", "max"])
def test_dim_extremum_preserves_first_nan_payload(kind):
    payloads = np.asarray([[0x7FC00011, 0x7FC00022, 0x3F800000]], dtype=np.uint32).view(np.float32)
    values = torch.from_numpy(payloads.copy())
    result = m2m.convert(DimExtremum(kind).eval(), (values,), backend="fx_importer")
    assert result.ok and not opaque_report(result.mlir_text)
    generic = next(op for op in result.module.walk() if op.name == "linalg.generic")
    expected = torch.min(values, dim=1) if kind == "min" else torch.max(values, dim=1)
    for order in permutations(range(payloads.shape[1])):
        best_value, best_index = np.float32(np.inf if kind == "min" else -np.inf), (1 << 63) - 1
        for lane_index in order:
            (best_value, best_index), _ = _evaluate_body(
                generic.body.block, payloads[0, lane_index], best_value, best_index, lane_index
            )
        assert np.asarray([best_value], dtype=np.float32).view(np.uint32).item() == (
            expected.values.numpy().view(np.uint32).item()
        )
        assert best_index == expected.indices.item() == 0


@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8])
def test_unsigned_and_boolean_extrema_refuse_signed_integer_lowering(dtype):
    value = torch.tensor([[0, 1, 255 if dtype == torch.uint8 else 1]], dtype=dtype)
    result = m2m.convert(DimExtremum("min").eval(), (value,), backend="fx_importer")
    assert result.ok
    assert opaque_report(result.mlir_text) == {"aten_min_dim": 1}


def test_signed_integer_extremum_remains_lowered():
    value = torch.tensor([[3, -2, -2]], dtype=torch.int64)
    result = m2m.convert(DimExtremum("min").eval(), (value,), backend="fx_importer")
    assert result.ok and not opaque_report(result.mlir_text)
    assert "arith.cmpi" in result.mlir_text
