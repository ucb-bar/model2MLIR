"""Integer right shift and integer floor / truncating division lower to exact integer arith.

Before these decompositions existed, ``aten.bitwise_right_shift``/``aten.__rshift__`` and a
rounding-mode ``aten.div``/``aten.floor_divide`` were OPAQUE calls, so a capture that needed an integer
floor-shift had to build it from float64 (cast, divide, floor, cast back) per element.

Two gates, as for every arithmetic decomposition here:

* **structural** -- no opaque ``func.call``, and the integer op that carries the semantics is present
  (``arith.shrsi`` / ``arith.floordivsi`` / ``arith.divsi``), never a float round trip;
* **numerical** -- the emitted body, evaluated with STRICT integer semantics (a shift amount outside
  ``[0, width)`` or a zero divisor is poison and fails the test rather than producing a value), agrees
  with torch on every int8 pair (exhaustive) and on the int64 boundary values: negative operands,
  ``INT64_MIN``/``INT64_MAX``, shifts of 0, 63, 64 and negative amounts.

This repo hosts no MLIR runtime, so the body is evaluated here op by op (``_eval_body``); every op it
meets must be one it implements, so a new op in the body cannot be silently skipped.
"""

from __future__ import annotations

import itertools

import pytest
import torch

import m2m
from m2m.coverage import opaque_report


class _Bin(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, a, b):
        return self.fn(a, b)


class _Un(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, a):
        return self.fn(a)


def _wrap(value: int, width: int) -> int:
    value &= (1 << width) - 1
    return value - (1 << width) if value >> (width - 1) else value


def _width(t) -> int:
    return int(str(t)[1:])


def _eval_body(block, values):
    """Evaluate a linalg.generic body on python ints with the arith semantics MLIR defines."""
    env = dict(zip(block.args, values))
    for op in block.ops:
        name = op.name
        if name == "linalg.yield":
            return env[op.operands[0]]
        args = [env[o] for o in op.operands]
        w = _width(op.results[0].type)
        if name == "arith.constant":
            r = op.value.value.data
        elif name in ("arith.extsi", "arith.trunci"):
            r = args[0]
        elif name == "arith.minui":
            r = min(args, key=lambda v: v & ((1 << w) - 1))
        elif name == "arith.shrsi":
            assert 0 <= args[1] < w, f"shrsi by {args[1]} on i{w} is poison"
            r = args[0] >> args[1]
        elif name == "arith.floordivsi":
            assert args[1] != 0, "floordivsi by zero"
            r = args[0] // args[1]
        elif name == "arith.divsi":
            assert args[1] != 0, "divsi by zero"
            q = abs(args[0]) // abs(args[1])
            r = q if (args[0] < 0) == (args[1] < 0) else -q
        else:
            raise AssertionError(f"body op {name} is not evaluated by this test")
        env[op.results[0]] = _wrap(r, w)
    raise AssertionError("body has no yield")


def _the_generic(module, op_kind: str):
    found = [
        op
        for op in module.walk()
        if op.name == "linalg.generic" and getattr(op.attributes.get("prov.op"), "data", None) == op_kind
    ]
    assert len(found) == 1, f"expected one {op_kind} generic, found {len(found)}"
    return found[0]


def _capture(model, inputs, op_kind):
    r = m2m.convert(model.eval(), inputs, backend="fx_importer")
    assert r.ok
    assert sum(opaque_report(r.mlir_text).values()) == 0, r.mlir_text
    assert "arith.sitofp" not in r.mlir_text and "math.floor" not in r.mlir_text
    return _the_generic(r.module, op_kind)


def _run(generic, *columns):
    """The generic evaluated element by element on 1-D operands of equal length."""
    block = generic.body.block
    n_in = len(generic.inputs)
    return [_eval_body(block, [*(c[i] for c in columns[:n_in]), 0]) for i in range(len(columns[0]))]


_INT8 = list(range(-128, 128))
_I64_EDGES = sorted(
    {0, 1, -1, 2, -2, 3, -3, 7, -7, 2**31, -(2**31), 2**62, -(2**62), 2**63 - 1, -(2**63), 2**63 - 2, -(2**63) + 1}
    | {s * (2**k + d) for k in (1, 15, 30, 31, 52, 53, 62) for d in (-1, 0, 1) for s in (1, -1)}
)
_SHIFT_AMOUNTS = list(range(-3, 67)) + [127, 1000, -1000, 2**40]


# ---- right shift --------------------------------------------------------------------------------


def test_shift_tensor_int8_exhaustive():
    pairs = list(itertools.product(_INT8, _INT8))
    a = torch.tensor([p[0] for p in pairs], dtype=torch.int8)
    s = torch.tensor([p[1] for p in pairs], dtype=torch.int8)
    gen = _capture(_Bin(torch.bitwise_right_shift), (a, s), "bitwise_right_shift")
    assert any(op.name == "arith.shrsi" for op in gen.body.block.ops)
    assert _run(gen, a.tolist(), s.tolist()) == torch.bitwise_right_shift(a, s).tolist()


def test_shift_tensor_int64_boundaries():
    pairs = list(itertools.product(_I64_EDGES, [v for v in _SHIFT_AMOUNTS if -(2**63) <= v < 2**63]))
    a = torch.tensor([p[0] for p in pairs], dtype=torch.int64)
    s = torch.tensor([p[1] for p in pairs], dtype=torch.int64)
    gen = _capture(_Bin(lambda x, y: x >> y), (a, s), "bitwise_right_shift")
    assert _run(gen, a.tolist(), s.tolist()) == (a >> s).tolist()


@pytest.mark.parametrize("amount", [0, 1, 7, 8, 12, 30, 62, 63, 64, 65, 300, -1, -64])
@pytest.mark.parametrize("spelling", ["operator", "function"])
def test_shift_by_python_int_int64(amount, spelling):
    a = torch.tensor(_I64_EDGES, dtype=torch.int64)
    fn = (lambda x: x >> amount) if spelling == "operator" else (lambda x: torch.bitwise_right_shift(x, amount))
    gen = _capture(_Un(fn), (a,), "bitwise_right_shift")
    names = [op.name for op in gen.body.block.ops]
    assert "arith.minui" not in names, "a python-int amount is clamped at capture time, not per element"
    assert _run(gen, a.tolist()) == fn(a).tolist()


@pytest.mark.parametrize("amount", [-300, -129, -1, 0, 3, 6, 7, 8, 9, 127, 128, 255, 263, 300])
def test_shift_by_python_int_int8_wraps_like_torch(amount):
    """torch casts a python-int operand into the tensor's dtype first: on int8, ``>> 263`` is ``>> 7``."""
    a = torch.tensor(_INT8, dtype=torch.int8)
    gen = _capture(_Un(lambda x: x >> amount), (a,), "bitwise_right_shift")
    assert _run(gen, a.tolist()) == (a >> amount).tolist()


def test_shift_promotes_a_narrower_amount():
    a = torch.tensor(_I64_EDGES, dtype=torch.int64)
    s = torch.tensor([(i * 7) % 70 - 3 for i in range(len(_I64_EDGES))], dtype=torch.int32)
    gen = _capture(_Bin(torch.bitwise_right_shift), (a, s), "bitwise_right_shift")
    assert _run(gen, a.tolist(), s.tolist()) == torch.bitwise_right_shift(a, s).tolist()


def test_shift_broadcasts_a_row_amount():
    a = torch.arange(-40, 40, dtype=torch.int64).reshape(8, 10) * 1001
    s = torch.arange(8, dtype=torch.int64).reshape(8, 1)
    r = m2m.convert(_Bin(torch.bitwise_right_shift).eval(), (a, s), backend="fx_importer")
    assert sum(opaque_report(r.mlir_text).values()) == 0
    assert "arith.shrsi" in r.mlir_text


def test_unsigned_shift_stays_opaque():
    """A uint8 is the same signless i8 as an int8; an arithmetic shift of it would type-check and be
    wrong, so it is left to the opaque path rather than guessed."""
    a = torch.tensor([0, 1, 128, 255], dtype=torch.uint8)
    r = m2m.convert(_Un(lambda x: x >> 1).eval(), (a,), backend="fx_importer")
    assert "arith.shrsi" not in r.mlir_text
    assert sum(opaque_report(r.mlir_text).values()) == 1


# ---- floor / truncating division ----------------------------------------------------------------

_NONZERO_INT8 = [v for v in _INT8 if v != 0]


@pytest.mark.parametrize(
    "spelling,mode",
    [
        ("div_floor", "floor"),
        ("floor_divide", "floor"),
        ("operator", "floor"),
        ("div_trunc", "trunc"),
    ],
)
def test_division_tensor_int8_exhaustive(spelling, mode):
    pairs = list(itertools.product(_INT8, _NONZERO_INT8))
    a = torch.tensor([p[0] for p in pairs], dtype=torch.int8)
    b = torch.tensor([p[1] for p in pairs], dtype=torch.int8)
    fn = {
        "div_floor": lambda x, y: torch.div(x, y, rounding_mode="floor"),
        "floor_divide": torch.floor_divide,
        "operator": lambda x, y: x // y,
        "div_trunc": lambda x, y: torch.div(x, y, rounding_mode="trunc"),
    }[spelling]
    kind = "floor_divide" if mode == "floor" else "trunc_divide"
    gen = _capture(_Bin(fn), (a, b), kind)
    want = "arith.floordivsi" if mode == "floor" else "arith.divsi"
    assert any(op.name == want for op in gen.body.block.ops)
    assert _run(gen, a.tolist(), b.tolist()) == fn(a, b).tolist()


def test_floor_division_rounds_toward_minus_infinity():
    """The point of floor mode: ``-7 // 2 == -4``, where C truncation gives -3."""
    a = torch.tensor([-7, 7, -7, 7, -1, -6], dtype=torch.int64)
    b = torch.tensor([2, -2, -2, 2, 3, 3], dtype=torch.int64)
    gen = _capture(_Bin(lambda x, y: torch.div(x, y, rounding_mode="floor")), (a, b), "floor_divide")
    assert _run(gen, a.tolist(), b.tolist()) == [-4, -4, 3, 3, -1, -2]


def test_floor_division_int64_boundaries():
    divisors = [v for v in _I64_EDGES if v != 0]
    pairs = [(x, y) for x in _I64_EDGES for y in divisors if not (x == -(2**63) and y == -1)]
    a = torch.tensor([p[0] for p in pairs], dtype=torch.int64)
    b = torch.tensor([p[1] for p in pairs], dtype=torch.int64)
    gen = _capture(_Bin(lambda x, y: torch.div(x, y, rounding_mode="floor")), (a, b), "floor_divide")
    assert _run(gen, a.tolist(), b.tolist()) == torch.div(a, b, rounding_mode="floor").tolist()


@pytest.mark.parametrize("divisor", [1, 2, 3, 7, 127, 1 << 30, -1, -3, -(1 << 30)])
def test_floor_division_by_python_int(divisor):
    a = torch.tensor(_I64_EDGES, dtype=torch.int64)
    a = a[a != -(2**63)] if divisor == -1 else a
    for fn in (lambda x: torch.div(x, divisor, rounding_mode="floor"), lambda x: x // divisor):
        gen = _capture(_Un(fn), (a,), "floor_divide")
        assert _run(gen, a.tolist()) == fn(a).tolist()


@pytest.mark.parametrize(
    "kind,fn",
    [
        ("floor_divide", lambda y: torch.div(1 << 30, y, rounding_mode="floor")),
        ("floor_divide", lambda y: (1 << 30) // y),
        ("bitwise_right_shift", lambda y: torch.bitwise_right_shift(-(1 << 40), y)),
    ],
)
def test_python_int_on_the_left(kind, fn):
    """A python-int first operand stays the FIRST operand: ``c // y``, not ``y // c``."""
    y = torch.tensor([v for v in range(-70, 70) if v != 0], dtype=torch.int64)
    gen = _capture(_Un(fn), (y,), kind)
    assert _run(gen, y.tolist()) == fn(y).tolist()


def test_true_division_mode_none_is_ordinary_division():
    a = torch.tensor([-7, 7, 3], dtype=torch.int64)
    b = torch.tensor([2, 2, -2], dtype=torch.int64)
    r = m2m.convert(_Bin(lambda x, y: torch.div(x, y, rounding_mode=None)).eval(), (a, b), backend="fx_importer")
    assert sum(opaque_report(r.mlir_text).values()) == 0
    assert "arith.divf" in r.mlir_text and "arith.floordivsi" not in r.mlir_text


def test_float_floor_division_stays_opaque():
    """torch's float floor division is not ``floor(a / b)`` (it is fmod-based); not written here."""
    a = torch.tensor([-7.5, 7.0], dtype=torch.float32)
    b = torch.tensor([2.0, -2.0], dtype=torch.float32)
    r = m2m.convert(_Bin(lambda x, y: torch.div(x, y, rounding_mode="floor")).eval(), (a, b), backend="fx_importer")
    assert "arith.floordivsi" not in r.mlir_text
    assert sum(opaque_report(r.mlir_text).values()) == 1
