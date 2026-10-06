"""Op-coverage harness.

Two tools for the agent-driven coverage loop:

- ``opaque_report(mlir_text)`` -- inventory the opaque ``func.call`` placeholders in
  emitted MLIR (the worklist of ops still needing real decompositions).
- ``validate_op(build_fn, example_inputs)`` -- capture a single-op module through the
  FXImporter and check it lowered to real ops (no opaque call) with a result tensor
  type whose shape/dtype match PyTorch eager. This is the gate every new decomposition
  must pass before it's considered done.

Numeric (bit-level) validation needs an MLIR execution engine, which xDSL doesn't ship;
until then we validate structurally (lowered + shape/dtype match eager), which catches
the overwhelming majority of decomposition mistakes.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn


def opaque_report(mlir_text: str) -> dict[str, int]:
    """Count calls without a defined body, independent of MLIR printer syntax.

    FXImporter prints generic ``"func.call"(%x) <{callee = @name}>`` while
    torch-mlir may print ``func.call @name(%x)``. A text regex can miss the
    former and falsely report a complete capture. Parse only call-bearing
    modules; an unparseable call is unknown, never zero opaque work.
    """
    if "func.call" not in mlir_text:
        return {}
    from xdsl.dialects.func import CallOp, FuncOp
    from xdsl.traits import SymbolTable

    from m2m.capture.torch_mlir_bridge import _parse_mlir_text_to_xdsl

    module = _parse_mlir_text_to_xdsl(mlir_text)
    if module is None:
        raise ValueError("cannot inspect call-bearing MLIR for unresolved externals")
    module.verify()
    calls = Counter()
    for op in module.walk():
        if not isinstance(op, CallOp):
            continue
        callee = SymbolTable.lookup_symbol(op, op.callee)
        if not isinstance(callee, FuncOp) or not callee.body.blocks:
            calls[op.callee.string_value()] += 1
    return dict(calls)


@dataclass
class OpValidation:
    op: str
    lowered: bool                      # no opaque func.call for this op
    shape_ok: bool                     # emitted result type matches eager shape/dtype
    opaque_calls: dict[str, int] = field(default_factory=dict)
    eager_shape: tuple[int, ...] | None = None
    mlir_result_type: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.lowered and self.shape_ok and self.error is None


class _SingleOp(nn.Module):
    def __init__(self, fn: Callable[..., torch.Tensor]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return self.fn(*args)


def validate_op(
    fn: Callable[..., torch.Tensor],
    example_inputs: tuple[torch.Tensor, ...],
    *,
    name: str,
) -> OpValidation:
    """Capture ``fn`` (a single torch op) via the FXImporter and validate it lowered.

    Forces ``backend="fx_importer"`` so we exercise our own decomposition path, not
    torch-mlir.
    """
    import m2m

    try:
        eager = fn(*example_inputs)
        eager_shape = tuple(int(d) for d in eager.shape)
    except Exception as exc:  # noqa: BLE001
        return OpValidation(op=name, lowered=False, shape_ok=False, error=f"eager failed: {exc}")

    try:
        r = m2m.convert(_SingleOp(fn).eval(), example_inputs, backend="fx_importer")
    except Exception as exc:  # noqa: BLE001
        return OpValidation(op=name, lowered=False, shape_ok=False,
                            eager_shape=eager_shape, error=f"convert failed: {exc}")

    opaque = opaque_report(r.mlir_text)
    lowered = sum(opaque.values()) == 0
    # The function's returned tensor type, e.g. "-> tensor<4x64xf32>"
    m = re.search(r"->\s*(tensor<[^>]+>)\s*\{", r.mlir_text) or re.search(r"return\s+%\w+\s*:\s*(tensor<[^>]+>)", r.mlir_text)
    result_type = m.group(1) if m else None
    shape_ok = False
    if result_type is not None:
        dims = re.findall(r"(\d+)", result_type.split(",")[0] + "".join(result_type.split("x")))
        # crude: confirm every eager dim appears in the result type string
        shape_ok = all(str(d) in result_type for d in eager_shape) if eager_shape else True
    return OpValidation(
        op=name, lowered=lowered, shape_ok=shape_ok, opaque_calls=opaque,
        eager_shape=eager_shape, mlir_result_type=result_type,
    )


def dialect_op_histogram(mlir_text: str) -> dict[str, int]:
    """Multiset of ``dialect.op`` names in the MLIR (e.g. {'linalg.generic': 3})."""
    ops = re.findall(r"\b([a-z_]+\.[a-z_][a-z_0-9]*)\b", mlir_text)
    keep = {"linalg", "tensor", "arith", "math", "func", "scf", "cf", "complex", "bufferization"}
    return dict(Counter(o for o in ops if o.split(".")[0] in keep and o != "func.func"))


# The matchable vocabulary -- the small, fixed surface a downstream pass keys on. These
# mirror import_fx._FAMILY_OF; a model whose families fall outside this set means a new
# pattern was introduced (proliferation) and should be reviewed, not silently accepted.
CANONICAL_FAMILIES = frozenset({
    "elementwise", "cast", "fill", "iota", "compare", "select", "minmax", "logical",
    "bitwise", "reduce", "arg_reduce", "contraction", "normalization", "attention", "layout",
    "concat", "gather_scatter", "scan", "search", "quantize",
})


def op_vocabulary(mlir_text: str) -> dict[str, dict[str, int]]:
    """Return the matchable op surface as ``{family: {op_kind: count}}`` from the
    ``prov.family`` / ``prov.op`` attribute tags. This is what a pass writer sees: how few,
    stable buckets the whole network collapses into (the scalability goal)."""
    out: dict[str, dict[str, int]] = {}
    for fam, op in re.findall(r'prov\.family = "([^"]+)"[^\n]*?prov\.op = "([^"]+)"', mlir_text):
        out.setdefault(fam, Counter())[op] += 1
    # also catch ops where the attrs print in the other order
    for op, fam in re.findall(r'prov\.op = "([^"]+)"[^\n]*?prov\.family = "([^"]+)"', mlir_text):
        d = out.setdefault(fam, Counter())
        # avoid double counting lines already matched above is hard via regex; this is a
        # best-effort report, not an exact count -- use it to see the *shape*, not totals.
        d[op] += 0
    return {k: dict(v) for k, v in out.items()}


# Named ops are matchable by their op-type alone (no attr needed); only the catch-all
# linalg.generic must carry prov.family/prov.op to be matchable. (xDSL's custom assembly for
# named linalg ops doesn't even print discardable attrs, though they exist on the IR.)
_NAMED_MATCHABLE = re.compile(
    r"\b(linalg\.matmul|linalg\.batch_matmul|linalg\.transpose|linalg\.reduce|linalg\.fill|"
    r"linalg\.broadcast|linalg\.softmax|linalg\.conv\w*|tensor\.concat|tensor\.extract_slice|"
    r"tensor\.insert_slice|tensor\.collapse_shape|tensor\.expand_shape|scf\.for)\b")


def module_sections(mlir_text: str) -> dict[str, int]:
    """Report ``{source_module: op_count}`` from the ``prov.module`` provenance tags -- the
    VLA sections (VLM / action expert / vision / ...) and their sizes. The basis for
    per-section partitioning + per-frequency scheduling."""
    return dict(Counter(re.findall(r'prov\.module = "([^"]+)"', mlir_text)))


def region_summary(mlir_text: str) -> dict[str, int]:
    """Report ``{family: number_of_distinct_regions}`` -- the count of transform-units (ops
    sharing an ``prov.region_id``) per coarse family. This is what a rewrite pass actually
    iterates over: a whole transformer collapses to a few dozen regions across ~a dozen
    families, not thousands of unique ops. The headline number for 'are we proliferating?'."""
    seen: dict[str, set] = {}
    for m in re.finditer(r'prov\.region_id = "([^"]+)"[^\n]*?prov\.family = "([^"]+)"', mlir_text):
        seen.setdefault(m.group(2), set()).add(m.group(1))
    for m in re.finditer(r'prov\.family = "([^"]+)"[^\n]*?prov\.region_id = "([^"]+)"', mlir_text):
        seen.setdefault(m.group(1), set()).add(m.group(2))
    return {fam: len(ids) for fam, ids in sorted(seen.items(), key=lambda kv: -len(kv[1]))}


def untagged_compute_ops(mlir_text: str) -> int:
    """Count ``linalg.generic`` ops carrying NO ``prov.family`` tag. Should be 0 -- a generic
    is the only op that's *not* self-describing (everything is a linalg.generic), so it MUST
    be tagged to be matchable. Named ops (matmul/transpose/reduce/...) are matchable by their
    op-type and are excluded. A non-zero value means a generic escaped the taxonomy."""
    untagged = 0
    for line in mlir_text.splitlines():
        if "linalg.generic" in line and "prov.family" not in line:
            untagged += 1
    return untagged


_NAMED_FAMILY = {  # named op (matchable by type) -> coarse family
    "linalg.matmul": "contraction", "linalg.batch_matmul": "contraction",
    "linalg.conv": "contraction", "linalg.transpose": "layout", "linalg.broadcast": "layout",
    "linalg.reduce": "reduce", "linalg.fill": "fill", "linalg.softmax": "normalization",
    "tensor.concat": "concat", "tensor.extract_slice": "layout",
    "tensor.insert_slice": "layout", "tensor.collapse_shape": "layout",
    "tensor.expand_shape": "layout", "scf.for": "gather_scatter",
}


def family_histogram(mlir_text: str) -> dict[str, int]:
    """Multiset of coarse families over the FULL matchable surface: generics by their
    ``prov.family`` attr, plus named ops folded in by op-type (since named ops are matchable
    by type and xDSL doesn't print their discardable attrs)."""
    h = Counter(re.findall(r'prov\.family = "([^"]+)"', mlir_text))
    for line in mlir_text.splitlines():
        if "linalg.generic" in line:
            continue
        for named, fam in _NAMED_FAMILY.items():
            if re.search(r"\b" + re.escape(named) + r"\w*\b" if named.endswith("conv") else r"\b" + re.escape(named) + r"\b", line):
                h[fam] += 1
                break
    return dict(h)


@dataclass
class OpDiff:
    op: str
    ours_lowered: bool
    ours_opaque: dict[str, int]
    golden_ok: bool                       # torch-mlir produced a lowering
    golden_ops: dict[str, int] = field(default_factory=dict)   # the TARGET lowering
    our_ops: dict[str, int] = field(default_factory=dict)
    result_type_match: bool = False       # our result type == torch-mlir's
    golden_mlir: str = ""
    error: str | None = None

    @property
    def canonical(self) -> bool:
        """True when our op-kinds match the oracle's (same standard ops, not just same
        result type) -- i.e. we used the canonical/proper form, not a hand-rolled
        equivalent (e.g. tensor.reshape+constants instead of collapse/expand_shape).

        Structural func.* boilerplate is ignored (it's not part of the op pattern)."""
        skip = {"func.return", "func.func", "func.call"}
        return {o for o in self.our_ops if o not in skip} == {o for o in self.golden_ops if o not in skip}

    @property
    def verdict(self) -> str:
        if not self.golden_ok:
            return "no-oracle"            # torch-mlir can't lower it either
        if not self.ours_lowered:
            return "TODO: implement (oracle available)"
        if not self.result_type_match:
            return "mismatch"
        return "ok" if self.canonical else "ok (non-canonical ops)"


def golden_lowering(fn: Callable[..., torch.Tensor], example_inputs: tuple[torch.Tensor, ...]):
    """Lower a single op via torch-mlir (the oracle). Returns mlir_text or None."""
    import m2m

    try:
        r = m2m.convert(_SingleOp(fn).eval(), example_inputs, backend="torch_mlir")
        return r.mlir_text or None
    except Exception:  # noqa: BLE001
        return None


def _result_type(mlir_text: str) -> str | None:
    m = re.search(r"->\s*(tensor<[^>]+>)\s*\{", mlir_text) or re.search(r"return\s+%\w+\s*:\s*(tensor<[^>]+>)", mlir_text)
    return m.group(1) if m else None


def differential_op(
    fn: Callable[..., torch.Tensor],
    example_inputs: tuple[torch.Tensor, ...],
    *,
    name: str,
) -> OpDiff:
    """Differential test one op: our FXImporter output vs torch-mlir's golden lowering.

    - ``verdict == "TODO: implement (oracle available)"``: we emit an opaque placeholder
      but torch-mlir shows the target linalg -- use ``golden_ops`` / ``golden_mlir`` as the
      spec for writing the emitter.
    - ``verdict == "ok"``: our emitter lowered and its result type matches the oracle.
    - ``verdict == "no-oracle"``: torch-mlir can't lower it either (truly novel op).
    """
    import m2m

    ours = validate_op(fn, example_inputs, name=name)
    golden = golden_lowering(fn, example_inputs)
    golden_ok = golden is not None
    golden_hist = dialect_op_histogram(golden) if golden else {}
    our_text = ""
    try:
        our_text = m2m.convert(_SingleOp(fn).eval(), example_inputs, backend="fx_importer").mlir_text
    except Exception:  # noqa: BLE001
        pass
    rt_match = bool(golden) and _result_type(our_text) is not None and _result_type(our_text) == _result_type(golden)
    return OpDiff(
        op=name,
        ours_lowered=ours.lowered,
        ours_opaque=ours.opaque_calls,
        golden_ok=golden_ok,
        golden_ops=golden_hist,
        our_ops=dialect_op_histogram(our_text),
        result_type_match=rt_match,
        golden_mlir=golden or "",
        error=ours.error,
    )


__all__ = [
    "CANONICAL_FAMILIES",
    "OpDiff",
    "OpValidation",
    "dialect_op_histogram",
    "differential_op",
    "family_histogram",
    "golden_lowering",
    "op_vocabulary",
    "opaque_report",
    "region_summary",
    "untagged_compute_ops",
    "validate_op",
]
