"""FP8 element types for Payload IR.

xDSL's builtin dialect does not ship Float8 types. MLIR upstream does
(``Float8E4M3FNType``, ``Float8E5M2Type``) and TorchAO / PyTorch both
produce tensors typed as ``torch.float8_e4m3fn`` or ``torch.float8_e5m2``.
Until xDSL gains these natively, CompGen ships its own implementations
that mirror MLIR's semantics so that a future MLIR round-trip works and
so that Phase-2 numerics passes (``set_numerics_policy``,
``demote_contraction_inputs``) can see FP8 as a real type rather than
a silent ``f16`` demotion.

Two types, matching MLIR exactly:

- ``Float8E4M3FNType`` -- 1 sign, 4 exponent, 3 mantissa bits. Bias 7.
  Finite range ±448. No infinity; NaN is encoded only as ``S.1111.111``
  (the "FN" = "Finite, NaN-only"). This is the H100 / TorchAO
  preferred activation + weight format.
- ``Float8E5M2Type`` -- 1 sign, 5 exponent, 2 mantissa bits. Bias 15.
  IEEE-754-like: has Inf and NaN. Matches the H100 gradient format.

Both types are tensor-legal: ``TensorType(Float8E4M3FNType(), [16, 16])``
parses and verifies without special-case wiring. They also participate
in the ``FixedBitwidthType`` hierarchy so buffer-layout passes can
compute element sizes correctly (1 byte per element).
"""

from __future__ import annotations

from typing import ClassVar

from xdsl.dialects.builtin import _FloatType
from xdsl.ir import ParametrizedAttribute
from xdsl.irdl import irdl_attr_definition


class _Float8Base(ParametrizedAttribute, _FloatType):
    """Shared implementation for FP8 types.

    Each subclass is a parameterless type attribute. Format metadata
    lives on class-level UPPERCASE constants (xDSL's IRDL machinery
    requires that convention for const ClassVars) and is exposed via
    lowercase instance properties for readability.
    """

    EXPONENT_BITS: ClassVar[int] = 0
    MANTISSA_BITS: ClassVar[int] = 0
    HAS_INFINITY: ClassVar[bool] = False
    HAS_NAN: ClassVar[bool] = False
    MAX_FINITE: ClassVar[float] = 0.0

    def iter_unpack(self, buffer, /):
        raise NotImplementedError(
            "FP8 encoding belongs to the original framework tensor"
        )

    def unpack(self, buffer, num, /):
        raise NotImplementedError(
            "FP8 encoding belongs to the original framework tensor"
        )

    def pack_into(self, buffer, offset, value):
        raise NotImplementedError(
            "FP8 encoding belongs to the original framework tensor"
        )

    def pack(self, values):
        raise NotImplementedError(
            "FP8 encoding belongs to the original framework tensor"
        )

    @property
    def bitwidth(self) -> int:
        return 8

    @property
    def compile_time_size(self) -> int:
        return 1

    @property
    def exponent_bits(self) -> int:
        return self.EXPONENT_BITS

    @property
    def mantissa_bits(self) -> int:
        return self.MANTISSA_BITS

    @property
    def has_infinity(self) -> bool:
        return self.HAS_INFINITY

    @property
    def has_nan(self) -> bool:
        return self.HAS_NAN

    @property
    def max_finite(self) -> float:
        return self.MAX_FINITE

    @property
    def exponent_bias(self) -> int:
        """IEEE-style exponent bias: ``2^(exponent_bits - 1) - 1``."""
        return (1 << (self.EXPONENT_BITS - 1)) - 1


@irdl_attr_definition
class Float8E4M3FNType(_Float8Base):
    """FP8 with 4 exponent + 3 mantissa bits, finite-only, NaN-only.

    Matches ``mlir::Float8E4M3FNType``. The "FN" suffix marks the
    format as *Finite, NaN-only*: there is no infinity encoding, and
    NaN occupies the single bit pattern ``S.1111.111``. This gives
    ±448 as the finite range, which is larger than an IEEE-compliant
    E4M3 would allow.

    This is the preferred activation + weight format on H100 for
    TorchAO's ``float8_weight_only`` / ``float8_dynamic_activation``
    paths.
    """

    name = "f8E4M3FN"

    EXPONENT_BITS: ClassVar[int] = 4
    MANTISSA_BITS: ClassVar[int] = 3
    HAS_INFINITY: ClassVar[bool] = False
    HAS_NAN: ClassVar[bool] = True
    MAX_FINITE: ClassVar[float] = 448.0


@irdl_attr_definition
class Float8E5M2Type(_Float8Base):
    """FP8 with 5 exponent + 2 mantissa bits, IEEE-like.

    Matches ``mlir::Float8E5M2Type``. Has full IEEE-style Inf + NaN
    encodings. The trade-off vs E4M3FN: E5M2 has wider dynamic range
    (±57344) but only 2 mantissa bits of precision, so it's the
    gradient/activation format when dynamic range matters more than
    precision.
    """

    name = "f8E5M2"

    EXPONENT_BITS: ClassVar[int] = 5
    MANTISSA_BITS: ClassVar[int] = 2
    HAS_INFINITY: ClassVar[bool] = True
    HAS_NAN: ClassVar[bool] = True
    MAX_FINITE: ClassVar[float] = 57344.0


__all__ = [
    "Float8E4M3FNType",
    "Float8E5M2Type",
]


def torch_fp8_type(dtype):
    """Preserve only exact supported framework formats; unknown FP8 refuses."""
    import torch

    if dtype is torch.float8_e4m3fn:
        return Float8E4M3FNType()
    if dtype is torch.float8_e5m2:
        return Float8E5M2Type()
    if str(dtype).startswith("torch.float8_"):
        raise ValueError("unsupported FP8 storage format: " + str(dtype))
    return None


def register_fp8_types():
    """Idempotent standard-type parser and arith float-constraint registration.

    These storage types encode no values and grant no arithmetic lowering.
    xDSL currently has a closed builtin float table/constraint union; register
    these exact two standard spellings without changing other parse behavior.
    """
    from xdsl.irdl.constraints import BaseAttr
    from xdsl.parser.attribute_parser import AttrParser
    from xdsl.utils.mlir_lexer import MLIRTokenKind

    types = {"f8E4M3FN": Float8E4M3FNType(), "f8E5M2": Float8E5M2Type()}
    if not getattr(AttrParser, "_m2m_fp8_parse_hook", False):
        original = getattr(AttrParser, "_parse_optional_integer_or_float_type", None)
        if original is None:
            raise RuntimeError("selected xDSL cannot parse standard FP8 storage types")

        def parse(self):
            token = self._current_token
            if token.kind == MLIRTokenKind.BARE_IDENT and token.text in types:
                self._consume_token()
                return types[token.text]
            return original(self)

        AttrParser._parse_optional_integer_or_float_type = parse
        AttrParser._m2m_fp8_parse_hook = True
    for name in ("arith", "math"):
        module = __import__("xdsl.dialects." + name, fromlist=["floatingPointLike"])
        constraint = getattr(
            getattr(module, "floatingPointLike", None), "elem_constr", None
        )
        if constraint is None or not hasattr(constraint, "_based_constrs"):
            # Preserve the selected xDSL's verifier; unsupported FP8 casts then
            # refuse instead of silently projecting into another storage type.
            continue
        for value in types.values():
            constraint._based_constrs.setdefault(type(value), BaseAttr(type(value)))


register_fp8_types()
