"""Explicit ordered FP8 arithmetic policy for actual ATen scaled-mm sources.

This is not an exact decomposition of every implementation of ``_scaled_mm``.
In particular, a CPU implementation may scale both F32 operands before BLAS.
The caller must independently authorize the scale/reduction-order difference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class OrderedFP8ScaledMMPolicy:
    """Default-off numerical permission, never inferred from source metadata.

    RNE and nontrapping, unobserved floating effects are execution-environment
    obligations. The emitted guards check the actual finite operand domain;
    they cannot establish a target's rounding mode or whole-model accuracy.
    """

    accept_scale_reordering: bool
    accept_reduction_order_difference: bool
    rounding: str
    floating_effects: str

    def __post_init__(self):
        if (
            type(self.accept_scale_reordering) is not bool
            or self.accept_scale_reordering is not True
            or type(self.accept_reduction_order_difference) is not bool
            or self.accept_reduction_order_difference is not True
            or type(self.rounding) is not str
            or type(self.floating_effects) is not str
            or self.rounding != "rne"
            or self.floating_effects != "nontrapping_unobserved"
        ):
            raise ValueError(
                "ordered FP8 requires explicit scale/reduction permission and RNE/nontrapping effects"
            )

    def record(self):
        require_policy(self)
        return {
            "schema": "m2m.explicit_scaled_mm_numeric_policy.v1",
            "name": "finite_e4m3fn_ordered_f32_postscale_v1",
            "source_operator": "aten._scaled_mm.default",
            "accept_scale_reordering": self.accept_scale_reordering,
            "accept_reduction_order_difference": self.accept_reduction_order_difference,
            "operand_domain": "finite E4M3FN A/B; finite F32 row/column scales",
            "guards": "complete runtime finite scans before contraction",
            "accumulation": "positive-zero F32 seed; increasing-K reduction; no fast-math flags",
            "backend_order_qualification_required": True,
            "scale_order": "F32(dot * row_scale) then F32(* column_scale)",
            "rounding": self.rounding,
            "floating_effects": self.floating_effects,
            "environment_condition_enforced_by_this_frontend": False,
            "source_bit_equivalence_claimed": False,
            "quality_or_target_admission": False,
            "quantizer_or_recipe_replaced": False,
        }


def require_policy(policy):
    if type(policy) is not OrderedFP8ScaledMMPolicy:
        raise ValueError("an explicit OrderedFP8ScaledMMPolicy is required")
    policy.__post_init__()
    return policy


def snapshot_policy(policy):
    """Immutable invocation permission; never a reference to mutable caller state."""
    require_policy(policy)
    return (
        policy.accept_scale_reordering,
        policy.accept_reduction_order_difference,
        policy.rounding,
        policy.floating_effects,
    )


def require_unchanged_policy(policy, snapshot):
    if snapshot_policy(policy) != snapshot:
        raise ValueError("explicit scaled-mm policy changed during conversion")


def sealed_policy_record(policy, snapshot):
    require_unchanged_policy(policy, snapshot)
    record = OrderedFP8ScaledMMPolicy(*snapshot).record()
    require_unchanged_policy(policy, snapshot)
    return record


def decomposition(policy):
    """Return an invocation-owned lowering; no global decomposition mutation."""
    snapshot = snapshot_policy(policy)
    record_json = json.dumps(sealed_policy_record(policy, snapshot), sort_keys=True)

    def lower(operands, meta, name):
        require_unchanged_policy(policy, snapshot)
        result = _lower(operands, meta, name, record_json)
        require_unchanged_policy(policy, snapshot)
        return result

    return lower


def _schema_defaults(schema):
    """Bind the actual pure overload; unknown optional defaults must refuse."""
    expected = [
        ("self", "Tensor", False, None),
        ("mat2", "Tensor", False, None),
        ("scale_a", "Tensor", False, None),
        ("scale_b", "Tensor", False, None),
        ("bias", "Optional[Tensor]", True, None),
        ("scale_result", "Optional[Tensor]", True, None),
        ("out_dtype", "Optional[int]", True, None),
        ("use_fast_accum", "bool", True, False),
    ]
    if (
        schema.name != "aten::_scaled_mm"
        or schema.overload_name != ""
        or len(schema.arguments) != len(expected)
        or len(schema.returns) != 1
        or str(schema.returns[0].type) != "Tensor"
        or schema.returns[0].alias_info is not None
    ):
        raise ValueError("unproved scaled-mm source overload/schema")
    for argument, (name, type_name, has_default, default) in zip(
        schema.arguments, expected, strict=True
    ):
        if (
            argument.name != name
            or str(argument.type) != type_name
            or argument.kwarg_only
            or argument.alias_info is not None
            or argument.has_default_value() is not has_default
            or type(argument.default_value) is not type(default)
            or argument.default_value != default
        ):
            raise ValueError("unproved scaled-mm source arguments/defaults/effects")
    return {argument.name: argument.default_value for argument in schema.arguments[4:]}


def _signature(operands, meta):
    import torch
    from xdsl.dialects.builtin import Float32Type, NoneAttr, TensorType

    from m2m.ir.types.float8 import Float8E4M3FNType

    args, kwargs = meta.get("_fx_args"), meta.get("_fx_kwargs")
    if (
        meta.get("_aten_target") != "aten._scaled_mm.default"
        or not isinstance(args, tuple)
        or not 4 <= len(args) <= 8
        or not isinstance(kwargs, dict)
        or set(kwargs) - {"bias", "scale_result", "out_dtype", "use_fast_accum"}
        or len(operands) != 4
        or set(meta.get("_fx_ssa_args", {})) != set(range(4))
        or any(meta["_fx_ssa_args"][i] is not value for i, value in enumerate(operands))
    ):
        raise ValueError(
            "ordered FP8 requires exact four positional tensor operands and supported optional schema"
        )
    options = _schema_defaults(torch.ops.aten._scaled_mm.default._schema)
    for i, key in enumerate(options, 4):
        if i < len(args):
            if key in kwargs:
                raise ValueError("duplicate scaled-mm optional argument")
            options[key] = args[i]
        elif key in kwargs:
            options[key] = kwargs[key]
    if (
        options["bias"] is not None
        or options["scale_result"] is not None
        or options["out_dtype"] is not torch.float32
        or type(options["use_fast_accum"]) is not bool
        or options["use_fast_accum"] is not False
    ):
        raise ValueError(
            "ordered FP8 supports only bias=None, scale_result=None, explicit outF32 and fast=False"
        )
    types = [value.type for value in operands]
    if any(
        not isinstance(t, TensorType) or not isinstance(t.encoding, NoneAttr)
        for t in types
    ):
        raise ValueError("ordered FP8 requires ordinary dense tensor values")
    if not all(
        isinstance(t.element_type, Float8E4M3FNType) for t in types[:2]
    ) or not all(isinstance(t.element_type, Float32Type) for t in types[2:]):
        raise ValueError(
            "ordered FP8 requires E4M3FN operands and F32 scales; no implicit conversion"
        )
    shapes = [tuple(t.get_shape()) for t in types]
    if any(
        len(shape) != 2 or any(type(d) is not int or not 0 < d < 1 << 63 for d in shape)
        for shape in shapes
    ):
        raise ValueError("ordered FP8 requires positive static rank-two dimensions")
    (m, k), (kb, n), sa, sb = shapes
    if (
        k != kb
        or sa != (m, 1)
        or sb != (1, n)
        or m * n > (1 << 61) - 1
        or any(a * b > (1 << 61) - 1 for a, b in shapes)
    ):
        raise ValueError(
            "ordered FP8 contraction, scale shapes or LP64 byte extents disagree"
        )
    expected_strides = [(k, 1), (1, k), (1, 1), (n, 1)]
    source_layouts = []
    for i, (shape, strides) in enumerate(zip(shapes, expected_strides, strict=True)):
        value = getattr(args[i], "meta", {}).get("val")
        actual_strides = (
            tuple(value.stride()) if isinstance(value, torch.Tensor) else ()
        )
        actual_shape = tuple(value.shape) if isinstance(value, torch.Tensor) else ()
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or any(type(d) is not int for d in actual_shape)
            or actual_shape != shape
            or len(actual_strides) != 2
            or any(type(s) is not int or not 0 <= s < 1 << 63 for s in actual_strides)
            or any(
                size > 1 and actual != expected
                for size, actual, expected in zip(
                    shape, actual_strides, strides, strict=True
                )
            )
            or value.dtype is not (torch.float8_e4m3fn if i < 2 else torch.float32)
        ):
            raise ValueError(
                "ordered FP8 source layout/type is not the proved row-A/column-B/per-row-scale layout"
            )
        # Singleton-axis strides do not affect any addressed logical element.
        # The real AO transposed [1,N] scale has stride [1,1], not [N,1].
        source_layouts.append(
            {
                "shape": list(shape),
                "strides": list(actual_strides),
                "dtype": str(value.dtype),
            }
        )
    result = meta.get("val")
    if (
        not isinstance(result, torch.Tensor)
        or result.dtype is not torch.float32
        or any(type(d) is not int for d in result.shape)
        or tuple(result.shape) != (m, n)
    ):
        raise ValueError("ordered FP8 result metadata is not exact F32[M,N]")
    return m, n, k, source_layouts


def _finite_guard(value, label):
    from xdsl.dialects.arith import AndIOp, CmpfOp, ConstantOp, ExtFOp
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        Float32Type,
        FloatAttr,
        IntegerAttr,
        TensorType,
        i1,
    )
    from xdsl.dialects.cf import AssertOp
    from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
    from xdsl.dialects.linalg.ops import FillOp, GenericOp, YieldOp
    from xdsl.dialects.math import AbsFOp
    from xdsl.dialects.tensor import EmptyOp, ExtractOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    rank = value.type.get_num_dims()
    f32 = Float32Type()
    boolean = TensorType(i1, [])
    empty = EmptyOp([], boolean)
    true = ConstantOp(IntegerAttr(1, i1), i1)
    fill = FillOp(inputs=[true.result], outputs=[empty.results[0]], res=[boolean])
    block = Block(arg_types=[value.type.element_type, i1])
    source = block.args[0]
    if not isinstance(source.type, Float32Type):
        cast = ExtFOp(source, f32)
        block.add_op(cast)
        source = cast.result
    limit = ConstantOp(FloatAttr(3.4028234663852886e38, f32), f32)
    absolute = AbsFOp(source)
    finite = CmpfOp(absolute.result, limit.result, "ole")
    valid = AndIOp(finite.result, block.args[1])
    block.add_ops([limit, absolute, finite, valid, YieldOp(valid.result)])
    scan = GenericOp(
        inputs=[value],
        outputs=[fill.results[0]],
        body=Region(block),
        indexing_maps=[
            AffineMapAttr(AffineMap.identity(rank)),
            AffineMapAttr(AffineMap(rank, 0, ())),
        ],
        iterator_types=[IteratorTypeAttr(IteratorType.REDUCTION)] * rank,
        result_types=[boolean],
    )
    scalar = ExtractOp(scan.results[0], [], i1)
    check = AssertOp(scalar.result, "explicit ordered FP8 requires finite " + label)
    return [empty, true, fill, scan, scalar, check]


def _lower(operands, meta, name, record_json):
    from xdsl.dialects.arith import AddfOp, ConstantOp, ExtFOp, MulfOp
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        Float32Type,
        FloatAttr,
        StringAttr,
        TensorType,
    )
    from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
    from xdsl.dialects.linalg.ops import FillOp, GenericOp, YieldOp
    from xdsl.dialects.tensor import EmptyOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap

    from m2m.ir.decompositions import DecompResult

    m, n, _, layouts = _signature(operands, meta)
    a, b, sa, sb = operands
    f32 = Float32Type()
    result_type = TensorType(f32, [m, n])
    ops = []
    for operand, label in zip(
        operands, ["A", "B", "row scales", "column scales"], strict=True
    ):
        ops.extend(_finite_guard(operand, label))
    empty = EmptyOp([], result_type)
    zero = ConstantOp(FloatAttr(0.0, f32), f32)
    fill = FillOp(inputs=[zero.result], outputs=[empty.results[0]], res=[result_type])
    block = Block(arg_types=[a.type.element_type, b.type.element_type, f32])
    ax, bx = ExtFOp(block.args[0], f32), ExtFOp(block.args[1], f32)
    product = MulfOp(ax.result, bx.result)
    total = AddfOp(block.args[2], product.result)
    block.add_ops([ax, bx, product, total, YieldOp(total.result)])
    d0, d1, d2 = (AffineDimExpr(i) for i in range(3))
    dot = GenericOp(
        inputs=[a, b],
        outputs=[fill.results[0]],
        body=Region(block),
        indexing_maps=[
            AffineMapAttr(AffineMap(3, 0, dims))
            for dims in [(d0, d2), (d2, d1), (d0, d1)]
        ],
        iterator_types=[
            IteratorTypeAttr(kind)
            for kind in [
                IteratorType.PARALLEL,
                IteratorType.PARALLEL,
                IteratorType.REDUCTION,
            ]
        ],
        result_types=[result_type],
    )
    dot.attributes["prov.scaled_mm_stage"] = StringAttr("raw_fp8_dot")
    dot.attributes["prov.scaled_mm_source_layouts"] = StringAttr(
        json.dumps(layouts, sort_keys=True)
    )
    ops.extend([empty, zero, fill, dot])
    result = dot.results[0]
    for scale, axis in [(sa, 0), (sb, 1)]:
        out = EmptyOp([], result_type)
        body = Block(arg_types=[f32, f32, f32])
        multiply = MulfOp(body.args[0], body.args[1])
        body.add_ops([multiply, YieldOp(multiply.result)])
        dimensions = (
            (AffineDimExpr(0), AffineConstantExpr(0))
            if axis == 0
            else (AffineConstantExpr(0), AffineDimExpr(1))
        )
        scaled = GenericOp(
            inputs=[result, scale],
            outputs=[out.results[0]],
            body=Region(body),
            indexing_maps=[
                AffineMapAttr(AffineMap.identity(2)),
                AffineMapAttr(AffineMap(2, 0, dimensions)),
                AffineMapAttr(AffineMap.identity(2)),
            ],
            iterator_types=[IteratorTypeAttr(IteratorType.PARALLEL)] * 2,
            result_types=[result_type],
        )
        scaled.attributes["prov.scaled_mm_stage"] = StringAttr(
            "row_scale" if axis == 0 else "column_scale"
        )
        ops.extend([out, scaled])
        result = scaled.results[0]
    contract = StringAttr(record_json)
    for op in ops:
        op.attributes["prov.explicit_numeric_policy"] = contract
    return DecompResult(ops=ops, result=result, pattern_hint="scaled_mm")
