"""QDQ-preservation pass: fold a lowered weight dequant into a ``quant_ext.dequantize`` op.

torchao weight-only int8/fp8 lowers its dequant to a `dtype_cast` (int->float) generic feeding
a `mul` (x scale) generic -- the quantization semantics are *implicit* in two generics. This
pass recognizes that pair via the taxonomy tags (`prov.op == "dtype_cast"` then `"mul"`) and
rewrites it to an explicit ``quant_ext.dequantize_per_{tensor,channel}`` op, so the
quantization is first-class and matchable (a target-aware fuse can then map it to a native
quantized matmul). Symmetric weight-only -> zero_point = 0.

Rewrites operate on a clone. Scale dependencies are preflighted for safe motion and SSA
availability before fusion; unsupported matches are left unchanged. A failed rewrite or
module verification returns the original module. Verification alone does not prove safe
motion or SSA dominance.
"""

from __future__ import annotations

from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import IntegerAttr, IntegerType, ModuleOp, StringAttr, TensorType
from xdsl.dialects.linalg import GenericOp, MatmulOp
from xdsl.dialects.tensor import SplatOp
from xdsl.ir import BlockArgument, Operation, SSAValue
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.traits import IsolatedFromAbove, Pure

from m2m.ir.quant.ops import DequantizePerChannelOp, DequantizePerTensorOp


def _tag(op) -> str | None:
    a = op.attributes.get("prov.op")
    return a.data if isinstance(a, StringAttr) else None


def _producer(value):
    """The op that defines ``value`` (None for block args)."""
    from xdsl.ir import Operation

    owner = getattr(value, "owner", None)
    return owner if isinstance(owner, Operation) else None


def _available_before(value: SSAValue, anchor: Operation) -> bool:
    """Check lexical SSA availability, including captures from ancestor blocks.

    Sibling blocks and other CFG dominance relationships are deliberately unsupported.
    Capturing a value across an isolated operation is never legal.
    """
    owner = value.owner
    block = value.block if isinstance(value, BlockArgument) else (
        owner.parent_block() if isinstance(owner, Operation) else None
    )
    if block is None:
        return False
    consumer = anchor
    while consumer.parent_block() is not block:
        parent = consumer.parent_op()
        if parent is None or parent.has_trait(IsolatedFromAbove):
            return False
        consumer = parent
    return isinstance(value, BlockArgument) or (
        owner is not consumer and owner.is_before_in_block(consumer)
    )


def _plan_producer_hoists_before(
    value: SSAValue, anchor: Operation
) -> tuple[Operation, ...] | None:
    """Preflight the whole dependency closure before moving any operation.

    Only region-free Pure producers in the anchor's block can move. Region captures,
    effects, missing/foreign definitions and dependencies on the anchor fail closed.
    The returned order defines every moved operand before its consumer.
    """
    block = anchor.parent_block()
    if block is None:
        return None
    planned: list[Operation] = []
    complete: set[Operation] = set()
    visiting: set[Operation] = set()

    def visit(operand: SSAValue) -> bool:
        if _available_before(operand, anchor):
            return True
        op = _producer(operand)
        if (
            op is None or op is anchor or op.parent_block() is not block
            or op.regions or not op.has_trait(Pure) or op in visiting
        ):
            return False
        if op in complete:
            return True
        visiting.add(op)
        if not all(visit(dependency) for dependency in op.operands):
            return False
        visiting.remove(op)
        complete.add(op)
        planned.append(op)
        return True

    return tuple(planned) if visit(value) else None


def fuse_qdq(module: ModuleOp) -> ModuleOp:
    """Fold `dtype_cast(int->float) -> mul(scale)` weight-dequant pairs into
    `quant_ext.dequantize_*`. Preflight motion on a clone and return it only if it
    verifies; failures return the original module unchanged."""
    work = module.clone()
    try:
        _fuse(work)
        work.verify()
        return work
    except Exception:  # noqa: BLE001
        return module


def _fuse(module: ModuleOp) -> None:
    """Recognize torchao's post-matmul weight dequant and move the scale onto the weight:

        mul( matmul(x, dtype_cast(w_i8 -> f32)), scale[N] )   (scale is per-output-channel)
      ==  matmul(x, quant_ext.dequantize_per_channel(w_i8, scale, zp, axis=last))

    Valid because the scale broadcasts over the matmul's output channels, which equals the
    weight's trailing (column) dim. Rewires by replacing the cast's single use (the matmul's
    RHS) with the dequantize, and the mul's uses with the matmul result."""
    for mul in list(module.walk()):
        if not isinstance(mul, GenericOp) or _tag(mul) != "mul" or len(mul.inputs) != 2:
            continue
        # one input is a matmul result, the other is the per-channel scale
        a, b = mul.inputs
        mm, scale = (_producer(a), b) if isinstance(_producer(a), MatmulOp) else (_producer(b), a)
        if not isinstance(mm, MatmulOp) or not isinstance(scale.type, TensorType):
            continue
        rhs = mm.inputs[1]                       # weight operand (already f32, from the cast)
        cast = _producer(rhs)
        if not isinstance(cast, GenericOp) or _tag(cast) != "dtype_cast" or not cast.inputs:
            continue
        w_i8 = cast.inputs[0]                     # the integer weight
        if not isinstance(w_i8.type, TensorType) or not isinstance(w_i8.type.element_type, IntegerType):
            continue
        if list(cast.results[0].uses) != [u for u in cast.results[0].uses if u.operation is mm] or \
                len(list(cast.results[0].uses)) != 1:
            continue  # the cast must feed only this matmul (safe to rewrite)

        # Verification alone does not prove SSA dominance or safe operation motion.
        # Refuse the whole match before changing its operations or source provenance.
        hoists = _plan_producer_hoists_before(scale, cast)
        if hoists is None or not _available_before(w_i8, cast):
            continue

        w_t = rhs.type                            # f32 weight type the matmul expects
        w_rank = len(w_t.get_shape())
        scale_shape = list(scale.type.get_shape())
        zp_elem = IntegerType(32)
        zero = ConstantOp(IntegerAttr(0, zp_elem), zp_elem)
        zp = SplatOp(zero.result, [], TensorType(zp_elem, scale_shape))
        deq = DequantizePerChannelOp(
            operands=[w_i8, scale, zp.results[0]], result_types=[w_t],
            properties={"axis": IntegerAttr(w_rank - 1, IntegerType(64)),
                        "input_dtype": StringAttr(str(w_i8.type.element_type))},
        )
        deq.attributes["prov.op"] = StringAttr("dequantize")
        deq.attributes["prov.family"] = StringAttr("quantize")
        from m2m.capture.trace import copy_mlir_sources

        # Both deleted operations survive as explicitly fused origins.
        copy_mlir_sources(deq, cast, mul, role="lowering")
        copy_mlir_sources(zero, cast, mul)
        copy_mlir_sources(zp, cast, mul)
        copy_mlir_sources(mm, mul, role="lowering")
        # Propagate the int_data/scale model attribute paths (set by the access-subclass
        # decomposition) onto the dequant. xDSL's printer drops attributes on tensor.empty,
        # so the tags can't ride on the elided inner-tensor empties through the text handoff;
        # the dequant serializes its attrs, so a consumer recovers the binding from them.
        def _src_quant_inner(val):
            op = getattr(val, "owner", None)
            for _ in range(8):
                if not hasattr(op, "attributes"):
                    return None
                qi = op.attributes.get("prov.quant_inner")
                if qi is not None:
                    return qi
                ins = getattr(op, "operands", ())
                op = getattr(ins[0], "owner", None) if ins else None
            return None
        _wk, _sk = _src_quant_inner(w_i8), _src_quant_inner(scale)
        if _wk is not None:
            deq.attributes["prov.quant_inner_w"] = _wk
        if _sk is not None:
            deq.attributes["prov.quant_inner_s"] = _sk
        # `scale`'s producer (e.g. a view/collapse_shape reshaping [N,1] -> [N]) was only
        # positioned after `mm` because its sole consumer used to be the post-matmul `mul`;
        # now that `deq` (which needs `scale`) is inserted before `cast`/`mm`, that producer
        # must be hoisted too, or the emitted IR uses `scale` before it's defined.
        for producer in hoists:
            producer.detach()
            Rewriter.insert_op(producer, InsertPoint.before(cast))
        Rewriter.insert_op([zero, zp, deq], InsertPoint.before(cast))
        cast.results[0].replace_all_uses_with(deq.results[0])   # matmul now reads the dequant
        mul.results[0].replace_all_uses_with(mm.results[0])     # drop the post-matmul scale
        Rewriter.erase_op(mul)
        Rewriter.erase_op(cast)
