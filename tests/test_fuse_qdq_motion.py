"""QDQ fusion must preserve SSA order and never move effectful scale producers."""

from __future__ import annotations

import io

import pytest
from xdsl.dialects import arith, scf, tensor
from xdsl.dialects.builtin import (
    AffineMapAttr, ArrayAttr, DenseIntOrFPElementsAttr, IntegerAttr, ModuleOp,
    StringAttr, TensorType, f32, i8, i64,
)
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, MatmulOp, YieldOp
from xdsl.ir import Block, Operation, Region
from xdsl.ir.affine import AffineDimExpr, AffineMap
from xdsl.printer import Printer

from m2m.transforms.fuse_qdq import fuse_qdq


MATRIX = TensorType(f32, [1, 1])
WEIGHT = TensorType(i8, [1, 1])
SCALE = TensorType(f32, [1])


def _zero():
    return arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(MATRIX, [0.0]))


def _collapse(value):
    groups = ArrayAttr([ArrayAttr([IntegerAttr(0, i64), IntegerAttr(1, i64)])])
    return tensor.CollapseShapeOp(
        operands=[value], result_types=[SCALE], properties={"reassociation": groups}
    )


def _append_chain(block, case):
    """Create a complete matched chain, with the scale initially after the matmul."""
    cast_out, mm_out, mul_out = _zero(), _zero(), _zero()
    cast_body = Block(arg_types=[i8, f32])
    converted = arith.SIToFPOp(cast_body.args[0], f32)
    cast_body.add_ops([converted, YieldOp(converted.results[0])])
    cast = GenericOp(
        [block.args[1]], [cast_out.results[0]], Region(cast_body),
        [AffineMapAttr(AffineMap.identity(2))] * 2,
        [IteratorTypeAttr.parallel()] * 2, [MATRIX],
    )
    cast.attributes["prov.op"] = StringAttr("dtype_cast")
    cast.attributes["prov.source_node_ids"] = ArrayAttr([StringAttr("prepared.cast")])
    cast.attributes["prov.origin_node_ids"] = ArrayAttr([StringAttr("original.cast")])
    mm = MatmulOp([block.args[0], cast.results[0]], [mm_out.results[0]], [MATRIX])
    mm.attributes["prov.source_node_ids"] = ArrayAttr([StringAttr("prepared.matmul")])
    mm.attributes["prov.origin_node_ids"] = ArrayAttr([StringAttr("original.matmul")])
    block.add_ops([cast_out, mm_out, mul_out, cast, mm])

    if case == "collapse":
        scale_op = _collapse(block.args[2])
        block.add_op(scale_op)
    elif case == "transitive":
        view = tensor.CastOp(block.args[2], MATRIX)
        scale_op = _collapse(view.results[0])
        block.add_ops([view, scale_op])
    elif case == "shared":
        view = tensor.CastOp(block.args[2], MATRIX)
        left, right = _collapse(view.results[0]), _collapse(view.results[0])
        scale_op = arith.AddfOp(left.results[0], right.results[0])
        block.add_ops([view, left, right, scale_op])
    elif case == "effect":
        block.add_op(CallOp("mutate_state", [], []))
        scale_op = CallOp("read_scale", [], [SCALE])
        block.add_op(scale_op)
    elif case == "region":
        cond = arith.ConstantOp.from_int_and_width(1, 1)
        captured = _collapse(block.args[2])
        scale_op = scf.IfOp(
            cond.results[0], [SCALE],
            Region(Block([scf.YieldOp(captured.results[0])])),
            Region(Block([scf.YieldOp(captured.results[0])])),
        )
        block.add_ops([cond, captured, scale_op])
    elif case == "matmul_dependency":
        scale_op = _collapse(mm.results[0])
        block.add_op(scale_op)
    elif case == "cast_dependency":
        scale_op = _collapse(cast.results[0])
        block.add_op(scale_op)
    elif case == "atomic_refusal":
        pure = _collapse(block.args[2])
        effect = CallOp("read_scale", [], [SCALE])
        scale_op = arith.AddfOp(pure.results[0], effect.results[0])
        block.add_ops([pure, effect, scale_op])
    elif case == "available":
        scale_op = _collapse(block.args[2])
        block.insert_op_before(scale_op, cast)
    elif case == "argument":
        scale_op = None
    elif case == "cycle":
        # Deliberately malformed SSA: structural verify() does not reject cycles.
        scale_op = tensor.CastOp(block.args[3], SCALE)
        first = tensor.CastOp(scale_op.results[0], SCALE)
        scale_op.operands = (first.results[0],)
        block.add_ops([first, scale_op])
    elif case == "detached":
        # Structural verification also permits an operand with no defining block.
        scale_op = _collapse(block.args[2])
    else:
        raise ValueError(case)

    mul_body = Block(arg_types=[f32, f32, f32])
    product = arith.MulfOp(mul_body.args[0], mul_body.args[1])
    mul_body.add_ops([product, YieldOp(product.results[0])])
    mul = GenericOp(
        [mm.results[0], block.args[3] if scale_op is None else scale_op.results[0]],
        [mul_out.results[0]], Region(mul_body),
        [
            AffineMapAttr(AffineMap.identity(2)),
            AffineMapAttr(AffineMap(2, 0, (AffineDimExpr(1),))),
            AffineMapAttr(AffineMap.identity(2)),
        ],
        [IteratorTypeAttr.parallel()] * 2, [MATRIX],
    )
    mul.attributes["prov.op"] = StringAttr("mul")
    mul.attributes["prov.source_node_ids"] = ArrayAttr([StringAttr("prepared.scale")])
    mul.attributes["prov.origin_node_ids"] = ArrayAttr([StringAttr("original.scale")])
    block.add_op(mul)
    return mul.results[0]


def _module(case):
    block = Block(arg_types=[MATRIX, WEIGHT, MATRIX, SCALE])
    result = _append_chain(block, case)
    block.add_op(ReturnOp(result))
    fn = FuncOp("forward", ([MATRIX, WEIGHT, MATRIX, SCALE], [MATRIX]), Region(block))
    return ModuleOp([
        FuncOp.external("mutate_state", [], []),
        FuncOp.external("read_scale", [], [SCALE]),
        fn,
    ])


def _text(module):
    stream = io.StringIO()
    Printer(stream=stream).print_op(module)
    return stream.getvalue()


def _assert_ssa_order(module):
    """Check uses nested inside regions as well as ordinary operation operands."""
    for op in module.walk():
        for value in op.operands:
            producer = value.owner
            if not isinstance(producer, Operation):
                continue
            defining_block = producer.parent_block()
            assert defining_block is not None
            consumer = op
            while consumer.parent_block() is not defining_block:
                consumer = consumer.parent_op()
                assert consumer is not None, "Value is not defined in an ancestor scope."
            assert producer is not consumer and producer.is_before_in_block(consumer)


def _fused(module):
    return [op for op in module.walk() if op.name == "quant_ext.dequantize_per_channel"]


@pytest.mark.parametrize("case", ["collapse", "transitive", "shared", "available", "argument"])
def test_late_pure_scale_dependencies_fuse_in_ssa_order(case):
    original = _module(case)
    original.verify()
    _assert_ssa_order(original)
    before = _text(original)
    result = fuse_qdq(original)
    result.verify()
    _assert_ssa_order(result)
    dequantize, = _fused(result)
    assert {v.data for v in dequantize.attributes["prov.source_node_ids"]} == {
        "prepared.cast", "prepared.scale"
    }
    assert {v.data for v in dequantize.attributes["prov.origin_node_ids"]} == {
        "original.cast", "original.scale"
    }
    assert _text(original) == before
    assert not any(
        op.attributes.get("prov.op") in (StringAttr("dtype_cast"), StringAttr("mul"))
        for op in result.walk()
    )
    if case == "shared":
        # A shared transitive dependency is moved once and still feeds both views.
        views = [op for op in result.walk() if isinstance(op, tensor.CastOp)]
        assert len(views) == 1
        assert len(list(views[0].results[0].uses)) == 2


@pytest.mark.parametrize(
    "case", ["effect", "region", "matmul_dependency", "cast_dependency", "atomic_refusal"]
)
def test_unsupported_scale_dependency_keeps_complete_match_unchanged(case):
    original = _module(case)
    original.verify()
    _assert_ssa_order(original)
    before = _text(original)
    result = fuse_qdq(original)
    result.verify()
    _assert_ssa_order(result)
    assert not _fused(result)
    assert _text(result) == before
    assert _text(original) == before


@pytest.mark.parametrize("case", ["cycle", "detached"])
def test_unavailable_scale_definitions_are_not_rewritten(case):
    original = _module(case)
    before = _text(original)
    result = fuse_qdq(original)
    assert not _fused(result)
    assert _text(result) == before


def test_refused_match_does_not_prevent_independent_safe_fusion():
    """Refusal leaves that chain intact without rolling back a separate safe match."""
    block = Block(arg_types=[MATRIX, WEIGHT, MATRIX, SCALE])
    refused = _append_chain(block, "atomic_refusal")
    fused = _append_chain(block, "transitive")
    block.add_op(ReturnOp(refused, fused))
    fn = FuncOp(
        "forward", ([MATRIX, WEIGHT, MATRIX, SCALE], [MATRIX, MATRIX]), Region(block)
    )
    original = ModuleOp([FuncOp.external("read_scale", [], [SCALE]), fn])
    original.verify()
    before = _text(original)
    result = fuse_qdq(original)
    result.verify()
    _assert_ssa_order(result)
    assert len(_fused(result)) == 1
    assert len([
        op for op in result.walk() if op.attributes.get("prov.op") == StringAttr("mul")
    ]) == 1
    assert _text(original) == before


@pytest.mark.parametrize("source", ["block_argument", "producer"])
def test_dominating_values_in_ancestor_block_remain_available(source):
    outer = Block(arg_types=[MATRIX, WEIGHT, MATRIX, SCALE])
    result = _append_chain(outer, "collapse")
    inner = Block()
    for op in list(outer.ops):
        op.detach()
        inner.add_op(op)
    inner.add_op(scf.YieldOp(result))
    cond = arith.ConstantOp.from_int_and_width(1, 1)
    outer.add_op(cond)
    if source == "producer":
        available = tensor.CastOp(outer.args[2], MATRIX)
        outer.add_op(available)
        collapse = next(op for op in inner.ops if isinstance(op, tensor.CollapseShapeOp))
        collapse.operands = (available.results[0],)
    branch = scf.IfOp(
        cond.results[0], [MATRIX], Region(inner),
        Region(Block([scf.YieldOp(outer.args[0])])),
    )
    outer.add_ops([branch, ReturnOp(branch.results[0])])
    original = ModuleOp([
        FuncOp("forward", ([MATRIX, WEIGHT, MATRIX, SCALE], [MATRIX]), Region(outer))
    ])
    original.verify()
    _assert_ssa_order(original)
    before = _text(original)
    fused = fuse_qdq(original)
    fused.verify()
    _assert_ssa_order(fused)
    assert len(_fused(fused)) == 1
    assert _text(original) == before


def test_scale_in_sibling_block_is_not_hoisted_into_this_block():
    original = _module("collapse")
    fn = next(op for op in original.body.block.ops if isinstance(op, FuncOp) and op.sym_name.data == "forward")
    entry = fn.body.block
    scale = next(op for op in entry.ops if isinstance(op, tensor.CollapseShapeOp))
    sibling = Block(arg_types=[MATRIX])
    scale.detach()
    scale.operands = (sibling.args[0],)
    sibling.add_ops([scale, ReturnOp(sibling.args[0])])
    fn.body.add_block(sibling)
    # Deliberately unavailable across CFG blocks; structural verify accepts the IR.
    original.verify()
    before = _text(original)
    result = fuse_qdq(original)
    assert not _fused(result)
    assert _text(result) == before
