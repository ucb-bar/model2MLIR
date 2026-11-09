"""Typed standard-IR lowering of scalar-boolean CPU source assertions.

No guard is inferred from values observed at capture time. The checks execute
on every invocation and preserve failure before downstream source arithmetic.
"""

from __future__ import annotations


def _operand(operands, meta):
    import torch
    from xdsl.dialects.builtin import TensorType, i1

    args, kwargs = meta.get("_fx_args"), meta.get("_fx_kwargs")
    if (
        meta.get("_aten_target") != "aten._assert_async.msg"
        or type(args) is not tuple
        or not args
        or not isinstance(args[0], torch.fx.Node)
        or type(kwargs) is not dict
        or kwargs
        or len(operands) != 1
        or type(meta.get("_fx_ssa_args")) is not dict
        or set(meta["_fx_ssa_args"]) != {0}
        or meta["_fx_ssa_args"][0] is not operands[0]
    ):
        raise ValueError(
            "finite/assert lowering requires exact ordered source operands"
        )
    value = args[0].meta.get("val")
    typ = operands[0].type
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype is not torch.bool
        or value.device.type != "cpu"
        or not isinstance(typ, TensorType)
        or typ.element_type != i1
        or tuple(value.shape) != typ.get_shape()
        or any(type(d) is not int or d < 0 for d in value.shape)
    ):
        raise ValueError("finite/assert source dtype/shape contradicts ranked SSA")
    return operands[0], args


def assert_async(operands, meta, name):
    import torch
    from xdsl.dialects.builtin import i1
    from xdsl.dialects.cf import AssertOp
    from xdsl.dialects.tensor import ExtractOp

    from m2m.ir.decompositions import DecompResult

    value, args = _operand(operands, meta)
    if (
        len(args) != 2
        or type(args[1]) is not str
        or value.type.get_shape() != ()
        or meta.get("val") is not None
    ):
        raise ValueError(
            "assert_async.msg requires one scalar bool, ordinary message and no results"
        )
    schema = torch.ops.aten._assert_async.msg._schema
    if (
        schema.name != "aten::_assert_async"
        or schema.overload_name != "msg"
        or schema.returns
        or len(schema.arguments) != 2
        or [
            (a.name, str(a.type), a.kwarg_only, a.has_default_value())
            for a in schema.arguments
        ]
        != [("self", "Tensor", False, False), ("assert_msg", "str", False, False)]
        or any(a.alias_info is not None for a in schema.arguments)
    ):
        raise ValueError("unproved assertion source schema/effects")
    scalar = ExtractOp(value, [], i1)
    check = AssertOp(scalar.result, args[1])
    return DecompResult([scalar, check], pattern_hint="assert")
