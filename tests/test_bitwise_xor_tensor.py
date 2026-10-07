"""Tensor XOR must lower as an integer bit operation, including broadcasts."""

import operator
from types import SimpleNamespace

import m2m
import pytest
import torch
from m2m.coverage import opaque_report
from m2m.ir.decompositions import decompose_bitwise_xor
from xdsl.dialects.builtin import TensorType, f32
from xdsl.ir import Block


class _Xor(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, left, right):
        return self.fn(left, right)


def _wrap(value, width):
    value &= (1 << width) - 1
    if width == 1:
        return bool(value)
    return value - (1 << width) if value >> (width - 1) else value


def _eval_body(block, left, right):
    values = dict(zip(block.args, (int(left), int(right), 0)))
    for op in block.ops:
        if op.name == "linalg.yield":
            return values[op.operands[0]]
        args = [values[value] for value in op.operands]
        width = int(str(op.results[0].type)[1:])
        if op.name == "arith.xori":
            value = args[0] ^ args[1]
        elif op.name == "arith.extsi":
            value = args[0]
        elif op.name == "arith.extui":
            value = args[0] & ((1 << int(str(op.operands[0].type)[1:])) - 1)
        elif op.name == "arith.trunci":
            value = args[0]
        else:
            raise AssertionError(f"unexpected XOR body operation: {op.name}")
        values[op.results[0]] = _wrap(value, width)
    raise AssertionError("missing XOR yield")


def _inputs(case):
    if case == "bool_tail":
        return (
            torch.tensor([False, True, False, True, True, False, True, False, True]),
            torch.tensor([True, True, False, False, True, True, False, False, False]),
        )
    if case == "int64_edges":
        values = [-(2**63), -(2**62), -65537, -129, -2, -1, 0, 1, 2, 3, 7, 127, 128, 65536, 2**31, 2**62, 2**63 - 1]
        return (
            torch.tensor(values, dtype=torch.int64),
            torch.tensor(list(reversed(values)), dtype=torch.int64),
        )
    if case == "int8_row_broadcast":
        return (
            torch.tensor(
                [[-128, -7, -1, 0, 1, 7, 127], [7, -1, 3, -4, 9, -10, 11], [0, 1, 2, 3, 4, 5, 6]], dtype=torch.int8
            ),
            torch.tensor([[-1, 3, -8, 15, -16, 64, -127]], dtype=torch.int8),
        )
    if case == "int8_column_broadcast":
        return (
            torch.tensor([[-128], [-1], [127]], dtype=torch.int8),
            torch.tensor(
                [[1, 2, 3, 4, 5, 6, 7], [-7, -6, -5, -4, -3, -2, -1], [127, 64, 32, 16, 8, 4, 2]], dtype=torch.int8
            ),
        )
    if case == "signed_promotion":
        return (
            torch.tensor([-128, -31, -1, 0, 1, 3, 7, 64, 127], dtype=torch.int8),
            torch.tensor([-(2**63), -65536, -1, 0, 1, 255, 2**31, 2**62, 2**63 - 1], dtype=torch.int64),
        )
    if case == "bool_signed_promotion":
        return (
            torch.tensor([True, False, True, True, False, False, True, False, True]),
            torch.tensor([-128, -31, -1, 0, 1, 3, 7, 64, 127], dtype=torch.int8),
        )
    if case == "uint8_same_width":
        return (
            torch.tensor([0, 1, 2, 3, 7, 15, 128, 254, 255], dtype=torch.uint8),
            torch.tensor([255, 254, 128, 127, 15, 7, 3, 2, 1], dtype=torch.uint8),
        )
    raise AssertionError(case)


@pytest.mark.parametrize(
    "fn,target", [(torch.bitwise_xor, "aten.bitwise_xor.Tensor"), (operator.xor, "aten.__xor__.Tensor")]
)
@pytest.mark.parametrize(
    "case",
    [
        "bool_tail",
        "int64_edges",
        "int8_row_broadcast",
        "int8_column_broadcast",
        "signed_promotion",
        "bool_signed_promotion",
        "uint8_same_width",
    ],
)
def test_tensor_xor_has_exact_typed_broadcast_body(fn, target, case):
    inputs = _inputs(case)
    model = _Xor(fn).eval()
    exported = torch.export.export(model, inputs)
    assert target in [str(node.target) for node in exported.graph.nodes if node.op == "call_function"]

    result = m2m.convert(model, inputs, backend="fx_importer")
    assert result.ok and opaque_report(result.mlir_text) == {}, result.mlir_text
    result.module.verify()
    generics = [
        op
        for op in result.module.walk()
        if op.name == "linalg.generic" and getattr(op.attributes.get("prov.op"), "data", None) == "bitwise_xor"
    ]
    assert len(generics) == 1
    generic = generics[0]
    assert [op.name for op in generic.body.block.ops].count("arith.xori") == 1
    assert len(generic.inputs) == 2

    expected = fn(*inputs)
    left, right = torch.broadcast_tensors(*inputs)
    observed = [
        _eval_body(generic.body.block, a, b) for a, b in zip(left.reshape(-1).tolist(), right.reshape(-1).tolist())
    ]
    if expected.dtype == torch.uint8:
        observed = [value & 255 for value in observed]
    assert observed == expected.reshape(-1).tolist()
    assert tuple(generic.results[0].type.get_shape()) == tuple(expected.shape)
    if case == "int8_row_broadcast":
        assert "(0, d1)" in str(generic.indexing_maps.data[1])
    if case == "int8_column_broadcast":
        assert "(d0, 0)" in str(generic.indexing_maps.data[0])


def test_float_xor_is_invalid_at_source():
    values = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32)
    with pytest.raises((RuntimeError, TypeError)):
        torch.bitwise_xor(values, values)

    # A malformed direct-import request must also remain opaque, not emit xori.
    operands = list(Block(arg_types=[TensorType(f32, [3]), TensorType(f32, [3])]).args)
    args = (SimpleNamespace(meta={"val": values}), SimpleNamespace(meta={"val": values}))
    result = decompose_bitwise_xor(
        operands, {"val": values, "_fx_args": args, "_aten_target": "aten.bitwise_xor.Tensor"}, "bitwise_xor"
    )
    assert [op.name for op in result.ops] == ["func.call"]


@pytest.mark.parametrize("signed_dtype", [torch.int16, torch.int64])
def test_unsigned_to_signed_promotion_remains_opaque(signed_dtype):
    # The shared promotion helper sign-extends non-i1 integers. Zero-extension
    # from uint8 needs separate typed source evidence before it can be lowered.
    unsigned = torch.tensor([0, 1, 127, 128, 255], dtype=torch.uint8)
    signed = torch.tensor([-1, 0, 1, 2, 3], dtype=signed_dtype)
    result = m2m.convert(_Xor(torch.bitwise_xor).eval(), (unsigned, signed), backend="fx_importer")
    assert result.ok and opaque_report(result.mlir_text) == {"aten_bitwise_xor_Tensor": 1}
