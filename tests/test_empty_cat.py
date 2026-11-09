"""Torch's special rank-one empty concatenation, including dtype promotion."""
import pytest
import torch
from xdsl.dialects.builtin import TensorType, f32
from xdsl.dialects.tensor import EmptyOp

import m2m
from m2m.coverage import opaque_report
from m2m.ir.decompositions import decompose_cat


@pytest.mark.parametrize("dim", [0, 1, 2, 3, -1, -2])
@pytest.mark.parametrize("empty_first", [False, True])
def test_rank_one_empty_can_accompany_rank_four(dim, empty_first):
    class Model(torch.nn.Module):
        def forward(self, empty, value):
            return torch.cat((empty, value) if empty_first else (value, empty), dim=dim)

    inputs = (torch.empty(0), torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4))
    expected = Model()(*inputs)
    result = m2m.convert(Model(), inputs, backend="fx_importer", capture_trace=True)
    assert result.ok, result.diagnostics
    assert opaque_report(result.mlir_text) == {}
    result.module.verify()
    function = next(op for op in result.module.walk() if op.name == "func.func")
    assert tuple(function.function_type.outputs.data[0].get_shape()) == tuple(expected.shape)
    assert function.function_type.outputs.data[0].element_type == f32


@pytest.mark.parametrize("empty_dtype,value_dtype,expected_dtype", [
    (torch.float32, torch.bfloat16, torch.float32),
    (torch.bfloat16, torch.float32, torch.float32),
    (torch.int64, torch.float32, torch.float32),
])
def test_omitted_empty_operand_still_controls_result_type(empty_dtype, value_dtype, expected_dtype):
    class Model(torch.nn.Module):
        def forward(self, empty, value):
            return torch.cat((empty, value), dim=2)

    inputs = (torch.empty(0, dtype=empty_dtype), torch.ones(1, 2, 3, 4, dtype=value_dtype))
    assert Model()(*inputs).dtype == expected_dtype
    result = m2m.convert(Model(), inputs, backend="fx_importer", capture_trace=True)
    assert result.ok, result.diagnostics
    assert opaque_report(result.mlir_text) == {}
    result.module.verify()
    concat = next(op for op in result.module.walk() if op.name == "tensor.concat")
    assert all(value.type.element_type == f32 for value in concat.operands)


def test_same_rank_empty_axis_and_all_empty_vectors_remain_valid():
    class Model(torch.nn.Module):
        def forward(self, empty, value):
            return torch.cat((empty, value), dim=0)

    for inputs in [(torch.empty(0, 4), torch.ones(3, 4)), (torch.empty(0), torch.empty(0))]:
        result = m2m.convert(Model(), inputs, backend="fx_importer", capture_trace=True)
        assert result.ok, result.diagnostics
        assert opaque_report(result.mlir_text) == {}
        result.module.verify()


def test_arbitrary_wrong_rank_empty_is_refused():
    empty, value = torch.empty(0, 0), torch.ones(1, 2, 3, 4)
    with pytest.raises((RuntimeError, IndexError)):
        torch.cat((empty, value), dim=2)
    operands = [EmptyOp([], TensorType(f32, list(tensor.shape))).tensor for tensor in (empty, value)]
    result = decompose_cat(operands, {"val": torch.ones(1, 2, 3, 4), "_fx_node_args": (None, 2)}, "cat")
    assert any(op.name == "func.call" for op in result.ops)
