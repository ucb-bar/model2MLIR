"""Python scalar coefficients must retain f32 opmath precision for half tensors."""

import math

import pytest
import torch
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region

from m2m.ir.decompositions import _binary_elementwise, _arith_build2

@pytest.mark.parametrize(
    "dtype,elem", [(torch.bfloat16, builtin.bf16), (torch.float16, builtin.f16)]
)
@pytest.mark.parametrize("name", ["AddfOp", "SubfOp", "MulfOp", "DivfOp"])
@pytest.mark.parametrize("left", [False, True])
def test_half_scalar_uses_wide_coefficient(dtype, elem, name, left):
    t = builtin.TensorType(elem, [128])
    block = Block(arg_types=[t])
    scalar = math.sqrt(960)
    args = (scalar, None) if left else (None, scalar)
    result = _binary_elementwise(
        list(block.args),
        {"val": torch.empty(128, dtype=dtype), "_fx_args": args},
        {"AddfOp":"add", "SubfOp":"sub", "MulfOp":"mul", "DivfOp":"div"}[name],
        _arith_build2(name, "AddiOp"),
    )
    assert result is not None
    block.add_ops(result.ops)
    block.add_op(func.ReturnOp(result.result))
    module = builtin.ModuleOp([func.FuncOp("forward", ([t], [t]), region=Region(block))])
    module.verify()
    assert result.result.type == t
    wide = name in {"MulfOp", "DivfOp"}
    expected_elem = builtin.f32 if wide else elem
    assert sum(op.name == "arith.truncf" for op in module.walk()) == int(wide)
    for op in module.walk():
        if op.name in {"arith.addf", "arith.subf", "arith.mulf", "arith.divf"}:
            assert op.results[0].type == expected_elem
    constants = [op for op in module.walk() if op.name == "arith.constant"]
    assert all(op.results[0].type == expected_elem for op in constants)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_framework_scalar_differs_from_early_half_round(dtype):
    torch.manual_seed(0)
    x = torch.randn(1024).to(dtype)
    scalar = math.sqrt(960)
    expected = x*scalar
    torch.testing.assert_close(expected, (x.float()*scalar).to(dtype), rtol=0, atol=0)
    assert not torch.equal(expected, x*torch.tensor(scalar, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("operation", [torch.add, torch.sub])
def test_framework_add_sub_round_scalar_first(dtype, operation):
    torch.manual_seed(0)
    x = torch.randn(10000).to(dtype)
    scalar = math.sqrt(960)
    expected = operation(x, scalar)
    torch.testing.assert_close(
        expected, operation(x, torch.tensor(scalar, dtype=dtype)), rtol=0, atol=0
    )
    assert not torch.equal(expected, operation(x.float(), scalar).to(dtype))
