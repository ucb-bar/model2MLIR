"""Half softmax sums and exponentials use framework f32 opmath precision."""

import pytest
import torch
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region

from m2m.ir.decompositions import build_softmax_body


@pytest.mark.parametrize("elem", [builtin.bf16, builtin.f16])
@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("shape,dim", [([2, 1024], -1), ([2, 1024, 3], 1)])
def test_half_softmax_uses_f32_until_final_output(elem, named, shape, dim):
    t = builtin.TensorType(elem, shape)
    block = Block(arg_types=[t])
    if named:
        from m2m.ir.linalg_ext.ops import SoftmaxOp

        op = SoftmaxOp(block.args[0], dim % len(shape), t)
        ops, result = [op], op.result
    else:
        ops, result = build_softmax_body(block.args[0], dim=dim)
    block.add_ops(ops)
    block.add_op(func.ReturnOp(result))
    module = builtin.ModuleOp(
        [func.FuncOp("forward", ([t], [t]), region=Region(block))]
    )
    if named:
        from m2m.transforms import expand_to_linalg

        expand_to_linalg(module)
    module.verify()
    reductions = [op for op in module.walk() if op.name == "linalg.reduce"]
    assert len(reductions) == 2
    assert all(op.results[0].type.element_type == builtin.f32 for op in reductions)
    for op in module.walk():
        if op.name in {"arith.maximumf", "arith.addf", "arith.subf", "arith.divf", "math.exp"}:
            assert op.results[0].type == builtin.f32
    assert sum(op.name == "arith.truncf" for op in module.walk()) == 1
    assert result.type == t


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_wide_softmax_matches_framework_half_output(dtype):
    torch.manual_seed(0)
    x = torch.cat([torch.zeros(1, 4096), torch.randn(1, 4096) * 2]).to(dtype)
    wide = x.float()
    ex = (wide - wide.max(-1, keepdim=True).values).exp()
    computed = (ex / ex.sum(-1, keepdim=True)).to(dtype)
    torch.testing.assert_close(computed, torch.softmax(x, -1), atol=0, rtol=0)
    assert float(computed[0, 0]) == 1 / 4096
