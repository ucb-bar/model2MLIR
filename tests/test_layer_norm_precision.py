"""Half LayerNorm must retain f32 intermediates, as the framework reference does."""

import pytest
import torch
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region

from m2m.ir.decompositions import build_layer_norm_body


@pytest.mark.parametrize("elem", [builtin.bf16, builtin.f16])
@pytest.mark.parametrize("affine", [None, "input", "f32"])
@pytest.mark.parametrize("shape,k", [([2, 768], 1), ([2, 3, 256], 2)])
def test_half_layer_norm_only_narrows_final_result(elem, affine, shape, k):
    t = builtin.TensorType(elem, shape)
    affine_elem = elem if affine == "input" else builtin.f32 if affine else None
    w = builtin.TensorType(affine_elem, shape[-k:]) if affine_elem else None
    block = Block(arg_types=[t, w, w] if w else [t])
    built = build_layer_norm_body(
        block.args[0],
        block.args[1] if w else None,
        block.args[2] if w else None,
        eps=1e-5,
        k=k,
    )
    ops, result = built
    block.add_ops(ops)
    block.add_op(func.ReturnOp(result))
    fn = func.FuncOp(
        "forward", ([a.type for a in block.args], [t]), region=Region(block)
    )
    module = builtin.ModuleOp([fn])
    module.verify()
    reductions = [op for op in module.walk() if op.name == "linalg.reduce"]
    assert len(reductions) == 2
    assert all(op.results[0].type.element_type == builtin.f32 for op in reductions)
    for op in module.walk():
        if op.name in {"arith.addf", "arith.subf", "arith.mulf", "arith.divf", "math.rsqrt"}:
            assert op.results[0].type == builtin.f32
    assert sum(op.name == "arith.truncf" for op in module.walk()) == 1
    assert result.type == t


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_half_layer_norm_framework_uses_wide_intermediates(dtype):
    torch.manual_seed(0)
    x = (torch.randn(2, 768) * 3 + 20).to(dtype)
    weight = torch.randn(768).to(dtype)
    bias = torch.randn(768).to(dtype)
    reference = torch.nn.functional.layer_norm(x, (768,), weight, bias, 1e-5)
    wide = x.float()
    mean = wide.mean(-1, keepdim=True)
    centered = wide - mean
    variance = (centered * centered).mean(-1, keepdim=True)
    computed = (
        centered * torch.rsqrt(variance + 1e-5) * weight.float() + bias.float()
    ).to(dtype)
    torch.testing.assert_close(computed, reference, atol=0, rtol=0)
