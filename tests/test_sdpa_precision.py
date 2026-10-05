"""Half SDPA preserves f32 scores/probabilities until the final result cast."""
import pytest
import torch
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region
from m2m.ir.decompositions import decompose_scaled_dot_product_attention
from m2m.ir.torchmlir_decomps import _sdpa

@pytest.mark.parametrize('dtype,elem',[(torch.bfloat16,builtin.bf16),(torch.float16,builtin.f16)])
@pytest.mark.parametrize('mask_kind',['none','bool','half'])
def test_direct_sdpa_half_intermediates(dtype,elem,mask_kind):
    t=builtin.TensorType(elem,[1,2,8,16]);types=[t,t,t]
    if mask_kind!='none':types.append(builtin.TensorType(builtin.i1 if mask_kind=='bool' else elem,[8,8]))
    block=Block(arg_types=types)
    result=decompose_scaled_dot_product_attention(list(block.args),{'val':torch.empty(1,2,8,16,dtype=dtype)},'attention')
    block.add_ops(result.ops);block.add_op(func.ReturnOp(result.result))
    module=builtin.ModuleOp([func.FuncOp('forward',(types,[t]),region=Region(block))]);module.verify()
    for op in module.walk():
        if op.name in {'arith.addf','arith.mulf','arith.divf','math.exp'}:assert op.results[0].type==builtin.f32
    assert sum(op.name=='arith.truncf' for op in module.walk())==1
    assert result.result.type==t

@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16])
@pytest.mark.parametrize('mask_kind',['none','bool','half'])
def test_export_decomposition_half_sdpa_matches_wide_reference(dtype,mask_kind):
    torch.manual_seed(0);q,k,v=(torch.randn(1,2,8,16).to(dtype) for _ in range(3))
    mask=None
    if mask_kind=='bool':mask=torch.ones(8,8,dtype=torch.bool).tril()
    if mask_kind=='half':mask=torch.randn(8,8).to(dtype)
    actual=_sdpa(q,k,v,mask)
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel(SDPBackend.MATH):
        expected=torch.nn.functional.scaled_dot_product_attention(q,k,v,mask)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(actual,torch.nn.functional.scaled_dot_product_attention(q,k,v,mask),rtol=.008,atol=.004)
