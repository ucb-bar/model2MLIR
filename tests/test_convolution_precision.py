"""Half convolution rounds once, after f32 reduction and bias addition."""
import pytest
import torch
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region
from m2m.ir.decompositions import decompose_convolution

@pytest.mark.parametrize('dtype,elem',[(torch.bfloat16,builtin.bf16),(torch.float16,builtin.f16)])
@pytest.mark.parametrize('bias',[False,True])
@pytest.mark.parametrize('lowering',['im2col','direct'])
def test_half_convolution_f32_bias_then_single_round(monkeypatch,dtype,elem,bias,lowering):
    monkeypatch.setenv('M2M_CONV_LOWERING',lowering)
    types=[builtin.TensorType(elem,[1,3,9,9]),builtin.TensorType(elem,[7,3,3,3])]
    if bias:types.append(builtin.TensorType(elem,[7]))
    out=builtin.TensorType(elem,[1,7,4,4]);block=Block(arg_types=types)
    meta={'val':torch.empty(1,7,4,4,dtype=dtype),'_aten_target':'aten.convolution.default','_fx_args':(None,None,None,[2,2],[0,0],[1,1],False,[0,0],1)}
    result=decompose_convolution(list(block.args),meta,'conv')
    block.add_ops(result.ops);block.add_op(func.ReturnOp(result.result))
    module=builtin.ModuleOp([func.FuncOp('forward',(types,[out]),region=Region(block))]);module.verify()
    for op in module.walk():
        if op.name in {'arith.addf','arith.mulf'}:assert op.results[0].type==builtin.f32
    assert sum(op.name=='arith.truncf' for op in module.walk())==1
    assert result.result.type==out
