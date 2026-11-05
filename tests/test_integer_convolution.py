"""Integral convolution must keep integer scalar bodies, bias and output dtype."""

import pytest
import torch
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region

import m2m
from m2m.coverage import opaque_report
from m2m.ir.decompositions import decompose_convolution


@pytest.mark.parametrize('lowering', ['im2col', 'direct', 'grouped_direct'])
@pytest.mark.parametrize('groups', [1, 2])
@pytest.mark.parametrize('bias', [False, True])
def test_integer_convolution_uses_integer_reduction(monkeypatch, lowering, groups, bias):
    monkeypatch.setenv('M2M_CONV_LOWERING', lowering)
    types = [builtin.TensorType(builtin.i32, [2, 4, 7, 8]),
             builtin.TensorType(builtin.i32, [6, 4 // groups, 3, 2])]
    if bias:
        types.append(builtin.TensorType(builtin.i32, [6]))
    block = Block(arg_types=types)
    out = builtin.TensorType(builtin.i32, [2, 6, 4, 5])
    result = decompose_convolution(list(block.args), {
        'val': torch.empty((2, 6, 4, 5), dtype=torch.int32),
        '_aten_target': 'aten.convolution.default',
        '_fx_args': (None, None, None, [2, 2], [1, 1], [1, 1], False, [0, 0], groups),
    }, 'integer_conv')
    block.add_ops(result.ops)
    block.add_op(func.ReturnOp(result.result))
    module = builtin.ModuleOp([func.FuncOp('forward', (types, [out]), region=Region(block))])
    module.verify()
    names = [op.name for op in module.walk()]
    assert 'func.call' not in names
    assert 'arith.muli' in names and 'arith.addi' in names
    assert not {'arith.mulf', 'arith.addf', 'arith.sitofp', 'arith.fptosi'} & set(names)
    assert result.result.type == out


@pytest.mark.parametrize('readout', ['native_i8_scale1', 'raw_i32'])
def test_public_nhwc_hwio_integer_convolution_capture_has_no_opaque(readout):
    class PublicConv(torch.nn.Module):
        def forward(self, x, w, bias):
            acc = torch.nn.functional.conv2d(
                x.permute(0, 3, 1, 2).contiguous().to(torch.int32),
                w.permute(3, 2, 0, 1).contiguous().to(torch.int32), bias)
            acc = acc.permute(0, 2, 3, 1).contiguous()
            if readout == 'raw_i32':
                return acc
            lower = torch.maximum(acc, torch.full_like(acc, -128))
            return torch.minimum(lower, torch.full_like(lower, 127)).to(torch.int8)

    inputs = (torch.randint(-3, 4, (2, 5, 6, 4), dtype=torch.int8),
              torch.randint(-3, 4, (2, 3, 4, 6), dtype=torch.int8),
              torch.arange(-3, 3, dtype=torch.int32))
    converted = m2m.convert(PublicConv().eval(), inputs, backend='fx_importer')
    assert converted.ok, converted.diagnostics
    assert not opaque_report(converted.mlir_text)
    converted.module.verify()
    assert 'arith.muli' in converted.mlir_text
    assert 'arith.addi' in converted.mlir_text
    assert 'arith.mulf' not in converted.mlir_text
