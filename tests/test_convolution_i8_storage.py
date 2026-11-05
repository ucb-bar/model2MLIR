"""Exact signed-extension storage recovery keeps complete convolution semantics."""
import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from xdsl.dialects import arith, builtin, linalg
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineExpr, AffineMap

import m2m
from m2m.ir.decompositions import _identity_signed_i8_storage
from test_activation_boundary_semantics import descriptor


def cast_generic(*, bits=8, unsigned=False, permuted=False, reduction=False, extra=False):
    source_type = builtin.TensorType(builtin.IntegerType(bits), [2, 2])
    output_type = builtin.TensorType(builtin.i32, [2, 2])
    outer = Block(arg_types=[source_type, output_type])
    block = Block(arg_types=[source_type.element_type, builtin.i32])
    extend = (arith.ExtUIOp if unsigned else arith.ExtSIOp)(block.args[0], builtin.i32)
    block.add_op(extend)
    if extra:
        block.add_op(arith.AddiOp(extend.result, extend.result))
    block.add_op(linalg.YieldOp(extend.result))
    identity = AffineMap.identity(2)
    first = AffineMap(2, 0, (AffineExpr.dimension(1), AffineExpr.dimension(0))) if permuted else identity
    generic = linalg.GenericOp(inputs=[outer.args[0]], outputs=[outer.args[1]],
        body=Region(block), indexing_maps=[builtin.AffineMapAttr(first), builtin.AffineMapAttr(identity)],
        iterator_types=[linalg.IteratorTypeAttr(linalg.IteratorType.REDUCTION if reduction else linalg.IteratorType.PARALLEL)] * 2,
        result_types=[output_type])
    return generic, outer.args[0]


def test_only_exact_identity_signed_extension_recovers_storage():
    generic, source = cast_generic()
    assert _identity_signed_i8_storage(generic.results[0]) is source
    for option in ({'bits':16}, {'unsigned':True}, {'permuted':True}, {'reduction':True}, {'extra':True}):
        generic, _ = cast_generic(**option)
        assert _identity_signed_i8_storage(generic.results[0]) is None
    assert _identity_signed_i8_storage(Block(arg_types=[builtin.TensorType(builtin.i32,[2,2])]).args[0]) is None


class Conv(torch.nn.Module):
    def __init__(self, groups, readout):
        super().__init__()
        self.groups, self.readout = groups, readout

    def forward(self, x, w, bias):
        value = torch.nn.functional.conv2d(x.to(torch.int32), w.to(torch.int32), bias,
                                          stride=(2,1), padding=(1,0), dilation=(1,1), groups=self.groups)
        if self.readout == 'raw_i32':
            return value
        return torch.minimum(torch.maximum(value, torch.full_like(value,-128)),
                             torch.full_like(value,127)).to(torch.int8)


def native_model(tmp_path, model, arrays, optimization):
    translate = os.environ.get('MODEL2MLIR_MLIR_TRANSLATE') or shutil.which('mlir-translate')
    clang = os.environ.get('MODEL2MLIR_CLANG') or shutil.which('clang')
    if not translate or not clang:
        pytest.skip('native convolution controls require MLIR/LLVM tools')
    opt = Path(translate).with_name('mlir-opt')
    examples = tuple(torch.zeros(a.shape,dtype=torch.from_numpy(a).dtype) for a in arrays)
    converted = m2m.convert(model,examples,backend='fx_importer')
    assert converted.ok, converted.diagnostics
    converted.module.verify()
    dots = [op for op in converted.module.walk() if op.name=='linalg.generic'
            and any(it.data==linalg.IteratorType.REDUCTION for it in op.iterator_types)]
    assert dots and all([arg.type for arg in op.body.block.args]==[builtin.i8,builtin.i8,builtin.i32] for op in dots)
    converted.module.body.block.first_op.attributes['llvm.emit_c_interface']=builtin.UnitAttr()
    source, lowered, llvm, library = (tmp_path / n for n in ('source.mlir','lowered.mlir','model.ll','model.so'))
    source.write_text(str(converted.module))
    pipeline=('builtin.module(one-shot-bufferize{bufferize-function-boundaries},'
              'buffer-results-to-out-params{modify-public-functions},convert-linalg-to-loops,'
              'expand-strided-metadata,lower-affine,convert-scf-to-cf,convert-math-to-libm,'
              'convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,'
              'convert-cf-to-llvm,reconcile-unrealized-casts)')
    subprocess.run([str(opt),str(source),'--pass-pipeline='+pipeline,'-o',str(lowered)],check=True,capture_output=True)
    subprocess.run([translate,'--mlir-to-llvmir',str(lowered),'-o',str(llvm)],check=True,capture_output=True)
    runtime=Path(translate).parent.parent/'lib'
    subprocess.run([clang,optimization,'-fPIC','-shared',str(llvm),'-lm','-L'+str(runtime),
                    '-lmlir_c_runner_utils','-Wl,-rpath,'+str(runtime),'-o',str(library)],check=True,capture_output=True)
    output=np.empty((2,6,4,7),dtype=np.int32 if model.readout=='raw_i32' else np.int8)
    entry=ctypes.CDLL(str(library))._mlir_ciface_forward
    entry.argtypes,entry.restype=[ctypes.c_void_p]*(len(arrays)+1),None
    entry(*(ctypes.byref(descriptor(a)) for a in (*arrays,output)))
    return output


@pytest.mark.parametrize('optimization',['-O0','-O2'])
@pytest.mark.parametrize('groups',[1,2])
@pytest.mark.parametrize('readout',['raw_i32','native_i8_scale1'])
def test_complete_native_i8_storage_convolution(tmp_path,monkeypatch,optimization,groups,readout):
    monkeypatch.setenv('M2M_CONV_LOWERING','im2col')
    palette=np.array([-128,-127,-1,0,1,126,127],dtype=np.int8)
    x=np.resize(palette,(2,4,7,8)).copy()
    w=np.resize(palette[::-1],(6,4//groups,3,2)).copy()
    bias=np.array([-65537,-129,-128,127,128,65539],dtype=np.int32)
    model=Conv(groups,readout)
    actual=native_model(tmp_path,model,(x,w,bias),optimization)
    # Independent int64 convolution, followed by the source's single final clamp.
    expected=np.broadcast_to(bias.reshape(1,6,1,1),(2,6,4,7)).astype(np.int64).copy()
    for n in range(2):
        for co in range(6):
            group=co//(6//groups)
            for oh in range(4):
                for ow in range(7):
                    for ci in range(4//groups):
                        for ky in range(3):
                            for kx in range(2):
                                ih,iw=oh*2-1+ky,ow+kx
                                if 0<=ih<7:
                                    expected[n,co,oh,ow]+=int(x[n,group*(4//groups)+ci,ih,iw])*int(w[co,ci,ky,kx])
    expected=np.clip(expected,-128,127).astype(np.int8) if readout!='raw_i32' else expected.astype(np.int32)
    np.testing.assert_array_equal(actual,expected)
