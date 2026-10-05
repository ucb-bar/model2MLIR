"""Matmul reductions use f32 opmath for half inputs and initialized accumulators."""

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region

from m2m.ir.decompositions import (
    _cast_tensor,
    decompose_bmm,
    decompose_matmul,
    decompose_mm,
)


def _case(form, k=4096):
    return {
        "rank2": ((2, k), (k, 3)),
        "mm": ((2, k), (k, 3)),
        "bmm": ((2, 2, k), (2, k, 3)),
        "batched": ((1, 2, 2, k), (1, 2, k, 3)),
        "broadcast": ((2, 2, k), (k, 3)),
    }[form]


def _inputs(form, dtype, pattern):
    k = {"long": 4096, "products": 2, "empty": 0}[pattern]
    shapes = _case(form, k)
    a, b = (torch.ones(s, dtype=dtype) for s in shapes)
    if pattern == "products":
        epsilon = 2 ** (-7 if dtype == torch.bfloat16 else -10)
        a[..., 0] = 1 + epsilon
        b[..., 0, :] = 1 + epsilon
        b[..., 1, :] = -(1 + 2 * epsilon)
    return a, b


def _module(form, dtype, elem, *, native=False, pattern="long"):
    inputs = _inputs(form, dtype, pattern)
    shapes = [tuple(x.shape) for x in inputs]
    expected = torch.matmul(*inputs)
    types = [builtin.TensorType(builtin.f32 if native else elem, s) for s in shapes]
    block = Block(arg_types=types)
    operands = list(block.args)
    if native and elem != builtin.f32:
        operands = []
        for value, shape in zip(block.args, shapes):
            ops, result = _cast_tensor(value, shape, elem)
            block.add_ops(ops)
            operands.append(result)
    result = ({"bmm": decompose_bmm, "mm": decompose_mm}.get(form, decompose_matmul))(
        operands, {"val": expected}, "matmul"
    )
    block.add_ops(result.ops)
    out = result.result
    if native and elem != builtin.f32:
        ops, out = _cast_tensor(out, list(expected.shape), builtin.f32)
        block.add_ops(ops)
    block.add_op(func.ReturnOp(out))
    function = func.FuncOp("forward", (types, [out.type]), Region(block))
    function.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
    module = builtin.ModuleOp([function])
    module.verify()
    return module, expected


@pytest.mark.parametrize(
    "dtype,elem",
    [
        (torch.bfloat16, builtin.bf16),
        (torch.float16, builtin.f16),
        (torch.float32, builtin.f32),
    ],
    ids=["bf16", "f16", "f32"],
)
@pytest.mark.parametrize("form", ["rank2", "mm", "bmm", "batched", "broadcast"])
def test_reduction_uses_f32_and_zero_init(form, dtype, elem):
    module, _ = _module(form, dtype, elem)
    for op in module.walk():
        if op.name in ("arith.mulf", "arith.addf"):
            assert op.results[0].type == builtin.f32
        if op.name == "linalg.matmul" or (
            op.name == "linalg.generic"
            and any(x.data.value == "reduction" for x in op.iterator_types)
        ):
            assert op.outputs[0].owner.name == "tensor.splat"
    assert sum(op.name == "arith.truncf" for op in module.walk()) == int(
        elem != builtin.f32
    )


def _descriptor(a):
    rank = a.ndim

    class Descriptor(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.c_int64 * rank),
            ("strides", ctypes.c_int64 * rank),
        ]

    return Descriptor(
        a.ctypes.data,
        a.ctypes.data,
        0,
        (ctypes.c_int64 * rank)(*a.shape),
        (ctypes.c_int64 * rank)(*(s // a.itemsize for s in a.strides)),
    )


@pytest.mark.parametrize(
    "dtype,elem",
    [
        (torch.bfloat16, builtin.bf16),
        (torch.float16, builtin.f16),
        (torch.float32, builtin.f32),
    ],
    ids=["bf16", "f16", "f32"],
)
@pytest.mark.parametrize("form", ["rank2", "mm", "bmm", "batched", "broadcast"])
@pytest.mark.parametrize("pattern", ["long", "products", "empty"])
def test_native_long_reduction_matches_torch(tmp_path, form, dtype, elem, pattern):
    translate = os.environ.get("MERLIN_MLIR_TRANSLATE") or shutil.which(
        "mlir-translate"
    )
    clang = os.environ.get("MERLIN_CLANG") or shutil.which("clang")
    if not translate or not clang:
        pytest.skip("native matmul tests require LLVM tools")
    opt = str(Path(translate).with_name("mlir-opt"))
    if not shutil.which(opt):
        pytest.skip("native matmul tests require mlir-opt")
    module, expected = _module(form, dtype, elem, native=True, pattern=pattern)
    src = tmp_path / "model.mlir"
    low = tmp_path / "llvm.mlir"
    ll = tmp_path / "model.ll"
    so = tmp_path / "model.so"
    src.write_text(str(module))
    pipeline = "builtin.module(one-shot-bufferize{bufferize-function-boundaries},buffer-results-to-out-params{modify-public-functions},convert-linalg-to-loops,expand-strided-metadata,lower-affine,convert-scf-to-cf,convert-math-to-libm,convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)"
    subprocess.run(
        [opt, str(src), "--pass-pipeline=" + pipeline, "-o", str(low)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [translate, "--mlir-to-llvmir", str(low), "-o", str(ll)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [clang, "-O2", "-fPIC", "-shared", str(ll), "-lm", "-o", str(so)],
        check=True,
        capture_output=True,
        text=True,
    )
    library = ctypes.CDLL(str(so))
    entry = library._mlir_ciface_forward
    entry.argtypes = [ctypes.c_void_p] * 3
    entry.restype = None
    arrays = [
        np.ascontiguousarray(x.float().numpy()) for x in _inputs(form, dtype, pattern)
    ] + [np.full(tuple(expected.shape), np.nan, dtype=np.float32)]
    desc = [_descriptor(a) for a in arrays]
    entry(*(ctypes.byref(d) for d in desc))
    np.testing.assert_array_equal(arrays[-1], expected.float().numpy())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("form", ["rank2", "mm", "bmm", "batched", "broadcast"])
def test_exported_half_matmul_keeps_wide_reduction(dtype, form):
    import m2m

    class Model(torch.nn.Module):
        def forward(self, a, b):
            if form == "mm":
                return torch.mm(a, b)
            if form == "bmm":
                return torch.bmm(a, b)
            return torch.matmul(a, b)

    result = m2m.convert(
        Model(),
        tuple(torch.ones(s, dtype=dtype) for s in _case(form, 8)),
        backend="fx_importer",
        level="linalg-on-tensors",
    )
    assert result.ok, result.diagnostics
    contractions = [
        op
        for op in result.module.walk()
        if op.name == "linalg.matmul"
        or (
            op.name == "linalg.generic"
            and any(x.data.value == "reduction" for x in op.iterator_types)
        )
    ]
    assert contractions
    assert all(op.results[0].type.element_type == builtin.f32 for op in contractions)
