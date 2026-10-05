"""Half SiLU uses f32 opmath throughout the activation before one narrowing."""

import ctypes
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region

from m2m.ir.decompositions import _cast_tensor, decompose_silu

CASES = [
    (torch.bfloat16, builtin.bf16),
    (torch.float16, builtin.f16),
    (torch.float32, builtin.f32),
]


def _input(dtype):
    generator = torch.Generator().manual_seed(192)
    x = torch.randn((2, 4096), generator=generator) * 7
    x[0, :13] = torch.tensor(
        [
            -100.0,
            -90.0,
            -30.0,
            -10.0,
            -1.0,
            -0.0,
            0.0,
            1.0,
            10.0,
            30.0,
            90.0,
            100.0,
            1e-7,
        ]
    )
    # Reciprocal multiplication hits the FP16 rounding midpoint here; direct
    # framework division rounds to the adjacent value instead.
    x[0, 13] = -2.724609375
    return x.to(dtype)


def _module(dtype, elem, native=False):
    x = _input(dtype)
    typ = builtin.TensorType(builtin.f32 if native else elem, x.shape)
    block = Block(arg_types=[typ])
    value = block.args[0]
    if native and elem != builtin.f32:
        ops, value = _cast_tensor(value, list(x.shape), elem)
        block.add_ops(ops)
    result = decompose_silu([value], {"val": torch.empty_like(x)}, "silu")
    block.add_ops(result.ops)
    value = result.result
    if native and elem != builtin.f32:
        ops, value = _cast_tensor(value, list(x.shape), builtin.f32)
        block.add_ops(ops)
    block.add_op(func.ReturnOp(value))
    fn = func.FuncOp("forward", ([typ], [value.type]), Region(block))
    fn.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
    module = builtin.ModuleOp([fn])
    module.verify()
    return module, x


@pytest.mark.parametrize("dtype,elem", CASES)
def test_silu_preserves_output_dtype_and_wide_intermediates(dtype, elem):
    module, _ = _module(dtype, elem)
    for op in module.walk():
        if op.name in {
            "arith.addf",
            "arith.mulf",
            "arith.divf",
            "math.exp",
            "arith.negf",
        }:
            assert op.results[0].type == builtin.f32
    assert sum(op.name == "arith.truncf" for op in module.walk()) == int(
        elem != builtin.f32
    )
    assert module.body.block.first_op.function_type.outputs.data[0].element_type == elem


@pytest.mark.parametrize("dtype,elem", CASES)
def test_exported_silu_retains_opmath(dtype, elem):
    import m2m

    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.silu(x)

    result = m2m.convert(
        Model(), (_input(dtype),), backend="fx_importer", level="linalg-on-tensors"
    )
    assert result.ok, result.diagnostics
    exp = [op for op in result.module.walk() if op.name == "math.exp"]
    assert len(exp) == 1 and exp[0].results[0].type == builtin.f32
    assert sum(op.name == "arith.truncf" for op in result.module.walk()) == int(
        elem != builtin.f32
    )


def _descriptor(a):
    class Descriptor(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.c_int64 * 2),
            ("strides", ctypes.c_int64 * 2),
        ]

    return Descriptor(
        a.ctypes.data,
        a.ctypes.data,
        0,
        (ctypes.c_int64 * 2)(*a.shape),
        (ctypes.c_int64 * 2)(*(s // a.itemsize for s in a.strides)),
    )


@pytest.mark.parametrize("dtype,elem", CASES)
@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
def test_native_silu_matches_framework(tmp_path, dtype, elem, optimization):
    translate = os.environ.get("MERLIN_MLIR_TRANSLATE") or shutil.which(
        "mlir-translate"
    )
    clang = os.environ.get("MERLIN_CLANG") or shutil.which("clang")
    if not translate or not clang:
        pytest.skip("native SiLU tests require LLVM tools")
    opt = str(Path(translate).with_name("mlir-opt"))
    if not shutil.which(opt):
        pytest.skip("native SiLU tests require mlir-opt")
    module, x = _module(dtype, elem, native=True)
    source, lowered, llvm = [
        tmp_path / n for n in ("model.mlir", "llvm.mlir", "model.ll")
    ]
    library = tmp_path / f"model_{dtype}_{optimization}.so"
    source.write_text(str(module))
    pipeline = "builtin.module(one-shot-bufferize{bufferize-function-boundaries},buffer-results-to-out-params{modify-public-functions},convert-linalg-to-loops,expand-strided-metadata,lower-affine,convert-scf-to-cf,convert-math-to-libm,convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)"
    subprocess.run(
        [opt, str(source), "--pass-pipeline=" + pipeline, "-o", str(lowered)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [translate, "--mlir-to-llvmir", str(lowered), "-o", str(llvm)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [clang, optimization, "-fPIC", "-shared", str(llvm), "-lm", "-o", str(library)],
        check=True,
        capture_output=True,
    )
    entry = ctypes.CDLL(str(library))._mlir_ciface_forward
    entry.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    entry.restype = None
    arrays = [x.float().numpy(), np.full(x.shape, np.nan, dtype=np.float32)]
    descriptors = [_descriptor(a) for a in arrays]
    entry(*(ctypes.byref(d) for d in descriptors))
    expected = torch.nn.functional.silu(x).float().numpy()
    if dtype == torch.float32:
        np.testing.assert_allclose(arrays[-1], expected, atol=1e-7, rtol=4e-7)
    else:
        np.testing.assert_array_equal(arrays[-1], expected)
