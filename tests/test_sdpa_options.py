"""SDPA options must survive both owned decomposition paths."""

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region

from m2m.ir.decompositions import decompose_scaled_dot_product_attention
from m2m.ir.torchmlir_decomps import _sdpa


def _direct(scale, causal, *, options=None):
    shapes = [(1, 2, 3, 4), (1, 2, 5, 4), (1, 2, 5, 6)]
    types = [builtin.TensorType(builtin.f32, s) for s in shapes]
    output = builtin.TensorType(builtin.f32, (1, 2, 3, 6))
    block = Block(arg_types=types)
    metadata = {
        "val": torch.empty(1, 2, 3, 6),
        "_fx_args": (None, None, None, None, 0.0, causal),
        "_fx_kwargs": {"scale": scale, **(options or {})},
    }
    result = decompose_scaled_dot_product_attention(
        list(block.args), metadata, "attention"
    )
    block.add_ops(result.ops)
    block.add_op(func.ReturnOp(result.result))
    function = func.FuncOp("forward", (types, [output]), Region(block))
    function.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
    module = builtin.ModuleOp([function])
    module.verify()
    return module


@pytest.mark.parametrize("option", [{"dropout_p": 0.25}, {"enable_gqa": True}])
def test_unsupported_options_refused_by_both_paths(option):
    q = torch.randn(1, 2, 3, 4)
    with pytest.raises(NotImplementedError):
        _sdpa(q, q, q, **option)
    with pytest.raises(NotImplementedError):
        _direct(None, False, options=option)


@pytest.mark.parametrize("scale", [None, 0.0, 0.125, -0.25])
@pytest.mark.parametrize("causal", [False, True])
def test_export_decomposition_matches_torch_math(scale, causal):
    torch.manual_seed(9)
    q, k, v = (torch.randn(*s) for s in [(1, 2, 3, 4), (1, 2, 5, 4), (1, 2, 5, 6)])
    with sdpa_kernel(SDPBackend.MATH):
        expected = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=causal
        )
    actual = _sdpa(q, k, v, scale=scale, is_causal=causal)
    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-6)


@pytest.fixture(scope="module")
def native_tools():
    translate = os.environ.get("MERLIN_MLIR_TRANSLATE") or shutil.which(
        "mlir-translate"
    )
    clang = os.environ.get("MERLIN_CLANG") or shutil.which("clang")
    if not translate or not clang:
        pytest.skip("native SDPA tests require mlir-translate and clang")
    opt = str(Path(translate).with_name("mlir-opt"))
    if not shutil.which(opt):
        pytest.skip("native SDPA tests require mlir-opt beside mlir-translate")
    return opt, translate, clang


def _descriptor(array):
    class Descriptor(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.c_int64 * 4),
            ("strides", ctypes.c_int64 * 4),
        ]

    return Descriptor(
        array.ctypes.data,
        array.ctypes.data,
        0,
        (ctypes.c_int64 * 4)(*array.shape),
        (ctypes.c_int64 * 4)(*(s // array.itemsize for s in array.strides)),
    )


@pytest.mark.parametrize(
    "scale,causal", [(None, False), (0.0, False), (0.125, True), (-0.25, True)]
)
def test_direct_native_matches_torch_math(tmp_path, native_tools, scale, causal):
    opt, translate, clang = native_tools
    source, llvm_mlir, llvm, shared = (
        tmp_path / p for p in ["input.mlir", "llvm.mlir", "input.ll", "native.so"]
    )
    source.write_text(str(_direct(scale, causal)))
    pipeline = (
        "builtin.module(one-shot-bufferize{bufferize-function-boundaries},"
        "buffer-results-to-out-params{modify-public-functions},convert-linalg-to-loops,"
        "expand-strided-metadata,lower-affine,convert-scf-to-cf,convert-math-to-libm,convert-arith-to-llvm,"
        "finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)"
    )
    subprocess.run(
        [opt, str(source), "--pass-pipeline=" + pipeline, "-o", str(llvm_mlir)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [translate, "--mlir-to-llvmir", str(llvm_mlir), "-o", str(llvm)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [clang, "-O2", "-fPIC", "-shared", str(llvm), "-lm", "-o", str(shared)],
        check=True,
        capture_output=True,
        text=True,
    )
    library = ctypes.CDLL(str(shared))
    entry = library._mlir_ciface_forward
    entry.argtypes = [ctypes.c_void_p] * 4
    entry.restype = None
    torch.manual_seed(9)
    q, k, v = (torch.randn(*s) for s in [(1, 2, 3, 4), (1, 2, 5, 4), (1, 2, 5, 6)])
    with sdpa_kernel(SDPBackend.MATH):
        expected = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=causal
        )
    output = np.zeros((1, 2, 3, 6), dtype=np.float32)
    arrays = [q.numpy(), k.numpy(), v.numpy(), output]
    descriptors = [_descriptor(a) for a in arrays]
    entry(*(ctypes.byref(d) for d in descriptors))
    torch.testing.assert_close(torch.from_numpy(output), expected, atol=3e-7, rtol=3e-6)


def test_positional_dropout_refused_before_emitting_ir():
    qtype = builtin.TensorType(builtin.f32, (1, 2, 3, 4))
    block = Block(arg_types=[qtype] * 3)
    with pytest.raises(NotImplementedError, match="dropout_p"):
        decompose_scaled_dot_product_attention(
            list(block.args),
            {"val": torch.empty(1, 2, 3, 4), "_fx_args": (None, None, None, None, 0.2)},
            "attention",
        )


def test_causal_and_explicit_mask_refused_like_torch_math():
    q = torch.randn(1, 2, 3, 4)
    mask = torch.ones(3, 3, dtype=torch.bool)
    with sdpa_kernel(SDPBackend.MATH), pytest.raises(RuntimeError, match="attn_mask"):
        torch.nn.functional.scaled_dot_product_attention(q, q, q, mask, is_causal=True)
    with pytest.raises(ValueError, match="attn_mask"):
        _sdpa(q, q, q, mask, is_causal=True)
    qtype = builtin.TensorType(builtin.f32, q.shape)
    block = Block(arg_types=[qtype] * 3 + [builtin.TensorType(builtin.i1, mask.shape)])
    with pytest.raises(ValueError, match="attn_mask"):
        decompose_scaled_dot_product_attention(
            list(block.args), {"val": q, "_fx_kwargs": {"is_causal": True}}, "attention"
        )
