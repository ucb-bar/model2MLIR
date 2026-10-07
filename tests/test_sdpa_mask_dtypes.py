"""SDPA mask admission and emitted opmath match the source dtype contract."""

import ctypes
import subprocess

import pytest
import test_sdpa_mask_semantics as masks
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from xdsl.dialects import builtin

from m2m.ir.torchmlir_decomps import _sdpa

native_tools = masks.native_tools
DTYPES = (
    (torch.float16, builtin.f16),
    (torch.bfloat16, builtin.bf16),
    (torch.float32, builtin.f32),
    (torch.float64, builtin.f64),
)


@pytest.mark.parametrize("query_dtype,query_elem", DTYPES)
@pytest.mark.parametrize("mask_dtype,mask_elem", DTYPES)
def test_mask_dtype_admission_matches_torch(
    query_dtype, query_elem, mask_dtype, mask_elem
):
    q, k, v = masks._inputs(query_dtype)
    mask = masks._mask("additive", mask_dtype)
    admitted = mask_dtype in (torch.float32, query_dtype)
    if admitted:
        with sdpa_kernel(SDPBackend.MATH):
            expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask)
        actual = _sdpa(q, k, v, mask)
        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-7)
        masks._direct((3, 5), mask_elem, query_elem)
    else:
        with (
            pytest.raises(RuntimeError, match="attn_mask dtype"),
            sdpa_kernel(SDPBackend.MATH),
        ):
            torch.nn.functional.scaled_dot_product_attention(q, k, v, mask)
        with pytest.raises(TypeError, match="mask.*query dtype"):
            _sdpa(q, k, v, mask)
        with pytest.raises(TypeError, match="mask.*query dtype"):
            masks._direct((3, 5), mask_elem, query_elem)


def _storage(tensor):
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.contiguous().view(torch.uint16).numpy()
    return tensor.contiguous().numpy()


def _native(module, inputs, query_dtype, tmp_path, native_tools):
    opt, translate, clang = native_tools
    source, lowered, llvm, shared = [
        tmp_path / name for name in ("input.mlir", "llvm.mlir", "input.ll", "native.so")
    ]
    source.write_text(str(module))
    pipeline = (
        "builtin.module(one-shot-bufferize{bufferize-function-boundaries},"
        "buffer-results-to-out-params{modify-public-functions},convert-linalg-to-loops,"
        "expand-strided-metadata,lower-affine,convert-scf-to-cf,convert-math-to-libm,"
        "convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,"
        "convert-cf-to-llvm,reconcile-unrealized-casts)"
    )
    for command in (
        [opt, str(source), "--pass-pipeline=" + pipeline, "-o", str(lowered)],
        [translate, "--mlir-to-llvmir", str(lowered), "-o", str(llvm)],
        [clang, "-O2", "-fPIC", "-shared", str(llvm), "-lm", "-o", str(shared)],
    ):
        subprocess.run(command, check=True, capture_output=True)
    output = torch.full((1, 2, 3, 6), torch.nan, dtype=query_dtype)
    arrays = [_storage(value) for value in (*inputs, output)]
    descriptors = [masks._descriptor(array) for array in arrays]
    library = ctypes.CDLL(str(shared))
    entry = library._mlir_ciface_forward
    entry.argtypes = [ctypes.c_void_p] * len(descriptors)
    entry.restype = None
    entry(*(ctypes.byref(value) for value in descriptors))
    return output


@pytest.mark.parametrize(
    "query_dtype,query_elem,mask_dtype,mask_elem",
    (
        (torch.float16, builtin.f16, torch.float16, builtin.f16),
        (torch.float16, builtin.f16, torch.float32, builtin.f32),
        (torch.bfloat16, builtin.bf16, torch.bfloat16, builtin.bf16),
        (torch.bfloat16, builtin.bf16, torch.float32, builtin.f32),
        (torch.float64, builtin.f64, torch.float32, builtin.f32),
        (torch.float64, builtin.f64, torch.float64, builtin.f64),
    ),
)
def test_direct_native_mask_opmath(
    tmp_path, native_tools, query_dtype, query_elem, mask_dtype, mask_elem
):
    q, k, v = masks._inputs(query_dtype)
    mask = masks._mask("additive", mask_dtype)
    module = masks._direct((3, 5), mask_elem, query_elem)
    actual = _native(module, (q, k, v, mask), query_dtype, tmp_path, native_tools)
    with sdpa_kernel(SDPBackend.MATH):
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[:, :, 1, :]) == 0
    rtol, atol = (
        (0.008, 0.004)
        if query_dtype == torch.bfloat16
        else (0.004, 0.001)
        if query_dtype == torch.float16
        else (3e-6, 3e-7)
    )
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("nonfinite", (torch.nan, torch.inf))
def test_nonfinite_score_rows_remain_nonfinite_native(
    tmp_path, native_tools, nonfinite
):
    q, k, v = masks._inputs()
    mask = torch.zeros((3, 5), dtype=torch.float32)
    mask[1] = nonfinite
    actual = _native(
        masks._direct((3, 5), builtin.f32),
        (q, k, v, mask),
        torch.float32,
        tmp_path,
        native_tools,
    )
    assert torch.isnan(actual[:, :, 1, :]).all()
    assert torch.isfinite(actual[:, :, [0, 2], :]).all()
    with sdpa_kernel(SDPBackend.MATH):
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask)
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-7, equal_nan=True)
