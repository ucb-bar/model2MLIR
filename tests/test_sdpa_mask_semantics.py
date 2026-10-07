"""SDPA empty rows and masks must retain the source operator's semantics."""

import ctypes
import subprocess

import numpy as np
import pytest
import test_sdpa_options as sdpa_options
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region

from m2m.ir.decompositions import decompose_scaled_dot_product_attention
from m2m.ir.torchmlir_decomps import _sdpa

native_tools = sdpa_options.native_tools


def _inputs(dtype=torch.float32):
    return tuple(
        torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape).to(dtype)
        / divisor
        for shape, divisor in zip(
            ((1, 2, 3, 4), (1, 2, 5, 4), (1, 2, 5, 6)), (17, 19, 23), strict=True
        )
    )


def _mask(kind, dtype=torch.float32):
    mask = torch.ones((3, 5), dtype=torch.bool)
    mask[1] = False
    mask[2, 3:] = False
    if kind == "bool":
        return mask
    return torch.where(
        mask, torch.tensor(0, dtype=dtype), torch.tensor(-torch.inf, dtype=dtype)
    )


def _direct(mask_shape, mask_elem, query_elem=builtin.f32):
    shapes = ((1, 2, 3, 4), (1, 2, 5, 4), (1, 2, 5, 6))
    types = [builtin.TensorType(query_elem, shape) for shape in shapes]
    types.append(builtin.TensorType(mask_elem, mask_shape))
    result_type = builtin.TensorType(query_elem, (1, 2, 3, 6))
    block = Block(arg_types=types)
    result = decompose_scaled_dot_product_attention(
        list(block.args), {"val": torch.empty(1, 2, 3, 6)}, "masked_attention"
    )
    block.add_ops(result.ops)
    block.add_op(func.ReturnOp(result.result))
    function = func.FuncOp("forward", (types, [result_type]), Region(block))
    function.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
    module = builtin.ModuleOp([function])
    module.verify()
    return module


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("kind", ["bool", "additive"])
def test_export_empty_rows_match_source(dtype, kind):
    q, k, v = _inputs(dtype)
    mask = _mask(kind, dtype)
    with sdpa_kernel(SDPBackend.MATH):
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask)
    actual = _sdpa(q, k, v, mask)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[:, :, 1, :]).item() == 0
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("shape", [(2, 7), (3, 3, 5), (1, 2, 2, 5)])
def test_direct_invalid_mask_broadcast_refused(shape):
    with pytest.raises(ValueError, match="mask.*broadcast"):
        _direct(shape, builtin.i1)


def test_direct_integer_mask_is_not_boolean():
    with pytest.raises(TypeError, match="mask.*boolean"):
        _direct((3, 5), builtin.i8)


def _descriptor(array):
    rank = array.ndim

    class Descriptor(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.c_int64 * rank),
            ("strides", ctypes.c_int64 * rank),
        ]

    return Descriptor(
        array.ctypes.data,
        array.ctypes.data,
        0,
        (ctypes.c_int64 * rank)(*array.shape),
        (ctypes.c_int64 * rank)(
            *(stride // array.itemsize for stride in array.strides)
        ),
    )


class _ExportedMaskedAttention(torch.nn.Module):
    def forward(self, q, k, v, mask):
        return _sdpa(q, k, v, mask)


@pytest.mark.parametrize("kind", ["bool", "additive"])
@pytest.mark.parametrize("path", ["direct", "exported"])
def test_native_empty_and_finite_rows(tmp_path, native_tools, kind, path):
    opt, translate, clang = native_tools
    source, lowered, llvm, shared = [
        tmp_path / name for name in ("input.mlir", "llvm.mlir", "input.ll", "native.so")
    ]
    q, k, v = _inputs()
    mask = _mask(kind)
    if path == "direct":
        source.write_text(
            str(_direct((3, 5), builtin.i1 if kind == "bool" else builtin.f32))
        )
    else:
        import m2m

        converted = m2m.convert(
            _ExportedMaskedAttention(), (q, k, v, mask), backend="fx_importer"
        )
        assert "opaque_aten" not in converted.mlir_text
        from xdsl.context import Context
        from xdsl.dialects import arith, linalg, math, tensor
        from xdsl.parser import Parser

        context = Context()
        for dialect in (
            builtin.Builtin,
            func.Func,
            arith.Arith,
            linalg.Linalg,
            math.Math,
            tensor.Tensor,
        ):
            context.load_dialect(dialect)
        module = Parser(context, converted.mlir_text).parse_module()
        function = next(
            op for op in module.body.block.ops if isinstance(op, func.FuncOp)
        )
        function.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
        module.verify()
        source.write_text(str(module))
    pipeline = (
        "builtin.module(one-shot-bufferize{bufferize-function-boundaries},"
        "buffer-results-to-out-params{modify-public-functions},convert-linalg-to-loops,"
        "expand-strided-metadata,lower-affine,convert-scf-to-cf,convert-math-to-libm,"
        "convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,"
        "convert-cf-to-llvm,reconcile-unrealized-casts)"
    )
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
        [clang, "-O2", "-fPIC", "-shared", str(llvm), "-lm", "-o", str(shared)],
        check=True,
        capture_output=True,
    )
    library = ctypes.CDLL(str(shared))
    entry = library._mlir_ciface_forward
    entry.argtypes = [ctypes.c_void_p] * 5
    entry.restype = None
    output = np.full((1, 2, 3, 6), np.nan, np.float32)
    descriptors = [
        _descriptor(a) for a in (q.numpy(), k.numpy(), v.numpy(), mask.numpy(), output)
    ]
    entry(*(ctypes.byref(descriptor) for descriptor in descriptors))
    with sdpa_kernel(SDPBackend.MATH):
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask)
    assert np.isfinite(output).all()
    assert np.count_nonzero(output[:, :, 1, :]) == 0
    torch.testing.assert_close(torch.from_numpy(output), expected, rtol=3e-6, atol=3e-7)


def test_nan_row_is_not_mistaken_for_empty():
    q, k, v = _inputs()
    q[:, :, 1, 0] = torch.nan
    actual = _sdpa(q, k, v)
    assert torch.isnan(actual[:, :, 1, :]).all()
    assert torch.isfinite(actual[:, :, [0, 2], :]).all()
