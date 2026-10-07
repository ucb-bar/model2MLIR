"""Neutral CPU cumsum examples through actual frontend and native lowering.

These examples test selected values and NaN classification, not a universal
NaN payload/signaling-NaN equivalence claim across runtimes or targets.
"""

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import m2m
import numpy as np
import pytest
import torch
from xdsl.dialects.builtin import UnitAttr


class _Cumsum(torch.nn.Module):
    def __init__(self, dim, dtype=None):
        super().__init__()
        self.dim = dim
        self.dtype = dtype

    def forward(self, x):
        return torch.cumsum(x, self.dim, dtype=self.dtype)


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
        (ctypes.c_int64 * rank)(*(stride // array.itemsize for stride in array.strides)),
    )


def _native_cumsum(tmp_path: Path, array, *, dim, optimization="-O2", dtype=None, example=None):
    translate = os.environ.get("MODEL2MLIR_MLIR_TRANSLATE") or shutil.which("mlir-translate")
    clang = os.environ.get("MODEL2MLIR_CLANG") or shutil.which("clang")
    if not translate or not clang:
        pytest.skip("native cumsum test requires MLIR/LLVM tools")
    opt = str(Path(translate).with_name("mlir-opt"))
    if not Path(opt).is_file():
        pytest.skip("native cumsum test requires mlir-opt beside mlir-translate")

    example = example if example is not None else torch.from_numpy(array)
    conversion = m2m.convert(_Cumsum(dim, dtype), (example,), backend="fx_importer", level="linalg-on-tensors")
    assert conversion.ok, conversion.diagnostics
    assert "func.call" not in conversion.mlir_text
    function = conversion.module.body.block.first_op
    function.attributes["llvm.emit_c_interface"] = UnitAttr()
    source, lowered, llvm, library = (
        tmp_path / name for name in ("source.mlir", "lowered.mlir", "model.ll", "model.so")
    )
    source.write_text(str(conversion.module), encoding="utf-8")
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
        text=True,
    )
    subprocess.run(
        [translate, "--mlir-to-llvmir", str(lowered), "-o", str(llvm)], check=True, capture_output=True, text=True
    )
    runner_lib = Path(translate).parent.parent / "lib"
    subprocess.run(
        [
            clang,
            optimization,
            "-fPIC",
            "-shared",
            str(llvm),
            "-lm",
            "-L" + str(runner_lib),
            "-lmlir_c_runner_utils",
            "-Wl,-rpath," + str(runner_lib),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    entry = ctypes.CDLL(str(library))._mlir_ciface_forward
    entry.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    entry.restype = None
    return entry, source, llvm


@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
def test_float32_cpu_cumsum_cancellation_matches_native_bits(tmp_path, optimization):
    inp = np.array([1e8, 1, -1e8], dtype=np.float32)
    entry, _source, _llvm = _native_cumsum(tmp_path, inp, dim=0, optimization=optimization)
    out = np.empty_like(inp)
    entry(ctypes.byref(_descriptor(inp)), ctypes.byref(_descriptor(out)))
    expected = torch.cumsum(torch.from_numpy(inp), 0).numpy()
    np.testing.assert_array_equal(out.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("shape,dim", [((2, 3), 0), ((2, 3), 1), ((2, 3), -1), ((3, 2), 0)])
def test_float32_cpu_cumsum_order_across_axes(tmp_path, shape, dim):
    inp = np.array([1e8, 1, -1e8, -0.0, 1, 2], dtype=np.float32).reshape(shape)
    entry, source, llvm = _native_cumsum(tmp_path, inp, dim=dim)
    text = source.read_text(encoding="utf-8")
    assert "scf.for" in text and "f64" in text
    assert "linalg.generic" not in text
    assert "reassoc" not in llvm.read_text(encoding="utf-8")
    out = np.empty_like(inp)
    entry(ctypes.byref(_descriptor(inp)), ctypes.byref(_descriptor(out)))
    expected = torch.cumsum(torch.from_numpy(inp), dim).numpy()
    np.testing.assert_array_equal(out.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize(
    "dtype,values",
    [
        (np.float16, [2048, 1, -2048]),
        (np.float32, [1e8, 1, -1e8]),
        (np.float64, [1e16, 1, -1e16]),
    ],
)
def test_cpu_cumsum_accumulation_dtype_matches_native(tmp_path, dtype, values):
    inp = np.array(values, dtype=dtype)
    entry, source, _llvm = _native_cumsum(tmp_path, inp, dim=0)
    text = source.read_text(encoding="utf-8")
    assert "scf.for" in text
    assert ("f32" if dtype == np.float16 else "f64") in text
    out = np.empty_like(inp)
    entry(ctypes.byref(_descriptor(inp)), ctypes.byref(_descriptor(out)))
    expected = torch.cumsum(torch.from_numpy(inp), 0).numpy()
    np.testing.assert_array_equal(out.view(f"u{out.itemsize}"), expected.view(f"u{out.itemsize}"))


@pytest.mark.parametrize(
    "out_dtype,np_dtype",
    [
        (torch.float16, np.float16),
        (torch.float32, np.float32),
        (torch.float64, np.float64),
    ],
)
def test_cpu_cumsum_explicit_output_dtype(tmp_path, out_dtype, np_dtype):
    inp = np.array([2048, 1, -2048], dtype=np.float16)
    entry, _source, _llvm = _native_cumsum(tmp_path, inp, dim=0, dtype=out_dtype)
    out = np.empty(inp.shape, dtype=np_dtype)
    entry(ctypes.byref(_descriptor(inp)), ctypes.byref(_descriptor(out)))
    expected = torch.cumsum(torch.from_numpy(inp), 0, dtype=out_dtype).numpy()
    np.testing.assert_array_equal(out.view(f"u{out.itemsize}"), expected.view(f"u{out.itemsize}"))


def test_cpu_cumsum_casts_each_input_before_explicit_dtype_scan(tmp_path):
    inp = np.array([16777217.0, -16777216.0], dtype=np.float64)
    entry, _source, _llvm = _native_cumsum(tmp_path, inp, dim=0, dtype=torch.float32)
    out = np.empty(inp.shape, dtype=np.float32)
    entry(ctypes.byref(_descriptor(inp)), ctypes.byref(_descriptor(out)))
    expected = torch.cumsum(torch.from_numpy(inp), 0, dtype=torch.float32).numpy()
    np.testing.assert_array_equal(out.view(np.uint32), expected.view(np.uint32))
    assert out[-1] == 0.0


@pytest.mark.parametrize("length", [1, 7, 8, 9, 17, 31, 256, 1025])
def test_cpu_cumsum_vector_tail_boundaries(tmp_path, length):
    pattern = [1e8, 1, -1e8, -0.0, 1, 2, -2, 3, -3]
    inp = np.array((pattern * ((length + len(pattern) - 1) // len(pattern)))[:length], dtype=np.float32)
    entry, _source, _llvm = _native_cumsum(tmp_path, inp, dim=0)
    out = np.empty_like(inp)
    entry(ctypes.byref(_descriptor(inp)), ctypes.byref(_descriptor(out)))
    expected = torch.cumsum(torch.from_numpy(inp), 0).numpy()
    np.testing.assert_array_equal(out.view(np.uint32), expected.view(np.uint32))


def test_cpu_cumsum_captured_device_is_required():
    from types import SimpleNamespace

    from m2m.ir.decompositions import decompose_cumsum
    from xdsl.dialects.builtin import Float32Type, TensorType
    from xdsl.ir import Block

    inp = Block(arg_types=[TensorType(Float32Type(), [3])]).args[0]
    for source_device, result_device in ((None, "cpu"), ("cuda", "cpu"), ("cpu", None), ("cpu", "cuda")):
        source_val = SimpleNamespace(device=SimpleNamespace(type=source_device))
        result_val = SimpleNamespace(device=SimpleNamespace(type=result_device), shape=(3,), dtype=torch.float32)
        source = SimpleNamespace(meta={"val": source_val})
        with pytest.raises(TypeError, match="captured CPU input and result devices"):
            decompose_cumsum([inp], {"_fx_args": (source, 0), "val": result_val}, "scan")


@pytest.mark.parametrize("shape", [(1 << 63,), (1 << 32, 1 << 32)])
def test_cpu_cumsum_rejects_unrepresentable_source_indices(shape):
    from m2m.ir.decompositions import _ordered_cpu_float_cumsum
    from xdsl.dialects.builtin import Float32Type, TensorType
    from xdsl.ir import Block

    inp = Block(arg_types=[TensorType(Float32Type(), [1])]).args[0]
    with pytest.raises(TypeError, match="64-bit source index"):
        _ordered_cpu_float_cumsum(inp, shape, 0, Float32Type())


def test_scalar_cpu_cumsum_remains_visible_opaque():
    result = m2m.convert(_Cumsum(0), (torch.tensor(1.0),), backend="fx_importer", level="linalg-on-tensors")
    assert result.ok, result.diagnostics
    assert "func.call" in result.mlir_text
    assert "scf.for" not in result.mlir_text


@pytest.mark.parametrize(
    "values",
    [
        [-0.0, -0.0, 0.0],
        [float("inf"), -float("inf"), 1.0],
        [float("nan"), 1.0, 2.0],
    ],
)
def test_cpu_cumsum_nonfinite_and_signed_zero(tmp_path, values):
    inp = np.array(values, dtype=np.float32)
    entry, _source, _llvm = _native_cumsum(tmp_path, inp, dim=0)
    out = np.empty_like(inp)
    entry(ctypes.byref(_descriptor(inp)), ctypes.byref(_descriptor(out)))
    expected = torch.cumsum(torch.from_numpy(inp), 0).numpy()
    np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))
    mask = ~np.isnan(expected)
    np.testing.assert_array_equal(out[mask].view(np.uint32), expected[mask].view(np.uint32))


def test_cpu_cumsum_empty_shape_is_not_divided_by_zero(tmp_path):
    inp = np.empty((0,), dtype=np.float32)
    entry, source, _llvm = _native_cumsum(tmp_path, inp, dim=0)
    assert "scf.for" not in source.read_text(encoding="utf-8")
    out = np.empty_like(inp)
    entry(ctypes.byref(_descriptor(inp)), ctypes.byref(_descriptor(out)))
    assert out.shape == torch.cumsum(torch.from_numpy(inp), 0).shape


@pytest.mark.parametrize("shape,dim", [((2, 0), 1), ((0, 2), 0)])
def test_cpu_cumsum_empty_multidimensional_shape(tmp_path, shape, dim):
    inp = np.empty(shape, dtype=np.float32)
    entry, source, _llvm = _native_cumsum(tmp_path, inp, dim=dim)
    assert "scf.for" not in source.read_text(encoding="utf-8")
    out = np.empty_like(inp)
    entry(ctypes.byref(_descriptor(inp)), ctypes.byref(_descriptor(out)))
    assert out.shape == torch.cumsum(torch.from_numpy(inp), dim).shape


def test_cpu_bfloat16_cumsum_uses_f32_accumulation(tmp_path):
    example = torch.tensor([256, 1, -256], dtype=torch.bfloat16)
    inp = example.view(torch.int16).numpy().view(np.uint16).copy()
    entry, source, _llvm = _native_cumsum(tmp_path, inp, dim=0, example=example)
    assert "f32" in source.read_text(encoding="utf-8")
    out = np.empty_like(inp)
    entry(ctypes.byref(_descriptor(inp)), ctypes.byref(_descriptor(out)))
    expected = torch.cumsum(example, 0).view(torch.int16).numpy().view(np.uint16)
    np.testing.assert_array_equal(out, expected)


def test_integer_cumsum_keeps_modular_existing_lowering(tmp_path):
    inp = np.array([np.iinfo(np.int64).max, 1, -3], dtype=np.int64)
    entry, source, _llvm = _native_cumsum(tmp_path, inp, dim=0)
    assert "linalg.generic" in source.read_text(encoding="utf-8")
    out = np.empty_like(inp)
    entry(ctypes.byref(_descriptor(inp)), ctypes.byref(_descriptor(out)))
    expected = torch.cumsum(torch.from_numpy(inp), 0).numpy()
    np.testing.assert_array_equal(out.view(np.uint64), expected.view(np.uint64))
