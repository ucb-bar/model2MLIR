"""A value-only pool must retain PyTorch's ordered winner bits after native lowering."""

import ctypes
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch
from xdsl.dialects.builtin import UnitAttr

import m2m


class _Pool(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.max_pool2d(x, 2)


class _PoolGeometry(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=(2, 3), stride=(1, 2), padding=(0, 1), dilation=(2, 1)
        )


class _PoolValuesFromTuple(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.max_pool2d(x, 2, return_indices=True)[0]


class _PoolValuesAndIndices(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.max_pool2d(x, 2, return_indices=True)


class _PoolWithDeadIndices(torch.nn.Module):
    def forward(self, x):
        values, indices = torch.nn.functional.max_pool2d(x, 2, return_indices=True)
        _ = indices
        return values


class _PoolWithConsumedIndices(torch.nn.Module):
    def forward(self, x):
        values, indices = torch.nn.functional.max_pool2d(x, 2, return_indices=True)
        return values, indices.to(torch.float32)


class _PoolIndicesOnly(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.max_pool2d(x, 2, return_indices=True)[1]


def test_pool_values_only_tuple_consumer_is_supported():
    inp = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    for model in (_PoolValuesFromTuple(), _PoolWithDeadIndices()):
        conversion = m2m.convert(model, (inp,), backend="fx_importer", level="linalg-on-tensors")
        assert conversion.ok, conversion.diagnostics
        assert "linalg.generic" in conversion.mlir_text


def test_live_pool_indices_refuse_instead_of_dropping_output():
    inp = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    for model in (_PoolValuesAndIndices(), _PoolWithConsumedIndices(), _PoolIndicesOnly()):
        conversion = m2m.convert(model, (inp,), backend="fx_importer", level="linalg-on-tensors")
        assert not conversion.ok
        assert any("live max_pool2d indices" in message for message in conversion.diagnostics)


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
        (ctypes.c_int64 * 4)(*(stride // array.itemsize for stride in array.strides)),
    )


def _native_pool(tmp_path: Path, *, optimization="-O2", model=None, input_shape=(1, 1, 2, 2)):
    translate = os.environ.get("MERLIN_MLIR_TRANSLATE") or shutil.which("mlir-translate")
    clang = os.environ.get("MERLIN_CLANG") or shutil.which("clang")
    if not translate or not clang:
        pytest.skip("native pool test requires MLIR/LLVM tools")
    opt = str(Path(translate).with_name("mlir-opt"))
    if not Path(opt).is_file():
        pytest.skip("native pool test requires mlir-opt beside mlir-translate")

    example = torch.zeros(input_shape, dtype=torch.float32)
    conversion = m2m.convert(model or _Pool(), (example,), backend="fx_importer", level="linalg-on-tensors")
    assert conversion.ok, conversion.diagnostics
    assert "func.call" not in conversion.mlir_text
    function = conversion.module.body.block.first_op
    function.attributes["llvm.emit_c_interface"] = UnitAttr()
    source, lowered, llvm, library = (tmp_path / name for name in (
        "source.mlir", "lowered.mlir", "model.ll", "model.so"
    ))
    source.write_text(str(conversion.module), encoding="utf-8")
    pipeline = (
        "builtin.module(one-shot-bufferize{bufferize-function-boundaries},"
        "buffer-results-to-out-params{modify-public-functions},convert-linalg-to-loops,"
        "expand-strided-metadata,lower-affine,convert-scf-to-cf,convert-math-to-libm,"
        "convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,"
        "convert-cf-to-llvm,reconcile-unrealized-casts)"
    )
    subprocess.run([opt, str(source), "--pass-pipeline=" + pipeline, "-o", str(lowered)],
                   check=True, capture_output=True, text=True)
    subprocess.run([translate, "--mlir-to-llvmir", str(lowered), "-o", str(llvm)],
                   check=True, capture_output=True, text=True)
    runner_lib = Path(translate).parent.parent / "lib"
    subprocess.run([clang, optimization, "-fPIC", "-shared", str(llvm), "-lm",
                    "-L" + str(runner_lib), "-lmlir_c_runner_utils",
                    "-Wl,-rpath," + str(runner_lib), "-o", str(library)],
                   check=True, capture_output=True, text=True)
    entry = ctypes.CDLL(str(library))._mlir_ciface_forward
    entry.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    entry.restype = None
    return entry, source, lowered, llvm


def _float_array(bits):
    return np.array(bits, dtype=np.uint32).view(np.float32).copy().reshape((1, 1, 2, 2))


@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
def test_native_value_pool_matches_ordered_pytorch_winner_bits(tmp_path, optimization):
    entry, source, _lowered, llvm = _native_pool(tmp_path, optimization=optimization)
    text = source.read_text(encoding="utf-8")
    assert "arith.cmpf uno" in text and "arith.cmpf ogt" in text
    assert "arith.maximumf" not in text
    llvm_text = llvm.read_text(encoding="utf-8")
    assert "fcmp uno float" in llvm_text and "fcmp ogt float" in llvm_text
    assert "select i1" in llvm_text and "llvm.maximum" not in llvm_text
    assert "fcmp fast" not in llvm_text and "reassoc" not in llvm_text
    for bits in (
        (0xc0400000, 0x40000000, 0xbf800000, 0x40800000),
        (0x80000000, 0x00000000, 0x00000000, 0x00000000),
        (0x00000000, 0x80000000, 0x80000000, 0x80000000),
        (0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000),
        (0x7fc00001, 0x3f800000, 0x7fc00002, 0x40000000),
        (0x3f800000, 0x7fc00001, 0x40000000, 0x7fc00002),
        (0xffc00001, 0x7fc00002, 0x3f800000, 0x40000000),
        (0x7fc00002, 0x3f800000, 0x7f800001, 0x40000000),
    ):
        inp = _float_array(bits)
        out = np.empty((1, 1, 1, 1), dtype=np.float32)
        entry(ctypes.byref(_descriptor(inp)), ctypes.byref(_descriptor(out)))
        expected = torch.nn.functional.max_pool2d(torch.from_numpy(inp), 2).numpy()
        np.testing.assert_array_equal(out.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
def test_native_value_pool_uses_captured_window_geometry(tmp_path, optimization):
    model = _PoolGeometry()
    inp = np.array([
        -9, -8, -7, -6, -5,
        -4, -3, -2, -1, 0,
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
    ], dtype=np.float32).reshape((1, 1, 4, 5))
    inp[0, 0, 0, 1] = -0.0
    inp[0, 0, 0, 2] = 0.0
    expected = model(torch.from_numpy(inp)).numpy()
    entry, source, _lowered, _llvm = _native_pool(
        tmp_path, model=model, input_shape=inp.shape, optimization=optimization
    )
    assert "linalg.generic" in source.read_text(encoding="utf-8")
    out = np.empty_like(expected)
    entry(ctypes.byref(_descriptor(inp)), ctypes.byref(_descriptor(out)))
    np.testing.assert_array_equal(out.view(np.uint32), expected.view(np.uint32))
