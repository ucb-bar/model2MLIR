"""Selected CPU ReLU/scalar clamp values survive ordinary native MLIR lowering.

These controls compare finite result bits and NaN classification. They do not
claim signaling-NaN, payload, FENV or cross-device equivalence.
"""

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from xdsl.dialects.builtin import UnitAttr

import m2m


class Activation(torch.nn.Module):
    def __init__(self, operation, lower=None, upper=None):
        super().__init__()
        self.operation, self.lower, self.upper = operation, lower, upper

    def forward(self, x):
        if self.operation == "relu":
            return torch.relu(x)
        return torch.clamp(x, min=self.lower, max=self.upper)


def descriptor(array):
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
        array.ctypes.data, array.ctypes.data, 0,
        (ctypes.c_int64 * rank)(*array.shape),
        (ctypes.c_int64 * rank)(*(stride // array.itemsize for stride in array.strides)),
    )


def native_activation(tmp_path, model, array, optimization):
    translate = os.environ.get("MODEL2MLIR_MLIR_TRANSLATE") or shutil.which("mlir-translate")
    clang = os.environ.get("MODEL2MLIR_CLANG") or shutil.which("clang")
    if not translate or not clang:
        pytest.skip("native activation controls require MLIR/LLVM tools")
    opt = Path(translate).with_name("mlir-opt")
    if not opt.is_file():
        pytest.skip("native activation controls require mlir-opt beside mlir-translate")

    # Runtime boundary values are supplied after compilation; they are not
    # constants or compiler-selected example values in the exported program.
    example = torch.zeros(array.shape, dtype=torch.from_numpy(array).dtype)
    converted = m2m.convert(model, (example,), backend="fx_importer", level="linalg-on-tensors")
    assert converted.ok, converted.diagnostics
    converted.module.verify()
    assert "func.call" not in converted.mlir_text
    converted.module.body.block.first_op.attributes["llvm.emit_c_interface"] = UnitAttr()
    source, lowered, llvm, library = (tmp_path / name for name in (
        "source.mlir", "lowered.mlir", "model.ll", "model.so"
    ))
    source.write_text(str(converted.module), encoding="utf-8")
    pipeline = (
        "builtin.module(one-shot-bufferize{bufferize-function-boundaries},"
        "buffer-results-to-out-params,convert-linalg-to-loops,"
        "expand-strided-metadata,lower-affine,convert-scf-to-cf,convert-math-to-libm,"
        "convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,"
        "convert-cf-to-llvm,reconcile-unrealized-casts)"
    )
    subprocess.run([str(opt), str(source), "--pass-pipeline=" + pipeline, "-o", str(lowered)], check=True, capture_output=True)
    subprocess.run([translate, "--mlir-to-llvmir", str(lowered), "-o", str(llvm)], check=True, capture_output=True)
    runner_lib = Path(translate).parent.parent / "lib"
    subprocess.run([
        clang, optimization, "-fPIC", "-shared", str(llvm), "-lm",
        "-L" + str(runner_lib), "-lmlir_c_runner_utils", "-Wl,-rpath," + str(runner_lib),
        "-o", str(library),
    ], check=True, capture_output=True)
    entry = ctypes.CDLL(str(library))._mlir_ciface_forward
    entry.argtypes, entry.restype = [ctypes.c_void_p, ctypes.c_void_p], None
    output = np.empty_like(array)
    entry(ctypes.byref(descriptor(array)), ctypes.byref(descriptor(output)))
    return output


CASES = [
    ("relu", None, None),
    ("clamp", 0.0, 2.0),
    ("clamp", -2.0, -0.0),
    ("clamp", 0.0, -0.0),
    ("clamp", 2.0, -2.0),
    ("clamp", float("nan"), 2.0),
    ("clamp", -2.0, float("nan")),
    ("clamp", 0.0, None),
    ("clamp", None, -0.0),
    ("clamp", float("nan"), None),
    ("clamp", None, float("nan")),
]


@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
@pytest.mark.parametrize("length", [7, 269], ids=["short", "long"])
@pytest.mark.parametrize("operation,lower,upper", CASES)
def test_native_f32_signed_zero_nan_and_boundaries(tmp_path, optimization, length, operation, lower, upper):
    values = [-np.inf, -3.0, -2.0, -1.0, -0.0, 0.0, 1.0, 2.0, 3.0, np.inf, np.nan, -0.0, 0.0]
    palette = values + list(reversed(values))
    array = np.array((palette * (length // len(palette) + 1))[:length], dtype=np.float32)
    model = Activation(operation, lower, upper)
    actual = native_activation(tmp_path, model, array, optimization)
    expected = model(torch.from_numpy(array)).numpy()
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    observed = ~np.isnan(expected)
    np.testing.assert_array_equal(actual.view(np.uint32)[observed], expected.view(np.uint32)[observed])


@pytest.mark.parametrize("dtype", [np.float16, np.float64])
@pytest.mark.parametrize("operation,lower,upper", CASES[:3])
def test_native_other_float_formats_preserve_zero_bits(tmp_path, dtype, operation, lower, upper):
    array = np.array([-3, -2, -0.0, 0.0, 2, 3, np.nan], dtype=dtype)
    model = Activation(operation, lower, upper)
    actual = native_activation(tmp_path, model, array, "-O2")
    expected = model(torch.from_numpy(array)).numpy()
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    finite = ~np.isnan(expected)
    unsigned = np.dtype("u" + str(array.itemsize))
    np.testing.assert_array_equal(actual.view(unsigned)[finite], expected.view(unsigned)[finite])
