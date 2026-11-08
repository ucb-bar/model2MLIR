"""Typed signed/unsigned integer dots agree with the real CPU ATen operator."""

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from test_activation_boundary_semantics import descriptor
from xdsl.dialects.builtin import UnitAttr

import m2m
from m2m.coverage import opaque_report


class IntMM(torch.nn.Module):
    def forward(self, a, b):
        return torch.ops.aten._int_mm.default(a, b)


def test_signless_storage_without_source_dtype_is_not_guessed():
    from xdsl.dialects.builtin import TensorType, i8
    from xdsl.ir import Block

    from m2m.ir.decompositions import decompose_int_mm

    args = Block(arg_types=[TensorType(i8, [2, 32]), TensorType(i8, [32, 4])]).args
    result = decompose_int_mm(
        list(args), {"val": torch.empty(2, 4, dtype=torch.int32)}, "unknown"
    )
    assert any(op.name == "func.call" for op in result.ops)


def operands(unsigned):
    palette = np.arange(256, dtype=np.uint8)
    if not unsigned:
        palette = palette.view(np.int8)
    a = np.resize(palette, (256, 32)).copy()
    b = np.resize(
        np.array([-128, -127, -1, 0, 1, 126, 127], dtype=np.int8), (32, 4)
    ).copy()
    return a, b


def lower(a, b):
    result = m2m.convert(
        IntMM(), (torch.from_numpy(a), torch.from_numpy(b)), backend="fx_importer"
    )
    assert result.ok and opaque_report(result.mlir_text) == {}
    result.module.verify()
    dots = [
        op
        for op in result.module.walk()
        if op.name == "linalg.generic"
        and getattr(op.attributes.get("prov.op"), "data", None) == "int_matmul"
    ]
    assert len(dots) == 1
    return result, dots[0]


def evaluate(block, a, b):
    """Execute the emitted body with independent modular integer operations."""
    output = np.zeros((a.shape[0], b.shape[1]), dtype=np.int32)
    for row in range(a.shape[0]):
        for col in range(b.shape[1]):
            acc = 0
            for k in range(a.shape[1]):
                values = dict(zip(block.args, [int(a[row, k]), int(b[k, col]), acc]))
                for op in block.ops:
                    args = [values[value] for value in op.operands]
                    if op.name == "linalg.yield":
                        acc = args[0]
                        continue
                    if op.name == "arith.extui":
                        value = args[0] & 255
                    elif op.name == "arith.extsi":
                        value = (args[0] & 127) - (args[0] & 128)
                    elif op.name == "arith.muli":
                        value = args[0] * args[1]
                    elif op.name == "arith.addi":
                        value = args[0] + args[1]
                    else:
                        raise AssertionError("unimplemented body operation: " + op.name)
                    value &= 0xFFFFFFFF
                    values[op.results[0]] = (
                        value - (1 << 32) if value >= 1 << 31 else value
                    )
            output[row, col] = acc
    return output


@pytest.mark.parametrize("unsigned", [False, True])
def test_all_input_byte_patterns_match_torch_cpu(unsigned):
    a, b = operands(unsigned)
    _, dot = lower(a, b)
    wanted = "uint8" if unsigned else "int8"
    assert dot.attributes["prov.int_mm_lhs_dtype"].data == wanted
    assert dot.attributes["prov.int_mm_rhs_dtype"].data == "int8"
    actual = evaluate(dot.body.block, a, b)
    expected = torch.ops.aten._int_mm.default(
        torch.from_numpy(a), torch.from_numpy(b)
    ).numpy()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        actual, (a.astype(np.int64) @ b.astype(np.int64)).astype(np.int32)
    )


@pytest.mark.parametrize("unsigned", [False, True])
def test_actual_native_compilation_preserves_operand_signedness(tmp_path, unsigned):
    translate = os.environ.get("MODEL2MLIR_MLIR_TRANSLATE") or shutil.which(
        "mlir-translate"
    )
    clang = os.environ.get("MODEL2MLIR_CLANG") or shutil.which("clang")
    if (
        not translate
        or not clang
        or not Path(translate).with_name("mlir-opt").is_file()
    ):
        pytest.skip("MLIR/LLVM tools are required for native int_mm control")
    a, b = operands(unsigned)
    result, _ = lower(a, b)
    result.module.body.block.first_op.attributes["llvm.emit_c_interface"] = UnitAttr()
    source, lowered, llvm, library = (
        tmp_path / name
        for name in ("source.mlir", "lowered.mlir", "model.ll", "model.so")
    )
    source.write_text(str(result.module))
    pipeline = (
        "builtin.module(one-shot-bufferize{bufferize-function-boundaries},"
        "buffer-results-to-out-params{modify-public-functions},convert-linalg-to-loops,"
        "expand-strided-metadata,lower-affine,convert-scf-to-cf,convert-math-to-libm,"
        "convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,"
        "convert-cf-to-llvm,reconcile-unrealized-casts)"
    )
    subprocess.run(
        [
            str(Path(translate).with_name("mlir-opt")),
            str(source),
            "--pass-pipeline=" + pipeline,
            "-o",
            str(lowered),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    subprocess.run(
        [translate, "--mlir-to-llvmir", str(lowered), "-o", str(llvm)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    subprocess.run(
        [clang, "-O2", "-fPIC", "-shared", str(llvm), "-lm", "-o", str(library)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    entry = ctypes.CDLL(str(library))._mlir_ciface_forward
    entry.argtypes, entry.restype = [ctypes.c_void_p] * 3, None
    actual = np.empty((a.shape[0], b.shape[1]), dtype=np.int32)
    entry(
        ctypes.byref(descriptor(a)),
        ctypes.byref(descriptor(b)),
        ctypes.byref(descriptor(actual)),
    )
    expected = torch.ops.aten._int_mm.default(
        torch.from_numpy(a), torch.from_numpy(b)
    ).numpy()
    np.testing.assert_array_equal(actual, expected)
