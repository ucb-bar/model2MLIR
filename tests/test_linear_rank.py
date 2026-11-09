"""Exported linear keeps source storage and emits a dimensionally valid matmul."""

from hashlib import sha256
from math import prod
import os
from pathlib import Path
import subprocess

import pytest
import torch

import m2m
from m2m.capture.torch_mlir_bridge import module_to_text
from m2m.ir.decompositions import decompose_linear
from xdsl.dialects import builtin
from xdsl.ir import Block, Region


def _raw(value):
    return value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


@pytest.fixture(scope="module")
def strict_opt():
    path = os.environ.get("M2M_MLIR_OPT")
    if not path:
        pytest.skip("strict linear verification requires M2M_MLIR_OPT")
    binary = Path(path)
    expected = os.environ.get("M2M_MLIR_OPT_SHA256")
    if expected:
        assert sha256(binary.read_bytes()).hexdigest() == expected
    return str(binary)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("shape", [(8,), (3, 8), (1, 3, 8), (2, 1, 3, 8)], ids=["rank1", "rank2", "rank3", "rank4"])
@pytest.mark.parametrize("bias", [False, True], ids=["nobias", "bias"])
def test_exported_linear_valid_rank_two_matmul(tmp_path, strict_opt, dtype, shape, bias):
    _check_exported_linear(tmp_path, strict_opt, dtype, shape, bias, 5)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("shape,n", [((0, 8), 5), ((2, 0, 8), 5), ((1, 3, 0), 5), ((1, 3, 8), 0)],
                         ids=["zero-rank2", "zero-leading", "zero-k", "zero-n"])
@pytest.mark.parametrize("bias", [False, True], ids=["nobias", "bias"])
def test_exported_linear_zero_extents(tmp_path, strict_opt, dtype, shape, n, bias):
    _check_exported_linear(tmp_path, strict_opt, dtype, shape, bias, n)


def _check_exported_linear(tmp_path, strict_opt, dtype, shape, bias, n):
    assert Path(m2m.__file__).absolute().parent.parent == Path(__file__).absolute().parents[1]
    torch.manual_seed(17)
    k = shape[-1]
    model = torch.nn.Linear(k, n, bias=bias, dtype=dtype).eval()
    source = torch.arange(prod(shape), dtype=torch.float32).reshape(shape).to(dtype)
    program = torch.export.export(model, (source,))
    storage = {name: (value.dtype, tuple(value.shape), _raw(value)) for name, value in program.state_dict.items()}
    graph, signature, source_bytes = str(program.graph), str(program.graph_signature), _raw(source)
    result = m2m.convert(program, (source,), backend="fx_importer", capture_trace=True, decompose=False)
    assert result.ok and result.path_taken == "fx_importer", result.diagnostics
    assert result.capture_trace["status"] == "complete", result.capture_trace["blockers"]
    assert "func.call" not in result.mlir_text
    result.module.verify()
    # Provenance names use an unregistered annotation namespace. Strip only
    # those annotations from a clone so the strict verifier checks every op.
    verification = result.module.clone()
    for op in verification.walk():
        for name in tuple(op.attributes):
            if name.startswith("prov."):
                del op.attributes[name]
    mlir = tmp_path / "linear.mlir"
    mlir.write_text(module_to_text(verification, generic=True))
    checked = subprocess.run(
        [strict_opt, str(mlir), "--mlir-disable-threading", "-o", os.devnull],
        capture_output=True, text=True, timeout=30,
    )
    assert checked.returncode == 0, checked.stderr
    matmuls = [op for op in result.module.walk() if op.name == "linalg.matmul"]
    assert len(matmuls) == 1
    matmul = matmuls[0]
    m = prod(shape[:-1])
    assert [tuple(x.type.get_shape()) for x in matmul.inputs] == [(m, k), (k, n)]
    assert tuple(matmul.results[0].type.get_shape()) == (m, n)
    expected_elem = "bf16" if dtype == torch.bfloat16 else "f32"
    assert all(str(value.type.element_type) == expected_elem for value in [*matmul.inputs, *matmul.results])
    assert all(str(op.results[0].type) == expected_elem for op in result.module.walk()
               if op.name in {"arith.mulf", "arith.addf"})
    assert not any(op.name in {"arith.extf", "arith.truncf"} for op in result.module.walk())
    transpose = matmul.inputs[1].owner
    assert transpose.name == "linalg.transpose" and tuple(transpose.inputs[0].type.get_shape()) == (n, k)
    assert list(transpose.permutation.get_values()) == [1, 0]
    returned = next(op for op in result.module.walk() if op.name == "func.return").arguments[0]
    assert tuple(returned.type.get_shape()) == (*shape[:-1], n)
    assert str(returned.type.element_type) == expected_elem
    if bias:
        assert returned.owner.name == "linalg.generic"
        assert any(op.name == "arith.addf" for op in returned.owner.walk())
    if len(shape) == 2:
        assert not any(op.name in {"tensor.collapse_shape", "tensor.expand_shape"} for op in result.module.walk())
    assert str(program.graph) == graph and str(program.graph_signature) == signature
    assert _raw(source) == source_bytes
    assert {name: (value.dtype, tuple(value.shape), _raw(value)) for name, value in program.state_dict.items()} == storage


@pytest.mark.parametrize("case", ["dynamic", "k-mismatch", "output-mismatch", "bias-mismatch"])
def test_unproved_linear_shapes_preserve_all_operands_in_refusal(case):
    shapes = [[3, 8], [5, 8], [5]]
    output_shape = [3, 5]
    if case == "dynamic":
        shapes[0][0] = builtin.DYNAMIC_INDEX
    elif case == "k-mismatch":
        shapes[1][1] = 9
    elif case == "output-mismatch":
        output_shape[1] = 6
    else:
        shapes[2][0] = 6
    block = Block(arg_types=[builtin.TensorType(builtin.bf16, shape) for shape in shapes])
    result = decompose_linear(list(block.args), {"val": torch.empty(output_shape, dtype=torch.bfloat16)}, "linear")
    assert len(result.ops) == 1 and result.ops[0].name == "func.call"
    assert tuple(result.ops[0].arguments) == tuple(block.args)


def test_bf16_rank_two_linear_keeps_matmul_and_bias_without_casts():
    """The static-shape guard keeps an ordinary BF16 linear on its linalg route."""
    from xdsl.dialects import func

    types = [builtin.TensorType(builtin.bf16, shape) for shape in ([3, 8], [5, 8], [5])]
    block = Block(arg_types=types)
    result = decompose_linear(list(block.args), {"val": torch.empty(3, 5, dtype=torch.bfloat16)}, "linear")
    block.add_ops(result.ops)
    block.add_op(func.ReturnOp(result.result))
    function = func.FuncOp.from_region("linear", types, [result.result.type], Region(block))
    function.verify()
    operations = list(function.walk())
    assert not any(op.name in {"func.call", "arith.extf", "arith.truncf",
                              "tensor.collapse_shape", "tensor.expand_shape"} for op in operations)
    matmuls = [op for op in operations if op.name == "linalg.matmul"]
    assert len(matmuls) == 1
    matmul = matmuls[0]
    assert matmul.inputs[0] is block.args[0]
    assert matmul.inputs[1].owner.name == "linalg.transpose"
    assert matmul.inputs[1].owner.inputs[0] is block.args[1]
    assert all(value.type.element_type == builtin.bf16 for value in [*matmul.inputs, *matmul.results])
    assert result.result.type == builtin.TensorType(builtin.bf16, [3, 5])
    assert result.result.owner.name == "linalg.generic"
    assert block.args[2] in result.result.owner.inputs
    assert any(op.name == "arith.addf" for op in result.result.owner.walk())


@pytest.mark.parametrize("mixed_operand", ["weight", "bias"])
def test_mixed_dtype_linear_preserves_operands_in_opaque_refusal(mixed_operand):
    """Unproved mixed precision remains explicit instead of inserting a cast."""
    types = [builtin.TensorType(builtin.bf16, [3, 8]),
             builtin.TensorType(builtin.f32 if mixed_operand == "weight" else builtin.bf16, [5, 8]),
             builtin.TensorType(builtin.f32 if mixed_operand == "bias" else builtin.bf16, [5])]
    block = Block(arg_types=types)
    result = decompose_linear(list(block.args), {"val": torch.empty(3, 5, dtype=torch.bfloat16)}, "linear")
    assert len(result.ops) == 1
    call = result.ops[0]
    assert call.name == "func.call" and call.callee.root_reference.data == "aten_linear"
    assert tuple(call.arguments) == tuple(block.args)
    assert [arg.type for arg in call.arguments] == types
    assert tuple(result.result.type.get_shape()) == (3, 5)
