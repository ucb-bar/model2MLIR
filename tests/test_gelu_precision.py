"""GELU preserves the requested approximation and framework opmath dtype."""

import pytest
import torch
import m2m
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region

from m2m.ir.decompositions import decompose_gelu


@pytest.mark.parametrize("dtype,elem", [
    (torch.bfloat16, builtin.bf16),
    (torch.float16, builtin.f16),
    (torch.float32, builtin.f32),
])
@pytest.mark.parametrize("mode", ["none", "tanh"])
def test_gelu_mode_and_opmath(dtype, elem, mode):
    t = builtin.TensorType(elem, [11])
    block = Block(arg_types=[t])
    result = decompose_gelu(
        [block.args[0]],
        {"val": torch.empty(11, dtype=dtype), "_fx_kwargs": {"approximate": mode}},
        "gelu",
    )
    block.add_ops(result.ops)
    block.add_op(func.ReturnOp(result.result))
    module = builtin.ModuleOp(
        [func.FuncOp("forward", ([t], [t]), region=Region(block))]
    )
    module.verify()
    names = [op.name for op in module.walk()]
    assert names.count("math.tanh") == (mode == "tanh")
    assert names.count("math.erf") == (mode == "none")
    for op in module.walk():
        if op.name in {"arith.addf", "arith.mulf", "math.erf", "math.tanh"}:
            assert op.results[0].type == builtin.f32
    assert names.count("arith.truncf") == (dtype != torch.float32)
    assert result.result.type == t


@pytest.mark.parametrize("mode", ["none", "tanh"])
def test_capture_preserves_gelu_mode(mode):
    class Gelu(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.gelu(x, approximate=mode)

    result = m2m.convert(
        Gelu(), (torch.randn(11, dtype=torch.bfloat16),), backend="fx_importer"
    )
    assert result.ok, result.diagnostics
    assert ("math.tanh" in result.mlir_text) == (mode == "tanh")
    assert ("math.erf" in result.mlir_text) == (mode == "none")
