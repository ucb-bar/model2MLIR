"""Zero-result source effects must not disappear through the tensor importer."""

from __future__ import annotations

import pytest
import torch

import m2m
from m2m.ir.finite_checks import assert_async


class Guarded(torch.nn.Module):
    def forward(self, x):
        torch._assert_async(torch.isfinite(x).all(), "finite source input required")
        return x + 1


def test_actual_export_preserves_guard_as_ordered_standard_ir():
    result = m2m.convert(
        Guarded(), (torch.ones(4),), backend="fx_importer", capture_trace=True
    )
    assert result.ok, result.diagnostics
    assert (
        "cf.assert" in result.mlir_text
        and "finite source input required" in result.mlir_text
    )
    assert result.capture_trace["status"] == "complete", result.capture_trace[
        "blockers"
    ]


@pytest.mark.parametrize(
    "bad",
    [
        "wrong_dtype",
        "not_scalar",
        "missing_input",
        "extra_input",
        "extra_kwarg",
        "bad_message",
        "claimed_result",
        "multiple_results",
        "wrong_source",
        "wrong_ssa",
    ],
)
def test_assert_source_controls_and_result_contract_refuse(bad):
    from xdsl.dialects.builtin import Float32Type, TensorType, i1
    from xdsl.ir import Block

    ep = torch.export.export(Guarded(), (torch.ones(4),), strict=True)
    node = next(
        n for n in ep.graph.nodes if n.target is torch.ops.aten._assert_async.msg
    )
    typ = TensorType(i1, [])
    value = Block(arg_types=[typ]).args[0]
    meta = dict(node.meta)
    meta.update(
        _fx_args=tuple(node.args),
        _fx_kwargs=dict(node.kwargs),
        _fx_ssa_args={0: value},
        _aten_target="aten._assert_async.msg",
    )
    operands = [value]
    if bad == "wrong_dtype":
        value = Block(arg_types=[TensorType(Float32Type(), [])]).args[0]
        operands = [value]
        meta["_fx_ssa_args"] = {0: value}
    elif bad == "not_scalar":
        value = Block(arg_types=[TensorType(i1, [1])]).args[0]
        operands = [value]
        meta["_fx_ssa_args"] = {0: value}
    elif bad == "missing_input":
        operands = []
    elif bad == "extra_input":
        operands.append(value)
    elif bad == "extra_kwarg":
        meta["_fx_kwargs"] = {"unknown": True}
    elif bad == "bad_message":
        meta["_fx_args"] = (node.args[0], object())
    elif bad == "claimed_result":
        meta["val"] = torch.tensor(True)
    elif bad == "multiple_results":
        meta["val"] = (torch.tensor(True), torch.tensor(True))
    elif bad == "wrong_source":
        meta["_aten_target"] = "aten._assert_async.default"
    elif bad == "wrong_ssa":
        meta["_fx_ssa_args"] = {1: value}
    with pytest.raises(ValueError):
        assert_async(operands, meta, "guard")


def test_guarded_export_throws_for_nonfinite_without_changing_finite_math():
    model = Guarded()
    x = torch.tensor([-0.0, torch.finfo(torch.float32).max, -1.0, 3.0])
    ep = torch.export.export(model, (x,), strict=True)
    assert torch.equal(ep.module()(x).view(torch.uint8), model(x).view(torch.uint8))
    for nonfinite in (float("inf"), float("-inf"), float("nan")):
        bad = x.clone()
        bad[0] = nonfinite
        with pytest.raises(RuntimeError, match="finite source input"):
            ep.module()(bad)
