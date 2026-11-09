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


class GuardedWithoutMessage(torch.nn.Module):
    def forward(self, x):
        torch.ops.aten._assert_async.default(torch.isfinite(x).all())
        return x + 1


@pytest.mark.parametrize("model_type", [Guarded, GuardedWithoutMessage])
def test_actual_export_preserves_guard_as_ordered_standard_ir(model_type):
    result = m2m.convert(
        model_type(), (torch.ones(4),), backend="fx_importer", capture_trace=True
    )
    assert result.ok, result.diagnostics
    message = (
        "finite source input required" if model_type is Guarded
        else "source tensor assertion failed"
    )
    assert (
        "cf.assert" in result.mlir_text
        and message in result.mlir_text
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


@pytest.mark.parametrize("model_type", [Guarded, GuardedWithoutMessage])
def test_guarded_export_throws_for_nonfinite_without_changing_finite_math(model_type):
    model = model_type()
    x = torch.tensor([-0.0, torch.finfo(torch.float32).max, -1.0, 3.0])
    ep = torch.export.export(model, (x,), strict=True)
    assert torch.equal(ep.module()(x).view(torch.uint8), model(x).view(torch.uint8))
    for nonfinite in (float("inf"), float("-inf"), float("nan")):
        bad = x.clone()
        bad[0] = nonfinite
        with pytest.raises(RuntimeError):
            ep.module()(bad)


def test_default_overload_cannot_accept_a_message_argument():
    from xdsl.dialects.builtin import TensorType, i1
    from xdsl.ir import Block

    ep = torch.export.export(GuardedWithoutMessage(), (torch.ones(4),), strict=True)
    node = next(n for n in ep.graph.nodes
                if n.target is torch.ops.aten._assert_async.default)
    value = Block(arg_types=[TensorType(i1, [])]).args[0]
    meta = dict(node.meta)
    meta.update(_fx_args=(node.args[0], "unproved message"), _fx_kwargs={},
                _fx_ssa_args={0: value}, _aten_target="aten._assert_async.default")
    with pytest.raises(ValueError, match="message arity"):
        assert_async([value], meta, "guard")


@pytest.mark.parametrize("defaults", [True, False])
def test_default_guard_preparation_preserves_original_and_global_effects(defaults):
    from torch._library.utils import is_impure

    from m2m.capture.torch_export import _prepare_exported_program

    x = torch.ones(4)
    ep = torch.export.export(GuardedWithoutMessage(), (x,), strict=True)
    original = str(ep.graph)
    effects = tuple(is_impure(op) for op in (
        torch.ops.aten._assert_async.default, torch.ops.aten._assert_async.msg))
    prepared, _ = _prepare_exported_program(
        ep, run_default_decompositions=defaults, export_decomposition_table=None
    )
    assert str(ep.graph) == original
    assert tuple(is_impure(op) for op in (
        torch.ops.aten._assert_async.default, torch.ops.aten._assert_async.msg)) == effects
    assert sum(n.target is torch.ops.aten._assert_async.msg
               for n in prepared.graph.nodes if n.op == "call_function") == 1
    assert torch.equal(prepared.module()(x), ep.module()(x))
    for number in (float("nan"), float("inf"), float("-inf")):
        bad = x.clone()
        bad[0] = number
        with pytest.raises(RuntimeError):
            prepared.module()(bad)


def test_source_guard_decomposition_override_refuses():
    from m2m.capture.torch_export import _prepare_exported_program

    ep = torch.export.export(GuardedWithoutMessage(), (torch.ones(4),), strict=True)
    with pytest.raises(ValueError, match="cannot be overridden"):
        _prepare_exported_program(
            ep, run_default_decompositions=True,
            export_decomposition_table={torch.ops.aten._assert_async.default: lambda _: None}
        )


@pytest.mark.parametrize("bad", [torch.tensor(1.), torch.ones(1, dtype=torch.bool)])
def test_default_preparation_refuses_nonboolean_or_nonscalar(bad):
    from m2m.ir.finite_checks import preserve_default_assert

    with pytest.raises(ValueError, match="CPU scalar bool"):
        preserve_default_assert(bad)
