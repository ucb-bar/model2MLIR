"""Verified EP names remain authoritative over FX's fresh builtin namespace."""

import copy

import pytest
import torch

import m2m
from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization
from m2m.capture.trace import snapshot_exported_program
from m2m.capture.trace_copy import copy_trace_program


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("traced", [False, True])
@pytest.mark.parametrize("as_export", [False, True])
def test_direct_linear_builtin_input_preserves_valid_signature(
    quantized, traced, as_export
):
    model = torch.nn.Linear(4, 4).eval().requires_grad_(False)
    quant = QuantizationConfig(scheme="int8_dyn_act_int8_weight") if quantized else None
    if quantized:
        pytest.importorskip("torchao")
        model = apply_quantization(model, quant)
    inputs = (torch.randn(1, 4),)
    ep = torch.export.export(model, inputs, strict=True)
    names = [(node.name, node.target) for node in ep.graph.nodes]
    assert ("input", "input") in names
    signature = copy.deepcopy(ep.graph_signature)
    states = dict(ep.state_dict)
    result = m2m.convert(
        ep if as_export else model,
        inputs,
        capture_trace=traced,
        backend="fx_importer",
        quantization=quant,
        quantization_preapplied=quantized,
    )
    assert result.ok, result.diagnostics
    assert [(node.name, node.target) for node in ep.graph.nodes] == names
    assert ep.graph_signature == signature
    assert all(ep.state_dict[name] is value for name, value in states.items())
    if traced and not quantized:
        assert result.capture_trace["status"] == "complete"


def test_builtin_and_suffixed_names_copy_without_signature_collision():
    class Model(torch.nn.Module):
        def forward(self, input, input_1):
            return input.sin() + input_1.cos()

    ep = torch.export.export(Model(), (torch.ones(2), torch.zeros(2)), strict=True)
    # Torch may already normalize parameter names; author a verified rename to
    # the legal pair to exercise an existing `input_1` alongside builtin input.
    from torch.export.graph_signature import TensorArgument

    formal = [node for node in ep.graph.nodes if node.op == "placeholder"]
    for node, spec, name in zip(
        formal, ep.graph_signature.input_specs, ("input", "input_1"), strict=True
    ):
        node.name = name
        node.target = name
        spec.arg = TensorArgument(name)
    ep.graph_module.recompile()
    ep.validate()
    snapshot = snapshot_exported_program(ep, stage="quantized")
    before = {node.name: copy.deepcopy(node.meta["custom"]) for node in ep.graph.nodes}
    owned = copy_trace_program(ep)
    assert snapshot_exported_program(owned, stage="quantized") == snapshot
    assert ep.graph_signature == owned.graph_signature
    assert [node.name for node in owned.graph.nodes if node.op == "placeholder"] == [
        "input",
        "input_1",
    ]
    assert before == {node.name: node.meta["custom"] for node in ep.graph.nodes}
    assert torch.equal(
        owned.module()(torch.ones(2), torch.zeros(2)),
        ep.module()(torch.ones(2), torch.zeros(2)),
    )


def test_hidden_tensor_in_opaque_metadata_is_refused_before_any_deepcopy():
    class Box:
        copied = False

        def __init__(self, tensor):
            self.tensor = tensor

        def __deepcopy__(self, memo):
            self.copied = True
            raise AssertionError("opaque deepcopy must never execute")

    model = torch.nn.Linear(4, 4).eval()
    ep = torch.export.export(model, (torch.ones(1, 4),), strict=True)
    node = next(iter(ep.graph.nodes))
    box = Box(model.weight)
    node.meta["custom"] = {"opaque_box": box}
    with pytest.raises(ValueError, match="opaque trace metadata"):
        copy_trace_program(ep)
    assert not box.copied and box.tensor is model.weight


def test_core_tracing_import_does_not_require_optional_annotation_capability():
    import os
    import subprocess
    import sys

    code = """
import builtins
import torch
actual_import = builtins.__import__
def import_without_optional_annotation(name, globals=None, locals=None, fromlist=(), level=0):
    if (name == 'torch.fx.traceback' and 'annotate' in (fromlist or ())
            and (globals or {}).get('__name__') == 'm2m.capture.dynamic_linear_trace'):
        raise ImportError('simulated missing optional source annotation capability')
    return actual_import(name, globals, locals, fromlist, level)
builtins.__import__ = import_without_optional_annotation
import m2m
from m2m.capture.dynamic_linear_trace import LinearInstrumentation, DynamicLinearLineageError
class Plain(torch.nn.Module):
    def forward(self, value): return value.sin() + 1
r = m2m.convert(Plain(), (torch.ones(2),), capture_trace=True, backend='fx_importer')
assert r.ok, r.diagnostics
try:
    LinearInstrumentation(torch.nn.Linear(4,4).eval())
except DynamicLinearLineageError as error:
    assert 'annotation support' in str(error)
else:
    raise AssertionError('missing optional annotation capability was not refused')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
