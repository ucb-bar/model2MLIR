"""Exported precision materializations retain provider mode restrictions."""

import copy
from types import SimpleNamespace

import pytest
import torch

import m2m
from m2m.capture import trace


@pytest.mark.parametrize("provider", ["legacy", "modern", "missing", "missing_api"])
def test_public_mode_helper_resolution_is_capability_based(monkeypatch, provider):
    seen = []

    def helper(module):
        seen.append(module)

    def imported(namespace):
        if namespace == "torch.ao.quantization":
            return (
                SimpleNamespace(allow_exported_model_train_eval=helper)
                if provider == "legacy"
                else SimpleNamespace()
            )
        assert namespace == "torchao.quantization.pt2e"
        if provider == "modern":
            return SimpleNamespace(allow_exported_model_train_eval=helper)
        if provider == "missing_api":
            return SimpleNamespace()
        raise ModuleNotFoundError("no optional provider", name="torchao")

    monkeypatch.setattr("importlib.import_module", imported)
    selected = trace._exported_train_eval_helper()
    assert selected is (None if provider in {"missing", "missing_api"} else helper)
    assert not seen  # Resolution never changes the mode or touches a module.


@pytest.mark.parametrize("problem", ["broken_import", "uncallable"])
def test_broken_installed_provider_is_not_silently_skipped(monkeypatch, problem):
    def imported(namespace):
        if problem == "broken_import":
            raise ModuleNotFoundError(
                "missing transitive dependency", name="unrelated_dependency"
            )
        return SimpleNamespace(allow_exported_model_train_eval=1)

    monkeypatch.setattr("importlib.import_module", imported)
    with pytest.raises(
        ModuleNotFoundError if problem == "broken_import" else TypeError
    ):
        trace._exported_train_eval_helper()


class ModeSource(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.5, 1.5], dtype=torch.bfloat16))
        self.register_buffer("offset", torch.tensor([0.25, -0.5], dtype=torch.bfloat16))

    def forward(self, value):
        scaled = value * self.weight + self.offset
        dropped = torch.nn.functional.dropout(scaled, p=0.5, training=self.training)
        return dropped * (2.0 if self.training else 3.0)


def _execute_with_seed(module, inputs):
    with torch.random.fork_rng():
        torch.manual_seed(73)
        return module(*inputs).detach().clone()


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("retarget", [False, True])
def test_real_materialization_preserves_original_mode_graph_state_and_guards(
    training, retarget
):
    model = ModeSource().train(training)
    inputs = (torch.arange(8, dtype=torch.bfloat16).reshape(4, 2),)
    exported = torch.export.export(model, inputs)
    graph = str(exported.graph)
    signature = str(exported.graph_signature)
    source_parameter = exported.state_dict["weight"]
    source_bytes = source_parameter.detach().contiguous().view(torch.uint8).clone()
    source_storage = source_parameter.untyped_storage().data_ptr()
    source_buffer = exported.state_dict["offset"]
    buffer_storage = source_buffer.untyped_storage().data_ptr()
    buffer_bytes = source_buffer.contiguous().view(torch.uint8).clone()
    source_method = model.train.__func__
    source_output = _execute_with_seed(model, inputs)
    expected_model = copy.deepcopy(model).to(torch.float32)

    staged, converted_inputs, original, receipt = m2m.materialize_frontend_precision(
        exported,
        inputs,
        dtype=torch.float32,
        retarget_float_dtype_arguments=retarget,
    )
    expected = _execute_with_seed(expected_model, converted_inputs)
    torch.testing.assert_close(
        _execute_with_seed(staged, converted_inputs), expected, rtol=0, atol=0
    )
    assert staged.weight.dtype is torch.float32
    assert staged.weight.untyped_storage().data_ptr() != source_storage
    assert staged.offset.untyped_storage().data_ptr() != buffer_storage
    assert model.training is training and model.train.__func__ is source_method
    assert exported.state_dict["weight"] is source_parameter
    assert source_parameter.untyped_storage().data_ptr() == source_storage
    assert exported.state_dict["offset"] is source_buffer
    assert source_buffer.untyped_storage().data_ptr() == buffer_storage
    assert torch.equal(source_buffer.contiguous().view(torch.uint8), buffer_bytes)
    assert torch.equal(
        source_parameter.detach().contiguous().view(torch.uint8), source_bytes
    )
    assert str(exported.graph) == graph and str(exported.graph_signature) == signature
    torch.testing.assert_close(
        _execute_with_seed(model, inputs), source_output, rtol=0, atol=0
    )
    assert receipt["original_graph_sha256"] == original["sha256"]

    # Neither the raw EP nor global Torch has had its mode guard replaced.
    with pytest.raises(NotImplementedError):
        exported.module().eval()
    with pytest.raises((AssertionError, RuntimeError)):
        staged(torch.ones(3, 2))

    converted = m2m.convert(
        staged,
        converted_inputs,
        backend="fx_importer",
        capture_trace=True,
        original_frontend_snapshot=original,
    )
    assert converted.ok, converted.diagnostics
    assert converted.capture_trace["graphs"]["original"]["sha256"] == original["sha256"]

    if trace._exported_train_eval_helper() is None:
        with pytest.raises(NotImplementedError):
            staged.eval()
        return
    # Use the actual provider API: its documented limited graph rewrites do not
    # recapture a Python branch specialized in the original export.
    staged.eval()
    branch = 2.0 if training else 3.0
    expected_eval = (converted_inputs[0] * staged.weight + staged.offset) * branch
    torch.testing.assert_close(staged(*converted_inputs), expected_eval, rtol=0, atol=0)
    staged.train()
    expected_train = ModeSource().float().train()
    expected_train.weight.detach().copy_(staged.weight)
    expected_train.offset.copy_(staged.offset)
    # The captured Python branch remains unchanged, even after train().
    expected_train_value = _execute_with_seed(expected_train, converted_inputs) * (
        branch / 2.0
    )
    torch.testing.assert_close(
        _execute_with_seed(staged, converted_inputs),
        expected_train_value,
        rtol=0,
        atol=0,
    )
    assert model.training is training
    assert torch.equal(
        source_parameter.detach().contiguous().view(torch.uint8), source_bytes
    )


@pytest.mark.parametrize("retarget", [False, True])
def test_core_only_materialization_retains_torch_mode_restrictions(
    monkeypatch, retarget
):
    monkeypatch.setattr(trace, "_exported_train_eval_helper", lambda: None)
    model = torch.nn.Linear(4, 3).eval()
    inputs = (torch.ones(2, 4),)
    staged, converted_inputs, _, _ = m2m.materialize_frontend_precision(
        model,
        inputs,
        dtype=torch.float32,
        retarget_float_dtype_arguments=retarget,
    )
    torch.testing.assert_close(
        staged(*converted_inputs), model(*inputs), rtol=0, atol=0
    )
    with pytest.raises(NotImplementedError):
        staged.eval()
    with pytest.raises(NotImplementedError):
        staged.train()
    # Ordinary module capture requests eval(), so it must still refuse.
    assert not m2m.convert(staged, converted_inputs, backend="fx_importer").ok
    # Public EP intake preserves the captured mode and needs no relaxed guard.
    program = torch.export.export(staged, converted_inputs)
    assert m2m.convert(program, converted_inputs, backend="fx_importer").ok


def test_mode_adapter_refuses_a_source_module():
    with pytest.raises(TypeError, match="owned GraphModule"):
        trace._allow_owned_exported_mode_api(torch.nn.Linear(2, 2))
