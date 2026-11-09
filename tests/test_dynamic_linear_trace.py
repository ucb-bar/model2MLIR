"""Source site tokens need exact callers, payloads, effects and selected outputs."""

from __future__ import annotations

import pytest
import torch

import m2m
from m2m.capture.dynamic_linear_trace import (
    DynamicLinearLineageError,
    LinearInstrumentation,
)
from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization

pytest.importorskip("torchao")


class Model(torch.nn.Module):
    def __init__(self, *, bias=True, tied=False):
        super().__init__()
        self.first = torch.nn.Linear(32, 32, bias=bias)
        self.second = torch.nn.Linear(32, 32, bias=bias)
        if tied:
            self.second.weight = self.first.weight
        self.switch = False

    def forward(self, value):
        hidden = self.first(value).relu()
        return self.second(value if self.switch else hidden) + value


def setup(*, bias=True, tied=False):
    torch.manual_seed(0)
    model = Model(bias=bias, tied=tied).eval().requires_grad_(False)
    value = torch.arange(32, dtype=torch.float32).reshape(1, 32) / 16
    return model, (value,), QuantizationConfig(scheme="int8_dyn_act_int8_weight")


@pytest.mark.parametrize("bias,tied", [(True, False), (False, False), (False, True)])
def test_actual_ao_pair_has_complete_source_correspondence_and_exact_export(bias, tied):
    model, inputs, quant = setup(bias=bias, tied=tied)
    baseline = model(*inputs).detach().clone()
    with LinearInstrumentation(model) as owner:
        assert torch.equal(model(*inputs), baseline)
        source = owner.capture_source(model, inputs)
        selected = apply_quantization(model, quant)
        exported, proof = owner.export_selected(
            source, selected, inputs, quantization=quant
        )
        result = m2m.convert(
            exported,
            inputs,
            quantization=quant,
            quantization_preapplied=True,
            original_frontend_snapshot=source.snapshot,
            backend="fx_importer",
            capture_trace=True,
        )
        assert result.ok and result.capture_trace["status"] == "complete"
        assert (
            proof["caller_graph_and_order_verified"]
            and proof["all_original_weight_encodings_verified"]
        )
        assert proof["selected_export_exact_eager_output"]
    assert (
        "forward" not in model.first.__dict__ and "forward" not in model.second.__dict__
    )


@pytest.mark.parametrize(
    "failure",
    ["caller", "weight", "bias", "source_state", "source_graph", "recipe", "callable"],
)
def test_changed_source_selected_relation_refuses(failure):
    model, inputs, quant = setup()
    with LinearInstrumentation(model) as owner:
        source = owner.capture_source(model, inputs)
        selected = apply_quantization(model, quant)
        if failure == "caller":
            selected.switch = True
        elif failure == "weight":
            selected.first.weight.qdata.view(-1)[0] += 1
        elif failure == "bias":
            selected.first.bias.view(-1)[0] += 1
        elif failure == "source_state":
            next(iter(source.exported.state_dict.values())).view(-1)[0] += 1
        elif failure == "source_graph":
            node = next(
                node
                for node in source.exported.graph.nodes
                if node.op == "call_function"
                and node.target is torch.ops.aten.relu.default
            )
            node.target = torch.ops.aten.neg.default
        elif failure == "recipe":
            quant = QuantizationConfig(
                scheme="int8_dyn_act_int8_weight", extra_args={"unknown": True}
            )
        elif failure == "callable":
            selected.first.forward = lambda value: value
        with pytest.raises(DynamicLinearLineageError):
            owner.export_selected(source, selected, inputs, quantization=quant)
        if failure == "callable":
            selected.first.forward = owner._sites["first"]["wrapper"]


def test_repeated_module_invocations_are_not_given_one_site_identity():
    class Repeat(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 32, bias=False)

        def forward(self, value):
            return self.linear(value) + self.linear(value + 1)

    model = Repeat().eval().requires_grad_(False)
    with (
        LinearInstrumentation(model) as owner,
        pytest.raises(DynamicLinearLineageError),
    ):
        owner.capture_source(model, (torch.ones(1, 32),))


def test_failed_instrumentation_is_atomic():
    model, _, _ = setup()
    model.second.train()
    with pytest.raises(DynamicLinearLineageError):
        LinearInstrumentation(model)
    assert (
        "forward" not in model.first.__dict__ and "forward" not in model.second.__dict__
    )


def test_alias_module_paths_decline_without_partial_wrappers():
    model, _, _ = setup()
    model.alias = model.first
    with pytest.raises(DynamicLinearLineageError):
        LinearInstrumentation(model)
    assert "forward" not in model.first.__dict__


def test_owner_close_restores_owned_wrappers_and_preserves_a_foreign_callable():
    model, _, _ = setup()
    owner = LinearInstrumentation(model)
    foreign = lambda value: value
    model.first.forward = foreign
    with pytest.raises(DynamicLinearLineageError, match="foreign"):
        owner.close()
    assert model.first.forward is foreign and "forward" not in model.second.__dict__


def test_actual_embedding_head_tie_is_split_only_by_the_declared_quantizer():
    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(32, 32)
            self.head = torch.nn.Linear(32, 32, bias=False)
            self.head.weight = self.embedding.weight

        def forward(self, tokens):
            return self.head(self.embedding(tokens)[:, -1, :])

    model = Tied().eval().requires_grad_(False)
    inputs = (torch.tensor([[1, 2]], dtype=torch.int64),)
    quant = QuantizationConfig(scheme="int8_dyn_act_int8_weight")
    assert model.embedding.weight is model.head.weight
    with LinearInstrumentation(model) as owner:
        source = owner.capture_source(model, inputs)
        selected = apply_quantization(model, quant)
        assert model.embedding.weight is not model.head.weight
        exported, _ = owner.export_selected(
            source, selected, inputs, quantization=quant
        )
        result = m2m.convert(
            exported,
            inputs,
            quantization=quant,
            quantization_preapplied=True,
            original_frontend_snapshot=source.snapshot,
            backend="fx_importer",
            capture_trace=True,
        )
        assert result.ok and result.capture_trace["status"] == "complete"


def test_shared_two_stage_writer_uses_one_transform_and_exact_preconverted_stages(
    tmp_path, monkeypatch
):
    import m2m.capture.torchao_pipeline as pipeline
    from m2m.capture.bundle import write_quantized_multi_program_bundle
    from m2m.capture.bundle_integrity import verify_bundle_integrity
    from m2m.capture.external_runtime import (
        ExternalRuntimeProgram,
        make_external_runtime_session,
    )

    class Stage(torch.nn.Module):
        def __init__(self, shared):
            super().__init__()
            self.shared = shared

        def forward(self, value, state):
            result = self.shared(value) + state
            return result, result

    shared, inputs, quant = setup(bias=False)
    value = inputs[0]

    def factory(selected):
        first = Stage(selected).eval()
        state = torch.zeros_like(value)
        with torch.no_grad():
            _, carried = first(value, state)
        child = {
            "kind": "test_recurrent",
            "paper_ready": False,
            "steps": 2,
            "stages": ["step"],
            "stage_schedule": [
                {
                    "name": "step",
                    "steps": 2,
                    "execution": "compiled_recurrent",
                    "timed": True,
                }
            ],
            "states": [{"name": "state", "input_index": 1, "output_index": 1}],
            "streams": [],
            "quality": {"output_index": 0},
        }
        root = {
            "kind": "test_recurrent",
            "paper_ready": False,
            "stages": ["first", "step"],
            "states": ["state"],
            "quality_program": "step",
            "stage_schedule": [
                {"name": "first", "steps": 1, "execution": "compiled", "timed": True},
                {
                    "name": "step",
                    "steps": 2,
                    "execution": "compiled_recurrent",
                    "timed": True,
                },
            ],
            "bindings": [
                {
                    "name": "state",
                    "from": {"program": "first", "output_index": 1},
                    "to": {"program": "step", "input_index": 1},
                }
            ],
        }
        return make_external_runtime_session(
            version=2,
            programs=(
                ExternalRuntimeProgram("first", first, (value, state), 1),
                ExternalRuntimeProgram(
                    "step", Stage(selected).eval(), (value / 2, carried), 2, child
                ),
            ),
            metadata=root,
        )

    source_stages = factory(shared)
    calls = []
    actual = pipeline.apply_quantization

    def transform(module):
        calls.append(module)
        return actual(module, quant)

    with LinearInstrumentation(shared) as owner:
        originals = {
            program.name: owner.capture_source(program.module, program.inputs)
            for program in source_stages.programs
        }

        def converter(program, out):
            out.mkdir(parents=True, exist_ok=True)
            source = originals[program.name]
            exported, _ = owner.export_selected(
                source, program.module, program.inputs, quantization=quant
            )
            return m2m.convert(
                exported,
                program.inputs,
                quantization=quant,
                quantization_preapplied=True,
                original_frontend_snapshot=source.snapshot,
                capture_trace=True,
                backend="fx_importer",
                weights_path=str(out / "weights.safetensors"),
            )

        output = tmp_path / "capture"
        result = write_quantized_multi_program_bundle(
            shared,
            transform,
            factory,
            output,
            shared_programs=("first", "step"),
            quantization={"scheme": quant.scheme, "transform_once": True},
            quant=quant,
            original_frontend_by_program={
                name: source.snapshot for name, source in originals.items()
            },
            shared_program_converter=converter,
            capture_trace=True,
        )
    assert calls == [shared] and len(result["programs"]) == 2
    verify_bundle_integrity(output)
    import json

    import numpy as np

    for name in ["first", "step"]:
        stage = output / "stages" / name
        assert (
            json.loads((stage / "frontend-trace.json").read_text())["status"]
            == "complete"
        )
        assert set(np.load(stage / "goldens.npz").files) == {"Y0", "Y1"}


def test_wrong_caller_never_leaks_partially_published_source_identities(monkeypatch):
    from m2m.capture.trace import graph_relation, snapshot_exported_program

    model, inputs, quant = setup()
    captured = []
    actual = torch.export.export

    def observed(*args, **kwargs):
        result = actual(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(torch.export, "export", observed)
    with LinearInstrumentation(model) as owner:
        source = owner.capture_source(model, inputs)
        selected = apply_quantization(model, quant)
        selected.switch = True
        with pytest.raises(DynamicLinearLineageError, match="caller"):
            owner.export_selected(source, selected, inputs, quantization=quant)
        relation = graph_relation(
            source.snapshot, snapshot_exported_program(captured[-1], stage="quantized")
        )
        assert relation["status"] == "diagnostic"
        assert all(
            not any(
                identity.startswith("g:original:")
                for identity in row.get("origin_node_ids", [])
            )
            for row in snapshot_exported_program(captured[-1], stage="quantized")[
                "nodes"
            ]
        )


def test_identical_foreign_model_does_not_grant_source_ownership():
    first, inputs, quant = setup()
    second, _, _ = setup()
    with LinearInstrumentation(first) as owner, LinearInstrumentation(second) as other:
        source = other.capture_source(second, inputs)
        selected = apply_quantization(first, quant)
        with pytest.raises(DynamicLinearLineageError, match="same source modules"):
            owner.export_selected(source, selected, inputs, quantization=quant)


@pytest.mark.parametrize("failure", ["none", "untraced", "wrong_weights", "exception"])
def test_shared_converter_failure_never_falls_back(tmp_path, failure):
    from m2m.capture.bundle import write_quantized_multi_program_bundle
    from m2m.capture.external_runtime import (
        ExternalRuntimeProgram,
        make_external_runtime_session,
    )

    class Program(torch.nn.Module):
        def __init__(self, shared):
            super().__init__()
            self.shared = shared

        def forward(self, value):
            return self.shared(value)

    shared, inputs, quant = setup(bias=False)
    calls = []

    def factory(selected):
        return make_external_runtime_session(
            version=2,
            programs=(ExternalRuntimeProgram("one", Program(selected), inputs, 1),),
            metadata={
                "kind": "test",
                "paper_ready": False,
                "stages": ["one"],
                "states": [],
                "quality_program": "one",
                "bindings": [],
                "stage_schedule": [
                    {"name": "one", "steps": 1, "execution": "compiled", "timed": True}
                ],
            },
        )

    def converter(program, out):
        calls.append(program.name)
        if failure == "none":
            return None
        if failure == "exception":
            raise RuntimeError("converter failed")
        out.mkdir(parents=True, exist_ok=True)
        weights = (
            out / "weights.safetensors"
            if failure == "untraced"
            else tmp_path / "different" / "weights.safetensors"
        )
        weights.parent.mkdir(parents=True, exist_ok=True)
        return m2m.convert(
            program.module,
            program.inputs,
            quantization=quant,
            quantization_preapplied=True,
            backend="fx_importer",
            capture_trace=failure != "untraced",
            weights_path=str(weights),
        )

    with pytest.raises((ValueError, RuntimeError)):
        write_quantized_multi_program_bundle(
            shared,
            lambda module: apply_quantization(module, quant),
            factory,
            tmp_path / "bundle",
            shared_programs=("one",),
            quantization={"scheme": quant.scheme},
            quant=quant,
            capture_trace=True,
            shared_program_converter=converter,
        )
    assert calls == ["one"]
