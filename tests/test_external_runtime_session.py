"""External runtimes receive exact modules, inputs, and an executable semantic driver plan."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from m2m.capture.external_runtime import (
    ExternalRuntimeInput,
    ExternalRuntimeOutput,
    ExternalRuntimeProgram,
    external_runtime_session,
    make_external_runtime_session,
)


class _StateStep(nn.Module):
    def forward(self, observation, state):
        updated = state + observation
        return updated * 2.0, updated


def test_v1_protocol_normalizes_stream_state_reset_and_logical_stages():
    model = _StateStep().eval()
    inputs = (torch.zeros(1), torch.ones(1))
    stream = torch.tensor([[1.0], [2.0], [3.0]])
    session = {
        "kind": "recurrent_frames", "paper_ready": True,
        # These are compiler-internal logical stages, not separate callable programs.
        "stages": ["encode", "recur", "predict"],
        "stage_schedule": [
            {"name": name, "steps": 3, "execution": "compiled_recurrent", "timed": True}
            for name in ("encode", "recur", "predict")
        ],
        "states": [{"name": "hidden", "input_index": 1, "output_index": 1}],
        "streams": [{"name": "frame", "input_index": 0, "values": stream}],
        "quality": {"output_index": 0},
        "parameters": {"sequence_length": 3},
        "provenance": {"source": "test"},
    }

    exported = external_runtime_session(model, inputs, session=session)

    assert exported.version == 1
    assert exported.kind == "recurrent_frames"
    assert exported.paper_ready is True
    assert exported.observations == 3
    assert len(exported.programs) == 1
    assert exported.programs[0].name == "forward"
    assert exported.programs[0].module is model
    assert exported.programs[0].inputs[1] is inputs[1]
    assert [(call.program, call.repeats, call.cadence) for call in exported.execution_schedule] == [
        ("forward", 3, "per_observation")]
    assert exported.streams[0].values is stream
    assert exported.streams[0].target == ExternalRuntimeInput("forward", 0)
    assert exported.routes[0].source == ExternalRuntimeOutput("forward", 1)
    assert exported.routes[0].target == ExternalRuntimeInput("forward", 1)
    assert [(binding.target.input_index, binding.kind) for binding in exported.input_bindings] == [
        (0, "stream"), (1, "state")]
    assert exported.observation_output == ExternalRuntimeOutput("forward", 0)
    assert exported.final_output == ExternalRuntimeOutput("forward", 0)
    assert exported.reset == "restore_initial_inputs"
    assert [row["name"] for row in exported.stage_schedule] == ["encode", "recur", "predict"]


class _Prefix(nn.Module):
    def forward(self, value):
        return value + 1.0


class _Step(nn.Module):
    def forward(self, observation, state):
        updated = observation + state
        return updated, updated


class _Final(nn.Module):
    def forward(self, state):
        return state[:, :1]


def _v2_capture():
    prefix, step, final = _Prefix(), _Step(), _Final()
    observations = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    step_session = {
        "kind": "generic_recurrent", "steps": 2,
        "states": [{"name": "state", "input_index": 1, "output_index": 1}],
        "streams": [{"name": "observation", "input_index": 0, "values": observations}],
        "quality": {"output_index": 0},
    }
    programs = (
        ExternalRuntimeProgram("prefix", prefix, (torch.zeros(1, 2),), 1),
        ExternalRuntimeProgram(
            "step", step, (observations[0], torch.zeros(1, 2)), 2, step_session),
        ExternalRuntimeProgram("final", final, (torch.zeros(1, 2),), 1),
    )
    root = {
        "kind": "generic_recurrent", "paper_ready": True,
        "stages": ["prefix", "step", "final"],
        "stage_schedule": [
            {"name": "prefix", "steps": 1, "timed": True},
            {"name": "step", "steps": 2, "timed": True},
            {"name": "final", "steps": 1, "timed": True},
        ],
        "quality_program": "step",
        "bindings": [
            {"name": "initial_state", "from": {"program": "prefix", "output_index": 0},
             "to": {"program": "step", "input_index": 1}},
            {"name": "final_state", "from": {"program": "step", "output_index": 0},
             "to": {"program": "final", "input_index": 0}},
        ],
    }
    return make_external_runtime_session(version=2, programs=programs, metadata=root)


def test_v2_protocol_has_explicit_before_observation_after_schedule_and_all_inputs():
    exported = _v2_capture()
    assert exported.observations == 2
    assert [(call.program, call.repeats, call.cadence) for call in exported.execution_schedule] == [
        ("prefix", 1, "once_before_observations"),
        ("step", 2, "per_observation"),
        ("final", 1, "once_after_observations"),
    ]
    assert exported.observation_output == ExternalRuntimeOutput("step", 0)
    assert exported.final_output == ExternalRuntimeOutput("final", 0)
    assert len(exported.input_bindings) == 4
    assert [binding.kind for binding in exported.input_bindings] == [
        "static", "stream", "state", "state"]
    # Cross-stage initial state, recurrent update, and final neural-output route are all explicit.
    assert [(route.source.program, route.target.program) for route in exported.routes] == [
        ("step", "step"), ("prefix", "step"), ("step", "final")]
    assert exported.bundle_programs()[1]["model"] is exported.programs[1].module


def test_generic_dispatch_uses_deferred_capability_without_workload_names():
    expected = _v2_capture()

    class Deferred:
        def external_runtime_session(self):
            return expected

    assert external_runtime_session(Deferred(), ()) is expected


def test_protocol_rejects_stream_shapes_that_differ_from_target_input():
    model = _StateStep()
    session = {
        "kind": "bad", "stages": ["forward"],
        "stage_schedule": [{"name": "forward", "steps": 2}],
        "states": [{"name": "state", "input_index": 1, "output_index": 1}],
        "streams": [{"name": "bad", "input_index": 0, "values": torch.zeros(2, 3)}],
        "quality": {"output_index": 0},
    }
    with pytest.raises(ValueError, match="stream element shape"):
        external_runtime_session(model, (torch.zeros(1), torch.zeros(1)), session=session)
