"""Semantic session artifacts preserve exported ABI indices and eager state trajectories."""
from __future__ import annotations

import numpy as np
import torch
import yaml
from torch import nn

from m2m.capture.bundle import write_multi_program_bundle, write_session_artifacts


class _StateStep(nn.Module):
    def forward(self, observation, hidden, cell):
        output = observation + hidden + cell
        return output, hidden + observation, cell + 1.0


def test_session_artifacts_resolve_inputs_after_weights_and_carry_state(tmp_path):
    model = _StateStep().eval()
    inputs = (torch.zeros(1), torch.zeros(1), torch.zeros(1))
    manifest = {
        "0": {"kind": "param", "name": "p_weight"},
        "1": {"kind": "input", "name": "observation"},
        "2": {"kind": "input", "name": "hidden"},
        "3": {"kind": "input", "name": "cell"},
    }
    session = {
        "kind": "recurrent_frames", "paper_ready": True,
        "stages": ["recurrent_step"],
        "stage_schedule": [{"name": "recurrent_step", "steps": 3,
                            "execution": "compiled_recurrent", "timed": True}],
        "parameters": {"sequence_length": 3, "batch": 1},
        "states": [
            {"name": "hidden_state", "input_index": 1, "output_index": 1},
            {"name": "cell_state", "input_index": 2, "output_index": 2},
        ],
        "streams": [{"name": "frame", "input_index": 0, "key": "frames",
                     "values": torch.tensor([[1.0], [2.0], [3.0]])}],
        "quality": {"key": "output0", "output_index": 0},
    }
    summary = write_session_artifacts(
        model, inputs, tmp_path, manifest=manifest,
        input_order={"observation": 0, "hidden": 1, "cell": 2}, session=session)
    contract = yaml.safe_load((tmp_path / "session_contract.yaml").read_text())
    assert contract["streams"][0]["input_arg"] == 1
    assert [(state["input_arg"], state["output_index"]) for state in contract["states"]] == [
        (2, 1), (3, 2)]
    assert contract["stage_schedule"] == session["stage_schedule"]
    assert contract["parameters"] == session["parameters"]
    with np.load(tmp_path / "session_goldens.npz") as goldens:
        np.testing.assert_allclose(goldens["output0"].ravel(), [1.0, 4.0, 8.0])
    assert summary == {"session_kind": "recurrent_frames", "session_steps": 3,
                       "session_states": 2, "paper_ready": True}


def test_session_artifacts_refuse_state_shape_changes(tmp_path):
    class Bad(nn.Module):
        def forward(self, observation, state):
            return observation, torch.cat([state, state])

    with np.testing.assert_raises_regex(ValueError, "changes shape"):
        write_session_artifacts(
            Bad(), (torch.zeros(1), torch.zeros(1)), tmp_path,
            manifest={"0": {"kind": "input", "name": "observation"},
                      "1": {"kind": "input", "name": "state"}},
            input_order={"observation": 0, "state": 1},
            session={"kind": "recurrent_frames", "stages": ["step"],
                     "states": [{"name": "state", "input_index": 1, "output_index": 1}],
                     "streams": [{"name": "frame", "input_index": 0, "key": "frames",
                                  "values": torch.ones(2, 1)}],
                     "quality": {"output_index": 0}})


def test_stream_free_session_uses_explicit_state_transition_steps(tmp_path):
    class Flow(nn.Module):
        def forward(self, state, timestep):
            updated = state + timestep
            return updated, timestep - 0.25

    session = {
        "kind": "action_chunk", "paper_ready": False,
        "stages": ["flow_denoise"], "steps": 4,
        "states": [
            {"name": "flow_state", "input_index": 0, "output_index": 0},
            {"name": "timestep", "input_index": 1, "output_index": 1},
        ],
        "streams": [], "quality": {"key": "actions", "output_index": 0},
    }
    summary = write_session_artifacts(
        Flow(), (torch.zeros(1), torch.ones(1)), tmp_path,
        manifest={"0": {"kind": "input", "name": "state"},
                  "1": {"kind": "input", "name": "timestep"}},
        input_order={"state": 0, "timestep": 1}, session=session)
    contract = yaml.safe_load((tmp_path / "session_contract.yaml").read_text())
    assert contract["steps"] == 4
    assert contract["streams"] == []
    assert summary["session_steps"] == 4
    with np.load(tmp_path / "session_goldens.npz") as goldens:
        np.testing.assert_allclose(goldens["actions"].ravel(), [1.0, 1.75, 2.25, 2.5])


def test_multi_program_writer_resolves_cross_stage_tuple_input_to_abi(tmp_path, monkeypatch):
    import json

    def fake_write(_model, _inputs, out, **_kwargs):
        out.mkdir(parents=True)
        # One weight precedes the runtime inputs, so tuple input 1 is ABI argument 2.
        (out / "weights.safetensors.manifest.json").write_text(json.dumps({
            "0": {"kind": "param", "name": "weight"},
            "1": {"kind": "input", "name": "tokens"},
            "2": {"kind": "input", "name": "state"},
        }))
        (out / "input_order.json").write_text(json.dumps({"tokens": 0, "state": 1}))
        return {"out": str(out)}

    monkeypatch.setattr("m2m.capture.bundle.write_bundle", fake_write)
    root = {
        "kind": "autoregressive_decode", "paper_ready": True,
        "stages": ["prefill", "decode"],
        "stage_schedule": [
            {"name": "prefill", "steps": 1, "execution": "compiled", "timed": True},
            {"name": "decode", "steps": 3, "execution": "compiled_recurrent", "timed": True},
        ],
        "parameters": {}, "states": ["state"], "quality_program": "decode",
        "bindings": [{"name": "state", "from": {"program": "prefill", "output_index": 0},
                      "to": {"program": "decode", "input_index": 1}}],
    }
    write_multi_program_bundle([
        {"name": "prefill", "model": object(), "inputs": (), "steps": 1},
        {"name": "decode", "model": object(), "inputs": (), "steps": 3},
    ], root, tmp_path)
    contract = yaml.safe_load((tmp_path / "session_contract.yaml").read_text())
    assert contract["version"] == 2
    assert contract["bindings"][0]["to"] == {"program": "decode", "input_arg": 2}
    assert contract["programs"][1] == {"name": "decode", "bundle": "stages/decode", "steps": 3}
