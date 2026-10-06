"""One transformed shared model supplies all quantized session stages."""

import hashlib
import json

import pytest
import torch
from torch import nn

from m2m.capture.bundle import write_multi_program_bundle, write_quantized_multi_program_bundle
from m2m.capture.bundle_integrity import verify_bundle_integrity
from m2m.capture.external_runtime import ExternalRuntimeProgram, make_external_runtime_session
from m2m.capture.torchao_pipeline import QuantizationConfig


class Shared(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)

    def forward(self, value):
        return self.linear(value)


class Prefix(nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared

    def forward(self, value):
        return self.shared(value)


class Step(nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared

    def forward(self, context, state):
        result = self.shared(context) + state
        return result, result


class Final(nn.Module):
    def forward(self, state):
        return state


def _session(shared, *, bad_route=False):
    value = torch.ones(1, 2)
    prefix = Prefix(shared).eval()
    context = prefix(value).detach()
    state = torch.zeros(1, 2)
    child = {"kind": "synthetic_recurrent", "paper_ready": False,
             "steps": 2, "stages": ["step"],
             "stage_schedule": [{"name": "step", "steps": 2,
                                 "execution": "compiled_recurrent", "timed": True}],
             "states": [{"name": "state", "input_index": 1, "output_index": 1}],
             "streams": [], "quality": {"output_index": 0}}
    programs = (
        ExternalRuntimeProgram("prefix", prefix, (value,), 1),
        ExternalRuntimeProgram("step", Step(shared).eval(), (context, state), 2, child),
        ExternalRuntimeProgram("final", Final().eval(),
                               (torch.zeros(1, 3 if bad_route else 2),), 1),
    )
    root = {
        "kind": "synthetic_recurrent", "paper_ready": False,
        "stages": [program.name for program in programs],
        "quality_program": "step", "states": ["state"],
        "stage_schedule": [
            {"name": program.name, "steps": program.steps, "execution": "compiled", "timed": True}
            for program in programs
        ],
        "bindings": [
            {"name": "context", "from": {"program": "prefix", "output_index": 0},
             "to": {"program": "step", "input_index": 0}},
            {"name": "final_state", "from": {"program": "step", "output_index": 0},
             "to": {"program": "final", "input_index": 0}},
        ],
    }
    return make_external_runtime_session(version=2, programs=programs, metadata=root)


def test_multi_program_bundle_reuses_exact_stage_conversions(tmp_path, monkeypatch):
    import m2m

    session = _session(Shared().eval())
    programs = session.bundle_programs()
    out = tmp_path / "reused"
    for program in programs:
        stage = out / "stages" / program["name"]
        stage.mkdir(parents=True)
        program["conversion_result"] = m2m.convert(
            program["model"], program["inputs"], backend="fx_importer",
            weights_path=str(stage / "weights.safetensors"), capture_trace=True,
        )

    def refuse_second_conversion(*args, **kwargs):
        raise AssertionError("a prepared stage must not be exported again")

    monkeypatch.setattr(m2m, "convert", refuse_second_conversion)
    summary = write_multi_program_bundle(
        programs, dict(session.metadata), out, capture_trace=True,
    )
    assert summary["n_programs"] == len(programs)
    verify_bundle_integrity(out)
    for program in programs:
        saved = (out / "stages" / program["name"] / "model.mlir").read_text()
        assert saved == program["conversion_result"].mlir_text


def test_quantized_session_binds_one_shared_model_and_stage_receipts(tmp_path):
    calls = []

    def quantize_once(model):
        calls.append(model)
        with torch.no_grad():
            weight = model.linear.weight
            weight.copy_(torch.quantize_per_tensor(weight, 0.125, 0, torch.qint8).dequantize())
        return model

    out = tmp_path / "session"
    result = write_quantized_multi_program_bundle(
        Shared().eval(), quantize_once, _session, out,
        shared_programs=("prefix", "step"),
        quantization={"kind": "synthetic_weight_roundtrip", "scale": 0.125},
        quant=QuantizationConfig(scheme="int8_weight_only"),
        capture_trace=True,
    )
    assert len(calls) == 1
    assert result["n_programs"] == 3
    verify_bundle_integrity(out)
    import yaml

    contract = yaml.safe_load((out / "session_contract.yaml").read_text())
    pointer = contract["quantized_session_binding"]
    binding_bytes = (out / pointer["path"]).read_bytes()
    assert pointer["sha256"] == hashlib.sha256(binding_bytes).hexdigest()
    binding = json.loads(binding_bytes)
    assert binding["shared_programs"] == ["prefix", "step"]
    assert len(binding["shared_tensors"]) == 1
    assert len(binding["routes"]) == 3  # prefix→step, recurrent state, step→final
    assert binding["program_abis"]["step"]["outputs"][0] == {
        "dtype": "torch.float32", "shape": [1, 2]}
    for name in ("prefix", "step"):
        assert "prov.quantization" in (out / "stages" / name / "model.mlir").read_text()
    assert "prov.quantization" not in (out / "stages" / "final" / "model.mlir").read_text()
    for name, receipt_digest in binding["stage_receipts"].items():
        receipt = (out / "stages" / name / "capture_receipt.json").read_bytes()
        assert hashlib.sha256(receipt).hexdigest() == receipt_digest
        assert json.loads(receipt)["materialized_abi"]["complete"] is True


def test_quantized_session_refuses_post_transform_route_mismatch(tmp_path):
    with pytest.raises(ValueError, match="post-quantization dtype/shape mismatch"):
        write_quantized_multi_program_bundle(
            Shared().eval(), lambda model: model,
            lambda model: _session(model, bad_route=True), tmp_path / "refused",
            shared_programs=("prefix", "step"), quantization={"kind": "test"},
            quant=QuantizationConfig(scheme="int8_weight_only"),
        )
    assert not (tmp_path / "refused").exists()


def test_global_preapplied_policy_without_config_still_fails_closed(tmp_path):
    session = _session(Shared().eval())
    with pytest.raises(ValueError, match="quantization_preapplied requires a quantization config"):
        write_multi_program_bundle(
            session.bundle_programs(), dict(session.metadata), tmp_path / "invalid",
            quantization_preapplied=True)
