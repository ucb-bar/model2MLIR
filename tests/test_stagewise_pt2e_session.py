"""End-to-end coarse proof for stage-wise PT2E and source-weight binding."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

from m2m.capture.bundle import write_stagewise_pt2e_multi_program_bundle
from m2m.capture.bundle_integrity import verify_bundle_integrity
from m2m.capture.external_runtime import ExternalRuntimeProgram, make_external_runtime_session
from m2m.capture.stagewise_pt2e import quantize_stagewise_pt2e_session
from m2m.capture.source_closure import (
    make_source_snapshot, verify_stagewise_source_snapshot_bundle,
)
from m2m.capture.torchao_pipeline import QuantizationConfig
from tests.test_quantized_multi_program_bundle import Shared, _session
from tests.test_stagewise_pt2e_boundary import FlowStage, PrefixStage, SharedMethodModel


def _convert(shared, session, *, source_snapshot=None):
    return quantize_stagewise_pt2e_session(
        shared, session,
        quant=QuantizationConfig(scheme="int8_static_act_int8_weight",
                                 calibration_samples=2),
        shared_programs=("prefix", "step"),
        calibration_inputs={
            "prefix": [session.programs[0].inputs],
            "step": [session.programs[1].inputs,
                     (torch.full((1, 2), 2.0), torch.zeros(1, 2))],
        },
        source_snapshot=source_snapshot,
    )


def test_stagewise_pt2e_binds_source_weight_routes_and_stage_receipts(tmp_path):
    shared = Shared().eval()
    converted = _convert(shared, _session(shared))
    assert converted.calibration_counts == {"prefix": 1, "step": 2}
    assert len(converted.frozen_weights) == 2
    prefix, step = converted.frozen_weights
    assert prefix["source_parameter"] == step["source_parameter"] == "linear.weight"
    assert prefix["source_sha256"] == step["source_sha256"]
    assert prefix["frozen_sha256"] == step["frozen_sha256"]
    assert prefix["scale_sha256"] == step["scale_sha256"]
    assert len(converted.routes) == 3
    assert converted.session.programs[0].module is not converted.session.programs[1].module

    out = tmp_path / "stagewise"
    summary = write_stagewise_pt2e_multi_program_bundle(
        converted, out, capture_trace=True)
    assert summary["n_programs"] == 3
    verify_bundle_integrity(out)
    binding = json.loads((out / "quantized-session-binding.json").read_text())
    assert binding["source_closure_verified"] is False
    assert binding["physical_weight_aliasing"] is False
    assert binding["quantization"]["scheme"] == converted.quant.scheme
    assert binding["quantization"]["calibration_samples"] == 2
    assert binding["frozen_weights"] == list(converted.frozen_weights)
    assert binding["calibration_counts"] == {"prefix": 1, "step": 2}
    for name, digest in binding["stage_receipts"].items():
        receipt = (out / "stages" / name / "capture_receipt.json").read_bytes()
        assert hashlib.sha256(receipt).hexdigest() == digest
        assert json.loads(receipt)["materialized_abi"]["complete"] is True


def test_stagewise_pt2e_supports_method_only_shared_model():
    shared = SharedMethodModel().eval()
    value = torch.ones(1, 2)
    programs = (
        ExternalRuntimeProgram("prefix", PrefixStage(shared).eval(), (value,), 1),
        ExternalRuntimeProgram("flow", FlowStage(shared).eval(), (value,), 2,
                               {"steps": 2, "quality": {"output_index": 0}}),
    )
    metadata = {
        "kind": "method_session", "stages": ["prefix", "flow"],
        "stage_schedule": [
            {"name": "prefix", "steps": 1}, {"name": "flow", "steps": 2},
        ],
        "quality_program": "flow",
        "bindings": [{"name": "prefix_to_flow",
                      "from": {"program": "prefix", "output_index": 0},
                      "to": {"program": "flow", "input_index": 0}}],
    }
    source = make_external_runtime_session(version=2, programs=programs, metadata=metadata)
    converted = quantize_stagewise_pt2e_session(
        shared, source,
        quant=QuantizationConfig(scheme="int8_static_act_int8_weight"),
        shared_programs=("prefix", "flow"),
        calibration_inputs={"prefix": [programs[0].inputs],
                            "flow": [programs[1].inputs]},
    )
    assert {item["source_parameter"] for item in converted.frozen_weights} == {"linear.weight"}
    assert len({item["frozen_sha256"] for item in converted.frozen_weights}) == 1
    assert converted.routes[0]["name"] == "prefix_to_flow"


def test_stagewise_snapshot_is_bound_and_independently_rechecked(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    loader = tmp_path / "loader.py"
    loader.write_text("# tiny fixture loader\n")
    model_source = tmp_path / "model.bin"
    model_source.write_bytes(b"tiny model source")
    runtime_source = tmp_path / "runtime.bin"
    runtime_source.write_bytes(b"tiny runtime source")
    snapshot = make_source_snapshot(
        package_root=Path(__file__).resolve().parents[1] / "m2m",
        loader=loader, model_roots={"checkpoint": model_source},
        support_roots={"runtime": runtime_source})
    shared = Shared().eval()
    converted = _convert(shared, _session(shared), source_snapshot=snapshot)
    out = tmp_path / "fresh-stagewise"
    write_stagewise_pt2e_multi_program_bundle(converted, out, capture_trace=False)
    verify_bundle_integrity(out)
    verify_stagewise_source_snapshot_bundle(out)
    binding = json.loads((out / "quantized-session-binding.json").read_text())
    assert binding["source_closure_verified"] is False
    assert binding["source_snapshot"]["source_closure_verified"] is False
    for name in ("prefix", "step", "final"):
        receipt = json.loads((out / "stages" / name / "capture_receipt.json").read_text())
        assert receipt["source"]["path"] == str(loader)
    model_source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="membership or bytes"):
        verify_stagewise_source_snapshot_bundle(out)


def test_stagewise_snapshot_rejects_a_different_m2m_package(tmp_path):
    fake_package = tmp_path / "m2m"
    fake_package.mkdir()
    (fake_package / "__init__.py").write_text("# unrelated package\n")
    loader = tmp_path / "loader.py"
    loader.write_text("# loader\n")
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    runtime = tmp_path / "runtime.bin"
    runtime.write_bytes(b"runtime")
    snapshot = make_source_snapshot(
        package_root=fake_package, loader=loader,
        model_roots={"checkpoint": model}, support_roots={"runtime": runtime})
    shared = Shared().eval()
    with pytest.raises(ValueError, match="not the executing package"):
        _convert(shared, _session(shared), source_snapshot=snapshot)


def test_stagewise_snapshot_refuses_source_drift_during_quantization(tmp_path, monkeypatch):
    import m2m.capture.stagewise_pt2e as pipeline

    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    loader = tmp_path / "loader.py"
    loader.write_text("# loader\n")
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    runtime = tmp_path / "runtime.bin"
    runtime.write_bytes(b"runtime")
    snapshot = make_source_snapshot(
        package_root=Path(__file__).resolve().parents[1] / "m2m", loader=loader,
        model_roots={"checkpoint": model}, support_roots={"runtime": runtime})
    original = pipeline.apply_quantization

    def change_source(*args, **kwargs):
        converted = original(*args, **kwargs)
        model.write_bytes(b"changed during quantization")
        return converted

    monkeypatch.setattr(pipeline, "apply_quantization", change_source)
    shared = Shared().eval()
    with pytest.raises(ValueError, match="membership or bytes"):
        _convert(shared, _session(shared), source_snapshot=snapshot)


def test_stagewise_pt2e_retains_eager_reference_for_paper_ready_flow():
    shared = Shared().eval()
    source = _session(shared)
    programs = list(source.programs)
    child = dict(programs[1].session)
    child["paper_ready"] = True
    programs[1] = ExternalRuntimeProgram(
        programs[1].name, programs[1].module, programs[1].inputs,
        programs[1].steps, child)
    root = dict(source.metadata)
    root["paper_ready"] = True
    source = make_external_runtime_session(
        version=2, programs=tuple(programs), metadata=root)
    converted = _convert(shared, source)
    quality = converted.session.programs[1].session["quality"]
    assert quality["reference_values"] is not None


def test_stagewise_pt2e_rejects_bad_route_before_writing(tmp_path):
    shared = Shared().eval()
    with pytest.raises(ValueError, match="post-PT2E dtype/shape mismatch"):
        _convert(shared, _session(shared, bad_route=True))
    assert not (tmp_path / "stagewise").exists()


def test_stagewise_pt2e_requires_explicit_calibration():
    shared = Shared().eval()
    session = _session(shared)
    with pytest.raises(ValueError, match="explicit calibration inputs"):
        quantize_stagewise_pt2e_session(
            shared, session,
            quant=QuantizationConfig(scheme="int8_static_act_int8_weight"),
            shared_programs=("prefix", "step"),
            calibration_inputs={"prefix": [session.programs[0].inputs]},
        )


def test_stagewise_writer_rejects_changed_frozen_encoding(tmp_path):
    shared = Shared().eval()
    converted = _convert(shared, _session(shared))
    first = converted.session.programs[0].module
    with torch.no_grad():
        first._scale_0[0] *= 2
    out = tmp_path / "refused"
    with pytest.raises(ValueError, match="frozen weight encoding changed"):
        write_stagewise_pt2e_multi_program_bundle(converted, out)
    assert not out.exists()


def test_stagewise_writer_rejects_mislabelled_quantization(tmp_path):
    shared = Shared().eval()
    converted = _convert(shared, _session(shared))
    out = tmp_path / "refused"
    with pytest.raises(ValueError, match="metadata differs"):
        write_stagewise_pt2e_multi_program_bundle(
            converted, out, quantization={"scheme": "int8_weight_only"})
    assert not out.exists()
