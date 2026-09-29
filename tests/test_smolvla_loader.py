"""SmolVLA's pinned pretrained loader preserves LeRobot config choice semantics."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


pytest.importorskip("lerobot")


def _loader():
    path = Path(__file__).resolve().parents[1] / "workloads" / "smolvla" / "loader.py"
    spec = importlib.util.spec_from_file_location("smolvla_loader_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pretrained_config_uses_registered_dispatcher_and_exact_revision(monkeypatch):
    loader = _loader()
    expected = loader.SmolVLAConfig()
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(loader.PreTrainedConfig, "from_pretrained", load)
    assert loader._load_pretrained_config() is expected
    assert calls == [((loader._MODEL_ID,), {
        "revision": loader._MODEL_REVISION, "local_files_only": True})]
    assert len(loader._MODEL_REVISION) == 40


def test_pretrained_config_rejects_wrong_registered_subclass(monkeypatch):
    loader = _loader()
    monkeypatch.setattr(loader.PreTrainedConfig, "from_pretrained", lambda *args, **kwargs: object())
    with pytest.raises(TypeError, match="expected SmolVLAConfig"):
        loader._load_pretrained_config()


def test_deferred_stagewise_capability_passes_one_source_and_explicit_calibration(monkeypatch):
    from torch import nn
    from m2m.capture import stagewise_pt2e

    loader = _loader()
    shared = nn.Linear(2, 2)
    capture = loader.SmolVLAMultiProgramCapture(
        shared, (), paper_ready=False, full_checkpoint=False, provenance={})
    source_session = object()
    monkeypatch.setattr(capture, "external_runtime_session", lambda: source_session)
    calls = []

    def convert(*args, **kwargs):
        calls.append((args, kwargs))
        return "converted"

    monkeypatch.setattr(stagewise_pt2e, "quantize_stagewise_pt2e_session", convert)
    quant = object()
    calibration = {"prefix_encode": [()], "flow_denoise": [()]}
    assert capture.shared_model is shared
    assert capture.stagewise_pt2e_session(
        quant, calibration_inputs=calibration) == "converted"
    assert calls == [((shared, source_session), {
        "quant": quant,
        "shared_programs": ("prefix_encode", "flow_denoise"),
        "calibration_inputs": calibration,
    })]
