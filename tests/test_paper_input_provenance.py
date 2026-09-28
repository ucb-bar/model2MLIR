"""Paper capture inputs fail closed and carry byte-level provenance."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn


def _loader(name: str):
    path = Path(__file__).resolve().parents[1] / "workloads" / name / "loader.py"
    spec = importlib.util.spec_from_file_location(f"{name}_loader_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lstmnetvit_requires_attributed_external_trajectory_for_paper(tmp_path, monkeypatch):
    loader = _loader("lstmnetvit")
    corpus = tmp_path / "trajectory.npz"
    np.savez(corpus, frames=np.zeros((2, 1, 1, 60, 90), np.float32),
             desired_velocity=np.ones((2, 1, 1), np.float32),
             quaternions=np.array([[[1, 0, 0, 0]], [[1, 0, 0, 0]]], np.float32))
    monkeypatch.setenv("VITFLY_SESSION_NPZ", str(corpus))
    monkeypatch.setenv("VITFLY_PAPER_READY", "1")
    example = (torch.zeros(1, 1, 60, 90), torch.zeros(1, 1), torch.zeros(1, 4))
    with pytest.raises(ValueError, match="SESSION_SOURCE"):
        loader._session_streams(2, example)
    monkeypatch.setenv("VITFLY_SESSION_SOURCE", "vitfly/evaluation-flight-trajectory/v1")
    streams, provenance, ready = loader._session_streams(2, example)
    assert ready is True and streams["frames"].shape[0] == 2
    assert provenance["synthetic_session"] is False
    assert len(provenance["session_sha256"]) == 64


def test_resnet_requires_v2_preprocessing_and_hashes_inputs_and_weights(tmp_path, monkeypatch):
    loader = _loader("resnet50_v1_5")
    corpus = tmp_path / "images.npz"
    np.savez(corpus, images=np.zeros((2, 3, 224, 224), np.float32))
    monkeypatch.setenv("M2M_RESNET_INPUT_NPZ", str(corpus))
    monkeypatch.setenv("M2M_RESNET_INPUT_SOURCE", "imagenet/validation-subset/v1")
    monkeypatch.setenv("M2M_RESNET_PAPER_READY", "1")
    monkeypatch.setenv("M2M_SESSION_STEPS", "2")
    with pytest.raises(RuntimeError, match="PREPROCESSING"):
        loader._images()
    monkeypatch.setenv("M2M_RESNET_PREPROCESSING", "IMAGENET1K_V2")
    images, provenance, ready = loader._images()
    assert ready is True and images.shape == (2, 1, 3, 224, 224)
    assert provenance["synthetic_inputs"] is False
    assert len(provenance["input_sha256"]) == 64
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    assert loader._state_dict_sha256(model) == loader._state_dict_sha256(model)
    assert len(loader._state_dict_sha256(model)) == 64


def test_resnet_can_keep_calibration_separate_from_measured_input(tmp_path, monkeypatch):
    loader = _loader("resnet50_v1_5")
    measured = torch.zeros((1, 1, 3, 224, 224))
    corpus = tmp_path / "calibration.npz"
    np.savez(corpus, images=np.ones((2, 3, 224, 224), np.float32))
    monkeypatch.setenv("M2M_RESNET_CALIBRATION_NPZ", str(corpus))
    monkeypatch.setenv("M2M_RESNET_CALIBRATION_SOURCE", "independent/calibration/v1")

    calibration, provenance = loader._calibration_images(measured)
    assert calibration.shape == (2, 1, 3, 224, 224)
    assert not torch.equal(calibration[0], measured[0])
    assert provenance["calibration_source"] == "independent/calibration/v1"
    assert len(provenance["calibration_sha256"]) == 64

    holder = nn.Module()
    holder.calibration_images = calibration
    samples = list(loader.get_calibration_inputs(holder, (measured[0],)))
    assert len(samples) == 2
    assert all(len(sample) == 1 and sample[0].shape == (1, 3, 224, 224)
               for sample in samples)
