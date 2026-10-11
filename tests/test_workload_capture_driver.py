"""Coarse checks for the generic workload capture driver's session protocol."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from workloads import capture


class _Step(nn.Module):
    def forward(self, value):
        return value + 1


def test_driver_writes_three_program_session_with_stage_receipts(monkeypatch):
    class Session:
        def write_bundle(self, out, *, quant=None):
            assert quant is None
            from m2m.capture.bundle import write_multi_program_bundle

            names = ("prefix", "flow", "action")
            programs = [
                {"name": name, "model": _Step(), "inputs": (torch.ones(1),), "steps": 1}
                for name in names
            ]
            root = {
                "kind": "test_session", "paper_ready": False,
                "stages": names,
                "stage_schedule": [
                    {"name": name, "steps": 1, "execution": "compiled", "timed": False}
                    for name in names
                ],
                "quality_program": "flow", "parameters": {}, "states": [],
            }
            return write_multi_program_bundle(programs, root, out)

    loader = types.ModuleType("loader")
    loader.get_model_and_inputs = lambda: (Session(), ())
    # Only restore our synthetic loader. patch.dict(sys.modules, ...) restores
    # the *entire* import table, dropping Torch modules loaded by the capture
    # while their native TORCH_LIBRARY registrations remain process-global.
    monkeypatch.setitem(sys.modules, "loader", loader)
    with tempfile.TemporaryDirectory(prefix="m2m-session-driver-") as directory:
        workload_root = Path(directory)
        (workload_root / "tiny_session").mkdir()
        output = io.StringIO()
        with patch.object(capture, "WORKLOADS", workload_root):
            with contextlib.redirect_stdout(output):
                capture._inner_capture("tiny_session", ["fp32"], "linalg-on-tensors")
        marker = "__CAPTURE_RESULT__ "
        result = json.loads(output.getvalue().split(marker, 1)[1])["fp32"]
        assert result["ok"] is True and result["opaque"] == 0, result
        assert result["source_closure_verified"] is False
        assert "no source-closure or execution proof" in result["qualification"]
        root = workload_root / "tiny_session" / "tiny_session_session"
        assert (root / "session_contract.yaml").is_file()
        assert result["bundle_integrity"] == str(root / "bundle_integrity.json")
        integrity = json.loads((root / "bundle_integrity.json").read_text())
        assert integrity["source_closure_verified"] is False
        assert "session_contract.yaml" in {row["path"] for row in integrity["files"]}
        assert [row["name"] for row in result["programs"]] == ["prefix", "flow", "action"]
        for name in ("prefix", "flow", "action"):
            stage = root / "stages" / name
            assert (stage / "model.mlir").is_file()
            assert (stage / "capture_receipt.json").is_file()


def test_driver_refuses_session_dtype_cast_instead_of_silently_capturing_fp32(monkeypatch):
    class Session:
        def write_bundle(self, _out, *, quant=None):
            raise AssertionError("unsupported dtype cast must not enter the writer")

    loader = types.ModuleType("loader")
    loader.get_model_and_inputs = lambda: (Session(), ())
    monkeypatch.setitem(sys.modules, "loader", loader)
    with tempfile.TemporaryDirectory(prefix="m2m-session-driver-") as directory:
        workload_root = Path(directory)
        (workload_root / "tiny_session").mkdir()
        output = io.StringIO()
        with patch.object(capture, "WORKLOADS", workload_root):
            with contextlib.redirect_stdout(output):
                capture._inner_capture("tiny_session", ["bf16"], "linalg-on-tensors")
        result = json.loads(output.getvalue().split("__CAPTURE_RESULT__ ", 1)[1])["bf16"]
        assert result["ok"] is False
        assert "does not support the bf16 dtype-cast format" in result["error"]


def test_driver_rejects_stage_symlink_before_reading_artifacts():
    class EscapingSession:
        def __init__(self, outside):
            self.outside = outside

        def write_bundle(self, out, *, quant=None):
            assert quant is None
            out.mkdir()
            (out / "session_contract.yaml").write_text(
                "stages: [prefix]\nprograms:\n- name: prefix\n  bundle: stages/prefix\n")
            (out / "stages").symlink_to(self.outside, target_is_directory=True)
            return {"out": str(out), "n_programs": 1}

    with tempfile.TemporaryDirectory(prefix="m2m-session-escape-") as directory:
        root = Path(directory)
        outside = root / "outside"
        outside.mkdir()
        try:
            capture._capture_multi_program(EscapingSession(outside), root, "bad", None)
        except ValueError as exc:
            assert "bundle path contains a symlink" in str(exc)
        else:
            raise AssertionError("session capture accepted an escaping stage symlink")


def test_driver_rejects_receipt_listed_weight_tamper():
    class TamperedSession:
        def write_bundle(self, out, *, quant=None):
            assert quant is None
            from m2m.capture.bundle import write_multi_program_bundle

            summary = write_multi_program_bundle(
                [{"name": "step", "model": _Step(),
                  "inputs": (torch.ones(1),), "steps": 1}],
                {"kind": "test_session", "paper_ready": False,
                 "stages": ["step"],
                 "stage_schedule": [{"name": "step", "steps": 1,
                                     "execution": "compiled", "timed": False}],
                 "quality_program": "step", "parameters": {}, "states": []},
                out,
            )
            with (out / "stages" / "step" / "weights.safetensors").open("ab") as stream:
                stream.write(b"tampered-after-receipt")
            return summary

    with tempfile.TemporaryDirectory(prefix="m2m-session-tamper-") as directory:
        try:
            capture._capture_multi_program(TamperedSession(), Path(directory), "bad", None)
        except ValueError as exc:
            assert "receipt does not bind weights.safetensors" in str(exc)
        else:
            raise AssertionError("session capture accepted a tampered weight artifact")


def test_driver_rejects_root_contract_tamper_after_bundle_receipt():
    class TamperedSession:
        def write_bundle(self, out, *, quant=None):
            assert quant is None
            from m2m.capture.bundle import write_multi_program_bundle

            summary = write_multi_program_bundle(
                [{"name": "step", "model": _Step(),
                  "inputs": (torch.ones(1),), "steps": 1}],
                {"kind": "test_session", "paper_ready": False,
                 "stages": ["step"],
                 "stage_schedule": [{"name": "step", "steps": 1,
                                     "execution": "compiled", "timed": False}],
                 "quality_program": "step", "parameters": {}, "states": []},
                out,
            )
            contract = out / "session_contract.yaml"
            contract.write_text(contract.read_text().replace("test_session", "evil_session"))
            return summary

    with tempfile.TemporaryDirectory(prefix="m2m-session-root-tamper-") as directory:
        try:
            capture._capture_multi_program(TamperedSession(), Path(directory), "bad", None)
        except ValueError as exc:
            assert "integrity receipt does not bind session_contract.yaml" in str(exc)
        else:
            raise AssertionError("session capture accepted a tampered root contract")


def test_capture_driver_passes_output_root_and_explicit_env_after_toml_defaults(monkeypatch, tmp_path):
    called = {}
    monkeypatch.setattr(capture, "_ensure_venv", lambda *_args, **_kwargs: Path(sys.executable))

    def run(command, *, env, capture_output, text):
        called.update(command=command, env=env)
        return types.SimpleNamespace(stdout='__CAPTURE_RESULT__ {"fp32": {"ok": true}}\n')

    monkeypatch.setattr(capture.subprocess, "run", run)
    output_root = tmp_path / "captures"
    capture._run_one("tiny_llama", ["fp32"], "linalg-on-tensors", build=False,
                     output_root=output_root,
                     env_overrides={"M2M_LLAMA_LAYERS": ""})
    assert called["env"]["M2M_LLAMA_LAYERS"] == ""
    assert called["command"][-2:] == ["--output-root", str(output_root)]


def test_capture_driver_uses_explicit_worker_python_without_creating_venv(monkeypatch):
    monkeypatch.setattr(capture, "_ensure_venv", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("must not create a venv")))
    seen = {}

    def run(command, *, env, capture_output, text):
        seen["command"] = command
        return types.SimpleNamespace(stdout='__CAPTURE_RESULT__ {"fp32": {"ok": true}}\n')

    monkeypatch.setattr(capture.subprocess, "run", run)
    capture._run_one("tiny_llama", ["fp32"], "linalg-on-tensors", build=False,
                     python=Path(sys.executable))
    assert seen["command"][0] == sys.executable


def test_capture_driver_places_session_under_requested_root(monkeypatch, tmp_path):
    loader = types.ModuleType("loader")
    loader.get_model_and_inputs = lambda: (types.SimpleNamespace(write_bundle=lambda *_a, **_k: {}), ())
    monkeypatch.setitem(sys.modules, "loader", loader)
    seen = {}

    def capture_session(_model, output_dir, stem, _quant):
        seen["output_dir"] = output_dir
        seen["stem"] = stem
        return {"ok": True, "opaque": 0}

    monkeypatch.setattr(capture, "_capture_multi_program", capture_session)
    root = tmp_path / "captures"
    with contextlib.redirect_stdout(io.StringIO()):
        capture._inner_capture("smolvla", ["fp32"], "linalg-on-tensors", output_root=root)
    assert seen == {"output_dir": root / "smolvla", "stem": "smolvla"}
