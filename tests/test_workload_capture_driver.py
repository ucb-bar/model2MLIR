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
