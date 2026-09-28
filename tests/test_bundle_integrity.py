"""Bounded checks for multi-program byte inventory, not semantic closure."""

from __future__ import annotations

import json

import pytest

from m2m.capture.bundle_integrity import write_bundle_integrity, verify_bundle_integrity


def test_bundle_integrity_binds_root_and_stage_session_files(tmp_path):
    (tmp_path / "session_contract.yaml").write_text("stages: [step]\n")
    (tmp_path / "session_inputs.npz").write_bytes(b"root-inputs")
    stage = tmp_path / "stages" / "step"
    stage.mkdir(parents=True)
    (stage / "model.mlir").write_text("module {}\n")
    (stage / "session_goldens.npz").write_bytes(b"stage-goldens")
    receipt_path = write_bundle_integrity(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    names = [record["path"] for record in receipt["files"]]
    assert names == sorted(names)
    assert names == ["session_contract.yaml", "session_inputs.npz", "stages/step/model.mlir",
                     "stages/step/session_goldens.npz"]
    assert receipt["source_closure_verified"] is False
    assert verify_bundle_integrity(tmp_path) == receipt_path
    exact_receipt = receipt_path.read_bytes()
    assert write_bundle_integrity(tmp_path).read_bytes() == exact_receipt

    (stage / "session_goldens.npz").write_bytes(b"alter-goldens")
    with pytest.raises(ValueError, match="does not bind stages/step/session_goldens.npz"):
        verify_bundle_integrity(tmp_path)


def test_bundle_integrity_rejects_unlisted_files_and_symlinks(tmp_path):
    (tmp_path / "session_contract.yaml").write_text("stages: []\n")
    write_bundle_integrity(tmp_path)
    (tmp_path / "new-session-file.npz").write_bytes(b"new")
    with pytest.raises(ValueError, match="file roster"):
        verify_bundle_integrity(tmp_path)
    (tmp_path / "new-session-file.npz").unlink()
    (tmp_path / "escaped").symlink_to(tmp_path / "session_contract.yaml")
    with pytest.raises(ValueError, match="symlink"):
        verify_bundle_integrity(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        write_bundle_integrity(tmp_path)


def test_bundle_integrity_rejects_receipt_path_escape(tmp_path):
    (tmp_path / "session_contract.yaml").write_text("stages: []\n")
    receipt_path = write_bundle_integrity(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    receipt["files"][0]["path"] = "../session_contract.yaml"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="unsafe path"):
        verify_bundle_integrity(tmp_path)
