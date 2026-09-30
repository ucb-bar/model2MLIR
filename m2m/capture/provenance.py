"""Byte-bound receipt for a captured model bundle.

The receipt records observed sources and outputs, not a claim that an arbitrary
Python loader's transitive code/data closure is known.  A caller with a .pt2
source passes that archive explicitly; it is hashed once, when the bundle is
complete.  The receipt itself is deliberately outside the hash list.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _tool_identity() -> dict:
    root = Path(__file__).resolve().parents[2]
    package = root / "m2m"
    if package.is_symlink() or not package.is_dir():
        raise ValueError("M2M Python package source is absent or indirect")
    sources = {}
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"M2M Python source is indirect: {relative}")
        sources[relative.as_posix()] = _sha256_file(path)
    if not sources:
        raise ValueError("M2M Python package source is empty")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    # A source-tree workload CLI is an invocation, not part of the installed
    # m2m package. Record whichever actual entrypoint ran independently instead
    # of making every installed wheel appear to lack two unused CLI scripts.
    entrypoint = None
    if sys.argv and sys.argv[0]:
        candidate = Path(sys.argv[0])
        if candidate.is_file():
            resolved = candidate.resolve()
            entrypoint = {"path": str(resolved), **_file_record(resolved)}
    return {
        "commit": commit,
        "source_sha256": sources,
        "source_inventory_scope": "m2m_package_python_sources_only",
        "source_inventory_status": "complete",
        "unavailable_sources": [],
        "executed_entrypoint": entrypoint,
    }


def capture_receipt(out: str | Path, *, source_path: str | Path | None = None) -> dict:
    """Describe the bytes a bundle writer actually published.

    ``source_path`` is an observed input artifact, such as a torch.export .pt2
    or a loader file; it is not proof of dependency closure or authorship.
    """
    out = Path(out)
    required = (
        "model.mlir", "weights.safetensors", "weights.safetensors.manifest.json",
        "inputs.npz", "golden.npy", "extra.npz", "input_order.json",
    )
    artifacts = {name: _file_record(out / name) for name in required}
    for name in ("frontend-trace.json", "meta.json"):
        if (out / name).is_file():
            artifacts[name] = _file_record(out / name)
    if "meta.json" in artifacts:
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        agreement = (meta.get("integerization_receipt") or {}).get("golden_agreement") or {}
        reference = agreement.get("output")
        if reference is not None:
            if not isinstance(reference, dict) or reference.get("path") != "integer-reference.json":
                raise ValueError("integer reference must point to the bundle-local artifact")
            reference_path = out / "integer-reference.json"
            if not reference_path.is_file() or reference_path.is_symlink():
                raise ValueError("integer reference artifact is absent or symlinked")
            record = _file_record(reference_path)
            if reference.get("sha256") != record["sha256"]:
                raise ValueError("integer reference artifact differs from its metadata digest")
            artifacts["integer-reference.json"] = record
    if "frontend-trace.json" in artifacts:
        trace = json.loads((out / "frontend-trace.json").read_text(encoding="utf-8"))
        if trace.get("mlir", {}).get("sha256") != artifacts["model.mlir"]["sha256"]:
            raise ValueError("frontend trace does not bind the bundle's exact model.mlir bytes")
        if trace.get("mlir", {}).get("bytes") != artifacts["model.mlir"]["bytes"]:
            raise ValueError("frontend trace MLIR byte count differs from the bundle")
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        pointer = meta.get("frontend_trace") or {}
        if pointer.get("path") != "frontend-trace.json" or pointer.get("sha256") != artifacts["frontend-trace.json"]["sha256"]:
            raise ValueError("bundle metadata does not bind its exact frontend trace")
    lifted = {}
    with np.load(out / "extra.npz", allow_pickle=False) as extras:
        for name in sorted(extras.files):
            if not name.startswith("c_lifted_tensor_"):
                continue
            value = np.asarray(extras[name])
            lifted[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
    manifest = json.loads((out / "weights.safetensors.manifest.json").read_text())
    expected_lifted = sorted({
        str(entry.get("name")) for entry in manifest.values()
        if isinstance(entry, dict) and str(entry.get("name", "")).startswith("c_lifted_tensor_")
    })
    if expected_lifted != sorted(lifted):
        raise ValueError(
            "lifted-constant ABI is incomplete: "
            f"manifest requires {expected_lifted}, extra.npz provides {sorted(lifted)}"
        )
    input_order = json.loads((out / "input_order.json").read_text())
    with np.load(out / "inputs.npz", allow_pickle=False) as inputs:
        materialized_inputs = set(inputs.files)
    expected_inputs = {f"in{i}" for i in range(len(input_order))}
    if materialized_inputs != expected_inputs or sorted(input_order.values()) != list(range(len(input_order))):
        raise ValueError(
            "input ABI is incomplete: input_order must bind every positional in<i> once"
        )
    source = None
    if source_path is not None:
        path = Path(source_path).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"capture source is not a file: {path}")
        source = {"path": str(path), **_file_record(path)}
    try:
        torchao_version = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        torchao_version = None
    return {
        "schema": "m2m.capture-receipt.v1",
        "tool": _tool_identity(),
        "framework": {"torch": torch.__version__, "torchao": torchao_version},
        "source": source,
        "source_closure_verified": False,
        "artifacts": artifacts,
        "materialized_abi": {"complete": True, "inputs": len(input_order),
                             "lifted_constants": expected_lifted},
        "lifted_constants": lifted,
    }


def write_capture_receipt(out: str | Path, *, source_path: str | Path | None = None) -> dict:
    """Write the receipt last, only after all required bundle outputs exist."""
    receipt = capture_receipt(out, source_path=source_path)
    (Path(out) / "capture_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return receipt
