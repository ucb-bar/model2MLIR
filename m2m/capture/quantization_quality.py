"""Diagnostic pre-quantization quality evidence, separate from compiler fidelity."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _same_tensor_bytes(left, right):
    return (
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(
            left.detach().contiguous().reshape(-1).view(torch.uint8),
            right.detach().contiguous().reshape(-1).view(torch.uint8),
        )
    )


def capture_prequant_outputs(model, inputs, flatten):
    """Snapshot a pure eval forward with immutable parameters.

    Inputs and registered buffers are cloned; PyTorch RNG states are restored.
    Detected mutations are refused without exposing those copies to the caller.
    Original parameter references are reused, so mutable parameters and opaque
    Python state require an independently supplied reference or session capture.
    """
    if any(not isinstance(value, torch.Tensor) for value in inputs):
        raise TypeError("quantization quality inputs must be tensors")
    memo = {}

    def clone(value):
        if id(value) not in memo:
            memo[id(value)] = value.detach().clone()
        return memo[id(value)]

    copied_inputs = tuple(clone(value) for value in inputs)
    original_buffers = dict(model.named_buffers(remove_duplicate=False))
    copied_buffers = {name: clone(value) for name, value in original_buffers.items()}
    snapshots = {name: value.detach().clone() for name, value in copied_buffers.items()}
    registries = [(module, module._buffers.copy()) for module in model.modules()]
    devices = sorted(
        {
            value.device.index
            for value in [*inputs, *original_buffers.values(), *model.parameters()]
            if value.device.type == "cuda"
        }
    )
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            result = torch.func.functional_call(model, copied_buffers, copied_inputs)
            outputs = tuple(value.detach().cpu().clone() for value in flatten(result))
    finally:
        changed = any(
            not _same_tensor_bytes(value, original)
            for value, original in zip(copied_inputs, inputs, strict=True)
        )
        changed |= set(copied_buffers) != set(snapshots)
        changed |= any(
            not _same_tensor_bytes(copied_buffers[name], original)
            for name, original in snapshots.items()
            if name in copied_buffers
        )
        for module, original in registries:
            changed |= set(module._buffers) != set(original)
            module._buffers.clear()
            module._buffers.update(original)
        if changed:
            raise ValueError(
                "automatic quality reference requires a pure eval forward; input/buffer mutation detected"
            )
    return outputs


def _errors(reference, selected):
    if reference.shape != selected.shape:
        return {"status": "INCOMPARABLE", "reason": "output shapes differ"}
    if np.iscomplexobj(reference) or np.iscomplexobj(selected):
        return {
            "status": "UNAVAILABLE",
            "reason": "complex output metrics are unsupported",
        }
    for array in (reference, selected):
        if array.dtype.kind in "iu" and (
            np.any(array > 2**53)
            or (array.dtype.kind == "i" and np.any(array < -(2**53)))
        ):
            return {
                "status": "UNAVAILABLE",
                "reason": "integer magnitude exceeds exact float64 representation",
                "unequal_elements": int(np.count_nonzero(reference != selected)),
            }
    a, b = reference.astype(np.float64), selected.astype(np.float64)
    nonfinite = int(np.count_nonzero(~np.isfinite(a) | ~np.isfinite(b)))
    if nonfinite:
        return {
            "status": "NONFINITE",
            "nonfinite_pairs": nonfinite,
            "reference_nan": int(np.count_nonzero(np.isnan(a))),
            "selected_nan": int(np.count_nonzero(np.isnan(b))),
            "reason": "finite numerical error metrics are not qualified",
        }
    with np.errstate(over="ignore", invalid="ignore"):
        delta = b - a
        l2 = float(np.linalg.norm(delta.reshape(-1)))
        norm = float(np.linalg.norm(a.reshape(-1)))
        metrics = {
            "l2_error": l2,
            "max_abs_error": float(np.max(np.abs(delta), initial=0)),
            "rms_error": l2 / math.sqrt(a.size) if a.size else 0.0,
            "relative_l2_error": l2 / norm if norm else 0.0 if l2 == 0 else None,
        }
    if any(
        value is not None and not math.isfinite(value) for value in metrics.values()
    ):
        return {"status": "UNAVAILABLE", "reason": "metric arithmetic overflow"}
    return {"status": "MEASURED", "elements": int(a.size), **metrics}


def write_quantization_quality(
    out, reference, selected, *, reference_origin, quantization, numpy_safe, tensor_abi
):
    """Store all outputs and observational errors without an acceptance gate."""
    out = Path(out)
    selected_arrays = {
        f"out{i}": np.ascontiguousarray(numpy_safe(value))
        for i, value in enumerate(selected)
    }
    np.savez(out / "quantization_selected_outputs.npz", **selected_arrays)
    baseline_arrays = None
    if reference is not None:
        baseline_arrays = {
            f"out{i}": np.ascontiguousarray(numpy_safe(value))
            for i, value in enumerate(reference)
        }
        np.savez(out / "prequant_reference.npz", **baseline_arrays)
    artifacts = {}
    for name in ("quantization_selected_outputs.npz", "prequant_reference.npz"):
        if name == "prequant_reference.npz" and baseline_arrays is None:
            continue
        path = out / name
        artifacts[name] = {
            "path": name,
            "sha256": _sha256(path),
        }
    rows = []
    if reference is not None:
        for index, (before, after) in enumerate(zip(reference, selected)):
            rows.append(
                {
                    "output_index": index,
                    "reference_abi": tensor_abi(before),
                    "selected_abi": tensor_abi(after),
                    **_errors(
                        baseline_arrays[f"out{index}"], selected_arrays[f"out{index}"]
                    ),
                }
            )
    report = {
        "schema": "m2m.quantization-quality.v1",
        "status": "UNKNOWN"
        if reference is None
        else "INCOMPARABLE"
        if len(reference) != len(selected)
        else "MEASURED"
        if all(row["status"] == "MEASURED" for row in rows)
        else "PARTIAL",
        "scope": "prequant_reference_to_selected_quantized_eager_outputs_on_bundle_inputs",
        "reference_origin": reference_origin,
        "quantization": str(quantization),
        "reference_output_count": None if reference is None else len(reference),
        "selected_output_count": len(selected),
        "outputs": rows,
        "artifacts": artifacts,
        "bundle_evidence": {
            name: _sha256(out / name)
            for name in (
                "golden.npy",
                "inputs.npz",
                "model.mlir",
                "weights.safetensors",
            )
        },
        "compiler_correctness_reference": "golden.npy remains the selected quantized model output",
        "acceptance_gate_applied": False,
        "task_quality_or_calibration_claimed": False,
        "reference_checkpoint_equivalence": "UNKNOWN for externally supplied outputs"
        if reference_origin == "caller_supplied"
        else "observed before quantization; pure eval and immutable parameters assumed, opaque Python side effects unproved",
    }
    if reference is None:
        report["reason"] = (
            "quantization preapplied without an independently supplied prequant reference"
        )
    destination = out / "quantization-quality.json"
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return {
        "path": destination.name,
        "sha256": _sha256(destination),
        "status": report["status"],
    }
