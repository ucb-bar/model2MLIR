"""Explicit, target-independent quantization adapter loading.

Adapters own their numerical transforms.  m2m only freezes selected input bytes,
checks the returned manifest, and carries it through the frontend boundary.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENTRY_POINT_GROUP = "m2m.quantization_adapters"
MANIFEST_SCHEMA = "m2m.quantization_manifest.v1"


@dataclass(frozen=True)
class ExternalQuantizationConfig:
    adapter_id: str
    contract_path: str | Path
    policy_path: str | Path
    expected_contract_sha256: str | None = None
    expected_policy_sha256: str | None = None

    @property
    def scheme(self) -> str:
        return f"external:{self.adapter_id}"


def _read_selected(path: str | Path, expected: str | None, role: str) -> tuple[bytes, str]:
    source = Path(path).expanduser().resolve(strict=True)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"selected {role} sha256 differs: expected {expected}, observed {digest}")
    return raw, digest


def _entry_point(adapter_id: str) -> Any:
    if not adapter_id or ":" in adapter_id or "/" in adapter_id:
        raise ValueError("adapter_id must be a nonempty installed entry-point name")
    matches = [entry for entry in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
               if entry.name == adapter_id]
    if len(matches) != 1:
        raise ValueError(f"expected one installed quantization adapter {adapter_id!r}, found {len(matches)}")
    return matches[0]


def apply_external_quantization(
    model: Any,
    example_inputs: tuple[Any, ...],
    config: ExternalQuantizationConfig,
    *,
    original_frontend_snapshot: dict[str, Any] | None = None,
) -> Any:
    """Run only the explicitly selected installed adapter and validate its receipt."""
    if not example_inputs:
        raise ValueError("external quantization requires example_inputs")
    contract, contract_sha = _read_selected(config.contract_path, config.expected_contract_sha256, "contract")
    policy, policy_sha = _read_selected(config.policy_path, config.expected_policy_sha256, "policy")
    entry = _entry_point(config.adapter_id)
    adapter = entry.load()
    result = adapter(
        model,
        tuple(example_inputs),
        contract_bytes=contract,
        policy_bytes=policy,
        original_frontend_snapshot=original_frontend_snapshot,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("quantization adapter must return (captured_model, manifest)")
    captured, manifest = result
    validate_manifest(manifest, config, contract_sha=contract_sha, policy_sha=policy_sha)
    captured._m2m_external_quantization_manifest = manifest
    return captured


def validate_manifest(
    manifest: Any,
    config: ExternalQuantizationConfig,
    *,
    contract_sha: str | None = None,
    policy_sha: str | None = None,
) -> dict[str, Any]:
    if contract_sha is None:
        _, contract_sha = _read_selected(config.contract_path, config.expected_contract_sha256, "contract")
    if policy_sha is None:
        _, policy_sha = _read_selected(config.policy_path, config.expected_policy_sha256, "policy")
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("external quantization manifest has an invalid schema")
    for field, expected in (("adapter_id", config.adapter_id),
                            ("contract_sha256", contract_sha), ("policy_sha256", policy_sha)):
        if manifest.get(field) != expected:
            raise ValueError(f"external quantization manifest {field} differs from selection")
    sites = manifest.get("sites")
    if not isinstance(sites, list) or not sites:
        raise ValueError("external quantization manifest needs a nonempty site census")
    ids = []
    for site in sites:
        if not isinstance(site, dict) or not isinstance(site.get("site_id"), str):
            raise ValueError("external quantization site needs a stable site_id")
        if site.get("status") not in {"quantized", "host", "skipped"}:
            raise ValueError("external quantization site needs an explicit status")
        ids.append(site["site_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("external quantization manifest has duplicate site IDs")
    json.dumps(manifest, sort_keys=True, allow_nan=False)
    return manifest


def manifest_digest(manifest: dict[str, Any]) -> str:
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()
