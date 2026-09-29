"""Exact, caller-owned input snapshot for a fresh stagewise capture.

The snapshot names the M2M package, loader, model inputs (code/checkpoints), and
support inputs (interpreter/runtime libraries) explicitly. It checks every file,
directory, and link in those roots before and after capture. This does *not*
observe arbitrary runtime reads or prove that the process was isolated; those
need a separate sealed issuer and replay verifier. The closure flag stays false.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SCHEMA = "m2m.stagewise-source-snapshot.v1"
SCOPE = "declared_input_roots_exact_membership_and_bytes_only"
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _root(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"source owner root must be an absolute path without '..': {value}")
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"source owner root traverses a symlink: {part}")
    if not path.exists() or not (path.is_file() or path.is_dir()):
        raise ValueError(f"source owner root is absent or not a plain file/directory: {path}")
    return path


def _owners(
    package_root: str | Path, loader: str | Path,
    model_roots: Mapping[str, str | Path], support_roots: Mapping[str, str | Path],
    data_roots: Mapping[str, str | Path] | None,
) -> list[tuple[str, str, Path]]:
    if not model_roots or not support_roots:
        raise ValueError("source snapshot needs explicit model and support owners")
    package, loader_path = _root(package_root), _root(loader)
    if package.name != "m2m" or not package.is_dir() or not (package / "__init__.py").is_file():
        raise ValueError("M2M package owner must be the complete m2m package directory")
    if not loader_path.is_file():
        raise ValueError("loader owner must be one plain file")
    records = [("m2m_package", "m2m", package), ("loader", "loader", loader_path)]
    for role, roots in (("model", model_roots), ("support", support_roots),
                        ("data", data_roots or {})):
        if not isinstance(roots, Mapping):
            raise TypeError(f"{role} roots must be a name-to-path mapping")
        for name, value in sorted(roots.items()):
            if not isinstance(name, str) or not _LABEL.fullmatch(name):
                raise ValueError(f"invalid {role} owner name: {name!r}")
            records.append((role, name, _root(value)))
    paths = [path for _, _, path in records]
    for index, path in enumerate(paths):
        if any(path == other or path.is_relative_to(other) or other.is_relative_to(path)
               for other in paths[:index]):
            raise ValueError(f"source owner roots overlap: {path}")
    return records


def _inside(path: Path, roots: list[Path]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in roots)


def _members(root: Path, owned_roots: list[Path]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    def visit(path: Path, name: str) -> None:
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            result.append({"path": name, "kind": "directory"})
            for child in sorted(path.iterdir(), key=lambda item: item.name):
                child_name = child.name if name == "." else f"{name}/{child.name}"
                visit(child, child_name)
        elif stat.S_ISREG(mode):
            result.append({"path": name, "kind": "file", "bytes": path.stat().st_size,
                           "sha256": _sha256(path)})
        elif stat.S_ISLNK(mode):
            target = os.readlink(path)
            direct = Path(os.path.abspath(target if os.path.isabs(target)
                                          else path.parent / target))
            try:
                resolved = path.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise ValueError(f"source snapshot has a dangling or cyclic link: {path}") from exc
            if not _inside(direct, owned_roots) or not _inside(resolved, owned_roots):
                raise ValueError(f"source snapshot link escapes declared owners: {path}")
            if not (resolved.is_file() or resolved.is_dir()):
                raise ValueError(f"source snapshot link has a special target: {path}")
            result.append({"path": name, "kind": "symlink", "target": target})
        else:
            raise ValueError(f"source snapshot contains a special entry: {path}")

    visit(root, ".")
    return result


def make_source_snapshot(
    *, package_root: str | Path, loader: str | Path,
    model_roots: Mapping[str, str | Path], support_roots: Mapping[str, str | Path],
    data_roots: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Record every member of explicit input roots; do not infer process closure."""
    owners = _owners(package_root, loader, model_roots, support_roots, data_roots)
    roots = [path for _, _, path in owners]
    recorded = [
        {"role": role, "name": name, "root": str(path), "members": _members(path, roots)}
        for role, name, path in owners
    ]
    if any(member["kind"] == "symlink" for member in recorded[0]["members"]):
        raise ValueError("M2M package source owner contains a symlink")
    return {
        "schema": SCHEMA, "scope": SCOPE, "source_closure_verified": False,
        "owners": recorded,
    }


def verify_source_snapshot(snapshot: Mapping[str, Any]) -> None:
    """Refuse any changed owner, member roster, content, link, or schema field."""
    if not isinstance(snapshot, Mapping) or set(snapshot) != {
            "schema", "scope", "source_closure_verified", "owners"}:
        raise ValueError("source snapshot has an invalid schema")
    if (snapshot["schema"] != SCHEMA or snapshot["scope"] != SCOPE
            or snapshot["source_closure_verified"] is not False
            or not isinstance(snapshot["owners"], list)):
        raise ValueError("source snapshot has an invalid scope or closure claim")
    roles: dict[str, dict[str, str]] = {"model": {}, "support": {}, "data": {}}
    package = loader = None
    for record in snapshot["owners"]:
        if not isinstance(record, dict) or set(record) != {"role", "name", "root", "members"}:
            raise ValueError("source snapshot has a malformed owner")
        role, name, path = record["role"], record["name"], record["root"]
        if not isinstance(path, str):
            raise ValueError("source snapshot has a non-path owner")
        if role == "m2m_package" and name == "m2m" and package is None:
            package = path
        elif role == "loader" and name == "loader" and loader is None:
            loader = path
        elif role in roles and isinstance(name, str) and _LABEL.fullmatch(name) and name not in roles[role]:
            roles[role][name] = path
        else:
            raise ValueError("source snapshot has duplicate or unknown ownership")
    if package is None or loader is None:
        raise ValueError("source snapshot omits package or loader ownership")
    observed = make_source_snapshot(
        package_root=package, loader=loader, model_roots=roles["model"],
        support_roots=roles["support"], data_roots=roles["data"])
    if snapshot != observed:
        raise ValueError("source snapshot differs from exact current input membership or bytes")


def verify_source_snapshot_file(path: str | Path) -> dict[str, Any]:
    source = _root(Path(path).absolute())
    if not source.is_file():
        raise ValueError("source snapshot manifest must be a plain file")
    try:
        snapshot = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("source snapshot manifest is unreadable") from exc
    verify_source_snapshot(snapshot)
    return snapshot


def verify_stagewise_source_snapshot_bundle(root: str | Path) -> None:
    """Independently replay source and stage-receipt links of a fresh bundle.

    This verifies the declared source snapshot against *current* inputs. It is
    not an execution replay or a certificate for an earlier capture process.
    """
    bundle = _root(Path(root).absolute())
    if not bundle.is_dir():
        raise ValueError("stagewise snapshot bundle must be a directory")
    binding_path = bundle / "quantized-session-binding.json"
    if not binding_path.is_file():
        raise ValueError("stagewise source snapshot binding is absent or linked")
    _root(binding_path)
    try:
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("stagewise source snapshot binding is unreadable") from exc
    if (not isinstance(binding, dict)
            or binding.get("schema") != "m2m.stagewise-pt2e-binding.v1"
            or binding.get("scope") != "one_source_model_verified_semantic_weights_separate_stage_storage"
            or binding.get("source_closure_verified") is not False):
        raise ValueError("stagewise source snapshot binding has invalid scope")
    pointer = binding.get("source_snapshot")
    if (not isinstance(pointer, dict) or set(pointer) != {
            "path", "sha256", "verified_before_and_after_capture", "source_closure_verified"}
            or pointer["path"] != "source-snapshot.json"
            or pointer["verified_before_and_after_capture"] is not True
            or pointer["source_closure_verified"] is not False
            or not isinstance(pointer["sha256"], str)):
        raise ValueError("stagewise source snapshot pointer is malformed")
    manifest_path = bundle / pointer["path"]
    if not manifest_path.is_file():
        raise ValueError("stagewise source snapshot manifest is absent or linked")
    _root(manifest_path)
    if _sha256(manifest_path) != pointer["sha256"]:
        raise ValueError("stagewise source snapshot manifest differs from its binding")
    manifest = verify_source_snapshot_file(manifest_path)
    loader = next(owner for owner in manifest["owners"] if owner["role"] == "loader")
    loader_member = loader["members"][0]
    package = next(owner for owner in manifest["owners"] if owner["role"] == "m2m_package")
    package_sources = {
        f"m2m/{member['path']}": member["sha256"]
        for member in package["members"]
        if member["kind"] == "file" and member["path"].endswith(".py")
        and "__pycache__" not in Path(member["path"]).parts
    }
    receipts = binding.get("stage_receipts")
    if not isinstance(receipts, dict) or not receipts:
        raise ValueError("stagewise source snapshot binding omits stage receipts")
    stages_root = bundle / "stages"
    if not stages_root.is_dir():
        raise ValueError("stagewise source snapshot bundle omits stages")
    _root(stages_root)
    observed_stages = list(stages_root.iterdir())
    if (any(not stage.is_dir() or stage.is_symlink() for stage in observed_stages)
            or set(receipts) != {stage.name for stage in observed_stages}):
        raise ValueError("stagewise source snapshot stage roster differs from its binding")
    for name, expected in receipts.items():
        if (not isinstance(name, str) or not _LABEL.fullmatch(name)
                or not isinstance(expected, str)):
            raise ValueError("stagewise source snapshot binding has an invalid stage receipt")
        receipt_path = bundle / "stages" / name / "capture_receipt.json"
        if not receipt_path.is_file():
            raise ValueError(f"stage {name!r} receipt is absent")
        _root(receipt_path)
        if _sha256(receipt_path) != expected:
            raise ValueError(f"stage {name!r} receipt differs from its binding")
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"stage {name!r} receipt is unreadable") from exc
        source = receipt.get("source") if isinstance(receipt, dict) else None
        tool = receipt.get("tool") if isinstance(receipt, dict) else None
        if (not isinstance(receipt, dict)
                or receipt.get("schema") != "m2m.capture-receipt.v1"
                or receipt.get("source_closure_verified") is not False
                or not isinstance(source, dict)
                or source.get("path") != loader["root"]
                or source.get("bytes") != loader_member["bytes"]
                or source.get("sha256") != loader_member["sha256"]):
            raise ValueError(f"stage {name!r} receipt does not bind the snapshot loader")
        if (not isinstance(tool, dict)
                or tool.get("source_inventory_scope") != "m2m_package_python_sources_only"
                or tool.get("source_inventory_status") != "complete"
                or tool.get("source_sha256") != package_sources):
            raise ValueError(f"stage {name!r} receipt does not bind the snapshot M2M package")


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--bundle", action="store_true", help="verify fresh stagewise binding and source snapshot")
    args = parser.parse_args()
    if args.bundle:
        verify_stagewise_source_snapshot_bundle(args.path)
    else:
        verify_source_snapshot_file(args.path)
    print("declared source snapshot matches current files; process source closure remains unverified")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
