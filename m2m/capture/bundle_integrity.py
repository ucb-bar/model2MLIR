"""Deterministic byte inventory for a multi-program capture directory.

This binds the files *in* a completed bundle, including root and stage session
artifacts. It is not a source-dependency, semantic, or execution proof. The
receipt itself is excluded to avoid a self-referential hash.
"""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path, PurePosixPath


RECEIPT_NAME = "bundle_integrity.json"
SCHEMA = "m2m.multi-program-integrity.v1"


def _bundle_files(root: Path) -> list[tuple[str, Path]]:
    """Inventory plain files only; refuse symlinks even in unused subtrees."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("multi-program bundle root must be a plain directory")
    files: list[tuple[str, Path]] = []

    def visit(directory: Path) -> None:
        for path in sorted(directory.iterdir(), key=lambda child: child.name):
            mode = path.lstat().st_mode
            relative = path.relative_to(root).as_posix()
            if stat.S_ISLNK(mode):
                raise ValueError(f"multi-program bundle contains a symlink: {relative}")
            if stat.S_ISDIR(mode):
                visit(path)
            elif stat.S_ISREG(mode):
                if relative != RECEIPT_NAME:
                    files.append((relative, path))
            else:
                raise ValueError(f"multi-program bundle contains a non-file: {relative}")

    visit(root)
    return sorted(files)


def _record(relative: str, path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": relative, "bytes": size, "sha256": digest.hexdigest()}


def _safe_record_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("multi-program integrity receipt has an unsafe path")
    relative = PurePosixPath(value)
    if (relative.is_absolute() or str(relative) != value or ".." in relative.parts
            or value == RECEIPT_NAME):
        raise ValueError(f"multi-program integrity receipt has an unsafe path: {value}")
    return value


def write_bundle_integrity(root: str | Path) -> Path:
    """Write a sorted, byte-bound inventory after all program artifacts exist."""
    root = Path(root)
    files = _bundle_files(root)
    if "session_contract.yaml" not in {name for name, _ in files}:
        raise ValueError("multi-program bundle omitted session_contract.yaml")
    receipt = {
        "schema": SCHEMA,
        "scope": "captured_bundle_file_integrity_only",
        "source_closure_verified": False,
        "files": [_record(name, path) for name, path in files],
    }
    destination = root / RECEIPT_NAME
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def verify_bundle_integrity(root: str | Path) -> Path:
    """Require an exact file roster and verify every observed byte/hash pair."""
    root = Path(root)
    files = _bundle_files(root)
    destination = root / RECEIPT_NAME
    if not destination.is_file():
        raise ValueError("multi-program bundle omitted bundle_integrity.json")
    try:
        receipt = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("multi-program integrity receipt is unreadable") from exc
    if (not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA
            or receipt.get("scope") != "captured_bundle_file_integrity_only"
            or receipt.get("source_closure_verified") is not False
            or not isinstance(receipt.get("files"), list)):
        raise ValueError("multi-program integrity receipt has an invalid schema or scope")
    records = receipt["files"]
    names = []
    for record in records:
        if (not isinstance(record, dict)
                or set(record) != {"path", "bytes", "sha256"}
                or type(record["bytes"]) is not int or record["bytes"] < 0
                or not isinstance(record["sha256"], str)
                or len(record["sha256"]) != 64
                or any(ch not in "0123456789abcdef" for ch in record["sha256"])):
            raise ValueError("multi-program integrity receipt has a malformed file record")
        names.append(_safe_record_path(record["path"]))
    if names != sorted(set(names)) or "session_contract.yaml" not in names:
        raise ValueError("multi-program integrity receipt has an invalid file roster")
    observed = [name for name, _ in files]
    if names != observed:
        raise ValueError("multi-program integrity receipt does not match the bundle file roster")
    for expected, (name, path) in zip(records, files, strict=True):
        if expected != _record(name, path):
            raise ValueError(f"multi-program integrity receipt does not bind {name}")
    return destination
