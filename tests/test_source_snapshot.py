"""Tiny, dependency-free issuer/replay checks for stagewise input snapshots."""

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "m2m" / "capture" / "source_closure.py"
SPEC = importlib.util.spec_from_file_location("source_snapshot_under_test", SOURCE)
closure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(closure)


class SourceSnapshotTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / "m2m"
        self.package.mkdir()
        (self.package / "__init__.py").write_text("# m2m fixture\n")
        capture = self.package / "capture"
        capture.mkdir()
        (capture / "source.py").write_text("SOURCE = 1\n")
        self.loader = self.root / "loader.py"
        self.loader.write_text("def load(): return 1\n")
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "weights.bin").write_bytes(b"weights-1")
        self.support = self.root / "support"
        self.support.mkdir()
        (self.support / "runtime.so").write_bytes(b"runtime-1")

    def snapshot(self):
        return closure.make_source_snapshot(
            package_root=self.package, loader=self.loader,
            model_roots={"checkpoint": self.model},
            support_roots={"runtime": self.support})

    def test_exact_ownership_membership_bytes_and_link_target(self):
        snapshot = self.snapshot()
        closure.verify_source_snapshot(snapshot)
        self.assertFalse(snapshot["source_closure_verified"])
        self.assertEqual([owner["role"] for owner in snapshot["owners"]],
                         ["m2m_package", "loader", "model", "support"])
        self.assertEqual(snapshot["owners"][2]["members"][1]["sha256"],
                         hashlib.sha256(b"weights-1").hexdigest())

        (self.model / "weights.bin").write_bytes(b"weights-2")
        with self.assertRaisesRegex(ValueError, "membership or bytes"):
            closure.verify_source_snapshot(snapshot)
        (self.model / "weights.bin").write_bytes(b"weights-1")
        (self.support / "new.py").write_text("NEW = 1\n")
        with self.assertRaisesRegex(ValueError, "membership or bytes"):
            closure.verify_source_snapshot(snapshot)

        (self.support / "new.py").unlink()
        (self.support / "lib-link").symlink_to("runtime.so")
        linked = self.snapshot()
        closure.verify_source_snapshot(linked)
        (self.support / "lib-link").unlink()
        (self.support / "lib-link").symlink_to(self.loader)
        with self.assertRaisesRegex(ValueError, "membership or bytes"):
            closure.verify_source_snapshot(linked)
        cross_owner = self.snapshot()
        closure.verify_source_snapshot(cross_owner)

    def test_refuses_unowned_link_and_forged_closure(self):
        (self.model / "outside").symlink_to("/etc/hosts")
        with self.assertRaisesRegex(ValueError, "escapes declared owners"):
            self.snapshot()
        (self.model / "outside").unlink()
        snapshot = self.snapshot()
        snapshot["source_closure_verified"] = True
        with self.assertRaisesRegex(ValueError, "closure claim"):
            closure.verify_source_snapshot(snapshot)
        with self.assertRaisesRegex(ValueError, "overlap"):
            closure.make_source_snapshot(
                package_root=self.package, loader=self.loader,
                model_roots={"checkpoint": self.model},
                support_roots={"duplicate": self.model})

    def test_independent_process_replays_fresh_bundle_links(self):
        snapshot = self.snapshot()
        bundle = self.root / "bundle"
        bundle.mkdir()
        manifest = bundle / "source-snapshot.json"
        manifest.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
        loader_member = snapshot["owners"][1]["members"][0]
        package_sources = {
            f"m2m/{member['path']}": member["sha256"]
            for member in snapshot["owners"][0]["members"]
            if member["kind"] == "file" and member["path"].endswith(".py")
        }
        receipts = {}
        for name in ("prefix", "flow"):
            stage = bundle / "stages" / name
            stage.mkdir(parents=True)
            receipt = {"schema": "m2m.capture-receipt.v1", "source_closure_verified": False,
                       "source": {"path": str(self.loader), "bytes": loader_member["bytes"],
                                  "sha256": loader_member["sha256"]},
                       "tool": {"source_inventory_scope": "m2m_package_python_sources_only",
                                "source_inventory_status": "complete",
                                "source_sha256": package_sources}}
            receipt_path = stage / "capture_receipt.json"
            receipt_path.write_text(json.dumps(receipt, sort_keys=True))
            receipts[name] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        binding = {
            "schema": "m2m.stagewise-pt2e-binding.v1", "source_closure_verified": False,
            "scope": "one_source_model_verified_semantic_weights_separate_stage_storage",
            "source_snapshot": {
                "path": manifest.name, "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "verified_before_and_after_capture": True, "source_closure_verified": False},
            "stage_receipts": receipts,
        }
        (bundle / "quantized-session-binding.json").write_text(json.dumps(binding))

        command = [sys.executable, str(SOURCE), str(bundle), "--bundle"]
        passed = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(passed.returncode, 0, passed.stderr)
        self.assertIn("closure remains unverified", passed.stdout)
        (self.model / "weights.bin").write_bytes(b"changed")
        refused = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("membership or bytes", refused.stderr)

        (self.model / "weights.bin").write_bytes(b"weights-1")
        flow_path = bundle / "stages" / "flow" / "capture_receipt.json"
        flow = json.loads(flow_path.read_text())
        flow["source"]["path"] = str(self.model / "weights.bin")
        flow_path.write_text(json.dumps(flow, sort_keys=True))
        binding["stage_receipts"]["flow"] = hashlib.sha256(flow_path.read_bytes()).hexdigest()
        (bundle / "quantized-session-binding.json").write_text(json.dumps(binding))
        refused = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("does not bind the snapshot loader", refused.stderr)

        flow["source"]["path"] = str(self.loader)
        flow["tool"]["source_sha256"]["m2m/capture/source.py"] = "0" * 64
        flow_path.write_text(json.dumps(flow, sort_keys=True))
        binding["stage_receipts"]["flow"] = hashlib.sha256(flow_path.read_bytes()).hexdigest()
        (bundle / "quantized-session-binding.json").write_text(json.dumps(binding))
        refused = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("does not bind the snapshot M2M package", refused.stderr)


if __name__ == "__main__":
    unittest.main()
