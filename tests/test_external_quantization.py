"""The frontend boundary admits only an explicitly selected adapter receipt."""

import hashlib

import pytest
import torch
from torch import nn

from m2m.api import convert
from m2m.capture.external_quantization import ExternalQuantizationConfig, validate_manifest
from m2m.capture.bundle import write_bundle
from m2m.capture.provenance import capture_receipt


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


class _Entry:
    name = "fixture"

    def __init__(self, adapter):
        self.adapter = adapter

    def load(self):
        return self.adapter


def test_external_adapter_binds_selected_bytes_and_mlir(tmp_path, monkeypatch):
    contract = tmp_path / "contract.yaml"
    policy = tmp_path / "policy.yaml"
    contract.write_bytes(b"contract: selected\n")
    policy.write_bytes(b"policy: selected\n")

    def adapter(model, inputs, *, contract_bytes, policy_bytes, original_frontend_snapshot):
        assert contract_bytes == contract.read_bytes()
        assert policy_bytes == policy.read_bytes()
        return model, {
            "schema": "m2m.quantization_manifest.v1", "adapter_id": "fixture",
            "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
            "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
            "sites": [{"site_id": "one", "status": "host"}],
        }

    monkeypatch.setattr("m2m.capture.external_quantization._entry_point",
                        lambda _: _Entry(adapter))
    config = ExternalQuantizationConfig("fixture", contract, policy)
    result = convert(_Model().eval(), (torch.ones(2, 4),),
                     quantization=config, backend="fx_importer")
    assert result.ok, result.diagnostics
    assert result.quantization_manifest["sites"][0]["site_id"] == "one"
    assert "prov.quantization_manifest_sha256" in result.mlir_text


def test_external_adapter_rejects_changed_selected_contract(tmp_path):
    contract = tmp_path / "contract.yaml"
    policy = tmp_path / "policy.yaml"
    contract.write_bytes(b"changed")
    policy.write_bytes(b"policy")
    config = ExternalQuantizationConfig("fixture", contract, policy,
                                        expected_contract_sha256="0" * 64)
    with pytest.raises(ValueError, match="selected contract sha256 differs"):
        convert(_Model(), (torch.ones(2, 4),), quantization=config)


def test_preserved_accelerator_site_is_distinct_from_host(tmp_path):
    contract = tmp_path / "contract.yaml"
    policy = tmp_path / "policy.yaml"
    contract.write_bytes(b"contract: selected\n")
    policy.write_bytes(b"policy: selected\n")
    config = ExternalQuantizationConfig("fixture", contract, policy)
    manifest = {
        "schema": "m2m.quantization_manifest.v2", "adapter_id": "fixture",
        "contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
        "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
        "sites": [
            {"site_id": "functional:mm", "status": "preserved", "execution_route": "simt_float"},
            {"site_id": "functional:matmul", "status": "quantized", "format": "mxfp8"},
            {"site_id": "module:host", "status": "host"},
        ],
    }
    assert validate_manifest(manifest, config) is manifest
    del manifest["sites"][0]["execution_route"]
    with pytest.raises(ValueError, match="execution route"):
        validate_manifest(manifest, config)
    manifest["sites"][0]["execution_route"] = "simt_float"
    manifest["schema"] = "m2m.quantization_manifest.v1"
    with pytest.raises(ValueError, match="explicit status"):
        validate_manifest(manifest, config)


def test_preserved_site_manifest_survives_conversion(tmp_path, monkeypatch):
    contract = tmp_path / "contract.yaml"
    policy = tmp_path / "policy.yaml"
    contract.write_bytes(b"contract: selected\n")
    policy.write_bytes(b"policy: selected\n")

    def adapter(model, _inputs, *, contract_bytes, policy_bytes, original_frontend_snapshot):
        return model, {
            "schema": "m2m.quantization_manifest.v2", "adapter_id": "fixture",
            "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
            "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
            "sites": [{"site_id": "module:linear", "status": "preserved",
                       "execution_route": "simt_float"}],
        }

    monkeypatch.setattr("m2m.capture.external_quantization._entry_point",
                        lambda _: _Entry(adapter))
    result = convert(_Model().eval(), (torch.ones(2, 4),),
                     quantization=ExternalQuantizationConfig("fixture", contract, policy),
                     backend="fx_importer")
    assert result.ok, result.diagnostics
    assert result.quantization_manifest["sites"][0]["execution_route"] == "simt_float"
    assert "prov.quantization_manifest_sha256" in result.mlir_text


def test_external_exported_program_materializes_bundle(tmp_path, monkeypatch):
    contract = tmp_path / "contract.yaml"
    policy = tmp_path / "policy.yaml"
    contract.write_bytes(b"contract: selected\n")
    policy.write_bytes(b"policy: selected\n")

    def adapter(model, inputs, *, contract_bytes, policy_bytes, original_frontend_snapshot):
        return torch.export.export(model, inputs), {
            "schema": "m2m.quantization_manifest.v1", "adapter_id": "fixture",
            "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
            "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
            "sites": [{"site_id": "one", "status": "host"}],
        }

    monkeypatch.setattr("m2m.capture.external_quantization._entry_point",
                        lambda _: _Entry(adapter))
    bundle = tmp_path / "bundle"
    write_bundle(_Model().eval(), (torch.ones(2, 4),), bundle,
                 quant=ExternalQuantizationConfig("fixture", contract, policy),
                 capture_regions=False, capture_trace=True)
    assert (bundle / "model.mlir").is_file()
    assert (bundle / "weights.safetensors.manifest.json").is_file()
    assert (bundle / "capture_receipt.json").is_file()
    assert (bundle / "quantization-manifest.json").is_file()
    assert "prov.quantization_manifest_sha256" in (bundle / "model.mlir").read_text()
    assert "quantization-manifest.json" in capture_receipt(bundle)["artifacts"]
    (bundle / "quantization-manifest.json").write_text("{}\n")
    with pytest.raises(ValueError, match="quantization manifest"):
        capture_receipt(bundle)
    (bundle / "quantization-manifest.json").unlink()
    with pytest.raises(ValueError, match="quantization manifest"):
        capture_receipt(bundle)


class _FakeQuantLinear(nn.Module):
    """A replacement module carrying no lineage back to the original node it replaces."""

    def __init__(self, linear):
        super().__init__()
        self.weight = nn.Parameter(torch.round(linear.weight.detach() * 16) / 16)

    def forward(self, x):
        return (torch.round(x * 16) / 16) @ self.weight.t()


def _replacing_adapter(calls):
    def adapter(model, inputs, *, contract_bytes, policy_bytes, original_frontend_snapshot):
        calls.append(len(inputs))
        model.linear = _FakeQuantLinear(model.linear)
        return model, {
            "schema": "m2m.quantization_manifest.v1", "adapter_id": "fixture",
            "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
            "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
            "sites": [{"site_id": "module:linear", "status": "quantized", "format": "fixture"}],
        }

    return adapter


def _selected(tmp_path):
    contract = tmp_path / "contract.yaml"
    policy = tmp_path / "policy.yaml"
    contract.write_bytes(b"contract: selected\n")
    policy.write_bytes(b"policy: selected\n")
    return ExternalQuantizationConfig("fixture", contract, policy)


def test_trace_reports_a_replacing_quantizer_without_lineage_as_diagnostic(tmp_path, monkeypatch):
    """Module replacement leaves both graphs complete but their correspondence unproven.

    The trace must say so: `diagnostic` with the original -> quantized blocker, never `complete`,
    even though the MLIR itself has no opaque calls.
    """

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 32, bias=False)

        def forward(self, x):
            return self.linear(x) @ x.transpose(-1, -2)

    calls = []
    monkeypatch.setattr("m2m.capture.external_quantization._entry_point",
                        lambda _: _Entry(_replacing_adapter(calls)))
    result = convert(Tiny().eval(), (torch.ones(32, 32),), quantization=_selected(tmp_path),
                     backend="fx_importer", capture_trace=True)
    assert result.ok, result.diagnostics
    assert result.capture_trace["status"] == "diagnostic"
    assert "original -> quantized correspondence incomplete" in result.capture_trace["blockers"]
    assert result.capture_trace["graphs"]["original"]["status"] == "complete"
    assert result.capture_trace["graphs"]["quantized"]["status"] == "complete"
    assert calls == [1]


def test_coverage_report_quantizes_once_with_the_example_inputs(tmp_path, monkeypatch):
    from m2m.api import coverage_report

    calls = []
    monkeypatch.setattr("m2m.capture.external_quantization._entry_point",
                        lambda _: _Entry(_replacing_adapter(calls)))
    report = coverage_report(_Model32().eval(), (torch.ones(32, 32),), quantization=_selected(tmp_path))
    assert report["valid"] is True
    assert report["num_ops"] > 0
    # One adapter call, and it received the example inputs (an external quantizer requires them).
    assert calls == [1]


class _Model32(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 32)

    def forward(self, x):
        return self.linear(x)
