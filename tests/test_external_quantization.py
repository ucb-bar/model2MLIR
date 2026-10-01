"""The frontend boundary admits only an explicitly selected adapter receipt."""

import hashlib

import pytest
import torch
from torch import nn

from m2m.api import convert
from m2m.capture.external_quantization import ExternalQuantizationConfig
from m2m.capture.bundle import write_bundle


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
    assert "prov.quantization_manifest_sha256" in (bundle / "model.mlir").read_text()
