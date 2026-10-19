"""Weight externalization: graph stays inspectable, real data goes to safetensors."""

from __future__ import annotations

import json
import os

import m2m
import pytest
import torch
import torch.nn as nn


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 16)
        self.b = nn.Linear(16, 8)

    def forward(self, x):  # noqa: D102
        return self.b(torch.relu(self.a(x)))


def test_weights_externalized_and_recoverable(tmp_path):
    pytest_safetensors = __import__("pytest").importorskip("safetensors")
    from safetensors.torch import load_file

    m = _MLP().eval()
    wp = str(tmp_path / "mlp.safetensors")
    r = m2m.convert(m, (torch.randn(2, 8),), backend="fx_importer", weights_path=wp)
    assert r.ok
    assert "prov.weights_file" in r.mlir_text          # graph references the data file

    st = load_file(wp)                                 # real data is recoverable
    assert set(st.keys()) == {"a.weight", "a.bias", "b.weight", "b.bias"}
    assert torch.allclose(st["a.weight"], m.a.weight)
    assert torch.allclose(st["b.bias"], m.b.bias)

    manifest = json.load(open(wp + ".manifest.json"))  # arg-index -> weight/dtype/shape
    weights = {v["weight"]: v for v in manifest.values() if "weight" in v}
    assert weights["a.weight"]["shape"] == [16, 8]
    assert weights["a.weight"]["kind"] == "param"


def test_parameter_free_program_has_a_real_empty_weight_container(tmp_path):
    from m2m.capture.bundle import write_bundle
    from m2m.capture.provenance import capture_receipt
    from safetensors.torch import load_file

    class Stateless(nn.Module):
        def forward(self, x):
            return x[:, :2] + 1

    write_bundle(Stateless(), (torch.randn(1, 4),), tmp_path,
                 capture_regions=False, capture_trace=True)
    assert load_file(str(tmp_path / "weights.safetensors")) == {}
    receipt = capture_receipt(tmp_path)
    assert receipt["artifacts"]["weights.safetensors"]["bytes"] > 0
    assert receipt["materialized_abi"]["complete"] is True


def test_explicit_weight_reuse_checks_bytes_and_keeps_original_inode(tmp_path):
    from m2m.transforms.externalize import externalize_weights
    from xdsl.dialects.builtin import ModuleOp

    model = _MLP().eval()
    example = (torch.randn(2, 8),)
    source, destination = tmp_path / "original.safetensors", tmp_path / "reused.safetensors"
    original = m2m.convert(model, example, backend="fx_importer", weights_path=str(source))
    assert original.ok
    before = source.read_bytes()
    result = externalize_weights(original.exported_program, ModuleOp([]), str(destination), reuse_source=source)
    assert os.path.samefile(source, destination)
    assert source.read_bytes() == before == destination.read_bytes()
    assert result["reused"]["source"] == str(source.resolve())
    assert json.loads((tmp_path / "reused.safetensors.manifest.json").read_text()) == json.loads(
        (tmp_path / "original.safetensors.manifest.json").read_text())


@pytest.mark.parametrize("change", ["value", "shape", "dtype", "names", "symlink", "existing"])
def test_weight_reuse_refuses_changed_or_ambiguous_storage(tmp_path, change):
    from m2m.transforms.externalize import externalize_weights
    from safetensors.torch import load_file, save_file
    from xdsl.dialects.builtin import ModuleOp

    model = _MLP().eval()
    source, destination = tmp_path / "original.safetensors", tmp_path / "reused.safetensors"
    result = m2m.convert(model, (torch.randn(2, 8),), backend="fx_importer", weights_path=str(source))
    tensors = {name: tensor.clone() for name, tensor in load_file(source).items()}
    if change == "value":
        tensors["a.weight"][0, 0] += 1
    elif change == "shape":
        tensors["a.weight"] = tensors["a.weight"].reshape(-1)
    elif change == "dtype":
        tensors["a.weight"] = tensors["a.weight"].to(torch.float64)
    elif change == "names":
        tensors["other"] = tensors.pop("a.weight")
    elif change == "symlink":
        alias = tmp_path / "alias.safetensors"
        alias.symlink_to(source)
        source = alias
    else:
        destination.write_bytes(b"existing")
    if change not in {"symlink", "existing"}:
        save_file(tensors, source)
    with pytest.raises(ValueError, match="reuse|reused"):
        externalize_weights(result.exported_program, ModuleOp([]), str(destination), reuse_source=source)
    if change != "existing":
        assert not destination.exists()


@pytest.mark.parametrize("change", [None, "nan_payload", "zero_sign"])
def test_weight_reuse_preserves_nonfinite_payloads_and_signed_zero(tmp_path, change):
    from m2m.transforms.externalize import externalize_weights
    from safetensors.torch import save_file
    from xdsl.dialects.builtin import ModuleOp

    class Special(nn.Module):
        def __init__(self):
            super().__init__()
            self.values = nn.Parameter(torch.tensor(
                [0x7fc12345, -2147483648, 0x7f800000, -8388608], dtype=torch.int32).view(torch.float32))

        def forward(self, x):
            return x * self.values

    model = Special()
    exported = torch.export.export(model, (torch.ones(4),))
    values = model.values.detach().clone()
    if change == "nan_payload":
        values.view(torch.int32)[0] = 0x7fc12346
    elif change == "zero_sign":
        values.view(torch.int32)[1] = 0
    source, destination = tmp_path / "source.safetensors", tmp_path / "selected.safetensors"
    save_file({"values": values}, source)
    if change is None:
        externalize_weights(exported, ModuleOp([]), str(destination), reuse_source=source)
        assert os.path.samefile(source, destination)
    else:
        with pytest.raises(ValueError, match="tensor bytes differ"):
            externalize_weights(exported, ModuleOp([]), str(destination), reuse_source=source)
        assert not destination.exists()


def test_weight_reuse_refuses_source_changed_after_tensor_validation(tmp_path, monkeypatch):
    import m2m.transforms.externalize as externalize
    from xdsl.dialects.builtin import ModuleOp

    source = tmp_path / "source.safetensors"
    converted = m2m.convert(_MLP().eval(), (torch.randn(2, 8),),
                            backend="fx_importer", weights_path=str(source))
    real_hash = externalize._file_sha256
    calls = []

    def mutate_before_second_hash(path):
        calls.append(path)
        if len(calls) == 2:
            with source.open("r+b") as stream:
                stream.seek(-1, 2)
                last = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([last[0] ^ 1]))
        return real_hash(path)

    monkeypatch.setattr(externalize, "_file_sha256", mutate_before_second_hash)
    destination = tmp_path / "selected.safetensors"
    with pytest.raises(ValueError, match="source changed during validation"):
        externalize.externalize_weights(converted.exported_program, ModuleOp([]), str(destination),
                                         reuse_source=source)
    assert not destination.exists()
