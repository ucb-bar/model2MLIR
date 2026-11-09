"""Weight externalization: graph stays inspectable, real data goes to safetensors."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import m2m


def _raw(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def test_exported_nonpersistent_buffers_and_lifted_constants_keep_source_bytes(tmp_path):
    from safetensors.torch import load_file

    class Roles(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([-1.5, 0.0], dtype=torch.bfloat16))
            self.register_buffer("codes", torch.tensor([0, 8, 126, 254], dtype=torch.uint8))
            self.register_buffer("exponents", torch.tensor([-127, 127], dtype=torch.int32))
            self.register_buffer("volatile_codes", torch.tensor([128, 64, 16], dtype=torch.uint8),
                                 persistent=False)
            self.register_buffer("volatile_exponents", torch.tensor([-9, 0, 11], dtype=torch.int32),
                                 persistent=False)

        def forward(self, x):
            return (x, self.weight, self.codes, self.exponents, self.volatile_codes,
                    self.volatile_exponents, torch.tensor(-0.0, dtype=torch.bfloat16))

    model = Roles().eval()
    inputs = (torch.tensor([[2.0, -3.0]], dtype=torch.bfloat16),)
    exported = torch.export.export(model, inputs)
    signature = exported.graph_signature
    nonpersistent = {spec.target for spec in signature.input_specs
                     if spec.kind.name == "BUFFER" and spec.persistent is False}
    assert nonpersistent == {"volatile_codes", "volatile_exponents"}
    assert nonpersistent <= set(exported.constants)
    assert nonpersistent.isdisjoint(exported.state_dict)
    assert len(signature.inputs_to_lifted_tensor_constants) == 1
    source = {**exported.state_dict, **exported.constants}
    before = {name: (str(value.dtype), tuple(value.shape), _raw(value))
              for name, value in source.items()}
    graph_before = str(exported.graph)
    specs_before = repr(signature.input_specs)
    model_before = {name: _raw(value) for name, value in model.named_buffers()}
    persistent_before = set(model.state_dict())
    input_before = _raw(inputs[0])

    path = tmp_path / "roles.safetensors"
    result = m2m.convert(exported, inputs, backend="fx_importer", decompose=False,
                         capture_trace=True,
                         weights_path=str(path))
    assert result.ok, result.diagnostics
    restored = load_file(str(path))
    assert set(restored) == set(source)
    manifest = json.loads(path.with_suffix(path.suffix + ".manifest.json").read_text())
    placeholders = [node for node in exported.graph.nodes if node.op == "placeholder"]
    assert len(manifest) == len(placeholders)
    specs = {spec.arg.name: spec for spec in signature.input_specs}
    for index, node in enumerate(placeholders):
        spec, row = specs[node.name], manifest[str(index)]
        if spec.kind.name == "USER_INPUT":
            assert row == {"kind": "input", "name": node.name}
            continue
        expected_kind = {"PARAMETER": "param", "BUFFER": "buffer",
                         "CONSTANT_TENSOR": "constant"}[spec.kind.name]
        expected = source[spec.target]
        expected_row = {"weight": spec.target, "kind": expected_kind,
                        "dtype": str(expected.dtype).replace("torch.", ""),
                        "shape": list(expected.shape)}
        if spec.kind.name == "CONSTANT_TENSOR":
            expected_row["name"] = node.name
        assert row == expected_row
        assert restored[spec.target].dtype == expected.dtype
        assert tuple(restored[spec.target].shape) == tuple(expected.shape)
        assert _raw(restored[spec.target]) == before[spec.target][2]
    assert str(exported.graph) == graph_before
    assert repr(signature.input_specs) == specs_before
    assert {name: (str(value.dtype), tuple(value.shape), _raw(value))
            for name, value in {**exported.state_dict, **exported.constants}.items()} == before
    assert {name: _raw(value) for name, value in model.named_buffers()} == model_before
    assert set(model.state_dict()) == persistent_before
    assert _raw(inputs[0]) == input_before


def test_tensor_subclass_dead_descriptor_is_still_a_stub(tmp_path):
    from safetensors.torch import load_file
    from xdsl.dialects.builtin import ModuleOp
    from m2m.transforms.externalize import externalize_weights

    class Descriptor(torch.Tensor):
        def dequantize(self):
            raise AssertionError("dead descriptor must never be dequantized")

    value = torch.Tensor._make_subclass(
        Descriptor, torch.ones(2, 3, dtype=torch.bfloat16), require_grad=False)
    exported = SimpleNamespace(
        graph=SimpleNamespace(nodes=[SimpleNamespace(op="placeholder", name="p_weight")]),
        graph_signature=SimpleNamespace(inputs_to_parameters={"p_weight": "weight"},
                                        inputs_to_buffers={}),
        state_dict={"weight": value})
    path = tmp_path / "stub.safetensors"
    externalize_weights(exported, ModuleOp([]), str(path))
    assert json.loads(path.with_suffix(path.suffix + ".manifest.json").read_text()) == {
        "0": {"weight": "weight", "kind": "param", "dtype": "bfloat16",
              "shape": [2, 3], "stub": True}}
    stored = load_file(str(path))["weight"]
    assert stored.dtype == torch.bfloat16 and tuple(stored.shape) == (1,)
    assert _raw(stored) == b"\0\0"


@pytest.mark.parametrize("depth", [0, 2])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_lifted_constant_archive_binding_preserves_bundle_receipt(tmp_path, depth, dtype):
    from safetensors.torch import load_file
    from m2m.capture.bundle import write_bundle
    from m2m.capture.provenance import capture_receipt

    class Literal(nn.Module):
        def forward(self, x):
            return x + torch.tensor(-0.0, dtype=getattr(torch, dtype))

    class Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x)

    model = Literal()
    for _ in range(depth):
        model = Wrapper(model)
    write_bundle(model, (torch.ones(2, dtype=getattr(torch, dtype)),), tmp_path,
                 capture_regions=False, capture_trace=True)
    manifest = json.loads((tmp_path / "weights.safetensors.manifest.json").read_text())
    constant, = [row for row in manifest.values() if row["kind"] == "constant"]
    assert constant["name"] == "c_" + "model_" * depth + "lifted_tensor_0"
    assert constant["weight"] == "model." * depth + "lifted_tensor_0"
    assert constant["dtype"] == dtype and constant["shape"] == []
    assert _raw(load_file(str(tmp_path / "weights.safetensors"))[constant["weight"]]) == \
        _raw(torch.tensor(-0.0, dtype=getattr(torch, dtype)))
    receipt = capture_receipt(tmp_path)
    assert receipt["materialized_abi"]["complete"] is True
    assert set(receipt["lifted_constants"]) == {constant["name"]}


def test_unserializable_tensor_remains_a_best_effort_manifest_error(tmp_path):
    from safetensors.torch import load_file
    from xdsl.dialects.builtin import ModuleOp
    from m2m.transforms.externalize import externalize_weights

    exported = SimpleNamespace(
        graph=SimpleNamespace(nodes=[SimpleNamespace(op="placeholder", name="p_weight")]),
        graph_signature=SimpleNamespace(inputs_to_parameters={"p_weight": "weight"},
                                        inputs_to_buffers={}),
        state_dict={"weight": torch.empty(2, device="meta")})
    path = tmp_path / "unserializable.safetensors"
    externalize_weights(exported, ModuleOp([]), str(path))
    row = json.loads(path.with_suffix(path.suffix + ".manifest.json").read_text())["0"]
    assert row["weight"] == "weight" and row["kind"] == "param" and row["error"]
    assert load_file(str(path)) == {}


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
    from safetensors.torch import load_file

    from m2m.capture.bundle import write_bundle
    from m2m.capture.provenance import capture_receipt

    class Stateless(nn.Module):
        def forward(self, x):
            return x[:, :2] + 1

    write_bundle(Stateless(), (torch.randn(1, 4),), tmp_path,
                 capture_regions=False, capture_trace=True)
    assert load_file(str(tmp_path / "weights.safetensors")) == {}
    receipt = capture_receipt(tmp_path)
    assert receipt["artifacts"]["weights.safetensors"]["bytes"] > 0
    assert receipt["materialized_abi"]["complete"] is True
