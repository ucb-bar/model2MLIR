"""A capture receipt binds the actual bundle bytes and observed source."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import sympy
import torch

from m2m.capture.bundle import _lifted_constants, write_bundle
from m2m.capture.provenance import capture_receipt
from m2m.capture.torch_export import _serialize_range_constraints


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)


def test_bundle_receipt_binds_source_tool_framework_and_materialized_abi(tmp_path):
    source = tmp_path / "example.pt2"
    source.write_bytes(b"observed source archive")
    bundle = tmp_path / "bundle"
    write_bundle(_Tiny().eval(), (torch.randn(2, 4),), bundle,
                 capture_regions=False, source_path=source)

    receipt = json.loads((bundle / "capture_receipt.json").read_text())
    assert receipt["schema"] == "m2m.capture-receipt.v1"
    assert receipt["source_closure_verified"] is False
    assert receipt["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert receipt["tool"]["commit"]
    assert receipt["tool"]["source_sha256"]["m2m/ir/decompositions.py"]
    assert receipt["tool"]["source_inventory_status"] == "complete"
    assert not receipt["tool"]["unavailable_sources"]
    assert all(name.startswith("m2m/") for name in receipt["tool"]["source_sha256"])
    assert receipt["framework"]["torch"] == torch.__version__
    assert receipt["lifted_constants"] == {}
    assert receipt["materialized_abi"] == {"complete": True, "inputs": 1, "lifted_constants": []}
    for name in ("model.mlir", "weights.safetensors", "inputs.npz", "golden.npy",
                 "extra.npz", "input_order.json"):
        record = receipt["artifacts"][name]
        data = (bundle / name).read_bytes()
        assert record == {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    # The original PT2 can lift a scalar. NumPy's ascontiguousarray promotes it
    # to [1], which would silently put the wrong ABI shape in the receipt.
    manifest_path = bundle / "weights.safetensors.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[str(len(manifest))] = {"kind": "constant", "name": "c_lifted_tensor_0"}
    manifest_path.write_text(json.dumps(manifest))
    np.savez(bundle / "extra.npz", c_lifted_tensor_0=np.array(1.0, dtype=np.float32))
    assert capture_receipt(bundle)["lifted_constants"]["c_lifted_tensor_0"]["shape"] == []


def test_unbounded_export_constraint_does_not_break_lifted_constant_capture():
    exported = SimpleNamespace(range_constraints={
        sympy.Symbol("s0"): SimpleNamespace(lower=1, upper=sympy.oo),
    })
    assert _serialize_range_constraints(exported)[0].minimum == 1
    assert _serialize_range_constraints(exported)[0].maximum is None


def test_original_export_supplies_lifted_constants_without_second_export():
    from torch.export.graph_signature import ConstantArgument, InputKind, InputSpec

    exported = SimpleNamespace(
        constants={"lifted_tensor_0": torch.tensor([True, False])},
        state_dict={},
        graph_signature=SimpleNamespace(input_specs=[
            InputSpec(InputKind.CONSTANT_TENSOR, ConstantArgument("c_lifted_tensor_0", None),
                      "lifted_tensor_0"),
        ]),
    )
    extra = {}
    _lifted_constants(None, (), extra, exported_program=exported)
    assert extra["c_lifted_tensor_0"].tolist() == [True, False]


def test_traced_bundle_and_reused_conversion_bind_exact_bytes_without_recapture(tmp_path, monkeypatch):
    import m2m
    import m2m.capture.torch_export as frontend

    original_capture = frontend.capture_model
    calls = []

    def capture(*args, **kwargs):
        calls.append(True)
        return original_capture(*args, **kwargs)

    monkeypatch.setattr(frontend, "capture_model", capture)
    model, inputs = _Tiny().eval(), (torch.ones(2, 4),)
    out = tmp_path / "bundle"
    out.mkdir()
    (out / "meta.json").write_text(json.dumps({"loader_provenance": {"source": "authored"}}))
    write_bundle(model, inputs, out, capture_trace=True, capture_regions=False,
                 metadata={"workload_role": "iteration"})
    assert len(calls) == 1  # no separate lifted-constant capture
    trace = json.loads((out / "frontend-trace.json").read_text())
    meta = json.loads((out / "meta.json").read_text())
    receipt = json.loads((out / "capture_receipt.json").read_text())
    assert trace["status"] == "complete" and meta["workload_role"] == "iteration"
    assert meta["loader_provenance"] == {"source": "authored"}
    for name in ("model.mlir", "frontend-trace.json", "meta.json"):
        assert receipt["artifacts"][name]["sha256"] == hashlib.sha256((out / name).read_bytes()).hexdigest()

    reused = tmp_path / "reused"
    reused.mkdir()
    result = m2m.convert(model, inputs, backend="fx_importer", capture_trace=True,
                         weights_path=str(reused / "weights.safetensors"))
    expected_weights = (reused / "weights.safetensors").read_bytes()
    before = len(calls)
    write_bundle(model, inputs, reused, capture_trace=True, capture_regions=False,
                 conversion_result=result)
    assert len(calls) == before
    assert (reused / "weights.safetensors").read_bytes() == expected_weights
    assert (reused / "model.mlir").read_text() == result.mlir_text

    # Byte relocation/normalization requires a new receipt, never blind rebind.
    (reused / "model.mlir").write_text(result.mlir_text + "\n")
    import pytest
    with pytest.raises(ValueError, match="exact model.mlir"):
        capture_receipt(reused)


def test_static_pt2e_bundle_golden_is_actual_quantized_model(tmp_path, monkeypatch):
    import m2m.capture.torchao_pipeline as pipeline
    from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization

    torch.manual_seed(0)
    model, inputs = _Tiny().eval(), (torch.randn(2, 4),)
    expected = apply_quantization(model, QuantizationConfig(scheme="int8_static_act_int8_weight"),
                                  example_inputs=inputs)
    expected_golden = expected(*inputs).detach().numpy()
    # Use the actual returned PT2E GraphModule at the seam where the writer used
    # to ignore it; no second quantization instance is necessary.
    monkeypatch.setattr(pipeline, "apply_quantization", lambda *args, **kwargs: expected)
    out = tmp_path / "pt2e"
    write_bundle(model, inputs, out, capture_regions=False, capture_trace=True,
                 quant=QuantizationConfig(scheme="int8_static_act_int8_weight"))
    np.testing.assert_array_equal(np.load(out / "golden.npy"), expected_golden)
