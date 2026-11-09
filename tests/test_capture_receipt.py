"""A capture receipt binds the actual bundle bytes and observed source."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import sympy
import torch
from m2m.capture.bundle import (
    _lifted_constants,
    _numpy_safe,
    _opaque_call_count,
    _tensor_abi,
    write_bundle,
)
from m2m.capture.provenance import _package_python_sources, capture_receipt
from m2m.capture.torch_export import _serialize_range_constraints


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)


class _LiteralRange(torch.nn.Module):
    def __init__(self, start, end):
        super().__init__()
        self.start = start
        self.end = end

    def forward(self, x):
        return torch.arange(self.start, self.end, dtype=torch.int64, device=x.device)


def test_bundle_preserves_exact_integer_golden_and_receipt(tmp_path):
    ranges = (
        (2**24 + 1, 2**24 + 6),
        (-(2**63) + 9, -(2**63) + 14),
        (2**63 - 14, 2**63 - 9),
    )
    for index, (start, end) in enumerate(ranges):
        out = tmp_path / f"range-{index}"
        write_bundle(_LiteralRange(start, end).eval(), (torch.ones(1),), out,
                     capture_regions=False, capture_trace=True)
        expected = torch.arange(start, end, dtype=torch.int64).numpy()
        actual = np.load(out / "golden.npy", allow_pickle=False)
        assert actual.dtype == np.dtype("int64")
        np.testing.assert_array_equal(actual.view("uint64"), expected.view("uint64"))
        assert capture_receipt(out)["artifacts"]["golden.npy"]["sha256"] == hashlib.sha256(
            (out / "golden.npy").read_bytes()
        ).hexdigest()


def test_bundle_golden_numpy_storage_policy_leaves_representable_dtypes_exact():
    f32 = torch.tensor([-0.0, 1.25], dtype=torch.float32)
    i64 = torch.tensor([2**63 - 1, -(2**63)], dtype=torch.int64)
    np.testing.assert_array_equal(_numpy_safe(f32).view("uint32"), f32.numpy().view("uint32"))
    np.testing.assert_array_equal(_numpy_safe(i64).view("uint64"), i64.numpy().view("uint64"))
    assert _numpy_safe(f32).dtype == np.dtype("float32")
    assert _numpy_safe(i64).dtype == np.dtype("int64")
    bf16 = torch.tensor([1.25, -0.0], dtype=torch.bfloat16)
    assert _numpy_safe(bf16).dtype == np.dtype("float32")
    np.testing.assert_array_equal(_numpy_safe(bf16), bf16.float().numpy())


def test_runtime_input_name_and_unrelated_extras_do_not_become_constants(tmp_path):
    class Runtime(torch.nn.Module):
        def forward(self, c_model_model_lifted_tensor_input):
            return c_model_model_lifted_tensor_input

    value = torch.tensor([-0.0, 1.25])
    write_bundle(Runtime(), (value,), tmp_path, capture_regions=False, capture_trace=True)
    assert json.loads((tmp_path / "input_order.json").read_text()) == {
        "c_model_model_lifted_tensor_input": 0}
    np.savez(tmp_path / "extra.npz", c_model_model_lifted_tensor_input=value.numpy(),
             c_model_model_lifted_tensor_0=value.numpy())
    receipt = capture_receipt(tmp_path)
    assert receipt["lifted_constants"] == {}
    assert receipt["materialized_abi"] == {"complete": True, "inputs": 1, "lifted_constants": []}


def test_nested_lifted_constant_requires_its_declared_extra_and_closed_role(tmp_path):
    import pytest

    class Literal(torch.nn.Module):
        def forward(self, x):
            return x + torch.tensor(-0.0)

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Literal()

        def forward(self, x):
            return self.model(x)

    write_bundle(Wrapper(), (torch.ones(2),), tmp_path, capture_regions=False, capture_trace=True)
    path = tmp_path / "weights.safetensors.manifest.json"
    manifest = json.loads(path.read_text())
    index, row = next((index, row) for index, row in manifest.items() if row["kind"] == "constant")
    assert row["name"] == "c_model_lifted_tensor_0"
    for mutation in ("unknown_field", "runtime_role", "runtime_overlap", "missing_extra"):
        changed = {key: dict(value) for key, value in manifest.items()}
        if mutation == "unknown_field":
            changed[index]["unknown"] = True
        elif mutation == "runtime_role":
            changed[index] = {"kind": "input", "name": row["name"]}
        elif mutation == "runtime_overlap":
            order_path = tmp_path / "input_order.json"
            original_order = order_path.read_bytes()
            order_path.write_text(json.dumps({row["name"]: 0}))
        else:
            np.savez(tmp_path / "extra.npz", unrelated=np.array(-0.0, dtype="float32"))
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="fields|position|incomplete"):
            capture_receipt(tmp_path)
        if mutation == "runtime_overlap":
            order_path.write_bytes(original_order)


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
    assert receipt["tool"]["source_inventory_scope"] == "m2m_package_python_sources_only"
    assert not receipt["tool"]["unavailable_sources"]
    assert all(name.startswith("m2m/") for name in receipt["tool"]["source_sha256"])
    assert "m2m/capture/stagewise_pt2e.py" in receipt["tool"]["source_sha256"]
    assert "m2m/capture/external_runtime.py" in receipt["tool"]["source_sha256"]
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


def test_python_source_inventory_finds_new_nested_modules_and_flags_links(tmp_path):
    package = tmp_path / "m2m"
    nested = package / "capture"
    nested.mkdir(parents=True)
    (package / "__init__.py").write_text("# package\n")
    (nested / "future_stage.py").write_text("VALUE = 1\n")
    cache = nested / "__pycache__"
    cache.mkdir()
    (cache / "generated.py").write_text("VALUE = 0\n")

    hashes, unavailable = _package_python_sources(tmp_path)
    assert set(hashes) == {"m2m/__init__.py", "m2m/capture/future_stage.py"}
    assert hashes["m2m/capture/future_stage.py"] == hashlib.sha256(b"VALUE = 1\n").hexdigest()
    assert unavailable == []

    (nested / "linked.py").symlink_to(nested / "future_stage.py")
    hashes, unavailable = _package_python_sources(tmp_path)
    assert "m2m/capture/linked.py" not in hashes
    assert unavailable == ["m2m/capture/linked.py"]


def test_integer_reference_is_bound_to_capture_receipt(tmp_path):
    import pytest

    bundle = tmp_path / "bundle"
    write_bundle(_Tiny().eval(), (torch.randn(2, 4),), bundle, capture_regions=False)
    reference = bundle / "integer-reference.json"
    reference.write_text('{"outputs": [1]}\n')
    digest = hashlib.sha256(reference.read_bytes()).hexdigest()
    meta = {"integerization_receipt": {"golden_agreement": {
        "reference": "pt2e_integer", "output": {"path": reference.name, "sha256": digest},
    }}}
    (bundle / "meta.json").write_text(json.dumps(meta))
    receipt = capture_receipt(bundle)
    assert receipt["artifacts"][reference.name]["sha256"] == digest

    reference.write_text('{"outputs": [2]}\n')
    with pytest.raises(ValueError, match="differs"):
        capture_receipt(bundle)
    meta["integerization_receipt"]["golden_agreement"]["output"]["path"] = "../integer-reference.json"
    (bundle / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="bundle-local"):
        capture_receipt(bundle)


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
    (out / "meta.json").write_text(json.dumps({"loader_provenance": {"source": "authored"}, "opaque": 7}))
    write_bundle(model, inputs, out, capture_trace=True, capture_regions=False,
                 metadata={"workload_role": "iteration"})
    assert len(calls) == 1  # no separate lifted-constant capture
    trace = json.loads((out / "frontend-trace.json").read_text())
    meta = json.loads((out / "meta.json").read_text())
    receipt = json.loads((out / "capture_receipt.json").read_text())
    assert trace["status"] == "complete" and meta["workload_role"] == "iteration"
    assert meta["opaque"] == 0
    assert meta["weights"] == str(out / "weights.safetensors")
    assert meta["input_abi"] == [{"shape": [2, 4], "dtype": "f32"}]
    assert meta["output_abi"] == [{"shape": [2, 3], "dtype": "f32"}]
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


def test_opaque_count_is_structural_and_unknown_without_module():
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.dialects.func import CallOp

    assert _opaque_call_count(ModuleOp([CallOp("unlowered", [], [])])) == 1
    assert _opaque_call_count(None) is None
    assert _tensor_abi(torch.empty(2, 3, dtype=torch.bfloat16)) == {"shape": [2, 3], "dtype": "bf16"}


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


def test_bundle_extra_preserves_registered_buffer_dtypes_and_values(tmp_path):
    class Buffered(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("large_ids", torch.tensor([2**40 + 1], dtype=torch.int64))
            self.register_buffer("quantized", torch.tensor([-128, 127], dtype=torch.int8))
            self.register_buffer("mask", torch.tensor([True, False]))
            self.register_buffer("wide", torch.tensor([1.0 + 2**-40], dtype=torch.float64))
            self.register_buffer("bf16", torch.tensor([1.25], dtype=torch.bfloat16))

        def forward(self, x):
            return x + self.quantized.float().sum()

    model = Buffered().eval()
    write_bundle(model, (torch.ones(2),), tmp_path, capture_regions=False)
    with np.load(tmp_path / "extra.npz") as extra:
        for name, tensor in model.named_buffers():
            expected = tensor.float().numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
            actual = extra["buf::" + name]
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_exported_bundle_reuses_real_conversion_and_preserves_nested_roles(tmp_path, monkeypatch, dtype):
    import m2m
    import m2m.capture.torch_export as frontend
    from safetensors import safe_open

    class Leaf(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.5, -0.5], dtype=dtype))
            self.register_buffer("weight_codes", torch.tensor([129, 7], dtype=torch.uint8), persistent=False)
            self.register_buffer("weight_scale_exp", torch.tensor([-3, 1], dtype=torch.int32), persistent=False)

        def forward(self, x):
            codes = self.weight_codes.to(dtype)
            exponents = self.weight_scale_exp.to(dtype)
            return x * self.weight + codes * (exponents + 1) + torch.tensor(-0.0, dtype=dtype)

    class Nested(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Leaf()

        def forward(self, x):
            return self.model(x)

    def tensor_bytes(value):
        return value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()

    def record(ep):
        return {
            "graph": str(ep.graph), "signature": str(ep.graph_signature),
            "stores": {label: {name: (str(value.dtype), tuple(value.shape), tensor_bytes(value))
                               for name, value in values.items()}
                       for label, values in (("state_dict", ep.state_dict), ("constants", ep.constants))},
        }

    inputs = (torch.tensor([2.0, -1.0], dtype=dtype),)
    ep = torch.export.export(Nested().eval(), inputs)
    before = record(ep)
    expected = ep.module()(*inputs).detach()
    token_bytes = tensor_bytes(inputs[0])
    out = tmp_path / "bundle"
    out.mkdir()
    converted = m2m.convert(ep, inputs, backend="fx_importer", capture_trace=True,
                            decompose=False, weights_path=str(out / "weights.safetensors"))
    assert converted.ok and converted.exported_program is ep
    weights_before = (out / "weights.safetensors").read_bytes()
    manifest_before = (out / "weights.safetensors.manifest.json").read_bytes()
    assert record(ep) == before

    def unexpected_capture(*args, **kwargs):
        raise AssertionError("publication must not convert or export again")

    monkeypatch.setattr(m2m, "convert", unexpected_capture)
    monkeypatch.setattr(torch.export, "export", unexpected_capture)
    monkeypatch.setattr(frontend, "capture_frontend_artifact", unexpected_capture)
    forwards = []

    def observed_forward(module, args, output):
        if isinstance(module, torch.fx.GraphModule) and isinstance(output, torch.Tensor):
            forwards.append((output.device.type, output.dtype, tensor_bytes(output)))

    hook = torch.nn.modules.module.register_module_forward_hook(observed_forward)
    try:
        write_bundle(ep, inputs, out, capture_regions=False, capture_trace=True,
                     conversion_result=converted, exported_program=ep)
    finally:
        hook.remove()
    assert forwards == [("cpu", dtype, tensor_bytes(expected))]
    assert record(ep) == before and record(converted.exported_program) == before
    assert tensor_bytes(inputs[0]) == token_bytes
    assert (out / "weights.safetensors").read_bytes() == weights_before
    assert (out / "weights.safetensors.manifest.json").read_bytes() == manifest_before
    assert (out / "model.mlir").read_text() == converted.mlir_text
    trace = json.loads((out / "frontend-trace.json").read_text())
    assert trace == converted.capture_trace and trace["status"] == "complete"
    assert trace["mlir"]["sha256"] == hashlib.sha256(converted.mlir_text.encode()).hexdigest()
    manifest = json.loads(manifest_before)
    lifted = [spec for spec in ep.graph_signature.input_specs if spec.kind.name != "USER_INPUT"]
    targets = {spec.target for spec in lifted}
    assert {spec.kind.name for spec in lifted} == {"PARAMETER", "BUFFER", "CONSTANT_TENSOR"}
    with safe_open(str(out / "weights.safetensors"), framework="pt", device="cpu") as archive:
        assert set(archive.keys()) == targets
        for ordinal, spec in enumerate(ep.graph_signature.input_specs):
            if spec.kind.name == "USER_INPUT":
                continue
            source = ep.state_dict.get(spec.target, ep.constants.get(spec.target))
            row = manifest[str(ordinal)]
            assert row["weight"] == spec.target and not row.get("stub") and "error" not in row
            assert row["kind"] == {"PARAMETER": "param", "BUFFER": "buffer", "CONSTANT_TENSOR": "constant"}[spec.kind.name]
            actual = archive.get_tensor(spec.target)
            assert actual.dtype == source.dtype and tuple(actual.shape) == tuple(source.shape)
            assert tensor_bytes(actual) == tensor_bytes(source)
            if spec.kind.name == "CONSTANT_TENSOR":
                assert row["name"] == spec.arg.name
    constant = next(spec for spec in lifted if spec.kind.name == "CONSTANT_TENSOR")
    assert constant.arg.name.startswith("c_model_")
    receipt = capture_receipt(out)
    assert receipt["materialized_abi"] == {"complete": True, "inputs": 1,
                                         "lifted_constants": [constant.arg.name]}
    with np.load(out / "extra.npz", allow_pickle=False) as extra:
        np.testing.assert_array_equal(extra[constant.arg.name],
                                      ep.constants[constant.target].float().numpy())
    golden = np.load(out / "golden.npy", allow_pickle=False)
    expected_numpy = expected.float().numpy() if dtype == torch.bfloat16 else expected.numpy()
    assert golden.dtype == expected_numpy.dtype and golden.shape == expected_numpy.shape
    np.testing.assert_array_equal(golden.view("uint32"), expected_numpy.view("uint32"))


def test_bundle_still_evaluates_ordinary_training_modules(tmp_path):
    class Training(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(1.0)

        def forward(self, x):
            return self.dropout(x)

    model = Training()
    assert model.training and model.dropout.training
    inputs = (torch.ones(2),)
    write_bundle(model, inputs, tmp_path, capture_regions=False, capture_trace=True)
    assert not model.training and not model.dropout.training
    np.testing.assert_array_equal(np.load(tmp_path / "golden.npy", allow_pickle=False), inputs[0].numpy())
