"""Ordinary captures retain model quality independently of selected-model fidelity."""

import json

import numpy as np
import pytest
import torch

from m2m.capture.bundle import _tensor_outputs, write_bundle
from m2m.capture.provenance import capture_receipt
from m2m.capture.quantization_quality import capture_prequant_outputs
from m2m.capture.torchao_pipeline import QuantizationConfig


class Scale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.25))

    def forward(self, x):
        return x * self.weight


def quantize_in_place(model, config, **kwargs):
    with torch.no_grad():
        model.weight.fill_(1.0)
    return model


def test_normal_capture_retains_prequant_but_golden_is_selected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "m2m.capture.torchao_pipeline.apply_quantization", quantize_in_place
    )
    model = Scale().eval()
    value = torch.tensor([[1.0, -2.0], [3.0, 0.0]])
    summary = write_bundle(
        model,
        (value,),
        tmp_path,
        quant=QuantizationConfig("int8_weight_only"),
        capture_regions=False,
        capture_quantization_quality=True,
    )
    np.testing.assert_array_equal(np.load(tmp_path / "golden.npy"), value.numpy())
    with np.load(tmp_path / "prequant_reference.npz") as reference:
        np.testing.assert_array_equal(reference["out0"], (value * 1.25).numpy())
    report = json.loads((tmp_path / "quantization-quality.json").read_text())
    assert report["acceptance_gate_applied"] is False
    assert report["reference_origin"] == "observed_before_quantization"
    assert report["outputs"][0]["max_abs_error"] == 0.75
    assert report["outputs"][0]["relative_l2_error"] == pytest.approx(0.2)
    assert summary["quantization_quality"]["status"] == "MEASURED"
    receipt = capture_receipt(tmp_path)
    assert "prequant_reference.npz" in receipt["artifacts"]
    assert "quantization_selected_outputs.npz" in receipt["artifacts"]


def test_opt_out_adds_no_quality_artifacts_or_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "m2m.capture.torchao_pipeline.apply_quantization", quantize_in_place
    )
    summary = write_bundle(
        Scale(),
        (torch.ones(1, 2),),
        tmp_path,
        quant=QuantizationConfig("int8_weight_only"),
        capture_regions=False,
    )
    assert "quantization_quality" not in summary
    assert not (tmp_path / "quantization-quality.json").exists()
    assert not (tmp_path / "prequant_reference.npz").exists()


@pytest.mark.parametrize("supplied", (False, True))
def test_preapplied_reference_never_falls_back_to_quantized_golden(tmp_path, supplied):
    model, x = Scale(), torch.ones(1, 2)
    baseline = x * 2 if supplied else None
    write_bundle(
        model,
        (x,),
        tmp_path,
        quant=QuantizationConfig("int8_weight_only"),
        quantization_preapplied=True,
        capture_regions=False,
        capture_quantization_quality=True,
        prequant_reference=baseline,
    )
    report = json.loads((tmp_path / "quantization-quality.json").read_text())
    assert report["status"] == ("MEASURED" if supplied else "UNKNOWN")
    if not supplied:
        assert report["reference_checkpoint_equivalence"] == (
            "UNKNOWN; no prequant reference available"
        )
    assert report["reference_origin"] == (
        "caller_supplied" if supplied else "unavailable"
    )
    assert (tmp_path / "prequant_reference.npz").exists() is supplied
    if supplied:
        assert report["outputs"][0]["max_abs_error"] == 0.75


def test_output_snapshot_survives_in_place_parameter_mutation():
    model = Scale()
    before = capture_prequant_outputs(model, (torch.ones(2),), _tensor_outputs)
    quantize_in_place(model, None)
    torch.testing.assert_close(before[0], torch.full((2,), 1.25), rtol=0, atol=0)


def test_auto_reference_restores_rng():
    class Random(torch.nn.Module):
        def forward(self, x):
            return x + torch.rand_like(x)

    torch.manual_seed(12)
    state = torch.random.get_rng_state().clone()
    capture_prequant_outputs(Random(), (torch.ones(2),), _tensor_outputs)
    assert torch.equal(torch.random.get_rng_state(), state)


@pytest.mark.parametrize("mutate", ("input", "buffer"))
def test_auto_reference_refuses_and_restores_visible_mutation(mutate):
    class Mutable(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("counter", torch.tensor(0))

        def forward(self, x):
            if mutate == "input":
                x.add_(1)
            else:
                self.counter.add_(1)
            return x

    model, x = Mutable(), torch.ones(2)
    with pytest.raises(ValueError, match="pure eval"):
        capture_prequant_outputs(model, (x,), _tensor_outputs)
    assert model.counter.item() == 0 and torch.equal(x, torch.ones(2))


def test_nonfinite_reference_is_not_a_quality_pass(tmp_path):
    write_bundle(
        Scale(),
        (torch.tensor([[torch.nan, torch.inf]]),),
        tmp_path,
        quant=QuantizationConfig("int8_weight_only"),
        quantization_preapplied=True,
        capture_regions=False,
        capture_quantization_quality=True,
        prequant_reference=torch.tensor([[torch.nan, torch.inf]]),
    )
    row = json.loads((tmp_path / "quantization-quality.json").read_text())["outputs"][0]
    assert row["status"] == "NONFINITE" and "max_abs_error" not in row


def test_quality_digest_and_bundle_identity_are_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "m2m.capture.torchao_pipeline.apply_quantization", quantize_in_place
    )
    write_bundle(
        Scale(),
        (torch.ones(1, 2),),
        tmp_path,
        quant=QuantizationConfig("int8_weight_only"),
        capture_regions=False,
        capture_quantization_quality=True,
    )
    path = tmp_path / "prequant_reference.npz"
    original = path.read_bytes()
    path.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="quality output"):
        capture_receipt(tmp_path)
    path.write_bytes(original)
    np.save(tmp_path / "golden.npy", np.zeros((1, 2), np.float32))
    with pytest.raises(ValueError, match="stale"):
        capture_receipt(tmp_path)


@pytest.mark.parametrize(
    "options",
    (
        {"prequant_reference": torch.ones(1)},
        {"capture_quantization_quality": True},
        {
            "capture_quantization_quality": True,
            "quant": QuantizationConfig("int8_weight_only"),
            "session": {},
        },
    ),
)
def test_ambiguous_quality_api_refused(tmp_path, options):
    with pytest.raises(ValueError):
        write_bundle(
            Scale(), (torch.ones(1, 2),), tmp_path, capture_regions=False, **options
        )


def test_all_tuple_outputs_and_bf16_abi_retained(tmp_path):
    class Tuple(torch.nn.Module):
        def forward(self, x):
            return x, x + 1

    x = torch.tensor([[0.25, -1]], dtype=torch.bfloat16)
    write_bundle(
        Tuple(),
        (x,),
        tmp_path,
        quant=QuantizationConfig("int8_weight_only"),
        quantization_preapplied=True,
        capture_regions=False,
        capture_quantization_quality=True,
        prequant_reference=(x.clone(), (x + 2).clone()),
    )
    report = json.loads((tmp_path / "quantization-quality.json").read_text())
    assert report["selected_output_count"] == 2 and len(report["outputs"]) == 2
    assert report["outputs"][0]["reference_abi"]["dtype"] == "bf16"
    assert report["outputs"][1]["max_abs_error"] == 1
    with np.load(tmp_path / "quantization_selected_outputs.npz") as selected:
        assert selected.files == ["out0", "out1"]
        np.testing.assert_array_equal(
            selected["out0"], np.load(tmp_path / "golden.npy")
        )


def test_large_integer_metrics_do_not_claim_float64_exactness():
    from m2m.capture.quantization_quality import _errors

    a = np.array([2**63 - 2], np.int64)
    b = np.array([2**63 - 1], np.int64)
    row = _errors(a, b)
    assert row["status"] == "UNAVAILABLE" and row["unequal_elements"] == 1
    assert "max_abs_error" not in row


def test_real_torchao_dynamic_capture_uses_separate_reference(tmp_path):
    pytest.importorskip("torchao")
    torch.manual_seed(41)
    model = torch.nn.Sequential(torch.nn.Linear(32, 8)).eval()
    x = torch.linspace(-2, 2, 96).reshape(3, 32)
    before = model(x).detach().clone()
    write_bundle(
        model,
        (x,),
        tmp_path,
        quant=QuantizationConfig("int8_dyn_act_int8_weight"),
        capture_regions=False,
        capture_quantization_quality=True,
    )
    with np.load(tmp_path / "prequant_reference.npz") as reference:
        np.testing.assert_array_equal(reference["out0"], before.numpy())
    np.testing.assert_array_equal(
        np.load(tmp_path / "golden.npy"), model(x).detach().numpy()
    )
    report = json.loads((tmp_path / "quantization-quality.json").read_text())
    assert report["outputs"][0]["max_abs_error"] > 0
    assert report["task_quality_or_calibration_claimed"] is False


def test_quality_option_preserves_selected_compiler_payload_bytes(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "m2m.capture.torchao_pipeline.apply_quantization", quantize_in_place
    )
    ordinary, measured = tmp_path / "ordinary", tmp_path / "measured"
    for out, enabled in ((ordinary, False), (measured, True)):
        write_bundle(
            Scale(),
            (torch.tensor([[1.0, -2.0]]),),
            out,
            quant=QuantizationConfig("int8_weight_only"),
            capture_regions=False,
            capture_quantization_quality=enabled,
        )
    for name in (
        "weights.safetensors",
        "weights.safetensors.manifest.json",
        "inputs.npz",
        "golden.npy",
        "extra.npz",
        "input_order.json",
    ):
        assert (ordinary / name).read_bytes() == (measured / name).read_bytes()
    assert (ordinary / "model.mlir").read_text().replace(str(ordinary), "BUNDLE") == (
        measured / "model.mlir"
    ).read_text().replace(str(measured), "BUNDLE")


@pytest.mark.parametrize(
    "mutate",
    (
        "resize_input",
        "resize_buffer",
        "replace_buffer",
        "new_buffer",
        "null_buffer",
        "delete_buffer",
    ),
)
def test_refused_shape_and_registry_mutations_do_not_damage_caller(mutate):
    class Mutable(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("counter", torch.arange(2.0))

        def forward(self, x):
            if mutate == "resize_input":
                x.resize_(5)
            elif mutate == "resize_buffer":
                self.counter.resize_(5)
            elif mutate == "replace_buffer":
                self.counter = torch.ones(5)
            elif mutate == "null_buffer":
                self.counter = None
            elif mutate == "delete_buffer":
                del self.counter
            else:
                self.register_buffer("extra", torch.ones(1))
            return x

    model, x = Mutable(), torch.arange(2.0)
    pointer = model.counter
    with pytest.raises(ValueError, match="pure eval"):
        capture_prequant_outputs(model, (x,), _tensor_outputs)
    assert model.counter is pointer
    torch.testing.assert_close(x, torch.arange(2.0), rtol=0, atol=0)
    torch.testing.assert_close(model.counter, torch.arange(2.0), rtol=0, atol=0)
    assert set(model._buffers) == {"counter"}


def test_failing_baseline_restores_rng_and_caller():
    class Failing(torch.nn.Module):
        def forward(self, x):
            torch.rand(3)
            raise RuntimeError("deliberate model failure")

    x = torch.ones(3)
    before = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="deliberate"):
        capture_prequant_outputs(Failing(), (x,), _tensor_outputs)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(x, torch.ones(3))
