"""Real PT2E captures must prove the original-to-frozen weight relation."""

from __future__ import annotations

import copy
import json

import pytest
import torch
from torch import nn
from safetensors.torch import load_file

import m2m
from m2m.capture.torchao_pipeline import QuantizationConfig
from m2m.capture.weight_lineage import bind_frozen_weights, snapshot_effective_weights, snapshot_source_weights


class _DevelopmentNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, 2)
        with torch.no_grad():
            self.bn.running_mean.fill_(0.25)
            self.bn.running_var.fill_(0.5)
            self.bn.weight.fill_(1.75)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.bn(self.conv(x)).relu().mean((2, 3)))


def _verify_files(path, *, expected_granularity: str) -> dict:
    receipt = json.loads((path.parent / (path.name + ".quantization.json")).read_text())
    prequant = load_file(str(path.parent / (path.name + ".prequant.safetensors")))
    frozen = load_file(str(path))
    assert receipt["schema"] == "m2m.quant_weight_lineage.v1"
    assert receipt["formula"]["verified_against_frozen_bytes"] is True
    assert len(receipt["weights"]) == 2
    assert {row["qparams"]["granularity"] for row in receipt["weights"]} == {expected_granularity}
    for row in receipt["weights"]:
        assert row["fx_operand"]["index"] == 1
        assert row["original"]["key"] in prequant
        assert row["effective"]["key"] in prequant
        assert row["frozen"]["key"] in frozen
        assert frozen[row["frozen"]["key"]].dtype is torch.int8
        if expected_granularity == "per_channel":
            assert row["qparams"]["scale"]["key"] in frozen
            assert row["qparams"]["zero_point"]["key"] in frozen
            assert row["qparams"]["axis"] == 0
        else:
            float.fromhex(row["qparams"]["scale"]["literal_float64_hex"])
            assert row["qparams"]["axis"] is None
    conv = next(row for row in receipt["weights"] if "conv" in row["original"]["state_key"])
    assert not torch.equal(prequant[conv["original"]["key"]], prequant[conv["effective"]["key"]])
    return receipt


def test_per_channel_capture_records_exact_original_effective_and_frozen_weights(tmp_path):
    torch.manual_seed(9)
    model = _DevelopmentNet().eval()
    same_model = copy.deepcopy(model)
    inputs = (torch.randn(1, 3, 8, 8),)
    config = QuantizationConfig(scheme="int8_static_act_int8_weight", calibration_samples=1)
    path = tmp_path / "one" / "weights.safetensors"
    path.parent.mkdir()
    result = m2m.convert(
        model, inputs, quantization=config, calibration_inputs=[inputs],
        backend="fx_importer", decompose=False, weights_path=str(path),
    )
    assert result.ok, result.diagnostics
    receipt = _verify_files(path, expected_granularity="per_channel")
    assert len({row["weight_id"] for row in receipt["weights"]}) == 2
    repeat = tmp_path / "two" / "weights.safetensors"
    repeat.parent.mkdir()
    again = m2m.convert(
        same_model, inputs, quantization=config, calibration_inputs=[inputs],
        backend="fx_importer", decompose=False, weights_path=str(repeat),
    )
    assert again.ok, again.diagnostics
    assert (path.parent / (path.name + ".quantization.json")).read_bytes() == (
        repeat.parent / (repeat.name + ".quantization.json")
    ).read_bytes()
    assert (path.parent / (path.name + ".prequant.safetensors")).read_bytes() == (
        repeat.parent / (repeat.name + ".prequant.safetensors")
    ).read_bytes()


def test_per_tensor_recipe_shape_records_literal_qparams_and_refuses_mismatch(tmp_path):
    from torchao.quantization.pt2e import HistogramObserver, MinMaxObserver
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    from torchao.quantization.pt2e.quantizer import QuantizationAnnotation, QuantizationSpec, Quantizer

    class TensorQuantizer(Quantizer):
        def __init__(self):
            self.activation = QuantizationSpec(torch.int8, HistogramObserver.with_args(eps=2**-12), -128, 127, torch.per_tensor_symmetric)
            self.weight = QuantizationSpec(torch.int8, MinMaxObserver.with_args(eps=2**-12), -127, 127, torch.per_tensor_symmetric)

        def annotate(self, graph_module):
            for node in graph_module.graph.nodes:
                if node.target in (torch.ops.aten.conv2d.default, torch.ops.aten.linear.default):
                    node.meta["quantization_annotation"] = QuantizationAnnotation(
                        input_qspec_map={node.args[0]: self.activation, node.args[1]: self.weight},
                        _annotated=True,
                    )
            return graph_module

        def validate(self, graph_module):
            del graph_module

    torch.manual_seed(11)
    source = _DevelopmentNet().eval()
    inputs = (torch.randn(1, 3, 8, 8),)
    exported = torch.export.export(source, inputs).module()
    original = snapshot_source_weights(source, exported)
    assert set(original) == {"conv.weight", "fc.weight"}
    prepared = prepare_pt2e(exported, TensorQuantizer())
    prepared(*inputs)
    snapshot = snapshot_effective_weights(original, prepared)
    converted = convert_pt2e(prepared, fold_quantize=True)
    converted._m2m_weight_lineage = bind_frozen_weights(snapshot, converted)
    from torch.ao.quantization import allow_exported_model_train_eval

    allow_exported_model_train_eval(converted)
    path = tmp_path / "weights.safetensors"
    result = m2m.convert(
        converted, inputs, quantization=QuantizationConfig(scheme="int8_static_act_int8_weight"),
        quantization_preapplied=True, backend="fx_importer", decompose=False, weights_path=str(path),
    )
    assert result.ok, result.diagnostics
    receipt = _verify_files(path, expected_granularity="per_tensor")
    assert all(row["qparams"]["scale"].get("literal_float64_hex") for row in receipt["weights"])

    bad = copy.deepcopy(converted._m2m_weight_lineage)
    bad[0]["frozen"].zero_()
    with pytest.raises(ValueError, match="exported state bytes changed"):
        from m2m.capture.weight_lineage import write_weight_lineage

        write_weight_lineage(bad, path)
