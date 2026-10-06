"""Real PT2E capture must not silently skip the string-padding Conv2d overload."""

import m2m
import torch
from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference
from m2m.capture.pt2e_integerize import integerize_pt2e
from m2m.capture.pt2e_padding import normalize_conv2d_padding
from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization
from m2m.capture.trace import snapshot_exported_program
from m2m.coverage import opaque_report
from torch import nn


class PaddedConv(nn.Module):
    def __init__(self, padding, kernel_size, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size, padding=padding, dilation=dilation)
        self.fc = nn.Linear(4, 2)

    def forward(self, image):
        return self.fc(self.conv(image).mean((2, 3)))


def test_string_padding_is_quantized_integerized_and_source_accounted():
    # Cover valid, symmetric same, asymmetric same, and dilated same. No model
    # names or target shapes are part of this generic capture regression.
    for padding, kernel, dilation in (("valid", 3, 1), ("same", 3, 1), ("same", 4, 1), ("same", 4, 2)):
        torch.manual_seed(7)
        model = PaddedConv(padding, kernel, dilation).eval()
        inputs = (torch.randn(1, 3, 9, 9),)
        original = snapshot_exported_program(torch.export.export(model, inputs), stage="original")
        quantized = apply_quantization(
            model,
            QuantizationConfig(
                "int8_static_act_int8_weight", calibration_samples=1, extra_args={"weight_granularity": "per_tensor"}
            ),
            example_inputs=inputs,
            original_frontend_snapshot=original,
        )
        assert quantized._m2m_quantization_stats["annotated_contractions"] == 2
        normalizations = quantized._m2m_quantization_stats["conv2d_padding_normalizations"]
        source_conv = next(node for node in original["nodes"] if node["target"] == "aten.conv2d.padding")
        assert len(normalizations) == 1 and source_conv["id"] in normalizations[0]["source_ids"]
        reference = run_pt2e_integer_reference(quantized, inputs, expected_contractions=2)
        integer, receipt = integerize_pt2e(quantized, inputs)
        assert receipt["conv2d_integerized"] == 1 and receipt["linear_integerized"] == 1, receipt
        assert receipt["quantized_contractions_remaining"] == 0 and receipt["refusals"] == []
        torch.testing.assert_close(integer(*inputs), reference.output, atol=0, rtol=0)
        result = m2m.convert(
            integer, inputs, backend="fx_importer", capture_trace=True, original_frontend_snapshot=original
        )
        assert result.ok and opaque_report(result.mlir_text) == {}
        assert result.capture_trace["status"] == "complete", result.capture_trace
        assert 'prov.aten = "aten._int_mm.default"' in result.mlir_text


def test_padding_normalization_preserves_float_values_and_keyword_arguments():
    for padding, kernel, dilation in (("valid", 3, 1), ("same", 3, 1), ("same", (4, 3), 1), ("same", 4, 2)):
        torch.manual_seed(11)
        inputs = (torch.randn(1, 3, 9, 9),)
        exported = torch.export.export(PaddedConv(padding, kernel, dilation).eval(), inputs).module()
        expected = exported(*inputs)
        conv = next(node for node in exported.graph.nodes if node.target == torch.ops.aten.conv2d.padding)
        # Exercise the same schema binding with kwargs rather than the common
        # exporter's positional form; defaults must not overwrite supplied pads.
        arguments = list(torch.ops.aten.conv2d.padding._schema.arguments)
        conv.kwargs = {argument.name: value for argument, value in zip(arguments[2:], conv.args[2:])}
        conv.args = conv.args[:2]
        receipt = normalize_conv2d_padding(exported)
        assert len(receipt) == 1 and receipt[0]["padding"] == padding
        torch.testing.assert_close(exported(*inputs), expected, atol=0, rtol=0)


def test_string_padding_normalized_before_batch_norm_fold_keeps_source_lineage():
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding="same"), nn.BatchNorm2d(4), nn.ReLU()).eval()
    result = m2m.convert(
        model,
        (torch.randn(1, 3, 8, 8),),
        backend="fx_importer",
        capture_trace=True,
        quantization=QuantizationConfig("int8_static_act_int8_weight", calibration_samples=1),
    )
    assert result.ok and result.capture_trace["status"] == "complete"
    graphs = result.capture_trace["graphs"]
    bn = next(node for node in graphs["original"]["nodes"] if node["target"] == "aten.batch_norm.default")
    conv = next(node for node in graphs["original"]["nodes"] if node["target"] == "aten.conv2d.padding")
    folded = [
        node
        for node in graphs["quantized"]["nodes"]
        if node["target"] == "aten.conv2d.default" and bn["id"] in node["origin_node_ids"]
    ]
    assert len(folded) == 1 and conv["id"] in folded[0]["origin_node_ids"]


def test_owned_fp32_staging_retains_original_padding_ancestry():
    # BF16 is an original framework input, not a claim about device/host support.
    torch.manual_seed(13)
    source = PaddedConv("valid", 3).eval().to(torch.bfloat16)
    inputs = (torch.randn(1, 3, 9, 9, dtype=torch.bfloat16),)
    staged, staged_inputs, original, staging = m2m.materialize_frontend_precision(
        source,
        inputs,
        dtype=torch.float32,
        retarget_float_dtype_arguments=True,
    )
    assert staging["target_dtype"] == "torch.float32"
    assert source.conv.weight.dtype == inputs[0].dtype == torch.bfloat16
    quantized = apply_quantization(
        staged,
        QuantizationConfig("int8_static_act_int8_weight", calibration_samples=1),
        example_inputs=staged_inputs,
        original_frontend_snapshot=original,
    )
    conv = next(node for node in original["nodes"] if node["target"] == "aten.conv2d.padding")
    row = quantized._m2m_quantization_stats["conv2d_padding_normalizations"][0]
    assert conv["id"] in row["source_ids"]
    integer, receipt = integerize_pt2e(quantized, staged_inputs)
    assert receipt["conv2d_integerized"] == receipt["linear_integerized"] == 1
    assert integer(*staged_inputs).dtype == torch.float32


def test_stagewise_session_binds_normalized_convolution_to_source_weight():
    from m2m.capture.external_runtime import ExternalRuntimeProgram, make_external_runtime_session
    from m2m.capture.stagewise_pt2e import quantize_stagewise_pt2e_session

    shared = PaddedConv("same", 4).eval()
    image = torch.randn(1, 3, 9, 9)
    programs = tuple(
        ExternalRuntimeProgram(name, nn.Sequential(shared).eval(), (image,), 1) for name in ("first", "second")
    )
    source = make_external_runtime_session(
        version=2,
        programs=programs,
        metadata={
            "kind": "independent_padding_probe",
            "stages": ["first", "second"],
            "stage_schedule": [{"name": name, "steps": 1} for name in ("first", "second")],
            "quality_program": "second",
            "bindings": [],
        },
    )
    result = quantize_stagewise_pt2e_session(
        shared,
        source,
        quant=QuantizationConfig("int8_static_act_int8_weight", calibration_samples=1),
        shared_programs=("first", "second"),
        calibration_inputs={name: [(image,)] for name in ("first", "second")},
    )
    assert len(result.frozen_weights) == 4
    for name in ("conv.weight", "fc.weight"):
        weights = [row for row in result.frozen_weights if row["source_parameter"] == name]
        assert len(weights) == 2 and weights[0]["frozen_sha256"] == weights[1]["frozen_sha256"]
    conv = next(row for row in result.frozen_weights if row["source_parameter"] == "conv.weight")
    assert conv["source_operation"] == "aten.conv2d.padding"
    assert conv["frozen_operation"] == "aten.conv2d.default"
