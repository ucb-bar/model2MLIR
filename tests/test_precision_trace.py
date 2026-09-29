"""Precision qualification must distinguish native types from projections."""

import torch
from torch import nn

import m2m


def test_bf16_and_fp16_capture_keep_native_typed_edges():
    for dtype, spelling in ((torch.bfloat16, "bf16"), (torch.float16, "f16")):
        model = nn.Sequential(nn.Linear(8, 5), nn.ReLU()).eval().to(dtype)
        result = m2m.convert(model, (torch.ones(2, 8, dtype=dtype),),
                             backend="fx_importer", capture_trace=True)
        assert result.ok and result.capture_trace["status"] == "complete"
        assert f"x{spelling}>" in result.mlir_text
        assert result.capture_trace["precision"]["status"] == "no_known_projection"


def test_fp8_storage_projection_is_explicit_and_not_qualified():
    class StoredFP8(nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.register_buffer("weight", torch.ones(8, 5).to(dtype))

        def forward(self, x):
            return x @ self.weight.to(torch.float32)

    for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        result = m2m.convert(StoredFP8(dtype).eval(), (torch.ones(2, 8),),
                             backend="fx_importer", capture_trace=True)
        assert result.ok  # preserve the legacy diagnostic representation
        assert result.capture_trace["status"] == "diagnostic"
        precision = result.capture_trace["precision"]
        assert precision["status"] == "projected" and precision["projections"]
        assert all(row["frontend_dtype"] == str(dtype).removeprefix("torch.")
                   and row["mlir_storage_dtype"] == "f32" for row in precision["projections"])
        assert any("FP8 storage" in reason for reason in result.capture_trace["blockers"])


def test_owned_precision_conversion_keeps_exact_original_source_identity():
    for dtype in (torch.bfloat16, torch.float16):
        model = nn.Sequential(nn.Linear(8, 5), nn.ReLU()).eval()
        inputs = (torch.ones(2, 8),)
        baseline = m2m.capture_frontend_snapshot(model, inputs)
        integer_graph, converted_inputs, original, receipt = m2m.materialize_frontend_precision(
            model, inputs, dtype=dtype, original_frontend_snapshot=baseline)
        assert model[0].weight.dtype == inputs[0].dtype == torch.float32
        assert receipt["original_graph_sha256"] == baseline["sha256"]
        assert receipt["target_dtype"] == str(dtype)
        result = m2m.convert(integer_graph, converted_inputs, backend="fx_importer", capture_trace=True,
                             original_frontend_snapshot=original)
        assert result.ok and result.capture_trace["status"] == "complete", result.capture_trace
        assert result.capture_trace["graphs"]["original"]["sha256"] == baseline["sha256"]


def test_exact_reexport_uses_recorded_original_ids_not_transient_probe_ids():
    from m2m.capture.trace import attach_original_identity, snapshot_exported_program

    model = nn.Sequential(nn.Linear(8, 5), nn.ReLU()).eval()
    inputs = (torch.ones(2, 8),)
    original = m2m.capture_frontend_snapshot(model, inputs)
    exported = torch.export.export(model, inputs)
    actual = snapshot_exported_program(exported, stage="quantization_input")
    assert attach_original_identity(exported, original, actual)
    from torch.ao.quantization import allow_exported_model_train_eval

    module = exported.module()
    allow_exported_model_train_eval(module)
    result = m2m.convert(module, inputs, backend="fx_importer", capture_trace=True,
                         original_frontend_snapshot=original)
    trace = result.capture_trace
    assert result.ok and trace["status"] == "complete", (result.diagnostics, trace["blockers"])
    recorded_origins = {node["id"] for stage in ("original", "quantized")
                        for node in trace["graphs"][stage]["nodes"]}
    assert all(set(op["origin_node_ids"]) <= recorded_origins for op in trace["mlir"]["operations"])
