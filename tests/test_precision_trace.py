"""Precision qualification must distinguish native types from projections."""

import copy
import hashlib

import pytest
import torch
from torch import nn

import m2m


def test_owned_fp32_graph_retargeting_changes_only_float_dtype_decisions():
    class Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.2345], dtype=torch.bfloat16))

        def forward(self, x):
            cast = x.to(torch.bfloat16)
            factory = torch.full(x.shape, 0.375, dtype=torch.bfloat16)
            value = (cast * self.weight + factory).float()
            return value, x.to(torch.int64), torch.ones_like(x, dtype=torch.bool)

    model = Mixed().eval()
    inputs = (torch.tensor([1.2345]),)
    original_weight = model.weight.detach().clone()
    source = model(*inputs)
    baseline = m2m.capture_frontend_snapshot(model, inputs)
    staged, converted_inputs, original, receipt = m2m.materialize_frontend_precision(
        model, inputs, dtype=torch.float32, original_frontend_snapshot=baseline,
        retarget_float_dtype_arguments=True)
    outputs = staged(*converted_inputs)
    assert model.weight.dtype == torch.bfloat16
    torch.testing.assert_close(model.weight, original_weight, atol=0, rtol=0)
    assert original["sha256"] == baseline["sha256"]
    assert receipt["original_graph_sha256"] == baseline["sha256"]
    assert receipt["staged_graph_sha256"] != baseline["sha256"]
    assert receipt["dtype_decisions"]
    assert any(row["from_dtype"] == "torch.bfloat16" and row["to_dtype"] == "torch.float32"
               for row in receipt["dtype_decisions"])
    assert any(row["target"] == "aten.full.default" and row["location"] == "kwargs"
               for row in receipt["dtype_decisions"])
    assert any(row["from_dtype"] == row["to_dtype"] == "torch.int64"
               for row in receipt["dtype_decisions"])
    assert receipt["staged_precision_audit"]["non_target_floating_values"] == 0
    assert outputs[0].dtype == torch.float32
    max_abs = float((outputs[0] - source[0]).detach().abs().max())
    assert 0 < max_abs < 0.01
    assert outputs[1].dtype == torch.int64 and torch.equal(outputs[1], source[1])
    assert outputs[2].dtype == torch.bool and torch.equal(outputs[2], source[2])
    trace = m2m.capture_frontend_snapshot(staged, converted_inputs)
    assert all(value["dtype"] not in {"bfloat16", "float16"}
               for node in trace["nodes"] for value in node["results"])
    assert (m2m.capture_frontend_snapshot(staged, converted_inputs, stage="staged")["sha256"]
            == receipt["staged_graph_sha256"])
    converted = m2m.convert(staged, converted_inputs, backend="fx_importer", capture_trace=True,
                            original_frontend_snapshot=original)
    assert converted.ok and converted.capture_trace["status"] == "complete"


def test_owned_fp32_graph_retargeting_accepts_already_fp32_source():
    class NativeFP32(nn.Module):
        def forward(self, x):
            return x.to(torch.float32) + torch.full(x.shape, 0.25, dtype=torch.float32)

    model = NativeFP32().eval()
    inputs = (torch.tensor([1.2345]),)
    staged, converted_inputs, original, receipt = m2m.materialize_frontend_precision(
        model, inputs, dtype=torch.float32, retarget_float_dtype_arguments=True)
    torch.testing.assert_close(staged(*converted_inputs), model(*inputs), atol=0, rtol=0)
    assert receipt["original_graph_sha256"] == original["sha256"]
    assert all(row["from_dtype"] == row["to_dtype"] for row in receipt["dtype_decisions"])


def test_owned_fp32_graph_retargeting_clones_lifted_float_but_keeps_bool_constant():
    class Lifted(nn.Module):
        def forward(self, x):
            scalar = torch.tensor(0.3761, dtype=torch.bfloat16)
            mask = torch.tensor([True, False], dtype=torch.bool)
            return x + scalar.float(), mask

    model = Lifted().eval()
    inputs = (torch.ones(2),)
    source = model(*inputs)
    captured_source = torch.export.export(model, inputs)
    source_constants = {name: value.detach().clone()
                        for name, value in captured_source.constants.items()}
    staged, converted_inputs, original, receipt = m2m.materialize_frontend_precision(
        captured_source, inputs, dtype=torch.float32, retarget_float_dtype_arguments=True)
    outputs = staged(*converted_inputs)
    constants = receipt["constant_conversions"]
    assert len(constants) == 2
    floating = next(row for row in constants if row["from_dtype"] == "torch.bfloat16")
    boolean = next(row for row in constants if row["from_dtype"] == "torch.bool")
    assert floating["to_dtype"] == "torch.float32"
    assert floating["source_sha256"] != floating["staged_sha256"]
    assert boolean["to_dtype"] == "torch.bool"
    assert boolean["source_sha256"] == boolean["staged_sha256"]
    assert set(captured_source.constants) == set(source_constants)
    for name, value in captured_source.constants.items():
        assert value.dtype == source_constants[name].dtype
        torch.testing.assert_close(value, source_constants[name], atol=0, rtol=0)
    # A BF16 literal was rounded when the source graph was captured; changing
    # its storage cannot reconstruct the pre-rounding Python scalar.
    assert outputs[0].dtype == torch.float32 and torch.equal(outputs[0], source[0])
    assert torch.equal(outputs[1], source[1])
    assert receipt["staged_precision_audit"]["non_target_floating_values"] == 0
    returned_export = torch.export.export(staged, converted_inputs)
    assert (m2m.capture_frontend_snapshot(returned_export, stage="staged")["sha256"]
            == receipt["staged_graph_sha256"])
    assert (m2m.capture_frontend_snapshot(staged, converted_inputs, stage="staged")["sha256"]
            == receipt["staged_graph_sha256"])
    from torch.export.graph_signature import InputKind

    final_specs = [spec for spec in returned_export.graph_signature.input_specs
                   if spec.kind == InputKind.CONSTANT_TENSOR]
    assert len(final_specs) == len(constants)
    for row, spec in zip(constants, final_specs, strict=True):
        value = returned_export.constants[str(spec.target)]
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        assert row["staged_target"] == str(spec.target)
        assert row["to_dtype"] == str(value.dtype) and row["shape"] == list(value.shape)
        assert row["staged_sha256"] == hashlib.sha256(raw.numpy().tobytes()).hexdigest()
    exported_outputs = returned_export.module()(*converted_inputs)
    assert [value.dtype for value in exported_outputs] == [value.dtype for value in outputs]
    for actual, expected in zip(exported_outputs, outputs, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    converted = m2m.convert(staged, converted_inputs, backend="fx_importer", capture_trace=True,
                            original_frontend_snapshot=original)
    assert converted.ok and converted.capture_trace["status"] == "complete"
    from m2m.capture.trace import graph_relation

    source_graph = converted.capture_trace["graphs"]["quantized"]
    prepared_graph = converted.capture_trace["graphs"]["prepared"]
    noop_cast = next(node for node in source_graph["nodes"]
                     if node["target"] == "aten.to.dtype")
    changed = copy.deepcopy(source_graph)
    next(node for node in changed["nodes"] if node["id"] == noop_cast["id"])["kwargs"]["copy"] = True
    assert noop_cast["id"] in graph_relation(changed, prepared_graph)["unresolved_source_ids"]
    wrong_edge = copy.deepcopy(prepared_graph)
    add = next(node["id"] for node in wrong_edge["nodes"]
               if node["target"] == "aten.add.Tensor")
    edge = next(edge for edge in wrong_edge["edges"]
                if edge["consumer_node_id"] == add and edge["argument_path"] == "args/1")
    edge["dtype"] = "float16"
    assert noop_cast["id"] in graph_relation(source_graph, wrong_edge)["unresolved_source_ids"]


def test_owned_precision_conversion_default_preserves_explicit_source_casts_and_receipt():
    class ExplicitBF16(nn.Module):
        def forward(self, x):
            return x.to(torch.bfloat16).float()

    model = ExplicitBF16().eval()
    inputs = (torch.tensor([1.2345]),)
    unchanged, unchanged_inputs, _, receipt = m2m.materialize_frontend_precision(
        model, inputs, dtype=torch.float32)
    torch.testing.assert_close(unchanged(*unchanged_inputs), model(*inputs), atol=0, rtol=0)
    assert set(receipt) == {"schema", "scope", "original_graph_sha256", "target_dtype",
                            "state_conversions", "input_conversions", "sha256"}
    assert any(node.target == torch.ops.aten.to.dtype and node.args[1] == torch.bfloat16
               for node in unchanged.graph.nodes)


def test_fp32_stage_audit_rejects_missing_or_unrecognized_tensor_dtype():
    from m2m.capture.trace import _audit_staged_float_precision, snapshot_exported_program

    inputs = (torch.ones(2),)
    exported = torch.export.export(nn.Identity(), inputs)
    snapshot = snapshot_exported_program(exported, stage="staged")
    for invalid in (None, "unrecognized_float"):
        damaged = copy.deepcopy(snapshot)
        tensor = next(result for node in damaged["nodes"] for result in node["results"]
                      if result["kind"] == "tensor")
        tensor["dtype"] = invalid
        with pytest.raises(ValueError, match="lacks a dtype|unrecognized dtype"):
            _audit_staged_float_precision(exported, exported.module(), inputs,
                                          damaged, torch.float32)


def test_owned_fp32_graph_retargeting_rejects_complex_without_changing_default():
    class ComplexCast(nn.Module):
        def forward(self, x):
            return x.to(torch.complex64)

    model = ComplexCast().eval()
    inputs = (torch.ones(2),)
    legacy, legacy_inputs, _, legacy_receipt = m2m.materialize_frontend_precision(
        model, inputs, dtype=torch.float32)
    assert legacy(*legacy_inputs).dtype == torch.complex64
    assert "staged_precision_audit" not in legacy_receipt
    with pytest.raises(ValueError, match="complex dtype"):
        m2m.materialize_frontend_precision(
            model, inputs, dtype=torch.float32, retarget_float_dtype_arguments=True)

    with pytest.raises(ValueError, match="complex precision"):
        m2m.materialize_frontend_precision(
            nn.Identity(), (torch.ones(2, dtype=torch.complex64),),
            dtype=torch.float32, retarget_float_dtype_arguments=True)


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


def test_fp32_staged_independent_branches_keep_exact_fold_owners():
    """Re-export may reorder independent calls, not their typed dataflow."""
    from m2m.capture.torchao_pipeline import QuantizationConfig
    from m2m.capture.trace import (_reordered_graph_identity, attach_original_identity,
                                   pt2e_conv_bn_fold_candidates, snapshot_exported_program)

    class Branches(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Conv2d(3, 4, 3, padding=1)
            self.first_norm = nn.BatchNorm2d(4)
            self.second = nn.Conv2d(4, 4, 3, padding=1)
            self.second_norm = nn.BatchNorm2d(4)
            self.skip = nn.Conv2d(3, 4, 1)
            self.skip_norm = nn.BatchNorm2d(4)

        def forward(self, x):
            main = self.second_norm(self.second(self.first_norm(self.first(x)).relu()))
            residual = self.skip_norm(self.skip(x))
            return (main + residual).relu()

    model = Branches().eval()
    inputs = (torch.randn(1, 3, 8, 8),)
    original = m2m.capture_frontend_snapshot(model, inputs)
    staged, staged_inputs, original, _ = m2m.materialize_frontend_precision(
        model, inputs, dtype=torch.float32, original_frontend_snapshot=original,
        retarget_float_dtype_arguments=True)
    staged.eval()  # the recipe PT2E path makes this transition before its export
    exported = torch.export.export(staged, staged_inputs)
    actual = snapshot_exported_program(exported, stage="quantization_input")
    assert [node["target"] for node in original["nodes"]] != [
        node["target"] for node in actual["nodes"]]
    for mutation in ("edge", "constant", "operand_order"):
        changed = copy.deepcopy(actual)
        norms = [node for node in changed["nodes"] if node["target"] == "aten.batch_norm.default"]
        if mutation == "edge":
            norms[0]["args"][0] = copy.deepcopy(norms[-1]["args"][0])
        elif mutation == "constant":
            eps = next(i for i, value in enumerate(norms[0]["args"]) if value == 1e-5)
            norms[0]["args"][eps] = 2e-5
        else:
            add = next(node for node in changed["nodes"] if node["target"] == "aten.add.Tensor")
            add["args"] = list(reversed(add["args"]))
        assert _reordered_graph_identity(original, changed) is None, mutation
    assert attach_original_identity(exported, original, actual)
    assert len(pt2e_conv_bn_fold_candidates(exported.module(), original)) == 3

    # The generic PT2E entry currently expects eval() to return its receiver;
    # ExportedProgram.module().eval() mutates in place and returns None.
    staged, staged_inputs, original, _ = m2m.materialize_frontend_precision(
        model, inputs, dtype=torch.float32, original_frontend_snapshot=original,
        retarget_float_dtype_arguments=True)
    staged.eval()
    staged.eval = lambda: staged
    converted = m2m.convert(
        staged, staged_inputs, backend="fx_importer", capture_trace=True,
        original_frontend_snapshot=original,
        quantization=QuantizationConfig(scheme="int8_static_act_int8_weight"))
    assert converted.ok and converted.capture_trace["status"] == "complete", converted.diagnostics
    source = converted.capture_trace["graphs"]["original"]
    quantized = converted.capture_trace["graphs"]["quantized"]
    source_norms = [node["id"] for node in source["nodes"]
                    if node["target"] == "aten.batch_norm.default"]
    assert len(source_norms) == 3
    for norm_id in source_norms:
        assert len([node for node in quantized["nodes"]
                    if node["target"] == "aten.conv2d.default"
                    and norm_id in node["origin_node_ids"]]) == 1


def test_reordered_identity_refuses_indistinguishable_source_owners():
    from m2m.capture.trace import _reordered_graph_identity, snapshot_exported_program

    class Repeated(nn.Module):
        def forward(self, x):
            return torch.sin(x) + torch.sin(x)

    model = Repeated().eval()
    inputs = (torch.ones(2),)
    original = m2m.capture_frontend_snapshot(model, inputs)
    actual = snapshot_exported_program(torch.export.export(model, inputs), stage="quantization_input")
    assert sum(node["target"] == "aten.sin.default" for node in original["nodes"]) == 2
    assert _reordered_graph_identity(original, actual) is None
