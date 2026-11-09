"""Actual exported FP8 storage is retained, never projected into FP32 ABI."""

import json

import pytest
import torch

import m2m
from m2m.capture.torch_mlir_bridge import _parse_mlir_text_to_xdsl, module_to_text
from m2m.ir.import_fx import _torch_dtype_to_xdsl
from m2m.ir.types.float8 import register_fp8_types


@pytest.mark.parametrize(
    "dtype,spelling", [(torch.float8_e4m3fn, "f8E4M3FN"), (torch.float8_e5m2, "f8E5M2")]
)
def test_standard_parser_width_and_no_encoder(dtype, spelling):
    type_ = _torch_dtype_to_xdsl(dtype)
    assert (
        str(type_) == spelling and type_.bitwidth == 8 and type_.compile_time_size == 1
    )
    with pytest.raises(NotImplementedError):
        type_.pack([1.0])
    register_fp8_types()
    register_fp8_types()
    text = f"module {{func.func @identity(%x:tensor<3x{spelling}>)->tensor<3x{spelling}>{{func.return %x:tensor<3x{spelling}>}}}}"
    module = _parse_mlir_text_to_xdsl(text)
    assert module is not None
    module.verify()
    assert "builtin_ext" not in module_to_text(module)


@pytest.mark.parametrize(
    "name", ["float8_e4m3fnuz", "float8_e5m2fnuz", "float8_e8m0fnu"]
)
def test_unknown_formats_refuse_instead_of_becoming_f32(name):
    dtype = getattr(torch, name, None)
    if dtype is None:
        pytest.skip("framework format unavailable")
    with pytest.raises(ValueError, match="unsupported FP8 storage"):
        _torch_dtype_to_xdsl(dtype)


class Views(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.register_buffer(
            "weight", torch.arange(256, dtype=torch.uint8).view(dtype).reshape(16, 16)
        )

    def forward(self, x):
        return x.t().contiguous(), self.weight.t(), x.to(torch.float32)


@pytest.mark.parametrize(
    "dtype,spelling", [(torch.float8_e4m3fn, "f8E4M3FN"), (torch.float8_e5m2, "f8E5M2")]
)
def test_actual_export_input_weight_bytes_and_source_ids(tmp_path, dtype, spelling):
    model = Views(dtype).eval()
    x = torch.arange(256, dtype=torch.uint8).flip(0).view(dtype).reshape(16, 16)
    before = model.weight.view(torch.uint8).numpy().tobytes()
    inputs = x.view(torch.uint8).numpy().tobytes()
    exported = torch.export.export(model, (x,))
    result = m2m.convert(
        exported,
        (x,),
        backend="fx_importer",
        capture_trace=True,
        weights_path=str(tmp_path / "weights.safetensors"),
    )
    assert result.ok, result.diagnostics
    result.module.verify()
    assert "builtin_ext" not in result.mlir_text and spelling in result.mlir_text
    functions = [
        op
        for op in result.module.walk()
        if op.name == "func.func" and list(op.body.blocks)
    ]
    assert len(functions) == 1
    function = functions[0]
    assert [str(t.element_type) for t in function.function_type.inputs] == [
        spelling,
        spelling,
    ]
    assert [str(t.element_type) for t in function.function_type.outputs] == [
        spelling,
        spelling,
        "f32",
    ]
    assert len(function.attributes["prov.argument_source_node_ids"]) == 2
    trace = result.capture_trace
    assert trace["precision"]["status"] == "preserved", trace["precision"]
    assert (
        not trace["precision"]["projections"]
        and not trace["precision"]["unknown_values"]
    )
    assert trace["precision"]["arithmetic_qualified"] is False
    assert all(
        row["node_id"]
        and row["value_id"]
        and row["observed_mlir_storage_dtypes"] == [spelling]
        for row in trace["precision"]["values"]
    )
    assert any(op.name == "arith.extf" for op in result.module.walk())
    from safetensors.torch import load_file

    tensors = load_file(tmp_path / "weights.safetensors")
    assert len(tensors) == 1
    value = next(iter(tensors.values()))
    assert value.dtype is dtype and value.numel() == 256
    assert value.view(torch.uint8).numpy().tobytes() == before
    assert model.weight.view(torch.uint8).numpy().tobytes() == before
    assert x.view(torch.uint8).numpy().tobytes() == inputs
    manifest = json.loads((tmp_path / "weights.safetensors.manifest.json").read_text())
    assert manifest["0"]["dtype"] == str(dtype).removeprefix("torch.")
    parsed = _parse_mlir_text_to_xdsl(result.mlir_text)
    assert parsed is not None
    parsed.verify()


class Scaled(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.register_buffer("weight", torch.ones(16, 16).to(dtype).t())
        self.register_buffer("scale", torch.ones(1, 16))

    def forward(self, a, sa):
        return torch._scaled_mm(a, self.weight, sa, self.scale, out_dtype=torch.float32)


@pytest.mark.parametrize(
    "dtype,spelling", [(torch.float8_e4m3fn, "f8E4M3FN"), (torch.float8_e5m2, "f8E5M2")]
)
def test_scaled_mm_stays_explicit_opaque_and_diagnostic(dtype, spelling):
    model = Scaled(dtype).eval()
    inputs = (torch.ones(2, 16).to(dtype), torch.ones(2, 1))
    exported = torch.export.export(model, inputs)
    result = m2m.convert(exported, inputs, backend="fx_importer", capture_trace=True)
    assert result.ok
    result.module.verify()
    external = [
        op
        for op in result.module.walk()
        if op.name == "func.func" and not list(op.body.blocks)
    ]
    assert len(external) == 1 and "scaled_mm" in external[0].sym_name.data
    assert [str(t.element_type) for t in external[0].function_type.inputs][:2] == [
        spelling,
        spelling,
    ]
    precision = result.capture_trace["precision"]
    assert precision["status"] == "preserved" and not precision["projections"]
    assert precision["opaque_fp8_calls"] == [external[0].sym_name.data]
    assert result.capture_trace["status"] == "diagnostic"
    assert any(
        "unresolved external call" in v for v in result.capture_trace["blockers"]
    )
    assert precision["arithmetic_qualified"] is False


def test_storage_trace_detects_actual_projection_and_missing_mapping():
    from m2m.capture.fp8_storage_trace import observe_storage

    prepared = {
        "nodes": [
            {
                "id": "g:prepared:root:n0",
                "results": [
                    {
                        "id": "g:prepared:root:n0:v0",
                        "storage_dtype": "float8_e4m3fn",
                        "shape": [3],
                    }
                ],
            }
        ]
    }
    module = _parse_mlir_text_to_xdsl("""module {
      func.func @forward(%x:tensor<3xf32>)->tensor<3xf32>
      attributes {prov.argument_source_node_ids=["g:prepared:root:n0"]} {
      func.return %x:tensor<3xf32>}}
    """)
    assert module is not None
    projected = observe_storage(prepared, module)
    assert projected["status"] == "projected" and len(projected["projections"]) == 1
    unknown = observe_storage(prepared, None)
    assert unknown["status"] == "unknown" and len(unknown["unknown_values"]) == 1


@pytest.mark.parametrize("source_results", [1, 2])
def test_node_ancestry_cannot_certify_ambiguous_result_slots(source_results):
    from xdsl.dialects.builtin import (
        ArrayAttr,
        Float32Type,
        ModuleOp,
        StringAttr,
        TensorType,
    )
    from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
    from xdsl.ir import Block, Region

    from m2m.capture.fp8_storage_trace import observe_storage
    from m2m.ir.types import Float8E4M3FNType

    types = [TensorType(Float8E4M3FNType(), [3]), TensorType(Float32Type(), [3])]
    declaration = FuncOp.external("unknown_pair", [], types)
    call = CallOp("unknown_pair", [], types)
    call.attributes["prov.source_node_ids"] = ArrayAttr(
        [StringAttr("g:prepared:root:n0")]
    )
    block = Block()
    block.add_ops([call, ReturnOp(*call.results)])
    module = ModuleOp([declaration, FuncOp("forward", ([], types), Region(block))])
    module.verify()
    prepared = {
        "nodes": [
            {
                "id": "g:prepared:root:n0",
                "results": [
                    {
                        "id": f"g:prepared:root:n0:v{index}",
                        "storage_dtype": "float8_e4m3fn",
                        "shape": [3],
                    }
                    for index in range(source_results)
                ],
            }
        ]
    }
    report = observe_storage(prepared, module)
    assert report["status"] == "unknown"
    assert not report["values"] and not report["projections"]
    assert len(report["unknown_values"]) == source_results
    assert all(
        row["observed_mlir_storage_dtypes"] == ["f32", "f8E4M3FN"]
        for row in report["unknown_values"]
    )
