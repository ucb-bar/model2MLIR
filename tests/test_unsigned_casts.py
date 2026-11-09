"""Integer storage width does not establish the FX tensor's signedness."""
import m2m
import pytest
import torch
from m2m.ir.decompositions import decompose_to_copy
from xdsl.dialects.builtin import TensorType, i8
from xdsl.dialects.tensor import EmptyOp


@pytest.mark.parametrize("source_dtype", [torch.uint8, torch.int8])
@pytest.mark.parametrize("target_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_actual_fx_integer_cast_selects_its_source_signedness(source_dtype, target_dtype):
    class Cast(torch.nn.Module):
        def forward(self, x):
            return x.to(target_dtype).to(torch.float32)

    values = torch.arange(256, dtype=torch.int16) if source_dtype == torch.uint8 else torch.arange(-128, 128, dtype=torch.int16)
    inputs = (values.to(source_dtype),)
    result = m2m.convert(Cast(), inputs, backend="fx_importer", capture_trace=True)
    assert result.ok, result.diagnostics
    result.module.verify()
    names = [op.name for op in result.module.walk()]
    expected = "arith.uitofp" if source_dtype == torch.uint8 else "arith.sitofp"
    forbidden = "arith.sitofp" if source_dtype == torch.uint8 else "arith.uitofp"
    assert expected in names
    assert forbidden not in names
    assert result.capture_trace["status"] == "complete", result.capture_trace["blockers"]
    assert torch.equal(Cast()(*inputs), values.to(torch.float32))


@pytest.mark.parametrize("target_dtype", [torch.int16, torch.int32, torch.int64])
def test_uint8_widening_zero_extends_before_a_later_float_conversion(target_dtype):
    class Cast(torch.nn.Module):
        def forward(self, x):
            return x.to(target_dtype).to(torch.float32)

    result = m2m.convert(Cast(), (torch.arange(256).to(torch.uint8),), backend="fx_importer")
    assert result.ok, result.diagnostics
    names = [op.name for op in result.module.walk()]
    assert "arith.extui" in names
    assert "arith.extsi" not in names


@pytest.mark.parametrize("failure", ["missing", "width", "floating_source", "string_hint", "wrong_ssa", "unsupported_unsigned"])
def test_missing_or_contradictory_integer_source_metadata_is_refused(failure):
    source = EmptyOp([], TensorType(i8, [256])).tensor
    other = EmptyOp([], TensorType(i8, [256])).tensor
    meta = {"val": torch.zeros(256, dtype=torch.float32)}
    if failure == "width":
        meta["_fx_args"] = (torch.zeros(256, dtype=torch.int16),)
    elif failure == "floating_source":
        meta["_fx_args"] = (torch.zeros(256, dtype=torch.float32),)
    elif failure == "string_hint":
        meta["source_dtype"] = "uint8"  # A string hint is not a tensor source witness.
    elif failure == "wrong_ssa":
        meta["_fx_args"] = (torch.zeros(256, dtype=torch.uint8),)
        meta["_fx_ssa_args"] = {0: other}
    elif failure == "unsupported_unsigned":
        meta["_fx_args"] = (torch.zeros(256, dtype=torch.uint16),)
    with pytest.raises(ValueError, match="actual FX integer source dtype"):
        decompose_to_copy([source], meta, "cast")


def test_bool_float_conversion_remains_unsigned():
    class Cast(torch.nn.Module):
        def forward(self, x):
            return x.float()

    result = m2m.convert(Cast(), (torch.tensor([False, True]),), backend="fx_importer")
    assert result.ok
    names = [op.name for op in result.module.walk()]
    assert "arith.uitofp" in names
    assert "arith.sitofp" not in names
