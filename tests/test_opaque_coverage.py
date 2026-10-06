"""Opaque coverage is about unresolved calls, not one MLIR printer spelling."""

import m2m
import pytest
import torch
from m2m.coverage import opaque_report
from torch import nn


class _UnsupportedXor(nn.Module):
    def forward(self, values):
        return values ^ 65535


def test_actual_fx_generic_func_call_is_counted_as_opaque():
    values = torch.tensor([-(2**63), -1, 0, 2**63 - 1], dtype=torch.int64)
    result = m2m.convert(_UnsupportedXor().eval(), (values,), backend="fx_importer", capture_trace=True)
    assert result.ok and result.capture_trace["status"] == "complete"
    assert '"func.call"' in result.mlir_text
    assert opaque_report(result.mlir_text) == {"aten_bitwise_xor_Scalar": 1}


def test_internal_call_is_not_an_unresolved_external():
    text = """module {
  func.func private @helper(%x: i64) -> i64 {
    return %x : i64
  }
  func.func @main(%x: i64) -> i64 {
    %y = func.call @helper(%x) : (i64) -> i64
    return %y : i64
  }
}"""
    assert opaque_report(text) == {}


def test_nested_external_shadows_defined_outer_symbol():
    text = """builtin.module {
  func.func @helper(%x: i64) -> i64 { func.return %x : i64 }
  builtin.module @inner {
    func.func private @helper(i64) -> i64
    func.func @main(%x: i64) -> i64 {
      %y = func.call @helper(%x) : (i64) -> i64
      func.return %y : i64
    }
  }
}"""
    assert opaque_report(text) == {"helper": 1}


def test_unparseable_call_cannot_be_reported_as_zero():
    with pytest.raises(ValueError):
        opaque_report('module { "func.call"(')
