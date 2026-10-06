"""Static floating ranges must preserve the selected source operator's values."""

import numpy as np
import pytest
import torch
from m2m.ir.decompositions import decompose_arange
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr


def _lower(start, end, step, *, val=None, device="cpu"):
    if val is None:
        val = torch.empty_like(torch.arange(start, end, step, dtype=torch.float32))
    return decompose_arange(
        [],
        {
            "val": val,
            "_fx_args": (start, end, step),
            "_fx_kwargs": {"dtype": torch.float32, "device": torch.device(device)},
        },
        "arange",
    )


@pytest.mark.parametrize(
    "start,end,step",
    [
        (-0.75, 3.25, 0.5),
        (-0.3, 3.7, 0.2),
        (3.7, -0.3, -0.2),
        (-0.0, -1.0, -0.25),
        (1.0, 1.0, 0.25),
    ],
)
def test_literal_fp32_arange_is_source_value_exact(start, end, step):
    result = _lower(start, end, step)
    assert len(result.ops) == 1
    assert isinstance(result.ops[0], ConstantOp)
    data = result.ops[0].value
    assert isinstance(data, DenseIntOrFPElementsAttr)
    actual = np.asarray(list(data.get_values()), dtype=np.float32)
    expected = torch.arange(start, end, step, dtype=torch.float32).numpy()
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("length", [1, 7, 8, 9, 17])
def test_fractional_range_across_source_vector_and_tail_lengths(length):
    start, step = -0.3, 0.2
    end = start + length * step
    expected = torch.arange(start, end, step, dtype=torch.float32).numpy()
    assert expected.size == length
    result = _lower(start, end, step)
    actual = np.asarray(list(result.ops[0].value.get_values()), dtype=np.float32)
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize(
    "start,end,step,val,device",
    [
        (0.0, 4097.0, 1.0, torch.empty(4097), "cpu"),
        (torch.tensor(0.0), 4.0, 1.0, torch.empty(4), "cpu"),
        (0.0, 4.0, 1.0, torch.empty(4, device="meta"), "meta"),
        (0.0, 4.0, 1.0, torch.empty(4), "meta"),
        (0.0, 4.0, 1.0, torch.empty(3), "cpu"),
        (float("nan"), 4.0, 1.0, torch.empty(4), "cpu"),
        (0.0, 4.0, 0.0, torch.empty(4), "cpu"),
    ],
)
def test_unproved_fp32_arange_does_not_emit_a_literal(start, end, step, val, device):
    with pytest.raises(TypeError, match="FP32 arange"):
        _lower(start, end, step, val=val, device=device)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float64])
def test_non_fp32_arange_keeps_existing_iota_path(dtype):
    result = decompose_arange(
        [],
        {"val": torch.empty(4, dtype=dtype), "_fx_args": (0, 4, 1)},
        "arange",
    )
    assert len(result.ops) == 2
    assert not any(isinstance(op, ConstantOp) for op in result.ops)


def test_frontend_capture_retains_arange_origin_when_literal_folded():
    import m2m

    class Range(torch.nn.Module):
        def forward(self, x):
            return torch.arange(-0.3, x.shape[0] - 0.3, 0.2, dtype=torch.float32, device=x.device)

    result = m2m.convert(Range().eval(), (torch.zeros(4),), backend="fx_importer", capture_trace=True)
    assert result.ok, result.diagnostics
    assert 'prov.source_evaluation = "torch.arange:cpu:f32"' in result.mlir_text
    assert f'prov.source_cpu_capability = "{torch.backends.cpu.get_cpu_capability()}"' in result.mlir_text
    assert "prov.source_num_threads" in result.mlir_text
    assert "prov.source_num_interop_threads" in result.mlir_text
    assert 'prov.aten = "aten.arange.start_step"' in result.mlir_text
    assert "linalg.index" not in result.mlir_text
    assert result.capture_trace["status"] == "complete"
