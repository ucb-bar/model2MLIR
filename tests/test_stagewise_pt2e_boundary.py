"""Executable boundary for PT2E on a shared, method-based multi-program model.

This mirrors the capture shape of SmolVLA's embed_prefix and denoise_step without
depending on a checkpoint or LeRobot. It is a diagnostic test, not a quantized
multi-program capture implementation.
"""

import pytest
import torch
from torch import nn

from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization


class SharedMethodModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)

    def embed_prefix(self, value):
        return self.linear(value)

    def denoise_step(self, value):
        return self.linear(value) + value


class PrefixStage(nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared

    def forward(self, value):
        return self.shared.embed_prefix(value)


class FlowStage(nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared

    def forward(self, state):
        return self.shared.denoise_step(state)


def test_static_pt2e_cannot_transform_shared_method_model_as_one_session():
    shared = SharedMethodModel().eval()
    prefix, flow = PrefixStage(shared).eval(), FlowStage(shared).eval()
    assert prefix.shared is flow.shared is shared
    sample = (torch.ones(1, 2),)
    quant = QuantizationConfig(
        scheme="int8_static_act_int8_weight", calibration_samples=2
    )

    # The selected static API exports one module.forward. The shared model has
    # two stage methods and no one forward corresponding to the full session.
    with pytest.raises(NotImplementedError, match='missing the required "forward"'):
        apply_quantization(shared, quant, example_inputs=sample)

    # Quantizing each wrapper is executable, but PT2E returns two independent
    # graph modules. Neither retains the original shared object or its methods.
    prefix_q = apply_quantization(
        prefix, quant, example_inputs=sample, calibration_inputs=[sample]
    )
    flow_q = apply_quantization(
        flow,
        quant,
        example_inputs=sample,
        calibration_inputs=[sample, (torch.full((1, 2), 2.0),)],
    )
    assert prefix_q(sample[0]).shape == flow_q(sample[0]).shape == (1, 2)
    assert prefix_q._m2m_quantization_stats["calibration_samples"] == 1
    assert flow_q._m2m_quantization_stats["calibration_samples"] == 2
    assert prefix_q is not flow_q
    assert not any(module is shared for module in prefix_q.modules())
    assert not any(module is shared for module in flow_q.modules())
    assert not hasattr(prefix_q, "embed_prefix")
    assert not hasattr(flow_q, "denoise_step")
    # PT2E freezes two distinct int8 allocations for the one original weight.
    # Matching tensor values, if any, would not establish session weight identity.
    prefix_weight = next(t for t in prefix_q.buffers() if t.dtype == torch.int8)
    flow_weight = next(t for t in flow_q.buffers() if t.dtype == torch.int8)
    assert prefix_weight.data_ptr() != flow_weight.data_ptr()
