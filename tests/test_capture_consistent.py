"""Regression checks for checkpoint-preserving consistent captures."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


_CAPTURE_PATH = Path(__file__).resolve().parents[1] / "workloads" / "capture_consistent.py"
_SPEC = importlib.util.spec_from_file_location("m2m_capture_consistent", _CAPTURE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_CAPTURE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CAPTURE)


class _ZeroModel(torch.nn.Module):
    def __init__(self, *, paper_ready: bool) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 2))
        self.bias = torch.nn.Parameter(torch.ones(2))
        self.paper_ready = paper_ready


def test_paper_ready_checkpoint_parameters_are_never_perturbed():
    model = _ZeroModel(paper_ready=True)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    assert _CAPTURE._perturb_zero_parameters_for_smoke(model) == 0
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_smoke_only_zero_parameter_is_made_nondegenerate():
    model = _ZeroModel(paper_ready=False)

    assert _CAPTURE._perturb_zero_parameters_for_smoke(model) == 1
    assert torch.count_nonzero(model.weight) > 0
    assert torch.equal(model.bias, torch.ones_like(model.bias))
