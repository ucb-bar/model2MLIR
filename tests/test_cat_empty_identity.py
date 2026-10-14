"""PyTorch's 1-D empty cat identity must stay legal in tensor MLIR."""

import pytest
import torch

import m2m
from m2m.coverage import opaque_report


class _CacheStyleConcat(torch.nn.Module):
    def __init__(self, append: bool) -> None:
        super().__init__()
        self.append = append

    def forward(self, cached: torch.Tensor, fresh: torch.Tensor) -> torch.Tensor:
        empty = torch.tensor([], dtype=cached.dtype)
        parts = (empty, cached, fresh) if self.append else (empty, cached)
        return torch.cat(parts, dim=-2)


@pytest.mark.parametrize("append", [False, True])
def test_rank_one_empty_cat_identity_in_higher_rank_cache(append: bool) -> None:
    cached = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    fresh = torch.full_like(cached, 3)
    model = _CacheStyleConcat(append).eval()
    exported = torch.export.export(model, (cached, fresh))
    assert "aten.cat.default" in {
        str(node.target) for node in exported.graph.nodes if node.op == "call_function"
    }

    result = m2m.convert(model, (cached, fresh), backend="fx_importer",
                         fully_standard=True)
    assert result.ok, result.diagnostics
    assert opaque_report(result.mlir_text) == {}
    result.module.verify()
    assert result.mlir_text.count("tensor.concat") == int(append)
    assert torch.equal(model(cached, fresh),
                       torch.cat((cached, fresh), dim=-2) if append else cached)
