"""Small source-owned dynamic-intermediate capture with a static return ABI."""

import torch


class MaskCastScatter(torch.nn.Module):
    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        selected = values[mask].to(torch.int64)
        return torch.zeros_like(values, dtype=torch.int64).index_put_((mask,), selected)


def get_model_and_inputs() -> tuple[torch.nn.Module, tuple[torch.Tensor, torch.Tensor]]:
    values = torch.tensor([True, False, True, False, True, False], dtype=torch.bool)
    mask = torch.tensor([False, True, True, False, True, False], dtype=torch.bool)
    return MaskCastScatter().eval(), (values, mask)
