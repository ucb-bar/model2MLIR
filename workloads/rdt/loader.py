"""RDT-1B (RoboticsDiffusionTransformer, PyTorch) capture loader for m2m.

Builds the RDT diffusion transformer (the `RDT` class in models/rdt/model.py)
with a small/random config (no checkpoint download) and exposes ONE denoise step
-- the model's `forward` is exactly the per-step network the DDPM/DPM sampling loop
calls (`RDTRunner.conditional_sample`). The diffusion while-loop and the frozen
vision/text encoders (SigLIP/DINOv2 image tokens, T5 language tokens) live OUTSIDE
the captured graph: their outputs enter as already-adapted condition tensors.

    get_model_and_inputs() -> (nn.Module, tuple[Tensor, ...])

Env:
    M2M_RDT_DEPTH=N   number of RDT blocks (default 2 for a fast smoke; real 1B = 28)

The condition adaptors (lang/img/state MLPs) project encoder features to hidden_size.
Here we feed the RDT core directly with hidden-size condition tokens, so the captured
graph is the heavy transformer stack (self-attn + cross-attn + FFN x depth).
"""

from __future__ import annotations

import os
import sys

import torch
from torch import nn

# RDT is imported by package-relative path `models.rdt.model`; the repo root must
# be importable. Adjust if the clone lives elsewhere.
_RDT_REPO = os.environ.get("M2M_RDT_ROOT", "/path/to/RoboticsDiffusionTransformer")
if _RDT_REPO not in sys.path:
    sys.path.insert(0, _RDT_REPO)


class RDTDenoiseStep(nn.Module):
    """One RDT denoise step: state+action trajectory + lang/img conditions -> action.

    Takes ALREADY-ADAPTED hidden-size tokens. The conditioning encoders and the
    state/lang/img adaptor MLPs run host-side, outside the captured graph.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x, freq, t, lang_c, img_c, lang_mask):
        # img_mask left as None (all-valid) to keep the trace clean.
        return self.model(x, freq, t, lang_c, img_c, lang_mask=lang_mask, img_mask=None)


def get_model_and_inputs():
    from models.rdt.model import RDT

    depth = int(os.environ.get("M2M_RDT_DEPTH", "2"))

    # Small/random config mirroring configs/base.yaml (1B uses hidden=2048,
    # depth=28, heads=32). hidden_size must be divisible by num_heads.
    hidden_size = 2048
    num_heads = 32
    horizon = 64          # action_chunk_size
    output_dim = 128      # state_dim / action_dim
    max_lang_cond_len = 1024
    img_cond_len = 4096   # img_history_size * num_cameras * patches (encoder-dependent)

    dtype = torch.float32
    model = RDT(
        output_dim=output_dim,
        horizon=horizon,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        max_lang_cond_len=max_lang_cond_len,
        img_cond_len=img_cond_len,
        dtype=dtype,
    ).eval()

    b = 1
    lang_len = 32  # variable-length language; <= max_lang_cond_len
    # x: (B, T, D) state+action+mask trajectory ALREADY adapted to hidden_size.
    # The runner builds T = horizon + 1 (state token + horizon action tokens); the
    # model itself prepends timestep+freq tokens internally (-> horizon + 3).
    x = torch.randn(b, horizon + 1, hidden_size, dtype=dtype)
    freq = torch.tensor([25.0] * b, dtype=dtype)          # ctrl_freqs (B,)
    t = torch.tensor([0], dtype=dtype)                     # diffusion timestep (1,)
    lang_c = torch.randn(b, lang_len, hidden_size, dtype=dtype)
    img_c = torch.randn(b, img_cond_len, hidden_size, dtype=dtype)
    lang_mask = torch.ones(b, lang_len, dtype=torch.bool)

    inputs = (x, freq, t, lang_c, img_c, lang_mask)
    return RDTDenoiseStep(model).eval(), inputs
