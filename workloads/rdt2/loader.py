"""RDT2 (thu-ml/RDT2, PyTorch) capture loader for m2m.

RDT2's `RDT` (models/rdt/model.py) is an ACTION EXPERT: a DiT-style stack
(adaLN-Zero modulation + self-attn + cross-attn + SwiGLU FFN) that flow-matches an
action chunk, cross-attending to the KV cache of a frozen Qwen2.5-VL-7B VLM. We
capture ONE flow-matching step (`RDT.forward`, which `RDTRunner.conditional_sample`
calls in its Euler ODE loop), random init, no checkpoint download.

    get_model_and_inputs() -> (nn.Module, tuple[Tensor, ...])

The Qwen2.5-VL forward (image+text -> per-layer KV cache) and the act/state adaptor
MLPs run host-side, OUTSIDE the captured graph. Their outputs enter as: the adapted
noisy-action tokens `x` (B, T, D), the adapted state token `state_c` (B, 1, D), and
the per-layer language KV cache `lang_c_kv` (a list of (k, v) tensors, one pair per
RDT block). The 5-step inference loop is NOT captured.

Env:
    M2M_RDT2_DEPTH=N   number of RDT blocks (default 2 smoke; real default config = 14)
"""

from __future__ import annotations

import os
import sys

import torch
from torch import nn

_RDT2_REPO = os.environ.get("M2M_RDT2_ROOT", "/path/to/RDT2")
if _RDT2_REPO not in sys.path:
    sys.path.insert(0, _RDT2_REPO)


# Mirrors configs/rdt/post_train.yaml `model` block (hidden/heads sized so
# hidden_size / num_heads = 128 to match Qwen2.5-VL-7B; GQA num_kv_heads = 4).
def _rdt_config(depth: int) -> dict:
    return {
        "hidden_size": 1024,
        "depth": depth,
        "num_heads": 8,
        "num_kv_heads": 4,
        "num_register_tokens": 4,
        "norm_eps": 1e-5,
        "multiple_of": 256,
        "ffn_dim_multiplier": None,
        # manual attention keeps the trace free of flash-attn custom ops
        "use_flash_attn": False,
    }


class RDT2DenoiseStep(nn.Module):
    """One RDT2 flow-matching step: noised action tokens + VLM KV cache -> velocity.

    `lang_c_kv` is the list of per-block (k, v) tensors sliced from the Qwen2.5-VL
    KV cache by `selected_layers`. Passing it (instead of `lang_c`) selects the
    cross-attention-with-cache branch, which is the real deployment path.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x, t, state_c, *kv):
        # Rebuild the list[(k, v)] structure the model expects (one pair per block).
        lang_c_kv = [(kv[2 * i], kv[2 * i + 1]) for i in range(len(kv) // 2)]
        return self.model(
            x=x,
            t=t,
            lang_c=None,
            lang_c_kv=lang_c_kv,
            img_c=None,
            state_c=state_c,
            lang_mask=None,
            img_mask=None,
        )


def get_model_and_inputs():
    from models.rdt.model import RDT

    depth = int(os.environ.get("M2M_RDT2_DEPTH", "2"))
    cfg = _rdt_config(depth)
    hidden_size = cfg["hidden_size"]
    n_kv_heads = cfg["num_kv_heads"]
    head_size = hidden_size // cfg["num_heads"]  # 128
    num_reg = cfg["num_register_tokens"]

    horizon = 24       # action_chunk_size
    action_dim = 20    # output_size
    dtype = torch.float32

    # No image branch (img_pos_emb_config=None disables it, matching post_train.yaml
    # where img_adaptor is commented out and language enters via the VLM KV cache).
    model = RDT(
        horizon=horizon,
        output_size=action_dim,
        config=cfg,
        x_pos_emb_config=[("action", horizon), ("register", num_reg)],
        lang_pos_emb_config=None,
        max_lang_len=256,
        img_pos_emb_config=None,
        max_img_len=0,
        dtype=dtype,
    ).eval()

    b = 1
    lang_len = 64  # Qwen2.5-VL prompt sequence length (variable)

    # x: adapted noisy-action tokens (B, horizon, D) -- act_adaptor output.
    x = torch.randn(b, horizon, hidden_size, dtype=dtype)
    # t: flow-matching timestep in [0, 1], shape (1,) (broadcast to batch internally).
    t = torch.tensor([0.0], dtype=dtype)
    # state_c: adapted proprio state token (B, 1, D) -- state_adaptor output.
    state_c = torch.randn(b, 1, hidden_size, dtype=dtype)

    # lang_c_kv: per-block (k, v) from the Qwen2.5-VL cache. In model.py each
    # pair is `.transpose(1, 2)`'d to (B, n_kv_heads, seq_len, head_size) BEFORE
    # being handed to CrossAttention, whose `repeat_kv` then expects layout
    # (B, seq_len, n_kv_heads, head_size). Working back through that transpose,
    # the cache tensors fed in must be (B, n_kv_heads, seq_len, head_size).
    # One (k, v) pair per RDT block.
    kv = []
    for _ in range(depth):
        kv.append(torch.randn(b, n_kv_heads, lang_len, head_size, dtype=dtype))  # k
        kv.append(torch.randn(b, n_kv_heads, lang_len, head_size, dtype=dtype))  # v

    inputs = (x, t, state_c, *kv)
    return RDT2DenoiseStep(model).eval(), inputs
