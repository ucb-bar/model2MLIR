"""Xiaomi-Robotics-0 (XR0) capture loader for m2m.

XR0 is a VLA model: a Qwen3-VL-4B backbone encodes vision+language into a
KV-cache, and a DiT (Diffusion Transformer) head decodes an action chunk via
rectified flow (Euler integration over `num_steps`).

The full `XR0.forward(batch)` is NOT a capture target:
  - it instantiates Qwen3-VL via `from_pretrained("Qwen/Qwen3-VL-4B-Instruct")`
    (multi-GB download, flash_attention_2),
  - inference runs a Python `for`-loop Euler integrator (`_flow_generate`),
  - control flow depends on `prefix_length` resolved via `.item()`.

Capture unit: ONE DiT denoise step (`XR0.dit_forward`) -- a single rectified-flow
velocity prediction. This is pure tensor ops (AdaLN-modulated DiT decoder layers
cross-attending to the VLM KV-cache) with no data-dependent control flow when
`prefix_length == 0`. The VLM KV-cache, RoPE position embeds, attention mask and
projected state are supplied as already-computed input tensors (host-side prefix
encoding stays out of the graph), mirroring the smolVLA denoise-step convention.

We build only the DiT head + projectors + embedders from a small random config;
the Qwen3-VL backbone is never constructed, so no weights are downloaded.

Env:
    XR0_DIT_LAYERS=N   number of DiT decoder layers (default 2 for a fast smoke;
                       the real model uses 16)
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import torch
from torch import nn


def _load_xr0_module() -> types.ModuleType:
    """Import the real ``mibot.models.VLA.XR0`` module in isolation.

    Importing ``mibot`` normally triggers ``mibot/__init__`` -> ``mibot.data``
    -> ``lightning`` (a heavy training-only dep we don't need), and the full
    ``mibot.models`` package which imports the Qwen3-VL backbone wiring. We only
    need the pure-tensor DiT classes from ``XR0.py``. So we pre-register the
    parent packages as lightweight stubs and load ``XR0.py`` directly from its
    file, re-using the *real* class implementations without the download/lightning
    side-effects. ``XR0.py`` only needs ``MIMODEL`` (an mmengine Registry) and
    ``auto_cast`` from the mibot namespace, plus the real ``qwen3vl`` module
    (pure transformers, no weights downloaded at import).
    """
    if "mibot.models.VLA.XR0" in sys.modules:
        return sys.modules["mibot.models.VLA.XR0"]

    xr0_root = os.environ.get("M2M_XR0_ROOT", "/path/to/Xiaomi-Robotics-0/xr0")
    from mmengine import Registry

    # --- stub package tree so XR0.py's intra-package imports resolve ---
    def _pkg(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__path__ = []  # mark as package
        sys.modules[name] = mod
        return mod

    mibot = _pkg("mibot")
    mibot_models = _pkg("mibot.models")
    mibot_models.MIMODEL = Registry("MIMODEL")  # XR0.py: from mibot.models import MIMODEL
    mibot.models = mibot_models
    _pkg("mibot.models.VLA")
    _pkg("mibot.models.VLM")
    mibot_utils = _pkg("mibot.utils")

    # Real auto_cast (tiny, no heavy deps).
    mu_spec = importlib.util.spec_from_file_location(
        "mibot.utils.model_utils", f"{xr0_root}/mibot/utils/model_utils.py"
    )
    model_utils = importlib.util.module_from_spec(mu_spec)
    sys.modules["mibot.utils.model_utils"] = model_utils
    mu_spec.loader.exec_module(model_utils)
    mibot_utils.model_utils = model_utils

    # Real qwen3vl backbone module (pure transformers; no from_pretrained at import).
    qv_spec = importlib.util.spec_from_file_location(
        "mibot.models.VLM.qwen3vl", f"{xr0_root}/mibot/models/VLM/qwen3vl.py"
    )
    qwen3vl = importlib.util.module_from_spec(qv_spec)
    sys.modules["mibot.models.VLM.qwen3vl"] = qwen3vl
    qv_spec.loader.exec_module(qwen3vl)

    # Real XR0 module.
    xr0_spec = importlib.util.spec_from_file_location(
        "mibot.models.VLA.XR0", f"{xr0_root}/mibot/models/VLA/XR0.py"
    )
    xr0_mod = importlib.util.module_from_spec(xr0_spec)
    sys.modules["mibot.models.VLA.XR0"] = xr0_mod
    xr0_spec.loader.exec_module(xr0_mod)
    return xr0_mod

# Real XR0 defaults (see xr0/mibot/models/VLA/XR0.py and configs/model).
# Qwen3-VL-4B text config: hidden 2560, 36 layers, head_dim 128, 8 KV heads.
_VLM_NUM_LAYERS = 36
_VLM_KV_HEADS = 8
_HEAD_DIM = 128
_DIT_HIDDEN = 1024
_STATE_LEN, _STATE_DIM = 1, 32
_ACTION_LEN, _ACTION_DIM = 30, 32
_KV_HEADS_DIT = 8


class XR0DenoiseStep(nn.Module):
    """Wrap XR0.dit_forward as a clean tensor->tensor capture unit (one flow step)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, noisy_action, t, action_mask, state, cos, sin, attn_mask, *kv):
        m = self.model
        # Reassemble the flat (k0, v0, k1, v1, ...) tensor list into the
        # per-layer [(k, v), ...] structure dit_forward expects.
        past_key_values = [(kv[2 * i], kv[2 * i + 1]) for i in range(len(kv) // 2)]
        state_embed = m.state_projector(state)
        return m.dit_forward(
            noisy_action=noisy_action,
            t=t,
            action_mask=action_mask,
            state_embed=state_embed,
            position_embeds=(cos, sin),
            past_key_values=past_key_values,
            attn_mask=attn_mask,
            prefix_length=0,
        )


def _build_dit_head(dit_layers: int) -> nn.Module:
    """Build the DiT head + projectors standalone, without the Qwen3-VL backbone.

    XR0.__init__ calls Qwen3VLForConditionalGeneration.from_pretrained (a multi-GB
    download). We bypass that by constructing the lightweight sub-modules directly,
    re-using the real XR0 classes so dit_forward semantics are identical.
    """
    _xr0 = _load_xr0_module()
    DiT = _xr0.DiT
    MLPProjector = _xr0.MLPProjector
    TimestepEmbedder = _xr0.TimestepEmbedder
    from transformers import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextRotaryEmbedding,
    )

    class _DiTHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dit_hidden_size = _DIT_HIDDEN
            self.action_shape = (_ACTION_LEN, _ACTION_DIM)
            self.state_shape = (_STATE_LEN, _STATE_DIM)
            self.dit = DiT(hidden_size=_DIT_HIDDEN, kv_heads=_KV_HEADS_DIT, layer_num=dit_layers)
            self.state_projector = MLPProjector(_STATE_DIM, _DIT_HIDDEN, num_layers=2)
            self.action_projector = MLPProjector(_ACTION_DIM, _DIT_HIDDEN, num_layers=2)
            self.action_output_layer = MLPProjector(_DIT_HIDDEN, _ACTION_DIM, num_layers=2)
            self.t_embedder = TimestepEmbedder(_DIT_HIDDEN)
            self.t_projector = MLPProjector(_DIT_HIDDEN, 6 * _DIT_HIDDEN, bias=True)
            self.sink = nn.Embedding(1, _DIT_HIDDEN)
            # Small random text config for RoPE (no weights downloaded).
            txt_cfg = Qwen3VLTextConfig(
                hidden_size=_DIT_HIDDEN, head_dim=_HEAD_DIM, num_attention_heads=8,
                num_key_value_heads=_KV_HEADS_DIT, num_hidden_layers=2,
                rope_scaling={"rope_type": "default", "mrope_section": [16, 24, 24]},
            )
            self.rotary_emb = Qwen3VLTextRotaryEmbedding(txt_cfg)

        # dit_forward is defined on XR0; bind the unbound method here.
        dit_forward = staticmethod(None)

    head = _DiTHead()

    def dit_forward(
        self,
        noisy_action,
        t,
        action_mask,
        state_embed,
        position_embeds,
        past_key_values,
        attn_mask,
        prefix_length: int = 0,
    ):
        """Capture-faithful copy of ``XR0.dit_forward`` (prefix_length==0 path).

        Identical math to the upstream method; the ONLY change is the timestep
        squeeze ``t[:, 0, 0]`` -> ``t.reshape(t.shape[0])``. The upstream double
        rank-reducing ``select`` on a ``(B, 1, 1)`` tensor lowers to a chain that
        collapses to a rank-0 SSA value, tripping a divide-by-rank in m2m's
        ``aten.select.int`` decomposition. ``reshape`` is value-identical for a
        ``(B, 1, 1)`` input and lowers cleanly. ``prefix_length`` is 0, so the
        masked-assignment branch (also non-exportable) is excluded as designed.
        """
        t_embeds = self.t_embedder(t.reshape(t.shape[0]) * 1000)
        t_embeds = self.t_projector(t_embeds).view(t_embeds.shape[0], 6, -1)

        noisy_action = noisy_action * action_mask
        noisy_action = self.action_projector(noisy_action)

        sink = self.sink.weight[None].repeat(state_embed.shape[0], 1, 1)
        hidden_states = torch.cat([sink, state_embed, noisy_action], dim=1).contiguous()

        hidden_states = self.dit(hidden_states, past_key_values, attn_mask, position_embeds, t_embeds)

        hidden_states = hidden_states[:, -noisy_action.shape[1]:, :]
        output = self.action_output_layer(hidden_states)
        return output

    head.dit_forward = dit_forward.__get__(head, _DiTHead)
    # TimestepEmbedder hardcodes self.dtype=bfloat16 for its sinusoidal embed;
    # align it to fp32 so the embedding matches the fp32 MLP weights.
    head.t_embedder.dtype = torch.float32
    return head.to(torch.float32).eval()


def get_model_and_inputs():
    dit_layers = int(os.environ.get("XR0_DIT_LAYERS", "2"))
    head = _build_dit_head(dit_layers)

    b = 1
    q_len = _STATE_LEN + 1 + _ACTION_LEN  # sink + state + action tokens
    # Prefix (VLM cache) length is arbitrary for a single denoise step; pick small.
    prefix_len = 16
    total_len = prefix_len + q_len

    noisy_action = torch.randn(b, _ACTION_LEN, _ACTION_DIM, dtype=torch.float32)
    t = torch.ones(b, 1, 1, dtype=torch.float32) * 0.5
    action_mask = torch.ones(b, _ACTION_LEN, _ACTION_DIM, dtype=torch.float32)
    state = torch.randn(b, _STATE_LEN, _STATE_DIM, dtype=torch.float32)

    # RoPE (cos, sin) for the q_len DiT tokens, head_dim = 128.
    cos = torch.randn(b, q_len, _HEAD_DIM, dtype=torch.float32)
    sin = torch.randn(b, q_len, _HEAD_DIM, dtype=torch.float32)

    # Boolean attention mask: (B, 1, q_len, prefix_len + q_len).
    attn_mask = torch.ones(b, 1, q_len, total_len, dtype=torch.bool)

    # VLM KV-cache: the DiT consumes the tail `dit_layers` cache entries, each
    # (B, kv_heads, prefix_len, head_dim). Supply `dit_layers` (k, v) pairs.
    kv = []
    for _ in range(dit_layers):
        kv.append(torch.randn(b, _KV_HEADS_DIT, prefix_len, _HEAD_DIM, dtype=torch.float32))  # k
        kv.append(torch.randn(b, _KV_HEADS_DIT, prefix_len, _HEAD_DIM, dtype=torch.float32))  # v

    inputs = (noisy_action, t, action_mask, state, cos, sin, attn_mask, *kv)
    return XR0DenoiseStep(head).eval(), inputs
