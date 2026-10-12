"""NVIDIA Isaac-GR00T N1.7 (PyTorch) capture loader for m2m.

GR00T N1.7 is a VLA policy = a Cosmos-Reason2-2B (Qwen3-VL) VLM backbone + a
flow-matching DiT action head. The backbone is loaded via transformers
trust_remote_code and needs flash-attn + a newer transformers; it is NOT the
representative compute we want to capture and it is heavy to instantiate.

The capture unit here is the *action head DiT denoise compute* (`AlternateVLDiT`),
which is the flow-matching transformer that actually predicts action velocities --
the analogue of pi0.5/smolVLA `denoise_step`. It runs on ALREADY-EMBEDDED backbone
features (the VLM forward is host-side and out of the captured graph), plus the
state/action encoders and action decoder. This avoids the Qwen3-VL backbone and the
dict-based `prepare_input` collator preprocessing (both data-dependent / not
torch.export-friendly), while keeping the heavy diffusion graph.

    get_model_and_inputs() -> (nn.Module, tuple[Tensor, ...])

Env:
    M2M_GROOT_LAYERS=N   DiT transformer layers (fast smoke; default: config's 16)
"""

from __future__ import annotations

import os
import sys
import types

import torch
from torch import nn

# Make the upstream repo importable without requiring an editable install of gr00t.
_GROOT_REPO = os.environ.get("M2M_GROOT_ROOT", "/path/to/Isaac-GR00T")
if _GROOT_REPO not in sys.path:
    sys.path.insert(0, _GROOT_REPO)

# ``gr00t.model.__init__`` eagerly imports the full training/inference pipeline
# (dataset factory -> pandas, HF backbone setup, etc.), none of which is needed
# to instantiate the action-head DiT and all of which drags in heavy/optional
# deps. We only need the pure-torch modules under ``gr00t/model/gr00t_n1d7/`` and
# ``gr00t/model/modules/``. Pre-seed ``sys.modules`` with a lightweight package
# stub for ``gr00t.model`` so importing its leaf submodules does NOT execute the
# package ``__init__`` body. The subpackages (gr00t_n1d7, modules) resolve via
# the real filesystem path recorded on the stub's ``__path__``.
import gr00t  # safe: gr00t/__init__.py only installs opt-in HF patches
if "gr00t.model" not in sys.modules:
    _pkg_path = os.path.join(os.path.dirname(gr00t.__file__), "model")
    _stub = types.ModuleType("gr00t.model")
    _stub.__path__ = [_pkg_path]  # mark as a package so submodule imports work
    sys.modules["gr00t.model"] = _stub

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead


class Gr00tDenoiseStep(nn.Module):
    """One flow-matching denoise step of the GR00T N1.7 action head.

    Mirrors the body of `Gr00tN1d7ActionHead.get_action_with_features` for a single
    timestep, on already-embedded backbone features. Inputs are plain tensors so the
    graph is a clean tensor->tensor function (no dict collator, no VLM backbone).
    """

    def __init__(self, head: Gr00tN1d7ActionHead) -> None:
        super().__init__()
        self.head = head

    def forward(
        self,
        backbone_features,      # [B, S, backbone_embedding_dim] VLM features (vlln-normalized upstream)
        state,                  # [B, state_history_length, max_state_dim]
        actions,                # [B, action_horizon, action_dim] current noised trajectory x_t
        embodiment_id,          # [B] long, selects per-embodiment MLP weights
        backbone_attention_mask,  # [B, S] bool
        image_mask,             # [B, S] bool (True = image token)
        timesteps,              # [B] long, discretized diffusion timestep bucket
    ):
        h = self.head
        # VL self-attention / layernorm on backbone features (host already ran vlln in
        # the full model; here vlln/vl_self_attention default to Identity unless config
        # enables them, matching process_backbone_output).
        vl_embeds = h.vlln(backbone_features)
        vl_embeds = h.vl_self_attention(vl_embeds)

        state = state.view(state.shape[0], 1, -1)
        state_features = h.state_encoder(state, embodiment_id)

        action_features = h.action_encoder(actions, timesteps, embodiment_id)
        if h.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=action_features.device)
            pos_embs = h.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        sa_embs = torch.cat((state_features, action_features), dim=1)

        if h.config.use_alternate_vl_dit:
            model_output = h.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output = h.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps,
            )
        pred = h.action_decoder(model_output, embodiment_id)
        return pred[:, -h.action_horizon :]  # predicted velocity over the action chunk


def get_model_and_inputs():
    cfg = Gr00tN1d7Config()
    cfg.model_dtype = "float32"
    cfg.use_flash_attention = False
    layers = os.environ.get("M2M_GROOT_LAYERS")
    if layers:
        cfg.diffusion_model_cfg = {**cfg.diffusion_model_cfg, "num_layers": int(layers)}

    head = Gr00tN1d7ActionHead(cfg).to(torch.float32).eval()

    b = 1
    seq = 64  # backbone feature sequence length (image + text tokens); arbitrary for capture
    sd = cfg.max_state_dim
    sh = cfg.state_history_length
    ah, ad = cfg.action_horizon, cfg.max_action_dim
    bdim = cfg.backbone_embedding_dim

    backbone_features = torch.randn(b, seq, bdim, dtype=torch.float32)
    state = torch.randn(b, sh, sd, dtype=torch.float32)
    actions = torch.randn(b, ah, ad, dtype=torch.float32)
    embodiment_id = torch.zeros(b, dtype=torch.long)
    backbone_attention_mask = torch.ones(b, seq, dtype=torch.bool)
    image_mask = torch.zeros(b, seq, dtype=torch.bool)
    image_mask[:, : seq // 2] = True  # first half = image tokens
    timesteps = torch.zeros(b, dtype=torch.long)

    inputs = (
        backbone_features, state, actions, embodiment_id,
        backbone_attention_mask, image_mask, timesteps,
    )
    return Gr00tDenoiseStep(head).eval(), inputs
