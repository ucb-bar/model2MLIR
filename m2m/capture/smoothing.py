"""Floating-point-equivalent activation smoothing for decoder-only transformers (SmoothQuant).

A static per-tensor int8 activation scale is set by the largest value in the tensor. In Llama-family
decoders a few input channels of some projections carry values hundreds of times larger than the rest
(the "massive activations" of the first token and of delimiters), so one scale either clips them -- and
the model stops behaving like itself -- or spends the whole int8 range on them, leaving the other
channels a handful of levels.

Smoothing moves a per-channel factor ``s_j`` from the activation into the weight that consumes it,

    s_j = max|X_j| ** alpha / max|W[:, j]| ** (1 - alpha)
    X' = X / s,   W' = W * s        so   X' @ W'.T == X @ W.T

and folds the division into the operation that produces ``X``, where it is exact:

* attention and MLP inputs: the preceding RMSNorm's weight (``input_layernorm`` /
  ``post_attention_layernorm``) divides by ``s``; q/k/v resp. gate/up columns multiply by it;
* ``down_proj`` input (``act(gate) * up``): ``up_proj`` rows divide, ``down_proj`` columns multiply;
* ``o_proj`` input (attention-weighted values): ``v_proj`` rows divide, ``o_proj`` columns of every
  query head sharing that key/value head multiply.

The transformed model computes the same function in floating point (to rounding); only the int8
quantization that follows sees different ranges. Weights stay per-tensor or per-channel exactly as the
recipe says; the hardware sees ordinary int8 contractions. Which projections are smoothed, ``alpha``
and the statistics' token corpus are explicit arguments and are returned for the capture's provenance.
"""
from __future__ import annotations

import hashlib
from typing import Any

import torch
from torch import nn

GROUPS = ("qkv", "gate_up", "down", "o")


def _decoder_layers(lm: nn.Module):
    """The decoder layers of a HF causal LM whose layers expose Llama's projection names."""
    model = getattr(lm, "model", None)
    layers = getattr(model, "layers", None)
    if layers is None:
        raise TypeError("smoothing needs a causal LM with model.layers")
    needed = ("input_layernorm", "post_attention_layernorm", "self_attn", "mlp")
    for index, layer in enumerate(layers):
        if not all(hasattr(layer, name) for name in needed):
            raise TypeError(f"decoder layer {index} lacks the Llama projection structure")
        attention, mlp = layer.self_attn, layer.mlp
        if not all(isinstance(getattr(attention, f"{n}_proj", None), nn.Linear) for n in "qkvo"):
            raise TypeError(f"decoder layer {index} attention lacks q/k/v/o linear projections")
        if not all(isinstance(getattr(mlp, f"{n}_proj", None), nn.Linear) for n in ("gate", "up", "down")):
            raise TypeError(f"decoder layer {index} MLP lacks gate/up/down linear projections")
        for name in ("input_layernorm", "post_attention_layernorm"):
            if not isinstance(getattr(getattr(layer, name), "weight", None), torch.Tensor):
                raise TypeError(f"decoder layer {index} {name} carries no per-channel weight to fold into")
    return layers


@torch.no_grad()
def input_absmax(lm: nn.Module, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-input-channel ``max|x|`` of every linear's input over ``tokens`` ([segments, length])."""
    stats: dict[str, torch.Tensor] = {}
    hooks = []

    def hook(name):
        def observe(_module, args):
            value = args[0].detach().reshape(-1, args[0].shape[-1]).abs().amax(0).float()
            stats[name] = torch.maximum(stats[name], value) if name in stats else value

        return observe

    for name, module in lm.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_pre_hook(hook(name)))
    try:
        for row in tokens:
            lm(input_ids=row.reshape(1, -1), use_cache=False)
    finally:
        for handle in hooks:
            handle.remove()
    return stats


def _factor(activation: torch.Tensor, weight_columns: torch.Tensor, alpha: float) -> torch.Tensor:
    floor = torch.finfo(torch.float32).tiny
    s = activation.clamp(min=floor).pow(alpha) / weight_columns.clamp(min=floor).pow(1.0 - alpha)
    return s.clamp(min=1e-5, max=1e5)


@torch.no_grad()
def smooth_decoder(lm: nn.Module, tokens: torch.Tensor, *, alpha: float = 0.5,
                   groups: tuple[str, ...] = GROUPS) -> dict[str, Any]:
    """Smooth ``lm`` in place on statistics from ``tokens``; returns what was done (provenance)."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown smoothing groups {sorted(unknown)}")
    layers = _decoder_layers(lm)
    names = {id(module): name for name, module in lm.named_modules()}
    stats = input_absmax(lm, tokens)
    config = lm.config.get_text_config(decoder=True) if hasattr(lm.config, "get_text_config") else lm.config
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", heads))
    for layer in layers:
        attention, mlp = layer.self_attn, layer.mlp
        if "qkv" in groups:
            linears = (attention.q_proj, attention.k_proj, attention.v_proj)
            s = _factor(stats[names[id(attention.q_proj)]],
                        torch.cat([lin.weight.abs() for lin in linears]).amax(0).float(), alpha)
            layer.input_layernorm.weight.div_(s.to(layer.input_layernorm.weight.dtype))
            for lin in linears:
                lin.weight.mul_(s.to(lin.weight.dtype))
        if "gate_up" in groups:
            linears = (mlp.gate_proj, mlp.up_proj)
            s = _factor(stats[names[id(mlp.gate_proj)]],
                        torch.cat([lin.weight.abs() for lin in linears]).amax(0).float(), alpha)
            layer.post_attention_layernorm.weight.div_(s.to(layer.post_attention_layernorm.weight.dtype))
            for lin in linears:
                lin.weight.mul_(s.to(lin.weight.dtype))
        if "down" in groups:
            # Measured after gate/up smoothing: act(gate) * up is unchanged by it.
            s = _factor(stats[names[id(mlp.down_proj)]], mlp.down_proj.weight.abs().amax(0).float(), alpha)
            if mlp.up_proj.bias is not None:
                mlp.up_proj.bias.div_(s.to(mlp.up_proj.bias.dtype))
            mlp.up_proj.weight.div_(s.to(mlp.up_proj.weight.dtype)[:, None])
            mlp.down_proj.weight.mul_(s.to(mlp.down_proj.weight.dtype))
        if "o" in groups:
            head_dim = attention.o_proj.weight.shape[1] // heads
            group = heads // kv_heads
            act = stats[names[id(attention.o_proj)]].reshape(kv_heads, group, head_dim).amax(1)
            weight = attention.o_proj.weight.abs().float().reshape(-1, kv_heads, group, head_dim).amax(dim=(0, 2))
            s = _factor(act, weight, alpha)  # [kv_heads, head_dim]
            if attention.v_proj.bias is not None:
                attention.v_proj.bias.div_(s.reshape(-1).to(attention.v_proj.bias.dtype))
            attention.v_proj.weight.div_(s.reshape(-1).to(attention.v_proj.weight.dtype)[:, None])
            columns = s[:, None, :].expand(kv_heads, group, head_dim).reshape(-1)
            attention.o_proj.weight.mul_(columns.to(attention.o_proj.weight.dtype))
    digest = hashlib.sha256(tokens.detach().cpu().numpy().tobytes()).hexdigest()
    return {
        "method": "smoothquant_fp_equivalent",
        "alpha": float(alpha),
        "groups": list(groups),
        "statistics_segments": int(tokens.shape[0]),
        "statistics_tokens": int(tokens.shape[1]),
        "statistics_sha256": digest,
    }
