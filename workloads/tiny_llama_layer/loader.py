"""Capture one real TinyLlama decoder layer with a populated KV cache.

The wrapper calls the checkpoint's own ``LlamaDecoderLayer``. Its inputs are
hidden states and cache tensors produced by the same checkpoint on the token
IDs supplied through ``M2M_LLAMA_LAYER_INPUT_IDS``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class CachedDecoderLayer(nn.Module):
    def __init__(self, layer: nn.Module, layer_index: int) -> None:
        super().__init__()
        self.layer = layer
        self.layer_index = layer_index

    def forward(self, hidden: torch.Tensor, key_cache: torch.Tensor,
                value_cache: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        from transformers import DynamicCache

        cache = DynamicCache()
        cache.update(key_cache, value_cache, self.layer_index)
        return self.layer(
            hidden, attention_mask=attention_mask, past_key_values=cache,
            position_embeddings=(cos, sin))


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    from transformers import AutoModelForCausalLM, DynamicCache

    input_path = os.environ.get("M2M_LLAMA_LAYER_INPUT_IDS")
    if not input_path:
        raise ValueError("M2M_LLAMA_LAYER_INPUT_IDS must name captured inputs.npz")
    layer_index = int(os.environ.get("M2M_LLAMA_LAYER_INDEX", "5"))
    prefix_tokens = int(os.environ.get("M2M_LLAMA_PREFIX_TOKENS", "64"))
    query_tokens = int(os.environ.get("M2M_LLAMA_QUERY_TOKENS", "64"))
    with np.load(Path(input_path), allow_pickle=False) as capture:
        token_ids = torch.from_numpy(capture["in0"].copy()).long()
    if tuple(token_ids.shape) != (1, prefix_tokens + query_tokens):
        raise ValueError(
            f"expected token IDs [1, {prefix_tokens + query_tokens}], "
            f"got {tuple(token_ids.shape)}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager").eval()
    if not 0 <= layer_index < len(model.model.layers):
        raise ValueError(f"layer index {layer_index} is outside the checkpoint")

    with torch.no_grad():
        cache = DynamicCache(config=model.config)
        model(token_ids[:, :prefix_tokens], past_key_values=cache,
              use_cache=True)
        full = model(token_ids, output_hidden_states=True, use_cache=False)
        hidden = full.hidden_states[layer_index][:, prefix_tokens:].contiguous()
        key_cache = cache.layers[layer_index].keys.contiguous()
        value_cache = cache.layers[layer_index].values.contiguous()
        positions = torch.arange(prefix_tokens, prefix_tokens + query_tokens)[None]
        cos, sin = model.model.rotary_emb(hidden, positions)
        keys = torch.arange(prefix_tokens + query_tokens)[None, :]
        queries = torch.arange(prefix_tokens, prefix_tokens + query_tokens)[:, None]
        mask = torch.where(keys > queries, -1e9, 0.0)[None, None].to(torch.float32)
        section = CachedDecoderLayer(model.model.layers[layer_index], layer_index).eval()
        inputs = (hidden, key_cache, value_cache, cos, sin, mask)
        actual = section(*inputs)
        reference = full.hidden_states[layer_index + 1][:, prefix_tokens:]
        if not torch.allclose(actual, reference, atol=1e-6, rtol=1e-6):
            delta = (actual - reference).abs().max().item()
            raise ValueError(f"cached layer does not match full checkpoint: max abs {delta}")
    return section, inputs
