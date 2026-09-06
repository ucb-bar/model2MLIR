"""Fixed-shape causal-LM decode sessions for export and the Merlin runtime.

HuggingFace's dynamic cache grows on every token, which is the opposite of the ABI a compiled
runtime needs.  This module adapts ``StaticCache`` to three plain tensors:

``input_ids``
    One or more new tokens (the paper workloads use one token per measured decode step).
``kv_cache``
    ``[2, layers, batch, kv_heads, capacity, head_dim]``; key then value.
``position``
    One current-length value per layer.  A vector is intentional: layers own their cache state and
    the exported function must not update one shared scalar once per layer.

The wrapper clones state before asking Transformers to mutate it.  Consequently the exported ABI
is functional (updated state is returned explicitly), while eager execution has the same semantics.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def _cache_geometry(lm: nn.Module) -> tuple[int, int, int]:
    config = lm.config.get_text_config(decoder=True)
    layers = int(config.num_hidden_layers)
    kv_heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    return layers, kv_heads, head_dim


def _new_static_cache(lm: nn.Module, capacity: int):
    from transformers.cache_utils import StaticCache

    layers, kv_heads, head_dim = _cache_geometry(lm)
    cache = StaticCache(config=lm.config, max_cache_len=capacity)
    cache.early_initialization(
        batch_size=1, num_heads=kv_heads, head_dim=head_dim,
        dtype=next(lm.parameters()).dtype, device=next(lm.parameters()).device)
    if len(cache.layers) != layers:
        raise ValueError(f"StaticCache has {len(cache.layers)} layers, expected {layers}")
    return cache


def _new_decode_cache(lm: nn.Module, capacity: int):
    """Create a functional, fixed-capacity cache specialized for one-token decode.

    ``StaticLayer.update`` uses in-place ``index_copy_``. Torch functionalization turns that into
    ``aten.index_put``, which is not an executable m2m lowering today. The equivalent one-token
    update below is a broadcast ``where`` over the fixed cache slots and stays entirely in the
    compiler-owned tensor dialect.
    """
    from transformers.cache_utils import Cache, StaticLayer

    class _FunctionalDecodeLayer(StaticLayer):
        def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
            if not self.is_initialized:
                self.lazy_initialization(key_states, value_states)
            slots = torch.arange(self.max_cache_len, device=key_states.device)
            selected = (slots == self.cumulative_length.reshape(1)).reshape(1, 1, -1, 1)
            self.keys = torch.where(selected, key_states, self.keys)
            self.values = torch.where(selected, value_states, self.values)
            self.cumulative_length = self.cumulative_length + 1
            return self.keys, self.values

    layers, kv_heads, head_dim = _cache_geometry(lm)
    cache = Cache(layers=[_FunctionalDecodeLayer(max_cache_len=capacity) for _ in range(layers)])
    cache.early_initialization(
        batch_size=1, num_heads=kv_heads, head_dim=head_dim,
        dtype=next(lm.parameters()).dtype, device=next(lm.parameters()).device)
    return cache


def _new_prefill_cache(lm: nn.Module, capacity: int):
    """Functional cache whose one update fills a fixed-capacity prefix from position zero."""
    from transformers.cache_utils import Cache, StaticLayer

    class _FunctionalPrefillLayer(StaticLayer):
        def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
            if not self.is_initialized:
                self.lazy_initialization(key_states, value_states)
            token_count = key_states.shape[-2]
            if token_count > self.max_cache_len:
                raise ValueError("prefill token count exceeds fixed cache capacity")
            # The prefill stage always starts at position zero. Padding is functional and remains
            # standard tensor/linalg IR; StaticLayer.index_copy_ would become unsupported index_put.
            self.keys = torch.nn.functional.pad(
                key_states, (0, 0, 0, self.max_cache_len - token_count))
            self.values = torch.nn.functional.pad(
                value_states, (0, 0, 0, self.max_cache_len - token_count))
            self.cumulative_length = torch.full_like(self.cumulative_length, token_count)
            return self.keys, self.values

    layers, kv_heads, head_dim = _cache_geometry(lm)
    cache = Cache(layers=[_FunctionalPrefillLayer(max_cache_len=capacity) for _ in range(layers)])
    cache.early_initialization(
        batch_size=1, num_heads=kv_heads, head_dim=head_dim,
        dtype=next(lm.parameters()).dtype, device=next(lm.parameters()).device)
    return cache


class FixedCachePrefillStage(nn.Module):
    """Compiled fixed-shape prefix stage returning the decode stage's initial tensor state."""

    def __init__(self, lm: nn.Module, capacity: int) -> None:
        super().__init__()
        self.lm = lm.eval().requires_grad_(False)
        self.capacity = int(capacity)
        self._static_cache = _new_prefill_cache(self.lm, self.capacity)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        for layer in self._static_cache.layers:
            layer.cumulative_length = torch.zeros_like(layer.cumulative_length)
        output = self.lm(
            input_ids=input_ids, cache_position=positions, position_ids=positions.unsqueeze(0),
            past_key_values=self._static_cache, use_cache=True)
        next_cache = torch.stack((
            torch.stack([layer.keys for layer in self._static_cache.layers]),
            torch.stack([layer.values for layer in self._static_cache.layers]),
        ))
        next_position = torch.cat([
            layer.cumulative_length.reshape(1) for layer in self._static_cache.layers
        ])
        return output.logits[:, -1, :], next_cache, next_position


class FixedCacheDecodeStep(nn.Module):
    """Expose a Transformers causal LM as a fixed-state decode step."""

    def __init__(self, lm: nn.Module, capacity: int) -> None:
        super().__init__()
        if capacity < 1:
            raise ValueError("cache capacity must be positive")
        self.lm = lm.eval().requires_grad_(False)
        self.capacity = int(capacity)
        self._static_cache = _new_decode_cache(self.lm, self.capacity)

    def forward(self, input_ids: torch.Tensor, kv_cache: torch.Tensor,
                position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The private cache object is rebound to the public input tensors on every call. Its update
        # is functional, so callers' state is never mutated and the returned tensors are the only
        # next-state values.
        for layer_index, layer in enumerate(self._static_cache.layers):
            layer.keys = kv_cache[0, layer_index]
            layer.values = kv_cache[1, layer_index]
            layer.cumulative_length = position[layer_index:layer_index + 1]

        cache_position = (torch.arange(input_ids.shape[1], device=input_ids.device)
                          + position[0:1])
        output = self.lm(
            input_ids=input_ids,
            cache_position=cache_position,
            position_ids=cache_position.unsqueeze(0),
            past_key_values=self._static_cache,
            use_cache=True,
        )
        next_cache = torch.stack((
            torch.stack([layer.keys for layer in self._static_cache.layers]),
            torch.stack([layer.values for layer in self._static_cache.layers]),
        ))
        next_position = torch.cat([
            layer.cumulative_length.reshape(1) for layer in self._static_cache.layers
        ])
        return output.logits[:, -1, :], next_cache, next_position


def prefill_state(lm: nn.Module, input_ids: torch.Tensor, capacity: int) \
        -> tuple[torch.Tensor, torch.Tensor]:
    """Run the reference prefill once and return fixed-capacity KV/position tensors."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"prefill input_ids must have shape [1, tokens], got {tuple(input_ids.shape)}")
    if input_ids.shape[1] < 1 or input_ids.shape[1] >= capacity:
        raise ValueError("prefill must leave at least one cache slot for decode")
    lm.eval().requires_grad_(False)
    cache = _new_static_cache(lm, capacity)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    with torch.no_grad():
        lm(input_ids=input_ids, cache_position=positions, position_ids=positions.unsqueeze(0),
           past_key_values=cache, use_cache=True)
    keys = torch.stack([layer.keys.detach().clone() for layer in cache.layers])
    values = torch.stack([layer.values.detach().clone() for layer in cache.layers])
    lengths = torch.stack([
        torch.as_tensor(layer.cumulative_length, device=input_ids.device).reshape(-1)[0]
        for layer in cache.layers
    ]).to(dtype=torch.long)
    expected = torch.full_like(lengths, int(input_ids.shape[1]))
    if not torch.equal(lengths, expected):
        raise ValueError(f"prefill cache lengths disagree: {lengths.tolist()}")
    return torch.stack((keys, values)), lengths


def load_token_corpus(*, vocab_size: int, total_tokens: int, env_prefix: str) \
        -> tuple[torch.Tensor, dict[str, Any], bool]:
    """Load an externally tokenized paper corpus or make an explicit synthetic smoke corpus.

    ``<PREFIX>_TOKEN_IDS`` accepts ``.npy`` or ``.npz`` (``input_ids`` or the sole array).
    ``<PREFIX>_TOKEN_SOURCE`` is required before ``<PREFIX>_PAPER_READY=1`` can succeed.
    The returned provenance always includes the actual bytes' SHA-256.
    """
    path_value = os.environ.get(f"{env_prefix}_TOKEN_IDS")
    source = os.environ.get(f"{env_prefix}_TOKEN_SOURCE", "")
    requested_ready = os.environ.get(f"{env_prefix}_PAPER_READY", "0") == "1"
    if path_value:
        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"token corpus is absent: {path}")
        if path.suffix == ".npz":
            with np.load(path) as data:
                if "input_ids" in data.files:
                    array = np.asarray(data["input_ids"])
                elif len(data.files) == 1:
                    array = np.asarray(data[data.files[0]])
                else:
                    raise ValueError("token npz needs an input_ids array or exactly one array")
        else:
            array = np.asarray(np.load(path))
        flat = np.ascontiguousarray(array).reshape(-1)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        synthetic = False
    else:
        generator = torch.Generator().manual_seed(0)
        flat = torch.randint(0, vocab_size, (total_tokens,), generator=generator,
                             dtype=torch.long).numpy()
        digest = hashlib.sha256(np.ascontiguousarray(flat).tobytes()).hexdigest()
        synthetic = True
    if flat.size < total_tokens:
        raise ValueError(f"token corpus has {flat.size} tokens, needs at least {total_tokens}")
    flat = flat[:total_tokens].astype(np.int64, copy=False)
    if np.any(flat < 0) or np.any(flat >= vocab_size):
        raise ValueError(f"token corpus contains ids outside [0, {vocab_size})")
    ready = requested_ready and not synthetic and bool(source)
    if requested_ready and not ready:
        raise ValueError(
            f"{env_prefix}_PAPER_READY=1 requires an external token corpus and "
            f"{env_prefix}_TOKEN_SOURCE")
    provenance = {
        "token_source": source or "synthetic_seed_0",
        "token_sha256": digest,
        "synthetic_tokens": synthetic,
    }
    return torch.from_numpy(flat.copy()).reshape(1, -1), provenance, ready


def make_decode_session(lm: nn.Module, *, env_prefix: str, checkpoint: str,
                        full_checkpoint: bool, prefill_tokens: int = 128,
                        decode_tokens: int = 32) \
        -> tuple[FixedCacheDecodeStep, tuple[torch.Tensor, ...]]:
    """Build a stateful decode module and attach its capture session specification."""
    if prefill_tokens < 1 or decode_tokens < 1:
        raise ValueError("prefill_tokens and decode_tokens must be positive")
    capacity = prefill_tokens + decode_tokens
    tokens, token_provenance, corpus_ready = load_token_corpus(
        vocab_size=int(lm.config.vocab_size), total_tokens=capacity, env_prefix=env_prefix)
    if corpus_ready and not full_checkpoint:
        raise ValueError(
            f"{env_prefix}_PAPER_READY=1 requires the complete pretrained checkpoint")
    tokens = tokens.to(next(lm.parameters()).device)
    kv_cache, position = prefill_state(lm, tokens[:, :prefill_tokens], capacity)
    decode_stream = tokens[:, prefill_tokens:].transpose(0, 1).reshape(decode_tokens, 1, 1)
    wrapper = FixedCacheDecodeStep(lm, capacity).eval()
    paper_ready = bool(full_checkpoint and corpus_ready)
    wrapper.session_spec = {
        "kind": "autoregressive_decode",
        "paper_ready": paper_ready,
        "stages": ["prefill", "decode"],
        "stage_schedule": [
            {"name": "prefill", "steps": 1, "tokens": prefill_tokens,
             "execution": "eager_reference_initial_state", "timed": False},
            {"name": "decode", "steps": decode_tokens, "tokens_per_step": 1,
             "execution": "compiled_recurrent", "timed": True},
        ],
        "parameters": {
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "batch": 1,
            "prefill_policy": "untimed_eager_reference_initial_state",
            "timed_stages": ["decode"],
        },
        "states": [
            {"name": "kv_cache", "input_index": 1, "output_index": 1},
            {"name": "position", "input_index": 2, "output_index": 2},
        ],
        "streams": [{"name": "decode_token", "input_index": 0, "key": "decode_tokens",
                     "values": decode_stream}],
        "quality": {"key": "logits", "output_index": 0},
        "provenance": {
            "checkpoint": checkpoint,
            "full_checkpoint": bool(full_checkpoint),
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            **token_provenance,
        },
    }
    inputs = (decode_stream[0].clone(), kv_cache, position)
    return wrapper, inputs


class CausalMultiProgramCapture:
    """Deferred prefill+decode capture so shared weights are quantized exactly once."""

    def __init__(self, lm: nn.Module, tokens: torch.Tensor, *, checkpoint: str,
                 full_checkpoint: bool, paper_ready: bool, provenance: dict[str, Any],
                 prefill_tokens: int, decode_tokens: int) -> None:
        self.lm = lm
        self.tokens = tokens
        self.checkpoint = checkpoint
        self.full_checkpoint = bool(full_checkpoint)
        self.paper_ready = bool(paper_ready)
        self.provenance = dict(provenance)
        self.prefill_tokens = int(prefill_tokens)
        self.decode_tokens = int(decode_tokens)

    def _external_runtime_session(self, *, quality_reference: np.ndarray | None = None):
        """Build the exact prefill/decode modules and functional tensor routing contract."""
        from m2m.capture.external_runtime import (
            ExternalRuntimeProgram, make_external_runtime_session,
        )

        capacity = self.prefill_tokens + self.decode_tokens
        tokens = self.tokens.to(next(self.lm.parameters()).device)
        prefill_input = tokens[:, :self.prefill_tokens]
        decode_stream = tokens[:, self.prefill_tokens:].transpose(0, 1).reshape(
            self.decode_tokens, 1, 1)
        prefill = FixedCachePrefillStage(self.lm, capacity).eval()
        with torch.no_grad():
            _logits, kv_cache, position = prefill(prefill_input)
        decode = FixedCacheDecodeStep(self.lm, capacity).eval()
        decode_session = {
            "kind": "autoregressive_decode", "paper_ready": self.paper_ready,
            "stages": ["decode"],
            "stage_schedule": [{"name": "decode", "steps": self.decode_tokens,
                                "execution": "compiled_recurrent", "timed": True}],
            "parameters": {"decode_tokens": self.decode_tokens, "batch": 1},
            "states": [
                {"name": "kv_cache", "input_index": 1, "output_index": 1},
                {"name": "position", "input_index": 2, "output_index": 2},
            ],
            "streams": [{"name": "decode_token", "input_index": 0, "key": "decode_tokens",
                         "values": decode_stream}],
            "quality": {"key": "logits", "output_index": 0, "reference": "eager_fp32"},
            "provenance": {"checkpoint": self.checkpoint,
                           "full_checkpoint": self.full_checkpoint, **self.provenance},
        }
        if quality_reference is not None:
            decode_session["quality"]["reference_values"] = quality_reference
        root = {
            "kind": "autoregressive_decode", "paper_ready": self.paper_ready,
            "stages": ["prefill", "decode"],
            "stage_schedule": [
                {"name": "prefill", "steps": 1, "tokens": self.prefill_tokens,
                 "execution": "compiled", "timed": True},
                {"name": "decode", "steps": self.decode_tokens, "tokens_per_step": 1,
                 "execution": "compiled_recurrent", "timed": True},
            ],
            "parameters": {
                "prefill_tokens": self.prefill_tokens, "decode_tokens": self.decode_tokens,
                "batch": 1, "prefill_policy": "compiled_in_continuous_session",
                "timed_stages": ["prefill", "decode"],
                "diagnostic_timed_stages": ["decode"], "paper_primary_scope": "end_to_end",
            },
            "bindings": [
                {"name": "kv_cache", "from": {"program": "prefill", "output_index": 1},
                 "to": {"program": "decode", "input_index": 1}},
                {"name": "position", "from": {"program": "prefill", "output_index": 2},
                 "to": {"program": "decode", "input_index": 2}},
            ],
            "states": ["kv_cache", "position"], "quality_program": "decode",
            "provenance": {"checkpoint": self.checkpoint,
                           "full_checkpoint": self.full_checkpoint,
                           "prefill_tokens": self.prefill_tokens,
                           "decode_tokens": self.decode_tokens, **self.provenance},
        }
        return make_external_runtime_session(
            version=2,
            programs=(
                ExternalRuntimeProgram("prefill", prefill, (prefill_input,), 1),
                ExternalRuntimeProgram(
                    "decode", decode, (decode_stream[0].clone(), kv_cache, position),
                    self.decode_tokens, decode_session),
            ),
            metadata=root,
        )

    def external_runtime_session(self):
        """Public model-agnostic capability consumed by external runtime exporters."""
        return self._external_runtime_session()

    def write_bundle(self, out: str | Path, *, quant=None) -> dict[str, Any]:
        # Generate the quality trajectory from the untouched checkpoint before torchAO changes the
        # shared module. This reference is never reused as the compiler-correctness golden.
        reference = self.external_runtime_session()
        decode_program = reference.programs[1]
        from m2m.capture.bundle import capture_session_trajectory
        quality_reference = capture_session_trajectory(
            decode_program.module, decode_program.inputs, dict(decode_program.session or {}))
        del reference, decode_program

        if quant is not None:
            from m2m.capture.torchao_pipeline import apply_quantization
            self.lm = apply_quantization(self.lm, quant)
        capture = self._external_runtime_session(quality_reference=quality_reference)
        from m2m.capture.bundle import write_multi_program_bundle
        return write_multi_program_bundle(
            capture.bundle_programs(), dict(capture.metadata), out, quant=quant,
            quantization_preapplied=quant is not None)


def make_causal_program_capture(lm: nn.Module, *, env_prefix: str, checkpoint: str,
                                full_checkpoint: bool, prefill_tokens: int = 128,
                                decode_tokens: int = 32) -> tuple[CausalMultiProgramCapture, tuple]:
    """Create a deferred, fully compiled prefill+decode capture plan."""
    capacity = int(prefill_tokens) + int(decode_tokens)
    tokens, provenance, corpus_ready = load_token_corpus(
        vocab_size=int(lm.config.vocab_size), total_tokens=capacity, env_prefix=env_prefix)
    if corpus_ready and not full_checkpoint:
        raise ValueError(f"{env_prefix}_PAPER_READY=1 requires the complete pretrained checkpoint")
    capture = CausalMultiProgramCapture(
        lm.eval(), tokens, checkpoint=checkpoint, full_checkpoint=full_checkpoint,
        paper_ready=bool(full_checkpoint and corpus_ready), provenance=provenance,
        prefill_tokens=prefill_tokens, decode_tokens=decode_tokens)
    return capture, ()


def session_spec(model: nn.Module, _inputs: tuple[torch.Tensor, ...]) -> dict | None:
    """Loader-compatible access to a decode wrapper's attached session contract."""
    return getattr(model, "session_spec", None)
