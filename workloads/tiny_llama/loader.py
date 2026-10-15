"""TinyLlama (1.1B) causal-LM example -> MLIR.

    m2m coverage examples/tiny_llama.py
    m2m convert  examples/tiny_llama.py --out /tmp/tinyllama.mlir
    m2m convert  examples/tiny_llama.py --quant int8_weight_only --out /tmp/tinyllama_int8.mlir

Env:
    M2M_LLAMA_LAYERS=N   random N-layer architecture (fast smoke; default: full)
    M2M_LLAMA_CHECKPOINT_LAYERS=N
                         keep the first N layers of the loaded checkpoint
    M2M_SEQ=N            sequence length for the example input (default 8)
    M2M_LLAMA_ATTENTION=eager
                         expose attention matmuls for MX graph quantization
    M2M_LLAMA_SESSION=decode|e2e
                         ``decode`` exports the diagnostic recurrent step; ``e2e`` emits separate
                         compiled prefill+decode programs for the primary continuous session
    M2M_LLAMA_TOKEN_IDS=/path/to/input_ids.npy
                         tokenized prefill+decode corpus used by the decode session;
                         for a single forward, use exactly the 1-D token IDs
                         in the file instead of randomly generated inputs
    M2M_LLAMA_TOKEN_SOURCE=...
    M2M_LLAMA_PAPER_READY=1
                         opt in only with the full checkpoint and an attributed external corpus

Weights are loaded from the local HF cache (TinyLlama/TinyLlama-1.1B-Chat-v1.0).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import torch
from torch import nn

_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class _LogitsOnly(nn.Module):
    """Wrap a HF causal LM so export sees a clean tensor->tensor forward."""

    def __init__(self, lm: nn.Module, *, causal_mask_tokens: int | None = None) -> None:
        super().__init__()
        self.lm = lm
        if causal_mask_tokens is not None:
            # A prepared 4-D mask has the same causal semantics as the HF
            # generated mask, while keeping the captured forward free of
            # tracing-only packed-sequence mask operations.
            dtype = next(lm.parameters()).dtype
            mask = torch.full((1, 1, causal_mask_tokens, causal_mask_tokens),
                              torch.finfo(dtype).min, dtype=dtype).triu(1)
            self.register_buffer("causal_mask", mask, persistent=False)
        else:
            self.causal_mask = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.lm(input_ids=input_ids, attention_mask=self.causal_mask,
                      use_cache=False)
        return out.logits


def _single_forward_ids(path: str, *, seq: int, vocab: int) -> torch.Tensor:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("single-forward token IDs must be a regular .npy file")
    values = np.load(source, allow_pickle=False)
    if (values.ndim != 1 or values.shape != (seq,)
            or values.dtype not in (np.dtype("int32"), np.dtype("int64"))
            or np.any(values < 0) or np.any(values >= vocab)):
        raise ValueError("single-forward token IDs must be valid 1-D IDs of length M2M_SEQ")
    return torch.from_numpy(values.astype(np.int64, copy=True))[None]


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    from transformers import AutoConfig, AutoModelForCausalLM

    n_layers = os.environ.get("M2M_LLAMA_LAYERS")
    checkpoint_layers = os.environ.get("M2M_LLAMA_CHECKPOINT_LAYERS")
    seq = int(os.environ.get("M2M_SEQ", "8"))
    attention = os.environ.get("M2M_LLAMA_ATTENTION")
    if n_layers and checkpoint_layers:
        raise ValueError("choose random smoke layers or checkpoint layers, not both")

    if n_layers:
        # Fast smoke path: real Llama architecture, fewer layers, random init.
        cfg = AutoConfig.from_pretrained(_MODEL_ID)
        cfg.num_hidden_layers = int(n_layers)
        cfg.use_cache = False
        cfg.tie_word_embeddings = False  # avoid tied-weight swap during quantize_
        if attention:
            cfg._attn_implementation = attention
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID, dtype=torch.float32, use_cache=False, tie_word_embeddings=False,
            **({"attn_implementation": attention} if attention else {}),
        )
        if checkpoint_layers:
            count = int(checkpoint_layers)
            layers = model.model.layers
            if not 1 <= count <= len(layers):
                raise ValueError(f"checkpoint layer count must be within 1..{len(layers)}")
            model.model.layers = nn.ModuleList(list(layers[:count]))
            model.config.num_hidden_layers = count
            model.model.config.num_hidden_layers = count

    # Break any residual weight tying so torchao quantize_ can swap weights.
    lm_head = getattr(model, "lm_head", None)
    if isinstance(lm_head, nn.Linear):
        lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())

    if os.environ.get("M2M_LLAMA_SESSION") in {"decode", "e2e"}:
        from m2m.capture.causal_session import make_causal_program_capture, make_decode_session

        factory = (make_causal_program_capture
                   if os.environ.get("M2M_LLAMA_SESSION") == "e2e" else make_decode_session)
        return factory(
            model.eval(), env_prefix="M2M_LLAMA",
            checkpoint=_MODEL_ID, full_checkpoint=not bool(n_layers),
            prefill_tokens=int(os.environ.get("M2M_PREFILL_TOKENS", "128")),
            decode_tokens=int(os.environ.get("M2M_DECODE_TOKENS", "32")))

    token_path = os.environ.get("M2M_LLAMA_TOKEN_IDS")
    model = _LogitsOnly(model.eval(),
                        causal_mask_tokens=seq if token_path else None).eval()
    vocab = model.lm.config.vocab_size
    input_ids = (_single_forward_ids(token_path, seq=seq, vocab=vocab)
                 if token_path else torch.randint(0, vocab, (1, seq), dtype=torch.long))
    return model, (input_ids,)


def get_session_spec(model: nn.Module, inputs: tuple[torch.Tensor, ...]) -> dict | None:
    from m2m.capture.causal_session import session_spec

    return session_spec(model, inputs)
