"""DeepSeek-R1-Distill-Qwen-1.5B causal-LM -> MLIR.

A Qwen2 decoder, so it differs from the Llama-family workloads in ways the op inventory sees: QKV
projections carry a BIAS (Llama's do not), and the GQA ratio is 12 query heads to 2 KV heads.

Env:
    M2M_QWEN_LAYERS=N   truncate to N decoder layers. NOTE this also switches to from_config, i.e.
                        RANDOM INIT -- it is a smoke path, not a smaller version of the real model.
                        Left unset (the default) the pretrained checkpoint is loaded at full depth.
    M2M_SEQ=N           sequence length for the example input (default 8)
    M2M_QWEN_ATTENTION=eager
                        expose QK/PV contractions in the captured graph
    M2M_QWEN_SESSION=decode|e2e
                        export a fixed-shape decode step or separate compiled prefill and decode
                        programs with explicit KV and position state
    M2M_QWEN_TOKEN_IDS=/path/to/input_ids.npy
                        optional tokenized prefill/decode corpus
"""

from __future__ import annotations

import os

import torch
from torch import nn

_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


class _LogitsOnly(nn.Module):
    """Wrap a HF causal LM so export sees a clean tensor->tensor forward."""

    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        self.lm = lm

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm(input_ids=input_ids, use_cache=False).logits


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    from transformers import AutoConfig, AutoModelForCausalLM

    n_layers = os.environ.get("M2M_QWEN_LAYERS")
    seq = int(os.environ.get("M2M_SEQ", "8"))
    attention = os.environ.get("M2M_QWEN_ATTENTION")

    if n_layers:
        # Smoke path: real Qwen2 architecture, fewer layers, RANDOM INIT (no checkpoint).
        cfg = AutoConfig.from_pretrained(_MODEL_ID)
        cfg.num_hidden_layers = int(n_layers)
        cfg.use_cache = False
        cfg.tie_word_embeddings = False   # avoid a tied-weight swap during quantize_
        if attention:
            cfg._attn_implementation = attention
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID, dtype=torch.float32, use_cache=False, tie_word_embeddings=False,
            **({"attn_implementation": attention} if attention else {}),
        )

    # Break any residual weight tying so torchao quantize_ can swap weights.
    lm_head = getattr(model, "lm_head", None)
    if isinstance(lm_head, nn.Linear):
        lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())

    if os.environ.get("M2M_QWEN_SESSION") in {"decode", "e2e"}:
        from m2m.capture.causal_session import make_causal_program_capture, make_decode_session

        factory = (make_causal_program_capture
                   if os.environ.get("M2M_QWEN_SESSION") == "e2e" else make_decode_session)
        return factory(
            model.eval(), env_prefix="M2M_QWEN",
            checkpoint=_MODEL_ID, full_checkpoint=not bool(n_layers),
            prefill_tokens=int(os.environ.get("M2M_PREFILL_TOKENS", "128")),
            decode_tokens=int(os.environ.get("M2M_DECODE_TOKENS", "32")))

    model = _LogitsOnly(model.eval()).eval()
    input_ids = torch.randint(0, model.lm.config.vocab_size, (1, seq), dtype=torch.long)
    return model, (input_ids,)


def get_session_spec(model: nn.Module, inputs: tuple[torch.Tensor, ...]) -> dict | None:
    from m2m.capture.causal_session import session_spec

    return session_spec(model, inputs)
