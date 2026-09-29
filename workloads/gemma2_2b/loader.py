"""Gemma 2 2B (instruction-tuned) causal LM -> MLIR.

    m2m coverage workloads/gemma2_2b/loader.py
    m2m convert  workloads/gemma2_2b/loader.py --quant int8_weight_only --out /tmp/gemma2_2b_int8.mlir

Env:
    M2M_GEMMA_LAYERS=N   RANDOM-INIT model truncated to N decoder layers (fast smoke; unset = full 26)
    M2M_GEMMA_SLICE_LAYERS=N
                         the REAL pretrained model, keeping its FIRST N decoder layers -- a section of
                         Gemma 2 2B rather than a differently-shaped stand-in. Every shape a whole-model
                         capture has (hidden 2304, ffn 9216, vocab 256000, head_dim 256) is preserved,
                         because none of them depends on the layer count; only the repetition is cut.
                         Use it when the whole model does not fit the memory you have.
    M2M_GEMMA_ENTRY=ids|embeds
                         where the exported forward starts. ``ids`` (default) is the whole pipeline,
                         token ids -> logits. ``embeds`` enters one op later, at the (already
                         embed-scaled) hidden state, and replaces the embedding table with a one-row
                         placeholder so it is NOT exported. That table is 256000x2304 fp32 = 2.25 GiB --
                         torchao's int8_weight_only quantizes nn.Linear only, so it stays fp32 and is
                         over half the weight bytes of an int8 Gemma 2 2B capture while contributing no
                         matmul. The example input is the REAL embedding rows for the drawn token ids,
                         so the section's boundary tensor is the tensor the whole model would have fed it.
    M2M_SEQ=N            sequence length for the example input (default 8)
    M2M_GEMMA_SESSION=decode|e2e
                         ``decode`` exports the diagnostic recurrent step; ``e2e`` emits separate
                         compiled prefill+decode programs for the primary continuous session
    M2M_GEMMA_TOKEN_IDS=/path/to/input_ids.npy
                         tokenized prefill+decode corpus used by the decode session
    M2M_GEMMA_TOKEN_SOURCE=...
    M2M_GEMMA_PAPER_READY=1
                         opt in only with the full checkpoint and an attributed external corpus

Weights come from the local HF cache (google/gemma-2-2b-it); HF_HOME is set by capture.toml.

WHAT THIS ARCHITECTURE ADDS over the Llama-family loaders beside it, since each item is a place the
lowering can diverge rather than fail:

  * **Logit soft-capping.** Gemma 2 caps attention logits and final logits with ``tanh``, so the export
    contains a tanh on the score path that no Llama capture has. It is left ON: capturing with it disabled
    would export a model that is not the one being served.
  * **Alternating attention.** Even layers use sliding-window attention, odd layers global — so a
    2-layer smoke capture covers BOTH variants, which is why 2 (not 1) is the smoke default.
  * **Grouped-query attention** with head_dim 256 while hidden_size is 2304: head_dim is NOT
    hidden/heads here, and a contraction shape derived from that assumption would be wrong.
  * **Tied embeddings.** The LM head shares the embedding matrix, and torchao's ``quantize_`` swaps
    weights in place, so the tie is broken explicitly below — otherwise quantizing the head also
    quantizes the embedding lookup through the same storage.
  * **GeGLU** feed-forward (gate + up + tanh-approximated gelu), not SwiGLU.
"""

from __future__ import annotations

import os

import torch
from torch import nn

_MODEL_ID = "google/gemma-2-2b-it"


class _LogitsOnly(nn.Module):
    """Wrap the HF causal LM so export sees a clean tensor->tensor forward.

    ``entry`` selects which tensor that forward takes: token ids (the whole pipeline) or the hidden
    state the embedding lookup would have produced (a section that starts one op later).
    """

    def __init__(self, lm: nn.Module, entry: str = "ids") -> None:
        super().__init__()
        self.lm = lm
        self.entry = entry

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.entry == "embeds":
            return self.lm(inputs_embeds=x, use_cache=False).logits
        return self.lm(input_ids=x, use_cache=False).logits


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    from transformers import AutoConfig, AutoModelForCausalLM

    n_layers = os.environ.get("M2M_GEMMA_LAYERS")
    n_slice = os.environ.get("M2M_GEMMA_SLICE_LAYERS")
    entry = os.environ.get("M2M_GEMMA_ENTRY", "ids")
    seq = int(os.environ.get("M2M_SEQ", "8"))
    if entry not in ("ids", "embeds"):
        raise RuntimeError(f"M2M_GEMMA_ENTRY must be ids or embeds, not {entry!r}")
    if n_layers and n_slice:
        raise RuntimeError("set M2M_GEMMA_LAYERS (random init) or M2M_GEMMA_SLICE_LAYERS (pretrained "
                           "section), not both")
    if os.environ.get("M2M_GEMMA_SESSION") in {"decode", "e2e"} and entry != "ids":
        raise RuntimeError("M2M_GEMMA_SESSION=decode|e2e requires M2M_GEMMA_ENTRY=ids")

    if n_layers:
        # Smoke path: the real Gemma 2 architecture at fewer layers, randomly initialized. Keeps both
        # attention variants as long as N >= 2 (even = sliding window, odd = global).
        cfg = AutoConfig.from_pretrained(_MODEL_ID)
        cfg.num_hidden_layers = int(n_layers)
        cfg.use_cache = False
        cfg.tie_word_embeddings = False       # so quantize_ can swap the head without touching embeddings
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    else:
        # Load with the tie INTACT. Passing tie_word_embeddings=False here makes transformers look for an
        # `lm_head.weight` that a tied checkpoint does not contain, and it reports
        # `lm_head.weight | MISSING` and RANDOMLY INITIALIZES it — a capture of Gemma with a random output
        # head, whose golden is self-consistent and whose logits mean nothing.
        model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID, dtype=torch.float32, use_cache=False)

    # Untie AFTER loading: copy the embedding matrix into its own storage, so torchao's quantize_ can swap
    # the head's weight without also rewriting the embedding lookup, and the head keeps the real values.
    head = getattr(model, "lm_head", None)
    embed = model.get_input_embeddings()
    if isinstance(head, nn.Linear):
        src = head.weight if head.weight is not None else getattr(embed, "weight", None)
        if src is None:
            raise RuntimeError("no lm_head or embedding weight to untie from")
        head.weight = nn.Parameter(src.detach().clone())
        model.config.tie_word_embeddings = False

    # A SECTION of the real model: keep its first N decoder layers and nothing else changes. Gemma 2
    # alternates attention by layer index (even sliding-window, odd global) and picks the variant from
    # config.layer_types[i], and Gemma2Model.forward iterates layers[: config.num_hidden_layers] -- so
    # the config must be truncated alongside the ModuleList or the forward walks off the end / asks for
    # a layer type that is no longer there. An even N keeps the two variants balanced.
    if n_slice:
        keep = int(n_slice)
        inner = model.model
        if keep < 1 or keep > len(inner.layers):
            raise RuntimeError(f"M2M_GEMMA_SLICE_LAYERS={keep} outside 1..{len(inner.layers)}")
        inner.layers = nn.ModuleList(list(inner.layers)[:keep])
        model.config.num_hidden_layers = keep
        model.config.layer_types = list(model.config.layer_types)[:keep]

    vocab = model.config.vocab_size
    if os.environ.get("M2M_GEMMA_SESSION") in {"decode", "e2e"}:
        from m2m.capture.causal_session import make_causal_program_capture, make_decode_session

        factory = (make_causal_program_capture
                   if os.environ.get("M2M_GEMMA_SESSION") == "e2e" else make_decode_session)
        return factory(
            model.eval(), env_prefix="M2M_GEMMA",
            checkpoint=_MODEL_ID, full_checkpoint=not bool(n_layers or n_slice),
            prefill_tokens=int(os.environ.get("M2M_PREFILL_TOKENS", "128")),
            decode_tokens=int(os.environ.get("M2M_DECODE_TOKENS", "32")))

    input_ids = torch.randint(0, vocab, (1, seq), dtype=torch.long)

    if entry == "embeds":
        # The boundary tensor, taken from the REAL table (so it carries the embed_scale that
        # Gemma2TextScaledWordEmbedding applies inside the lookup -- Gemma2Model.forward does NOT
        # rescale an inputs_embeds it is handed). Then the table itself is replaced by a one-row
        # placeholder: nothing reads it once the lookup is out of the graph, and leaving the real one
        # attached would export 2.25 GiB of fp32 weights that the section never touches.
        embed = model.get_input_embeddings()
        with torch.no_grad():
            example = embed(input_ids).detach().clone()
        model.model.embed_tokens = nn.Embedding(1, model.config.hidden_size)
        model = _LogitsOnly(model.eval(), entry="embeds").eval()
        return model, (example,)

    model = _LogitsOnly(model.eval()).eval()
    return model, (input_ids,)


def get_session_spec(model: nn.Module, inputs: tuple[torch.Tensor, ...]) -> dict | None:
    from m2m.capture.causal_session import session_spec

    return session_spec(model, inputs)
