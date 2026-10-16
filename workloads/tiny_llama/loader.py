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
    M2M_LLAMA_CALIBRATION_TOKEN_IDS=/path/to/calibration.npz
                         ``input_ids`` [segments, prefill+decode] tokenized calibration segments; with
                         ``e2e`` each recipe-quantized stage is calibrated on them (prefill: every
                         segment's prefix; decode: the FP32 teacher-forced recurrence over its
                         continuation). Unset: stages calibrate on their capture example, as before.
    M2M_LLAMA_CALIBRATION_DECODE_STRIDE=N
                         sample every N-th decode step of each calibration segment (default 1)
    M2M_LLAMA_SPLIT_RATIO=r
                         FP-equivalent outlier channel splitting (m2m.capture.outlier_split) of every
                         gated MLP: a product channel whose calibrated absmax exceeds r x the median
                         channel is split into ceil(absmax / (r x median)) parts
    M2M_LLAMA_SPLIT_WIDTH_BUCKETS=n [M2M_LLAMA_SPLIT_WIDTH_ALIGN=a]
                         after the split (and any smoothing), pad every gated MLP's intermediate width
                         with zero channels up to one of n shared widths (multiples of a), chosen to
                         minimise the padding: exact, and n distinct MLP shapes instead of one per layer
    M2M_LLAMA_SMOOTH_ALPHA=a [M2M_LLAMA_SMOOTH_GROUPS=qkv,gate_up,down,o] [M2M_LLAMA_SMOOTH_SEGMENTS=32]
                         FP-equivalent SmoothQuant rescaling (m2m.capture.smoothing) on statistics of
                         the calibration prefixes, before quantization; needs the calibration corpus

Weights are loaded from the local HF cache (TinyLlama/TinyLlama-1.1B-Chat-v1.0).
"""

from __future__ import annotations

import json
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

    smoothing = _smooth(model.eval())

    if os.environ.get("M2M_LLAMA_SESSION") in {"decode", "e2e"}:
        from m2m.capture.causal_session import make_causal_program_capture, make_decode_session

        factory = (make_causal_program_capture
                   if os.environ.get("M2M_LLAMA_SESSION") == "e2e" else make_decode_session)
        result = factory(
            model.eval(), env_prefix="M2M_LLAMA",
            checkpoint=_MODEL_ID, full_checkpoint=not bool(n_layers),
            prefill_tokens=int(os.environ.get("M2M_PREFILL_TOKENS", "128")),
            decode_tokens=int(os.environ.get("M2M_DECODE_TOKENS", "32")))
        calibration = _calibration_tokens(model)
        if hasattr(result[0], "provenance"):
            # Recorded in the session contract beside the capture's own token provenance.
            if calibration is not None:
                result[0].provenance.update(calibration[1])
            if smoothing is not None:
                # A JSON string, so it survives the capture's scalar-only provenance record.
                result[0].provenance["range_transforms"] = json.dumps(smoothing, sort_keys=True)
        return result

    token_path = os.environ.get("M2M_LLAMA_TOKEN_IDS")
    model = _LogitsOnly(model.eval(),
                        causal_mask_tokens=seq if token_path else None).eval()
    vocab = model.lm.config.vocab_size
    input_ids = (_single_forward_ids(token_path, seq=seq, vocab=vocab)
                 if token_path else torch.randint(0, vocab, (1, seq), dtype=torch.long))
    return model, (input_ids,)


def _smooth(lm: nn.Module) -> dict | None:
    """FP-equivalent range transforms before any quantization.

    ``M2M_LLAMA_SPLIT_RATIO``: split every gated-MLP product channel whose calibrated absmax exceeds
    ratio x the median channel into equal parts (``m2m.capture.outlier_split``).
    ``M2M_LLAMA_SMOOTH_ALPHA``: SmoothQuant rescaling of ``M2M_LLAMA_SMOOTH_GROUPS`` (default all:
    qkv, gate_up, down, o). Both take their statistics from the first ``M2M_LLAMA_SMOOTH_SEGMENTS``
    (default 32) calibration prefixes, so the calibration corpus is required. Unset: untouched.
    """
    ratio = os.environ.get("M2M_LLAMA_SPLIT_RATIO", "")
    alpha = os.environ.get("M2M_LLAMA_SMOOTH_ALPHA", "")
    if not ratio and not alpha:
        return None
    calibration = _calibration_tokens(lm)
    if calibration is None:
        raise ValueError("FP-equivalent range transforms need M2M_LLAMA_CALIBRATION_TOKEN_IDS")
    from m2m.capture.outlier_split import split_outlier_channels
    from m2m.capture.smoothing import GROUPS, smooth_decoder

    segments = int(os.environ.get("M2M_LLAMA_SMOOTH_SEGMENTS", "32"))
    prefill = int(os.environ.get("M2M_PREFILL_TOKENS", "128"))
    statistics = calibration[0][:segments, :prefill]
    record = {"statistics_segments": int(statistics.shape[0]), "statistics_tokens": int(statistics.shape[1])}
    if ratio:
        def observe():
            for row in statistics:
                lm(input_ids=row.reshape(1, -1), use_cache=False)

        record["split"] = split_outlier_channels(lm, observe, ratio=float(ratio))
    if alpha:
        groups = tuple(g for g in os.environ.get("M2M_LLAMA_SMOOTH_GROUPS", ",".join(GROUPS)).split(",") if g)
        record["smooth"] = smooth_decoder(lm, statistics, alpha=float(alpha), groups=groups)
    buckets = os.environ.get("M2M_LLAMA_SPLIT_WIDTH_BUCKETS", "")
    if buckets:
        from m2m.capture.outlier_split import pad_gated_widths

        align = int(os.environ.get("M2M_LLAMA_SPLIT_WIDTH_ALIGN", "1"))
        record["pad"] = pad_gated_widths(lm, buckets=int(buckets), align=align)
    return record


def _calibration_tokens(lm: nn.Module):
    """The declared calibration corpus and its provenance, or None when none is declared."""
    path = os.environ.get("M2M_LLAMA_CALIBRATION_TOKEN_IDS", "")
    if not path:
        return None
    from m2m.capture.causal_session import load_calibration_tokens

    prefill = int(os.environ.get("M2M_PREFILL_TOKENS", "128"))
    decode = int(os.environ.get("M2M_DECODE_TOKENS", "32"))
    stride = int(os.environ.get("M2M_LLAMA_CALIBRATION_DECODE_STRIDE", "1"))
    tokens, provenance = load_calibration_tokens(
        path, vocab_size=int(lm.config.vocab_size), min_tokens=prefill + decode)
    return tokens, {**provenance, "calibration_decode_stride": stride}


def get_calibration_inputs(model: nn.Module, inputs: tuple[torch.Tensor, ...]):
    """Calibration samples for one recipe-quantized session stage, or None (use its example).

    The capture worker calls this once per quantized stage, with that stage's module.
    """
    if not os.environ.get("M2M_LLAMA_CALIBRATION_TOKEN_IDS", ""):
        return None
    from m2m.capture.causal_session import causal_calibration_inputs

    tokens, _provenance = _calibration_tokens(model.lm)
    return causal_calibration_inputs(
        model, tokens, prefill_tokens=int(os.environ.get("M2M_PREFILL_TOKENS", "128")),
        decode_stride=int(os.environ.get("M2M_LLAMA_CALIBRATION_DECODE_STRIDE", "1")))


def get_session_spec(model: nn.Module, inputs: tuple[torch.Tensor, ...]) -> dict | None:
    from m2m.capture.causal_session import session_spec

    return session_spec(model, inputs)
