"""BitVLA (1-bit VLA, OFT variant) capture loader for m2m.

BitVLA is a Llava-style VLA: a W1.58-A8 (BitNet) language model + a (optionally
1-bit) SigLIP vision tower, fine-tuned with OpenVLA-OFT to predict a chunk of
continuous actions in a single bi-directional forward pass (no autoregressive
generation loop).

`BitVLAForActionPrediction.predict_action` is NOT a capture target:
  - heavy host-side prep with `.item()` (`(input_ids == idx).sum().item()`),
    `masked_scatter`, numpy unnormalization, and a proprio/action-head plumbing,
  - requires the vendored `transformers` fork + the `prismatic` package +
    real checkpoints + an image processor.

Capture unit: the inner VLM forward on already-built `inputs_embeds`, i.e.
`LlavaForConditionalGeneration.forward(inputs_embeds=..., use_bi_attn=True)`
returning logits. This is exactly what `predict_action` calls after all the
host-side embedding assembly; it carries the full BitNet LM + bi-directional
attention and the W1.58-A8 BitLinear math (round/clamp/absmean quant, traced
through each autograd.Function's forward). Vision-tower image embedding and the
masked_scatter of image/proprio tokens stay out of the graph (done host-side),
mirroring the smolVLA / tiny_llama conventions.

We instantiate a SMALL random Llava+BitNet config (no weights downloaded). The
vendored BitVLA `transformers` fork MUST be installed (it hardcodes
`BitNetForCausalLM` + `SiglipVisionModel` and the `use_bi_attn` kwarg).

Env:
    BITVLA_LLM_LAYERS=N   BitNet decoder layers (default 2 fast smoke; real: 30)
    BITVLA_SEQ=N          prefix sequence length of the embeds (default 32)
"""

from __future__ import annotations

import os
import sys
import types
from enum import Enum

import torch
from torch import nn

# Make the OFT bitvla *modeling* code importable. The modeling file uses flat
# imports (`from configuration_bit_vla import ...`, `from bitvla_for_action_prediction
# import ...`), so both the package dir and its `model/` subdir go on sys.path.
_BITVLA_PKG = "/scratch/agustin/projects/BitVLA/openvla-oft/bitvla"
_BITVLA_MODEL = "/scratch/agustin/projects/BitVLA/openvla-oft/bitvla/model"
for _p in (_BITVLA_PKG, _BITVLA_MODEL):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _install_prismatic_stubs() -> None:
    """Pre-register lean stub `prismatic.*` modules in sys.modules.

    `bitvla_for_action_prediction` imports `prismatic.vla.constants` and
    `prismatic.training.train_utils` at MODULE LOAD time. Importing the real
    `prismatic` package runs `prismatic/__init__.py -> from .models import load`,
    which pulls heavy deps (draccus, etc.). None of those symbols are exercised by
    the captured inner-VLM forward, so we stub exactly what the module needs and
    block the real package's __init__ from ever running.
    """
    if "prismatic" in sys.modules:
        return

    prismatic = types.ModuleType("prismatic")
    prismatic.__path__ = []  # mark as package so submodule imports resolve here
    vla = types.ModuleType("prismatic.vla")
    vla.__path__ = []
    training = types.ModuleType("prismatic.training")
    training.__path__ = []

    constants = types.ModuleType("prismatic.vla.constants")

    class NormalizationType(str, Enum):
        NORMAL = "normal"
        BOUNDS = "bounds"
        BOUNDS_Q99 = "bounds_q99"

    constants.NormalizationType = NormalizationType
    constants.ACTION_DIM = 7
    constants.NUM_ACTIONS_CHUNK = 8
    constants.ACTION_PROPRIO_NORMALIZATION_TYPE = NormalizationType.BOUNDS_Q99
    constants.IGNORE_INDEX = -100
    constants.ACTION_TOKEN_BEGIN_IDX = 31743
    constants.STOP_INDEX = 2

    train_utils = types.ModuleType("prismatic.training.train_utils")

    def get_current_action_mask(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("host-side path; not part of the captured forward")

    def get_next_actions_mask(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("host-side path; not part of the captured forward")

    train_utils.get_current_action_mask = get_current_action_mask
    train_utils.get_next_actions_mask = get_next_actions_mask

    sys.modules["prismatic"] = prismatic
    sys.modules["prismatic.vla"] = vla
    sys.modules["prismatic.vla.constants"] = constants
    sys.modules["prismatic.training"] = training
    sys.modules["prismatic.training.train_utils"] = train_utils


_install_prismatic_stubs()


class _VLMLogits(nn.Module):
    """Run the inner bi-directional Llava VLM forward on inputs_embeds -> logits."""

    def __init__(self, vla: nn.Module) -> None:
        super().__init__()
        self.vla = vla
        # Resolve the parent Llava forward ONCE, at construction time, outside the
        # dynamo trace. `BitVLAForActionPrediction.forward` is the (non-exportable)
        # host-side action path; the capture unit is the inner VLM forward, which is
        # exactly `LlavaForConditionalGeneration.forward`. Importing it inside
        # `forward` trips dynamo on the transformers `_LazyModule`.
        from transformers import LlavaForConditionalGeneration

        self._vlm_forward = LlavaForConditionalGeneration.forward

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self._vlm_forward(
            self.vla,
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=None,
            pixel_values=None,
            labels=None,
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            use_bi_attn=True,  # OFT uses bi-directional attention over action tokens
        )
        return out.logits


def get_model_and_inputs():
    from configuration_bit_vla import Bitvla_Config

    llm_layers = int(os.environ.get("BITVLA_LLM_LAYERS", "2"))
    seq = int(os.environ.get("BITVLA_SEQ", "32"))

    # Small random BitNet text config (W1.58 LM) + small SigLIP vision config.
    # The vendored fork's LlavaForConditionalGeneration hardcodes BitNetForCausalLM
    # and SiglipVisionModel, so only the config dicts matter here.
    text_config = {
        "model_type": "BitNet",
        "vocab_size": 1024,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": llm_layers,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "max_position_embeddings": 2048,
        # Pin semantic configuration rather than inheriting a Transformers-version default.
        # BitVLA's vendored 4.51 BitNetConfig defaults to SiLU, while upstream 5.0 defaults to
        # ReLU-squared; omitting this made the same captured weights instantiate different math in
        # downstream compiler environments.
        "hidden_act": "silu",
        "tie_word_embeddings": False,
    }
    vision_config = {
        "model_type": "siglip_vision_model",
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_channels": 3,
        "image_size": 224,
        "patch_size": 16,
    }

    cfg = Bitvla_Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_index=10,
        vision_feature_layer=-1,
        vision_feature_select_strategy="full",
        norm_stats={},
        n_action_bins=256,
        # The vendored BitVLAForActionPrediction.__init__ reads `config.vocab_size`
        # directly (it normally comes from a real checkpoint config); LlavaConfig
        # does not surface it at top level, so set it to match the text config.
        vocab_size=text_config["vocab_size"],
    )

    from bitvla_for_action_prediction import BitVLAForActionPrediction

    vla = BitVLAForActionPrediction(cfg).to(torch.float32).eval()
    # P21-S4 native low-bit datapath: with BITVLA_NATIVE_QUANT=1, materialize the
    # packed-int2 ternary weights (BitLinear.quantize_weights()) so the captured
    # graph takes the NATIVE path -- packed int2 storage + dequantize_from_int2
    # (bit-unpack + per-tensor absmean scale) + matmul -- instead of the f32
    # fake-quant (absmean round/clamp) branch. Makes the native W1.58 ternary
    # storage + scale visible in the IR (the capture-gap the audit flagged).
    if os.environ.get("BITVLA_NATIVE_QUANT") == "1":
        import sys as _sys
        n_q = 0
        for mod in vla.modules():
            if type(mod).__name__ != "BitLinear" or getattr(mod, "weight", None) is None:
                continue
            # Replicate BitLinear.quantize_weights() WITHOUT its buggy bf16-vs-f32
            # self-check assert (which crashes on an f32 model). This materializes the
            # packed-int2 ternary weight + per-tensor absmean scale and flips the module
            # to the native dequantize_from_int2 (bit-unpack + scale) + matmul forward.
            q2 = _sys.modules[type(mod).__module__].quantize_to_int2
            packed, step, orig_shape, n_elems = q2(mod.weight.data)
            mod.register_buffer("q_weight", packed)
            mod.register_buffer("w_step", torch.tensor(step, dtype=torch.float32))
            mod.orig_shape = orig_shape
            mod.n_elems = n_elems
            mod.weight = None
            mod.enable_qlora = True
            n_q += 1
        print(f"[bitvla] native-quant: materialized packed-int2 ternary for {n_q} BitLinear")
    # Constants only matter for the host-side path we skip; set benign values.
    vla.set_constant(
        image_token_idx=10,
        proprio_pad_idx=11,
        ignore_idx=-100,
        action_token_begin_idx=12,
        stop_index=2,
    )

    hidden = text_config["hidden_size"]
    inputs_embeds = torch.randn(1, seq, hidden, dtype=torch.float32)
    attention_mask = torch.ones(1, seq, dtype=torch.long)
    return _VLMLogits(vla).eval(), (inputs_embeds, attention_mask)
