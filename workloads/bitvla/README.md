# BitVLA (1-bit VLA, OpenVLA-OFT variant)

BitVLA = W1.58-A8 (BitNet) language model + SigLIP-L vision tower, fine-tuned
with OpenVLA-OFT for LIBERO. ~3B params. Repo: https://github.com/ustcwhy/BitVLA
(uses a VENDORED `transformers` fork that hardcodes `BitNetForCausalLM` +
`SiglipVisionModel` and adds W1.58-A8 `BitLinear` + a `use_bi_attn` kwarg).

Capture unit: the inner bi-directional VLM forward on already-built
`inputs_embeds` -> logits (the call `predict_action` makes after host-side
embedding assembly). Carries the full BitNet LM + BitLinear quant math. Vision
encoding, `masked_scatter` of image/proprio tokens, and numpy action
unnormalization stay host-side (out of the graph).

Source model class: `BitVLAForActionPrediction(LlavaForConditionalGeneration)`
in `openvla-oft/bitvla/model/bitvla_for_action_prediction.py`.

The loader builds a SMALL random Llava+BitNet config — no weights downloaded.

## PyTorch-exportable?
**Maybe.** The captured inner forward is torch.export-friendly in principle
(BitLinear is pure `round/clamp/abs`-mean quant via `autograd.Function`, traced
through each `.forward`; bi-directional attention is a static mask). Risks:
- depends on the vendored `transformers` fork being installed (eager attn path).
- The full `predict_action` is NOT exportable (`.item()`, masked_scatter,
  numpy, image processor) — kept host-side by design.

## Venv (dedicated)
The `fx_importer` backend uses m2m's OWN FXImporter (no torch-mlir wheel needed).
m2m requires `enum.StrEnum` -> Python **3.11+**. torchao 0.17 needs `torch>=2.8`
(`torch.utils._pytree.register_constant`) and `torch.int1` (`torch>=2.6`), so the
whole stack is pinned to torch 2.8 / torchvision 0.23.
```bash
cd /path/to/bitvla_capture
uv venv --python 3.11
# VENDORED transformers fork (hardcodes BitNet + Siglip + use_bi_attn) + OFT deps:
uv pip install -e /path/to/BitVLA/transformers
uv pip install "torch==2.8.0" "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/cpu
uv pip install numpy timm accelerate
# m2m + deps (NOTE: torchao 0.17 / torch 2.8):
uv pip install -e /path/to/model2MLIR --no-deps
uv pip install xdsl structlog ml_dtypes "torchao==0.17.0"
```
Note: `bitvla_for_action_prediction` imports `prismatic.vla.constants` and
`prismatic.training.train_utils` at MODULE LOAD time. Importing the real
`prismatic` package runs `prismatic/__init__.py -> from .models import ... load`,
which pulls heavy deps (draccus, etc.) that fail. The loader's
`_install_prismatic_stubs()` pre-registers lean stub modules in `sys.modules`
(the constants + two mask fns) so the real package __init__ never runs — none of
those symbols are exercised by the captured inner-VLM forward.

`BitNet` registration: the vendored fork hardcodes `BitNetForCausalLM` +
`SiglipVisionModel` inside `transformers/models/llava/` (no AutoModel.register
needed). The loader just builds a `Bitvla_Config(text_config={"model_type":
"BitNet", ...})`; the fork's `LlavaConfig.__init__` routes that to its bundled
`BitNetConfig`. The loader also sets a top-level `vocab_size` on the config
(`BitVLAForActionPrediction.__init__` reads `config.vocab_size`, which LlavaConfig
does not surface) and resolves the parent `LlavaForConditionalGeneration.forward`
ONCE at construction time (importing it inside `forward` trips dynamo on the
transformers `_LazyModule`).

## Run
```bash
cd /path/to/bitvla_capture
M2M_BITVLA_ROOT=/path/to/BitVLA BITVLA_LLM_LAYERS=2 uv run --no-sync python -c "
import m2m, sys; sys.path.insert(0,'/path/to/model2MLIR/workloads/bitvla')
from loader import get_model_and_inputs
m,i=get_model_and_inputs(); r=m2m.convert(m,i)
open('/path/to/model2MLIR/workloads/bitvla/bitvla.mlir','w').write(r.mlir_text)
print(r.path_taken, r.mlir_text.count('linalg.'))"
```

## Status
CAPTURED (all 3 formats, BITVLA_LLM_LAYERS=2, seq=32, hidden=256, vocab=1024).
Input: inputs_embeds [1,32,256] f32 + attention_mask [1,32] i64 -> logits [1,32,1024].

- **fp32** `bitvla.mlir`: ok, path=fx_importer, linalg=812, total_opaque=16.
- **int8** `bitvla_int8.mlir`: ok, path=fx_importer, linalg=816, total_opaque=16.
- **fp8** `bitvla_fp8.mlir`: ok, path=fx_importer, linalg=815, total_opaque=16.

Opaque histogram (identical across all 3): `aten_mean_default`x4,
`aten_mean_default_1`x4, `aten_mean_default_2`x4, `aten_mean_default_3`x2,
`aten_slice_Tensor`x2. The `aten.mean`s are BitLinear's W1.58 absmean weight-quant
reductions (`weight.abs().mean()`); the importer leaves the reduction opaque.

BitNet x torchao caveat: the task's full-model `int8_weight_only` /
`float8_weight_only_e4m3` schemes FAIL on this model — every `BitLinear.forward`
runs `WeightQuant.apply(self.weight)` -> `weight.abs().mean()` in-graph, and
torchao's weight subclasses do not implement `aten.abs`
(`AffineQuantizedTensor`/`Float8Tensor` dispatch error). BitNet already does its
own in-graph W1.58 quant, so stacking torchao weight-only on the BitLinears is
both impossible and redundant. The int8/fp8 .mlir here therefore quantize ONLY the
one plain `nn.Linear` in the captured path — `lm_head` — via
`QuantizationConfig(scheme="none", per_module={"lm_head": <scheme>})`; the int8
file carries `i8` lm_head weights, the fp8 file carries `f8` lm_head weights, and
the BitLinear W1.58 math stays intact.
