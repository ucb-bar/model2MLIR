# AllenAI MolmoAct (PyTorch)

Molmo-style VLA: SigLIP2 ViT vision backbone + adapter + Qwen2-style LLM decoder that
autoregressively emits reasoning / depth / action tokens, then decodes action bins.

## Which repo
The task pointed at https://github.com/allenai/molmoact2, but that repo is **only a
FastAPI inference server** (`molmoact2-policy-server`): it loads the checkpoint from the
HF Hub via `AutoModelForImageTextToText.from_pretrained("allenai/MolmoAct2-DROID",
trust_remote_code=True)` and a hub-side `modeling_molmoact2.py`. No model class lives in
that repo (only example clients/servers for DROID and bimanual YAM robots, plus lerobot/
YAM/EVA_DROID git submodules, none fetched). MolmoAct2 uses a flow-matching action head.

The full PyTorch model definition lives in https://github.com/allenai/molmoact (the `olmo`
training package), cloned at /path/to/molmoact. This loader builds against
that — the autoregressive MolmoAct (v1) lineage, same ViT+LLM backbone.

Model classes (in `olmo/hf_model/molmoact/modeling_molmoact.py`):
- `MolmoActForActionReasoning(MolmoActConfig)` (PreTrainedModel, GenerationMixin) — full VLA;
  action output goes through `.generate(...)` (a loop, NOT capturable as one graph).
- `MolmoActForCausalLM(MolmoActLlmConfig)` — the LLM decoder, plain `input_ids -> logits`
  causal LM. **This is the capture unit** (tiny_llama style).
- `MolmoActVisionBackbone` — SigLIP2 ViT (separate capturable unit; not wrapped here).
- Configs (`configuration_molmoact.py`): `MolmoActConfig` (vit+adapter+llm), `MolmoActLlmConfig`
  (hidden 3584, 48 layers, 28 heads / 4 kv heads, head_dim 128, vocab 152064, rope_theta 1e6).

Capture unit: `MolmoActForCausalLM.forward(input_ids) -> logits` — the representative heavy
transformer, avoiding the generation loop, ViT preprocessing, and image-token splicing
(all data-dependent / host-side).

## PyTorch-exportable?
Yes (maybe, in practice) for the LLM causal-LM forward: it is a standard transformers-style
decoder (`wte -> rotary -> decoder layers -> RMSNorm -> lm_head`), tensor->tensor, built from
a config with random init. Caveats for torch.export: uses HF `DynamicCache`/`_update_causal_mask`
and `transformers` SDPA — set `use_cache=False` and `_attn_implementation="eager"` (done in the
loader), same handling as tiny_llama / pi0.5. The full `MolmoActForActionReasoning` is NOT
directly exportable (autoregressive `.generate` loop + image splicing control flow).

## Venv (dedicated)
The `olmo` package targets recent transformers; the model file imports `GradientCheckpointingLayer`,
`Cache`/`DynamicCache`, `FlashAttentionKwargs`, `ROPE_INIT_FUNCTIONS` from transformers, so a
modern transformers (>=4.51) is needed. The repo's own pyproject is heavy (full training stack);
for capture, only `olmo/hf_model/molmoact/` + transformers + torch are required. The loader puts
the repo on `sys.path` (no `pip install -e` of olmo needed) via `MOLMOACT_REPO`.

```bash
# molmoact already cloned (shallow, LFS skipped):
#   GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/allenai/molmoact /path/to/molmoact
uv venv --python 3.11 --directory /path/to/model2MLIR/workloads/molmoact
# minimal capture deps. NOTE: transformers MUST be 4.5x, NOT 5.x:
# transformers 5.x removed the "default" key from ROPE_INIT_FUNCTIONS, and the
# model file does ROPE_INIT_FUNCTIONS["default"] -> KeyError. 4.53.3 verified working.
uv pip install "torch==2.7.*" "transformers==4.53.*" einops numpy
# frontend (FXImporter mode) + quant:
uv pip install xdsl structlog ml_dtypes torchao
uv pip install -e /path/to/model2MLIR --no-deps
```

NOTE: invoke the workload venv explicitly (`./.venv/bin/python`); `uv run` from this
dir walks up and picks the parent `model2MLIR/.venv` instead.

## Run
```bash
MOLMOACT_REPO=/path/to/molmoact \
M2M_MOLMOACT_LAYERS=4 M2M_MOLMOACT_VOCAB=4096 \
uv run --no-sync python -c "
import m2m, sys; sys.path.insert(0,'/path/to/model2MLIR/workloads/molmoact')
from loader import get_model_and_inputs
m,i=get_model_and_inputs(); r=m2m.convert(m,i)
open('/path/to/model2MLIR/workloads/molmoact/molmoact.mlir','w').write(r.mlir_text)
print(r.path_taken, r.mlir_text.count('linalg.'))"
```

## Status
- CAPTURED (random-init LLM decoder; 4 layers, vocab 4096, seq 8; eager attention; no hub
  download). All 3 formats clean (ok=True, 0 opaque aten ops, fx_importer path):
  - fp32 -> molmoact.mlir       (470 linalg., ~214 KB)
  - int8 -> molmoact_int8.mlir  (554 linalg., ~276 KB, int8_weight_only)
  - fp8  -> molmoact_fp8.mlir   (521 linalg., ~226 KB, float8_weight_only_e4m3)
- transformers pinned to 4.53.3 (5.x breaks on ROPE_INIT_FUNCTIONS["default"], see Venv).
- The ViT backbone (`MolmoActVisionBackbone`) and the MolmoAct2 flow-matching head are
  separate future capture units.
