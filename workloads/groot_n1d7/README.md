# NVIDIA Isaac-GR00T N1.7 (PyTorch)

VLA policy: Cosmos-Reason2-2B (Qwen3-VL) VLM backbone + flow-matching DiT action head.
Source: https://github.com/NVIDIA/Isaac-GR00T (clone at /path/to/Isaac-GR00T).

Capture unit: one flow-matching denoise step of the **action head** (`AlternateVLDiT`
+ state/action encoders + action decoder), the analogue of pi0.5/smolVLA `denoise_step`.
It runs on already-embedded backbone features, so the Qwen3-VL backbone and the
dict-based collator preprocessing stay host-side, out of the captured graph.

Model classes:
- `gr00t.model.gr00t_n1d7.gr00t_n1d7.Gr00tN1d7` (PreTrainedModel) — full policy;
  `forward(inputs: dict)` runs collator + backbone + head (NOT capturable as-is).
- `gr00t.model.gr00t_n1d7.gr00t_n1d7.Gr00tN1d7ActionHead(config)` — the DiT head we wrap.
- Config: `gr00t.configs.model.gr00t_n1d7.Gr00tN1d7Config` (constructs with no args; random init).

Default dims (Gr00tN1d7Config): backbone_embedding_dim=2048, max_state_dim=132,
max_action_dim=132, action_horizon=40, hidden_size=1024, input_embedding_dim=1536,
DiT num_layers=16, num_attention_heads=32, num_inference_timesteps=4, use_alternate_vl_dit=True.

## PyTorch-exportable?
Yes for the action-head DiT (pure torch.nn.Module, tensor->tensor; constructed from a
config with random init, no checkpoint or backbone needed). The full `Gr00tN1d7` is NOT
directly exportable: `forward(dict)` runs a HF data collator and the trust_remote_code
Qwen3-VL backbone (data-dependent control flow + Python glue). The DiT head itself has no
data-dependent control flow once a single timestep is fixed (the denoise loop is unrolled
host-side, like pi0.5).

## Venv (dedicated)
GR00T pins `requires-python == 3.10.*` and (for the full backbone) flash-attn +
torch 2.7 + a newer transformers. The action-head capture only needs the modules under
`gr00t/model/modules/` + `gr00t/configs/`, which import `torch`, `transformers`,
`diffusers` (ModelMixin/ConfigMixin), and `dm-tree`.

```bash
# Isaac-GR00T already cloned (shallow, LFS skipped):
#   GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/NVIDIA/Isaac-GR00T /path/to/Isaac-GR00T
cd /path/to/Isaac-GR00T
uv venv --python 3.10
# Heavy/full install (flash-attn, backbone): `uv pip install -e .` — NOT needed for the head.
# Minimal capture deps for the action head only:
uv pip install "torch==2.7.*" "transformers>=4.51" diffusers dm-tree numpy
# Frontend (FXImporter mode):
uv pip install xdsl structlog ml_dtypes
uv pip install -e /path/to/model2MLIR --no-deps
```

## Run
```bash
cd /path/to/Isaac-GR00T
M2M_GROOT_ROOT=/path/to/Isaac-GR00T PYTHONPATH=. uv run --no-sync python -c "
import m2m, sys; sys.path.insert(0,'/path/to/model2MLIR/workloads/groot_n1d7')
from loader import get_model_and_inputs
m,i=get_model_and_inputs(); r=m2m.convert(m,i)
open('/path/to/model2MLIR/workloads/groot_n1d7/groot_n1d7.mlir','w').write(r.mlir_text)
print(r.path_taken, r.mlir_text.count('linalg.'))"
# M2M_GROOT_LAYERS=2 for a fast smoke (fewer DiT layers).
```

## Status
- CAPTURED. Random-init action head (no checkpoint / no backbone), DiT 16 layers,
  877M params. fp32 + int8 + fp8 all `ok=True` via `fx_importer`.
- Venv built with Python **3.11** (m2m needs `enum.StrEnum`, 3.11+; the 3.10 pin in
  GR00T's `pyproject.toml` is only for the full flash-attn backbone, not the head).
  Pinned `transformers==4.57.3` (5.x makes `PretrainedConfig` a dataclass, which breaks
  `Gr00tN1d7Config`'s field ordering) and **`torchao==0.11.0`** (0.17 + torch 2.7.1 fails
  fp8 export: `Float8Tensor` FakeTensor lacks `tensor_data_names` under dynamo when
  `timestep_encoder` does `next(self.parameters()).dtype`; int8/fp32 unaffected).
  Extra runtime dep: `tyro` (pulled by `gr00t/configs/model/__init__.py`).
- The loader pre-seeds a lightweight `sys.modules["gr00t.model"]` stub so importing the
  leaf head module does NOT run `gr00t/model/__init__.py` (which eagerly imports the
  dataset/pipeline stack -> pandas etc., none needed for the head).
- Per-format results (default 16 DiT layers): each `ok=True`, 1 opaque op
  `aten_bitwise_not_default(tensor<1x64xi1>) -> tensor<1x64xi1>` (the `~image_mask` in
  `AlternateVLDiT.forward`). linalg.: fp32=2257, int8=2721, fp8=2489.
