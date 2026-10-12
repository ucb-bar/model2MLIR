# Xiaomi-Robotics-0 (XR0)

VLA model (~4.7B): Qwen3-VL-4B backbone encodes vision+language into a KV-cache;
a DiT (Diffusion Transformer) head decodes a 30-step action chunk via rectified
flow (Euler integration). Repo: https://github.com/XiaomiRobotics/Xiaomi-Robotics-0
(`xr0/` package, HF `transformers >= 4.57.1`, mmengine registry).

Source model class: `XR0(nn.Module)` in `xr0/mibot/models/VLA/XR0.py`.

Capture unit: ONE DiT denoise step (`XR0.dit_forward`) — a single rectified-flow
velocity prediction. Pure AdaLN-modulated DiT decoder layers cross-attending to
the VLM KV-cache; no data-dependent control flow at `prefix_length == 0`. The
VLM KV-cache, RoPE (cos, sin), attention mask and state are fed as already-built
input tensors (host-side prefix encoding stays out of the graph), mirroring the
smolVLA denoise-step convention.

The loader builds ONLY the DiT head + projectors + embedders from a small random
config (re-using the real `DiT`/`MLPProjector`/`TimestepEmbedder`/`dit_forward`
classes). The Qwen3-VL backbone is never constructed, so **no weights are
downloaded** — XR0's own `_build_model` would call
`Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")`.

## PyTorch-exportable?
**Yes (the DiT step).** `dit_forward` at `prefix_length==0` is static tensor ops:
linear projections, RMSNorm, RoPE, `scaled_dot_product_attention`, AdaLN. No
`.item()`, no Python loop, no flash-attn inside the captured unit (the full
`forward` does use flash_attention_2 on the VLM + a Python Euler loop, both
excluded). The full `XR0.forward(batch)` is NOT exportable (Euler `for`-loop,
`prefix_length` via `.item()`, dict batch, flash-attn VLM).

## Venv (dedicated)
```bash
cd /path/to/xr0_capture
uv venv --python 3.12
uv pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers==4.57.1 mmengine==0.10.7 numpy
uv pip install -e /path/to/Xiaomi-Robotics-0/xr0 --no-deps
# m2m + torch-mlir:
uv pip install xdsl structlog ml_dtypes
uv pip install --pre torch-mlir \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
  -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
uv pip install -e /path/to/model2MLIR --no-deps
```
Note: `flash-attn` is only needed for the full VLM forward, which we do NOT
capture — CPU export of the DiT step does not require it.

## Run
```bash
cd /path/to/xr0_capture
M2M_XR0_ROOT=/path/to/Xiaomi-Robotics-0/xr0 XR0_DIT_LAYERS=2 uv run --no-sync python -c "
import m2m, sys; sys.path.insert(0,'/path/to/model2MLIR/workloads/xr0')
from loader import get_model_and_inputs
m,i=get_model_and_inputs(); r=m2m.convert(m,i)
open('/path/to/model2MLIR/workloads/xr0/xr0.mlir','w').write(r.mlir_text)
print(r.path_taken, r.mlir_text.count('linalg.'))"
```

## Status
Captured in 3 formats (`backend="fx_importer"`, CPU, no download, random init,
`XR0_DIT_LAYERS=2`). Importing `mibot` normally pulls `mibot.data` -> `lightning`
and the Qwen3-VL backbone wiring, so the loader loads `XR0.py` in isolation via
`importlib` with lightweight stub parent packages (`_load_xr0_module`), re-using
the real `DiT`/`MLPProjector`/`TimestepEmbedder` classes. No weights downloaded.

Two loader fixes were needed for export:
- `t_embedder.dtype` forced to fp32 (upstream hardcodes bf16 for the sinusoidal
  embedding, which mismatched the fp32 MLP weights).
- `dit_forward`'s `t[:, 0, 0]` squeeze replaced with `t.reshape(t.shape[0])`: the
  double rank-reducing `select` on a `(B,1,1)` tensor collapses to a rank-0 SSA
  value and trips a divide-by-rank in m2m's `aten.select.int` decomposition.
  `reshape` is value-identical and lowers cleanly.

| format | file | ok | linalg | opaque | histogram |
|--------|------|----|--------|--------|-----------|
| fp32 | `xr0.mlir` | yes | 431 | 1 | `aten_repeat_default: 1` |
| int8 | `xr0_int8.mlir` | yes | 507 | 1 | `aten_repeat_default: 1` |
| fp8  | `xr0_fp8.mlir`  | yes | 488 | 1 | `aten_repeat_default: 1` |

Single opaque op (all formats): `@aten_repeat_default(tensor<1x1x1024xf32>) ->
tensor<1x1x1024xf32>` — the `sink.weight[None].repeat(B,1,1)` broadcast (identity
at B=1).
