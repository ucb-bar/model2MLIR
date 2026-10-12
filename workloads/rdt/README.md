# RDT-1B (RoboticsDiffusionTransformer, PyTorch)

RDT diffusion transformer (`models.rdt.model.RDT`): DiT-style stack of self-attn +
cross-attn + FFN blocks conditioned on language (T5) and image (SigLIP/DINOv2) tokens.
1B config = hidden 2048, depth 28, heads 32. Capture unit: ONE denoise step (the
`RDT.forward` the DDPM/DPM sampling loop calls), random init, no checkpoint download.

The frozen vision/text encoders and the lang/img/state adaptor MLPs run host-side;
their outputs enter the captured graph as hidden-size condition tokens. The diffusion
while-loop (`RDTRunner.conditional_sample`, default 5 inference steps) is NOT captured.

## Venv (dedicated)
RDT pins torch + transformers 4.41 + diffusers 0.27.2 + timm 1.0.3. Only torch + timm
(for `Attention`/`Mlp`/`RmsNorm`) + diffusers are needed to instantiate `RDT`.

```bash
git clone https://github.com/thu-ml/RoboticsDiffusionTransformer /path/to/RoboticsDiffusionTransformer
cd /path/to/RoboticsDiffusionTransformer
uv venv --python 3.10
uv pip install torch timm==1.0.3 diffusers==0.27.2 transformers==4.41.0 packaging
# frontend (FXImporter mode; a matching cp310 torch-mlir wheel can be added too)
uv pip install xdsl structlog ml_dtypes
uv pip install -e /path/to/model2MLIR --no-deps
```

## Run
```bash
cd /path/to/RoboticsDiffusionTransformer
M2M_RDT_ROOT=/path/to/RoboticsDiffusionTransformer M2M_RDT_DEPTH=2 uv run --no-sync python -c "
import m2m, sys; sys.path.insert(0,'/path/to/model2MLIR/workloads/rdt')
from loader import get_model_and_inputs
m,i=get_model_and_inputs(); r=m2m.convert(m,i)
open('/path/to/model2MLIR/workloads/rdt/rdt.mlir','w').write(r.mlir_text)
print(r.path_taken, r.mlir_text.count('linalg.'))"
```
Env: `M2M_RDT_DEPTH=N` (number of RDT blocks; default 2 smoke, real 1B = 28).

## Feasibility
- PyTorch, torch.export-able: YES. `RDT.forward` is pure tensor ops (Linear, RmsNorm,
  GELU, SDPA self/cross-attn, residuals). No data-dependent control flow at the chosen
  inputs: the `if t.shape[0] == 1` branch is decided by a static shape, and the
  `for i, block in enumerate(self.blocks)` loop is Python-unrolled at trace time
  (the `conds[i%2]` alternation is constant per block).
- timm `use_fused_attn()` may pick SDPA vs. manual matmul attention; both are exportable.
  Force eager/manual if the fused path traces poorly.

## Status
- fp32: **captured → 0 opaque ops** (depth=2; 526 linalg ops; FXImporter). The only
  gap surfaced was `aten.squeeze.dims`, now lowered as a reshape — no other new ops needed.
- Dependency: `uv pip install timm` (RDT's blocks import `timm.models.vision_transformer`).
- int8/fp8: pending (apply a `QuantizationConfig` before capture).
