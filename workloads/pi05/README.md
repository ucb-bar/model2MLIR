# pi0.5 (openpi, PyTorch)

Full pi0.5: PaliGemma `gemma_2b` + `gemma_300m` action expert (~3.62B params, random init).
Capture unit: one flow-matching `denoise_step` (prefix SigLIP+Gemma pass + one expert pass).
Host-side observation preprocessing runs eagerly, outside the captured graph.

## Venv (dedicated)
openpi pins torch 2.7.1 + transformers 4.53.2 + a `transformers_replace` patch + jax.

```bash
git clone https://github.com/Physical-Intelligence/openpi /path/to/openpi
cd /path/to/openpi
uv venv --python 3.11 && uv pip install -e .
# apply openpi's transformers_replace over the installed transformers
T=$(uv run python -c "import transformers,os;print(os.path.dirname(transformers.__file__))")
cp -r src/openpi/models_pytorch/transformers_replace/* "$T/"
# add the frontend (FXImporter mode; a cp311 torch-mlir wheel can be added too)
uv pip install xdsl structlog ml_dtypes
uv pip install -e /path/to/model2MLIR --no-deps
```

## Run
```bash
cd /path/to/openpi
uv run --no-sync python -c "
import m2m, sys; sys.path.insert(0,'/path/to/model2MLIR/workloads/pi05')
from loader import get_model_and_inputs
m,i=get_model_and_inputs(); r=m2m.convert(m,i)
open('/path/to/model2MLIR/workloads/pi05/pi05.mlir','w').write(r.mlir_text)
print(r.path_taken, r.mlir_text.count('linalg.'))"
```

## Status
- fp32: captured → 1557 `linalg` (FXImporter mode; 2.2 MB). torch-mlir for torch 2.7.1 +
  gap-op decompositions would cut the opaque count.
- int8/fp8: pending.
