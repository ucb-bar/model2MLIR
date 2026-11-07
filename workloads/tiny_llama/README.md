# TinyLlama

Causal LM (TinyLlama-1.1B architecture). Full-model weights come from the
local Hugging Face cache. `M2M_LLAMA_LAYERS=2` in this workload constructs
a random-initialized two-layer smoke model; Merlin's
`examples/workloads/causal_lm_section/loader.py` instead keeps the first two
pretrained layers when a size-bounded hardware test is needed.

## Environment

Merlin's chip `run.sh setup` builds a compatible capture environment. Fetch
the desired TinyLlama checkpoint into that environment's Hugging Face cache
before requesting an offline pretrained capture. The Merlin section loader
documents the pinned revision and file checks.

## Run
```bash
cd /path/to/model2MLIR
# fast smoke: fewer layers
M2M_LLAMA_LAYERS=2 uv run --no-sync m2m convert workloads/tiny_llama/loader.py --out workloads/tiny_llama/tinyllama.mlir
# int8
M2M_LLAMA_LAYERS=2 uv run --no-sync m2m convert workloads/tiny_llama/loader.py --quant int8_weight_only --out workloads/tiny_llama/tinyllama_int8.mlir
```
Env: `M2M_LLAMA_LAYERS=N` (truncate layers), `M2M_SEQ=N` (sequence length).

## Status
- fp32: 282 `linalg`, 0 opaque.
- int8: 342 `linalg`, 0 opaque (quantization preserved).
- fp8: captures; blocked at xDSL float8 element typing.
