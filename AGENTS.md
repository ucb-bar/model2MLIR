# model2MLIR (`m2m`)

Lower **any** model — PyTorch today, JAX next — to **standard-dialect MLIR** (linalg-on-tensors
for torch; StableHLO for jax) that any downstream MLIR project can consume. The whole point: one
common output regardless of source framework.

## Publication

Do not open any new pull request without explicit user approval. Permission to
implement, test, upstream, or finish existing changes does not authorize a new
pull request. Follow the publication route the user has authorized.

## Install

Uses [uv](https://docs.astral.sh/uv/). Flat layout — `import m2m` works after a plain install.

```bash
git clone https://github.com/ucb-bar/model2MLIR && cd model2MLIR
uv venv && uv pip install -e .              # core (torch + xdsl)
uv pip install -e '.[quant]'                # + torchAO (int8/fp8)
uv pip install -e '.[jax]'                  # + jax frontend
# optional accelerator (when a matching cp/torch wheel exists):
uv pip install --pre torch-mlir -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
```

Models with pinned/incompatible stacks (the VLAs) get their **own venv** — see
`workloads/<model>/README.md`. The recipe there: make a dedicated venv, then
`uv pip install -e /path/to/model2MLIR --no-deps`.

## Use

### Python (one API for both frameworks)
```python
import m2m
# torch
r = m2m.convert(model, (example_input,))            # -> ConversionResult
r.mlir_text          # standard-dialect MLIR (the artifact)
r.frontend           # "torch" | "jax"
r.output_type        # "linalg-on-tensors" | "stablehlo"
open("model.mlir", "w").write(r.mlir_text)          # write one .mlir (or many: split as you like)

# jax (same entrypoint, auto-detected)
r = m2m.convert(jax_fn, (x, y))                      # -> StableHLO

# quantization is torchAO (int8 / fp8), captured end-to-end:
from m2m.capture.torchao_pipeline import QuantizationConfig
r = m2m.convert(model, inputs, quantization=QuantizationConfig(scheme="int8_weight_only"))
```

`convert(model, inputs, *, backend="auto", quantization=None, output_type="linalg-on-tensors")`:
- `backend="auto"` tries torch-mlir (0 opaque when it works), else the FXImporter.
- `backend="fx_importer"` forces our xDSL path (use for vision-heavy VLAs where torch-mlir OOMs).
- `backend="torch_mlir"` torch-mlir only.

### CLI (writes `.mlir`)
```bash
m2m lower-torch workloads/tiny_llama/loader.py --out tiny_llama.mlir
m2m lower-torch workloads/tiny_llama/loader.py --quant int8_weight_only --out tiny_llama_int8.mlir
m2m lower-jax   path/to/jax_loader.py --out fn.mlir
m2m convert     loader.py            # auto-detect torch/jax; stdout if no --out
```
A loader is a `.py` exposing `get_model_and_inputs() -> (model_or_fn, tuple_of_inputs)`.

### Claude Code skills
`/m2m-lower-torch <loader.py> [--quant ...] [--backend ...] [--out ...]` and
`/m2m-lower-jax <loader.py> [--out ...]` (in `.claude/commands/`).

## Architecture (two planes, no torch-mlir maintenance)

- **FXImporter (xDSL) — the owned substrate.** Pure-Python, portable, no build, no OOM. Lowers
  aten → linalg/tensor/arith. Coverage grows via the **self-update loop** (`m2m/capture/unsupported/`):
  detect unsupported op → introspect → synthesize decomposition → validate → register.
- **torch-mlir — opportunistic accelerator + oracle.** When available it lowers many ops to 0-opaque
  MLIR in one shot. Our decompose-first table removes the ops it can't legalize before it runs. And
  via `m2m.coverage.differential_op`, torch-mlir's lowering of a single op is the **golden reference**
  that teaches/validates our FXImporter emitters — so we never maintain torch-mlir converters.

## Coverage / adding op support

```python
from m2m.coverage import opaque_report, validate_op, differential_op
opaque_report(r.mlir_text)                 # {opaque_func: count} — the worklist
differential_op(fn, inputs, name="mul")    # ours vs torch-mlir golden; verdict + target lowering
```
To support a new op: implement its emitter in `m2m/ir/decompositions.py` (real `tensor`/`linalg`/
`arith` ops), then confirm `differential_op(...).verdict == "ok"`.

## Layout
```
m2m/
  api.py            convert(), ConversionResult (the common output)
  cli.py            m2m lower-torch | lower-jax | convert | coverage
  capture/          torch.export capture, torchAO, self-update loop (unsupported/)
  ir/               FXImporter + decomposition tables (aten->linalg), torch-mlir gap decomps
  frontends/jax.py  jax.export -> StableHLO
  coverage.py       opaque report + single-op validation + torch-mlir differential oracle
workloads/          one folder per model (loader.py + README with its venv)
```
