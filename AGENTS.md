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

Frontend snapshots preserve historical `results` slots and add versioned
`result_metadata` describing actual FX `val` presence, container and every
existing result ordinal. Missing metadata stays unobserved; explicit None is
recorded separately. This provenance grants no numerical or operator-effect
semantics and does not infer absent runtime outputs.

```python
from m2m.coverage import opaque_report, validate_op, differential_op
opaque_report(r.mlir_text)                 # {opaque_func: count} — the worklist
differential_op(fn, inputs, name="mul")    # ours vs torch-mlir golden; verdict + target lowering
```
To support a new op: implement its emitter in `m2m/ir/decompositions.py` (real `tensor`/`linalg`/
`arith` ops), then confirm `differential_op(...).verdict == "ok"`.

## Custom kernels (Triton) — the kernel registry

A model may call hand-written kernels — e.g. a `@torch.library.triton_op` like
`torch.ops.triton_exp.fused_mlp2` (see `workloads/mlp_basic_triton`). `torch.export` keeps each
as ONE custom-op node. If its qualname is in the global table `KERNEL_REGISTRY`
(`m2m/ir/kernel_registry.py`, the sibling of `decompositions.py`), the FXImporter emits it as a
**black-box kernel call** instead of an opaque op:

```mlir
func.func @forward(...) {
  %7 = func.call @triton_exp__fused_mlp2(%6, %0, %1, %2, %3) {prov.kernel = "triton_exp::fused_mlp2",
         prov.kernel_operands = ["x", "w1", "b1", "w2", "b2"],
         prov.family = "kernel", prov.op = "kernel", prov.region_id = ...}
         : (...) -> tensor<4x32xf32>
  ...
}
func.func private @triton_exp__fused_mlp2(...) -> tensor<4x32xf32> attributes {prov.kernel = "triton_exp::fused_mlp2"}
```

- Tensor inputs are SSA operands in the op **schema's** argument order (named by
  `prov.kernel_operands`); every output is an SSA result (multi-output kernels -> multi-result
  call). Non-tensor args (floats, ints, block sizes, omitted defaults) go in `prov.kernel_args`,
  which is **omitted** when the kernel has none.
- That is the whole contract. Where a kernel's implementation lives and what language it is in
  belong to the consumer's kernel table, not the IR.
- The private declaration is the hook a downstream lowering (merlin) points at the real kernel;
  it is placed **after** `@forward`, so the entry function stays the first `func.func`.
- `prov.kernel` is the stable identity (the symbol gets a `_<n>` suffix per new shape).
- Registered kernel calls are **not opaque**: `opaque_report` skips them, `kernel_report(text)`
  lists them, `workloads/capture.py` prints `kernels=N`. torch-mlir can't legalize custom ops,
  so `backend="auto"` goes straight to the FXImporter when a graph contains one.
- To add a kernel: append `KernelSpec("ns::name", description=...)` to `_BUILTIN_KERNELS`.
  Unregistered Triton ops still lower as opaque (with a diagnostic pointing at the registry).
- The consumer is merlin: `src/merlin/llvmlower/kernel_table.py` maps the same
  `prov.kernel` names to the Triton kernel exported as llvm-dialect MLIR (under
  `src/merlin/kernels/triton_llvmir/`) and links it into the model's MLIR before lowering. A
  kernel registered here needs a matching merlin entry before its captures can run; merlin's
  `test_table_keys_match_m2m_registry` fails until it has one.
- Not yet supported: in-place kernels (`mutates_args`, which export wraps in
  `auto_functionalized_v2`). A registered one is an import **error**, never silently wrong IR.
- How it is implemented, its limitations, and how to use it (the end-to-end MLP example):
  `docs/guides/triton_kernel_integration.md`.

## Layout
```
m2m/
  api.py            convert(), ConversionResult (the common output)
  cli.py            m2m lower-torch | lower-jax | convert | coverage
  capture/          torch.export capture, torchAO, self-update loop (unsupported/)
  ir/               FXImporter + decomposition tables (aten->linalg), torch-mlir gap decomps,
                    kernel_registry.py (custom kernels -> black-box func.call)
  frontends/jax.py  jax.export -> StableHLO
  coverage.py       opaque/kernel reports + single-op validation + torch-mlir differential oracle
workloads/          one folder per model (loader.py + README with its venv)
```
