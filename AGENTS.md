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

`capture/dynamic_linear_trace.py` provides opt-in `LinearInstrumentation` for
ordinary inference FP32 `nn.Linear` sites under the default TorchAO v2 dynamic
W8A8 recipe. Capture every original shared stage with `capture_source` before
one caller-owned `apply_quantization`; then `export_selected` checks actual
module-call boundaries, exact typed callers/call order, unchanged source and
non-quantized state, independently repacked original weights and complete
selected eager/export outputs. Failed checks publish no source IDs. Repeated
module invocations, custom Linear behavior, unknown effects and unsupported
recipes refuse. Use the owner as a context manager to restore temporary metadata
wrappers. This is source correspondence, not model accuracy or target execution.
The source-pair export retains an owned, process-local live seal over original
and selected graphs, proof data, state/field owners, storage, recipe metadata,
bytes and the exact selected input example. Traced conversion revalidates that
seal before any source snapshot or transformation. Later qdata/scales/zero-points,
biases, state, inputs or graph changes refuse; copying serialized proof metadata
without the actual owned binding cannot grant source correspondence. This is an
opt-in immutable-source contract, not a checkpoint or input repair mechanism.
Read-only `wrap_with_set_grad_enabled` regions may remain in both actual exports.
Their exact control value, ordered typed caller operands, schema-owned nonwriting
body, output slots and call order must match. Context projections grant no fresh
storage fact; writes through their aliases refuse. Nested context Linear sites,
unknown effects and shared/ambiguous context bodies remain unsupported. Contexts
are never erased to manufacture a source graph, and this source-pair proof does
not establish post-inline subclass-field lifting.
`write_quantized_multi_program_bundle(shared_program_converter=...)` can consume
these exact exports through an explicit successful `ConversionResult`; it keeps
actual stage objects, eager goldens, routing, shared tensor checks and file
integrity, and never falls back when the converter fails. Default capture is
unchanged. Example consumer:

```python
with LinearInstrumentation(shared_model) as lineage:
    source = lineage.capture_source(stage, inputs)
    apply_quantization(shared_model, quant)
    exported, proof = lineage.export_selected(source, stage, inputs, quantization=quant)
    converted = m2m.convert(exported, inputs, quantization=quant,
        quantization_preapplied=True, original_frontend_snapshot=source.snapshot,
        capture_trace=True, backend="fx_importer")
```

Tensor-subclass field lineage in `capture/subclass_trace.py` observes actual
state-field accesses before decomposition and binds only unique, unchanged live
storage and bytes in the prepared parameters. It rejects unknown source effects,
external input/state writes, changed source graphs/owners/fields and ambiguous
liftings. Exact multi-return tensor slots are separate storage owners; repeated
selection of one slot retains its intervening writes. This closes a specific
quantized-to-prepared proof and does not certify original-to-TorchAO attribution,
quantization quality, arbitrary tensor subclasses or target execution.
The field observer is bound to the archived raw quantized snapshot. Traced
conversion observes this actual source before HOP inlining, then transforms an
owned graph/custom-metadata copy with separately verified signature and the same
immutable state owners. The original graph remains live and unchanged for the
binding check; no checkpoint storage is copied. Supported read-only grad-context
bodies retain exact ordered input/output/schema proofs. Body-local field receivers
remain unsupported. Direct observation after a rewrite that changes public node
ordinals still declines before any snapshot can overwrite IDs; no post-rewrite
graph is relabelled as the raw source stage.

Graph copying preserves the verified exported node names and signature, including
builtin-like `input` and colliding suffixed names. Custom metadata containers are
owned; tensor storage and callable/operator identities are retained, supported Torch
provenance records are copied by their typed fields, and unknown opaque objects
refuse without invoking arbitrary deepcopy hooks. Linear source annotations are
an optional runtime capability, loaded only when that opt-in owner is requested.
Byte identities flatten scalar and one-element strided tensors before reinterpretation.
Typed FX/PT2E observer configuration and partial callback bindings retain owned
configuration fields without invoking callbacks or opaque deepcopy hooks.
Symbolic scalar/expression identities, typed shape paths and already-validated
range bounds are preserved without concretization, constraint evaluation or
removing runtime guards. Unknown nested objects or extra unproved fields refuse.

Disabled `aten.dropout` may be proved as a forward tensor identity only with
its selected schema, explicit `train=False`, a finite valid probability, exact
tensor metadata and typed consumer-edge bypass. Source and destination storage
versions must remain unchanged through every consuming edge. Training dropout,
unknown controls, wrong consumers and intervening alias writes decline; this does
not certify training/autograd semantics or numerical target execution.

`capture/call_boundary_trace.py` normalizes owned module-call signature slots
only through an exact no-op `aten.type_as` primitive with typed operand/result
and unchanged consumer-lifetime proofs. It leaves the arithmetic graph and raw
export untouched, records each actual module-call entry/slot and source IDs,
and keeps real dtype/device conversions as operations. This repairs framework
callee-signature maintenance when decomposition erases the identity; it never
clears call metadata or bypasses exported-program validation.

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
