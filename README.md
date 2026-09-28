# model2MLIR
Any model in Torch or JAX format to MLIR frontend dialects.

## Inspect frontend operations and lowering correspondence

```python
import json
import m2m

r = m2m.convert(model, inputs, backend="fx_importer", capture_trace=True,
                weights_path="artifacts/weights.safetensors")
assert r.ok, r.diagnostics
with open("artifacts/frontend-trace.json", "w") as f:
    json.dump(r.capture_trace, f, indent=2)
```

Create the output directory before calling. The trace contains the original
captured PyTorch graph, the actual quantized graph (unchanged when no quantization
was requested), the prepared/decomposed graph and typed producer–consumer edges.
Counts are **static captured call sites**, not runtime invocation counts or the
total Python API universe. Inputs, weights and buffers retain their signature roles;
tensor contents stay outside the trace and may be externalized to safetensors.

Graph-scoped identities follow owned decompositions, quantization input boundaries,
aliases, and MLIR expansion/fusion. Final correspondence is reconciled against the
exact returned MLIR bytes and their SHA-256 digest. Traced FXImporter output uses
generic MLIR serialization so custom printers cannot silently discard identity
attributes on operations such as `tensor.empty`.

Eliminated calls need explicit evidence: an unused tuple selection must be
in-bounds, exactly typed, and have no users; a vanished no-op cast or `detach_`
must have an exact typed producer-to-consumer bypass at every use. The `detach_`
proof covers forward tensor values only. It does not certify autograd metadata or
training semantics; those require a separate contract.

Data-dependent size guards are not silently discarded: supported
`sym_size`/range-constraint/assertion chains become `tensor.dim`, `arith.cmpi`,
and `cf.assert` in the MLIR, with exact prepared-call provenance. A later compiler
must preserve or explicitly discharge `cf.assert`; a complete frontend trace
alone does not prove its runtime implementation.

When a grad-disabled nested graph is inlined, the trace binds each inner
placeholder to its uniquely selected, exactly typed caller value. Identical
inner calls can then be matched through their typed arguments. A renamed,
decomposed, or ambiguously repeated call remains diagnostic rather than being
matched by operator name or FX ordinal.

If an external quantizer will mutate the model, snapshot first and pass that receipt:

```python
original = m2m.capture_frontend_snapshot(model, inputs, stage="original")
# Apply your quantizer here using public TorchAO interfaces.
r = m2m.convert(quantized_model, inputs, backend="fx_importer", capture_trace=True,
                quantization=config, quantization_preapplied=True,
                original_frontend_snapshot=original)
```

The portable PT2E quantizer accepts the same `original_frontend_snapshot` option
on `apply_quantization()` to preserve exact transformation origins. Other external
quantizers without lineage instrumentation can still expose both graphs, but their
correspondence remains diagnostic. Already-quantized loaders without an original
receipt are explicitly unknown; they never masquerade as the original float graph.
Capture failures do not trigger a hidden replacement export in traced conversion.

Check `r.capture_trace["status"]` and `blockers`. `complete` certifies structural
accounting only, not backend support, correct quantization, numerical execution, or
whole-model compilation. Opaque calls are still accounted for but do not prove a
usable implementation. Unavailable provenance or an unqualified backend remains
diagnostic. Existing callers without `capture_trace=True` keep normal serialization.

Precision is not inferred from a scheme name. BF16/FP16 tensors retain native
`bf16`/`f16` MLIR types. The current FXImporter/xDSL path projects Torch FP8
storage to FP32; the trace records each affected value in `precision.projections`
and remains diagnostic. This representation is not native FP8 lowering or an
execution qualification.

To change the precision of the captured static program while preserving its
original operation identities:

```python
import torch

cast_model, cast_inputs, original, cast_receipt = m2m.materialize_frontend_precision(
    model, inputs, dtype=torch.bfloat16)
r = m2m.convert(cast_model, cast_inputs, backend="fx_importer", capture_trace=True,
                original_frontend_snapshot=original)
```

The helper uses an owned materialized graph and fresh parameters/buffers, leaving
the source model unchanged. Its deterministic receipt lists exact state/input
precision conversions. It converts the captured static program, not dtype-dependent
Python branches: any different Python specialization needs a separate capture.

## Portable PT2E integer contractions

```python
from m2m.capture.pt2e_integerize import integerize_pt2e

# quantized_model is a converted PT2E GraphModule, not a TorchAO source edit.
integer_model, receipt = integerize_pt2e(quantized_model, inputs)
assert receipt["quantized_contractions_remaining"] == 0, receipt["refusals"]
# Use this same rewritten model for the integer host golden and capture.
```

The deterministic rewrite supports symmetric signed-int8 activations with a
scalar scale, and scalar or frozen per-output-channel weight scales, for
Linear, ungrouped NCHW Conv2d, and equal-shape-batch Matmul. It emits int8
contractions with an overflow-checked int32 accumulator, then applies the actual
activation and weight scales and float32 bias. Linear/Conv weights use axis 0;
Matmul weights use their last/output axis. Frozen scale/zero-point byte hashes,
axis, dtype, geometry and refusals appear in the receipt. Float64 scale tensors
remain float64; the scaled result is explicitly converted to PT2E's float32
output type.

Nonzero zero points, reduction/batch channel axes, per-channel activations,
dynamic qparams/shapes, grouped convolution and unsupported broadcast geometry
remain unchanged with explicit refusals. Integer accumulation changes floating
evaluation order: compare the portable and rewritten outputs using an authored
tolerance, and generate downstream golden outputs from the rewritten model.
Successful capture alone does not certify a backend's integer implementation.

## Materialize a traceable capture bundle

```python
from m2m.capture.bundle import write_bundle

summary = write_bundle(model, inputs, "artifacts/capture", capture_trace=True,
                        source_path="loader.py", metadata={"workload_role": "iteration"})
```

The bundle contains `model.mlir`, `weights.safetensors` and its ABI manifest,
inputs, actual selected-model goldens, `frontend-trace.json`, and merged
`meta.json` (caller provenance is retained). `capture_receipt.json` binds the
exact bytes, including the trace and its MLIR digest. Lifted constants come from
the same prepared export used for lowering, not a hidden replacement capture.
Session and multi-program helpers expose the same trace opt-in.

If a caller has already converted the actual selected model into this same output
directory, pass `conversion_result=r` to `write_bundle` to reuse it without
recapture or rewriting weights. With `quant` supplied, this also requires
`quantization_preapplied=True`. A trace that is diagnostic remains diagnostic in
the bundle; materialization does not turn it into an execution qualification.
