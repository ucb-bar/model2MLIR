# model2MLIR
Any model in Torch or JAX format to MLIR frontend dialects.

Target quantization adapters can be installed out of tree through the
`m2m.quantization_adapters` entry-point group. Select an adapter with exact
software contract and policy files using `--quant-adapter`, `--contract`, and
`--policy`. A successful conversion writes `<output>.quantization.json` beside
the MLIR file and stamps its SHA-256 digest in the module attributes. Adapter
implementations and target-specific format policies belong to their support
packages.

An adapter's `m2m.quantization_manifest.v2` may mark a site `preserved` with a
nonempty `execution_route` when the selected accelerator path uses its source
float dtype. This differs from a `host` fallback and from an unsupported
`skipped` site. The route is target-owned; model2MLIR records it without
interpreting or certifying the backend. The v1 manifest remains accepted for
existing adapters.

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
Exported tensor-metadata assertions introduced by a quantizer can inherit a
producer's source lineage only when every asserted size, stride, dtype, device,
or layout field matches that producer's captured type. A mismatch leaves the
original-to-quantized trace diagnostic.
Re-exported single-use tensor constants keep the lineage of their captured
placeholder and value-preserving copy. Tuple selectors reconstructed by export
are related to their original calls only when the tuple producer, selected
index, and result type match uniquely.

When a grad-disabled nested graph is inlined, it is flattened before export
decompositions so each inner call retains its own source identity. The trace
binds each inner placeholder to its uniquely selected, exactly typed caller
value before accepting that identity. Wrapper outputs and tuple selections
carry identity through exactly typed value substitutions, including a proven
same-dtype, no-copy `to` that export removes. Export may share an outer node's
mutable metadata with an inner node, so snapshots isolate each node's metadata
before stamping it. Unbound callers, real casts and ambiguous substitutions
remain diagnostic; this is still forward-value provenance, not autograd proof.

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
`bf16`/`f16` MLIR types. Exact Torch E4M3FN and E5M2 tensors retain standard
`f8E4M3FN`/`f8E5M2` signatures and one-byte external storage. Pure layout
operations and supported explicit casts preserve those types; unknown
FP8 formats refuse instead of becoming FP32. No local encoder or replacement
quantization recipe is introduced. The trace checks actual source-ID-mapped
MLIR storage, including positional arguments and existing checked aliases;
unproved mappings remain unknown. `precision.projections` records an actual
FP32 projection rather than assuming one from the framework dtype.

Typed storage is not arithmetic qualification. `aten._scaled_mm` remains an
explicit unresolved external call with typed FP8 operands, and its trace remains
diagnostic. Accumulation order, scale/bias evaluation, floating effects and
numeric permission require separate proofs before any target implementation or
mixed-precision policy can be selected.

FP32-to-E4M3FN narrowing has an additional overflow-domain guard: a standard
`arith.truncf` is emitted only when the actual incoming SSA producer is a
complete pure ordered clamp whose finite bounds lie within the format's finite
range. The original clamp and cast remain in the graph. An unbounded cast,
unknown producer, altered effects or unproved source binding instead remains an
explicit typed `aten_to_e4m3fn_unproved_cast_domain` external call and a diagnostic
trace. Example values or range hints cannot grant this proof. This avoids
silently representing a finite-saturating source cast as a standard cast that
can produce NaN on overflow; it introduces no replacement quantizer or clamp.
The guard establishes the overflow domain only: target arithmetic, floating
environment and NaN payload behavior still require separate qualification.

An explicitly selected numerical policy can retain an actual `_scaled_mm`
source as a finite E4M3FN contraction with a positive-zero F32 seed, followed
by separate F32 row-scale and column-scale multiplications:

```python
from m2m.ir.scaled_mm import OrderedFP8ScaledMMPolicy

policy = OrderedFP8ScaledMMPolicy(
    accept_scale_reordering=True,
    accept_reduction_order_difference=True,
    rounding="rne",
    floating_effects="nontrapping_unobserved",
)
r = m2m.convert(exported_program, inputs, backend="fx_importer",
                scaled_mm_policy=policy, capture_trace=True)
```

This is a distinct, default-off arithmetic contract, not a bit-exact claim about
the original CPU kernel. The selected Torch CPU rowwise implementation may
widen and scale each operand before BLAS; moving those multiplications after
the dot product changes rounding. The caller must explicitly accept both that
change and the reduction-order difference. The generated typed contraction
has no fast-math flags; increasing-K accumulation, RNE and unobserved floating
effects remain mandatory backend/execution qualifications. The trace remains
diagnostic and records this policy separately from source IDs.

Only actual E4M3FN rank-two A/B, F32 scales `[M,1]`/`[1,N]`, explicit F32 output,
`bias=None`, `scale_result=None` and `use_fast_accum=False` are admitted.
Source row-A/column-B layouts are checked, allowing irrelevant singleton-axis
strides and recording the actual strides. All four tensors receive complete
runtime finite checks. Unknown shapes/options/layouts/types refuse; they do
not fall back to another arithmetic implementation. Ordinary F32 matmul is
never converted to FP8 by this capability. Normal source decomposition may
expose the original public quantizer's primitives, but the selected path must
still match an actual `_scaled_mm` call after preparation. With `decompose=False`
any unresolved quantizer operation remains visible.

The emitted standard tensor IR is not a deployable program by itself. A target
must qualify the typed FP8 contraction, scalar finite-check/guard placement,
storage ABI and scale epilogue, or provide a separately verified FP8
legalization. Toolchains without FP8 extension lowering may refuse the LLVM
boundary. Neither policy metadata nor CPU agreement grants source-to-target
lineage, automatic routing, model accuracy or performance admission. Quantizer
scales and original weight bytes are not replaced, repaired or re-encoded.

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
It never requests a train/eval transition during materialization. On the owned
result it uses the public exported-graph mode adapter from Torch, or its newer
TorchAO PT2E namespace when available. That provider adapter supports only its
documented limited dropout/BatchNorm rewrites; it cannot recapture Python
branches specialized by the original mode. Without either adapter, Torch's
train/eval restriction remains intact. An already-captured result can still be
converted through `m2m.convert(torch.export.export(cast_model, cast_inputs),
cast_inputs, ...)` without changing that mode or disabling its input guards.

For a captured graph with explicit floating-point casts or typed factories,
opt in to audited FP32 staging:

```python
staged, staged_inputs, original, receipt = m2m.materialize_frontend_precision(
    model, inputs, dtype=torch.float32, retarget_float_dtype_arguments=True)
```

The opt-in rewrites only floating `dtype` operands identified by Torch operator
schemas and copies floating lifted tensor constants into owned FP32 storage.
Integer and boolean dtype operands and constants retain their values and types;
complex tensors are refused rather than treated as FP32. The source model and
captured constants are not mutated. The receipt binds the
original and staged graph hashes, typed dtype decisions, and source/staged
constant content hashes. Re-export must verify that no non-FP32 floating tensor
type remains. This is a precision change, **not** a numerical-equivalence or
application-accuracy guarantee: compare the source and staged outputs separately
for the selected inputs, and do not infer behavior of uncaptured Python branches.

## Portable PT2E integer contractions

```python
import torch
from m2m.capture.pt2e_integerize import integerize_pt2e
from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference

# quantized_model is a converted PT2E GraphModule, not a TorchAO source edit.
# Run the independent framework-side integer oracle on the same frozen graph.
reference = run_pt2e_integer_reference(quantized_model, inputs, expected_contractions=2)
integer_model, receipt = integerize_pt2e(quantized_model, inputs)
assert receipt["quantized_contractions_remaining"] == 0, receipt["refusals"]
torch.testing.assert_close(integer_model(*inputs), reference.output, atol=0, rtol=0)
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

The independent reference executes frozen per-tensor or per-output-channel
weight Q/DQ for Linear, Conv2d and equal-batch Matmul in PyTorch without
reading the rewritten graph or compiler IR. It validates the actual int8 range, scales, zero points,
axis and output dtype and accounts for the number of contractions executed.
Supply the expected contraction count from a separately recorded graph census;
the example's `2` is illustrative. Exact equality to the rewritten model on
selected inputs checks that particular integer execution, not all inputs,
model quality, accelerator lowering or source-closed capture provenance.

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
An installed external quantization adapter can return a `torch.export.ExportedProgram`.
`write_bundle(..., quant=ExternalQuantizationConfig(...))` executes that selected
program for goldens and saves `quantization-manifest.json`. The receipt binds
the manifest file, its canonical digest in `meta.json`, and the digest stamped
on `model.mlir`. The adapter and its policy remain outside model2MLIR.
Session and multi-program helpers expose the same trace opt-in.
For a multi-program session, `write_multi_program_bundle` also writes
`bundle_integrity.json`: a sorted SHA-256/byte-count inventory of every regular
file beneath the session root, including `session_contract.yaml`, stage
receipts, and any root or stage session inputs/goldens. The generic workload
driver verifies the whole inventory and rejects missing, added, changed, or
symlinked files before reporting success. This is captured-file integrity,
not proof of frontend source closure, numerical correctness, or execution.

If a caller has already converted the actual selected model into this same output
directory, pass `conversion_result=r` to `write_bundle` to reuse it without
recapture or rewriting weights. With `quant` supplied, this also requires
`quantization_preapplied=True`. A trace that is diagnostic remains diagnostic in
the bundle; materialization does not turn it into an execution qualification.
