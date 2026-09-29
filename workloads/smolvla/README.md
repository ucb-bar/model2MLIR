# smolVLA (lerobot)

smolVLA `VLAFlowMatching`: SmolVLM2-500M backbone + action expert (~0.45B params).
Capture unit: one flow-matching `denoise_step` (prefix pass + one expert pass), with one camera.

## Local smoke capture

Run from the model2MLIR repository root. The loader uses random weights unless
`M2M_SMOLVLA_PRETRAINED=1`; this command captures one denoise step, not a full
three-stage action session.

```bash
uv venv workloads/smolvla/.venv --python 3.12
uv pip install --python workloads/smolvla/.venv/bin/python 'lerobot[smolvla]==0.5.1' xdsl structlog ml_dtypes
uv pip install --python workloads/smolvla/.venv/bin/python -e . --no-deps
```

```bash
workloads/smolvla/.venv/bin/python - <<'PY'
import os
import sys
from pathlib import Path
import m2m

os.environ.pop("M2M_SMOLVLA_SESSION", None)
sys.path.insert(0, "workloads/smolvla")
from loader import get_model_and_inputs

model, inputs = get_model_and_inputs()
result = m2m.convert(model, inputs)
assert result.ok
output = Path("workloads/smolvla/smolvla.mlir")
output.write_text(result.mlir_text)
print(output, result.path_taken)
PY
```

The generic workload driver can instead invoke the loader's three-program
session contract: `python workloads/capture.py smolvla --formats fp32`.
It reads `capture.toml`, writes `workloads/smolvla/smolvla_session/` with separate
stage MLIR and capture receipts. This is the loader's three-stage capture path,
not the small smoke command above. An offline run with the full checkpoint and
synthetic inputs produced all three stage bundles with zero opaque operations;
each stage also lowered to LLVM IR. Those checks did not execute compiled code.
The `fp32` format label selects the capture request; it does not cast every
internal operation to FP32. The prefix and flow stages retain BF16 operations.

## Status

Merlin's separate session capture treats `prefix_encode`, recurrent `flow_denoise`, and
`action_decode` as separate programs. Capturing or lowering them is not evidence
of accelerator execution or full-model numerical accuracy. Stage receipts do
not establish frontend source closure. New captures made through the generic
driver include `bundle_integrity.json`, which byte-binds the root session
contract and all root/stage files, including session artifacts. The driver
verifies this receipt before reporting capture success. The earlier offline
capture predates this receipt and is not retroactively certified by it; file
integrity still does not prove source closure or numerical execution.

PT2E models may store a weight as `i8` while dequantizing it to `bf16` before a
floating contraction. Such a contraction is **not** an `i8 x i8 -> i32` operation.
The frontend preserves its typed `quant_ext.dequantize_per_tensor` operations
instead of silently relabeling it as W8A8.

For an `i8 x i8 -> i32` accelerator, a legal W8A8 rewrite must have a target
datapath for the integer contraction, valid frozen qparams and geometry, and an
explicit numerical contract for changing the source evaluation order. BF16 Q/DQ
rounds *each operand before* the floating contraction; integer accumulation then
FP32 scaling does not preserve that result. A four-element exported PT2E example
in `tests/test_pt2e_channel.py` differs by about 0.0075, and the integerizer
therefore refuses it with the observed dtype recorded. An explicit precision
conversion followed by a measured model accuracy gate would be a new program,
not an implicit consequence of selecting `int8` in a capture command.

New integerization receipts classify every selected contraction under
`precision_decisions`: `integerized_i32` records an actual integer graph rewrite,
`preserve_float_qdq` records a source-precision floating path, and `unresolved`
records other refusals. `precision_decision_counts` totals those rows. A mixed
exported-graph test exercises one FP32 Q/DQ integer rewrite beside one preserved
BF16 Q/DQ linear. Neither receipt classifies a node as device-executable. For an
int8-only target, the preserved BF16 paths need a qualified host placement with
explicit data transfers, or an explicit precision conversion with its own
numerical gate. Consumers must treat `quantized_contractions_remaining` as a
refusal gate for an all-W8A8 claim. A full session also requires an independent
numerical reference, host/device partition and transfer checks, and target
execution.
