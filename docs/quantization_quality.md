# Quantization quality in ordinary captures

Capture a pre-quantization reference separately from the selected model outputs:

```python
from m2m.capture.bundle import write_bundle
from m2m.capture.torchao_pipeline import QuantizationConfig

summary = write_bundle(
    model.eval(), inputs, "bundle",
    quant=QuantizationConfig("int8_dyn_act_int8_weight"),
    capture_quantization_quality=True,
)
```

The writer snapshots every flat tensor output before an in-place quantizer runs.
It writes `prequant_reference.npz`, `quantization_selected_outputs.npz`, and
`quantization-quality.json`. The report contains source output dtypes/shapes and
finite L2, relative L2, maximum absolute, and RMS errors for each comparable
output. Metadata and the capture receipt bind the report, output files, selected
model, inputs, weights, and golden bytes. `golden.npy` continues to come from the
selected quantized model forward used for compiler validation.

These are observations on the bundle inputs. No tolerance or acceptance gate is
applied, and no dataset/task quality or static calibration is inferred. Dynamic
activation quantization can be measured with this API without a static
calibration dataset. Existing compiler accuracy criteria remain caller-owned.

Automatic reference capture requires a pure evaluation forward with immutable
parameters. It clones inputs and registered buffers under `torch.func.functional_call`,
restores PyTorch RNG states, and refuses detected input/buffer mutations. Original
parameter references are reused without cloning the checkpoint; parameter mutations
and arbitrary Python side effects are not proven absent.
Stateful workloads use their session trajectory contract or an independently
captured reference. A prequantized model can supply a reference explicitly:

```python
write_bundle(
    selected_model, inputs, "bundle", quant=config,
    quantization_preapplied=True,
    capture_quantization_quality=True,
    prequant_reference=original_outputs,
)
```

The caller owns the provenance of supplied outputs. Without a supplied reference,
a prequantized capture emits an `UNKNOWN` report. Shape/count incompatibilities,
nonfinite outputs, complex output metrics, and integer values outside exact
float64 representation are recorded explicitly. The selected model golden is
never substituted for a missing original reference.

The option defaults to disabled; existing capture/session recipes keep their
behavior. Session capture already retains its separate trajectory quality
reference and uses that path.
