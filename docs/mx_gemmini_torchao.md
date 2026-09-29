# MX Gemmini TorchAO extension

The selected hardware contract is the `gemmini-mx-cleanup` RTL at
`f0167390b56fb315deea90ac1fc3983772e92d82`. This extension is local to
model2MLIR and selects `GemminiMxFPConfigs.standaloneMxFPConfig` /
`GemminiMxFPStandaloneConfig`: the pinned 16 by 16 standalone path with the
three symmetric FP4 E2M1, FP6 E3M2, and FP8 E4M3 PE modes. It uses TorchAO's
public `AOBaseConfig`, `register_quantize_module_handler`, and `quantize_`
APIs and requires no TorchAO fork. The selected formats are
`mx_gemmini_fp8`, `mx_gemmini_fp6`, and `mx_gemmini_fp4`.

## Use

```python
from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization

model = apply_quantization(
    model.eval(),
    QuantizationConfig(scheme="mx_gemmini_fp8"),
    example_inputs=(sample_input,),
)
print(model._m2m_quantization_stats)
```

For the one-layer TinyLlama architecture smoke test, set
`M2M_LLAMA_LAYERS=1`, `M2M_SEQ=32`, and `M2M_LLAMA_ATTENTION=eager` before
loading `workloads.tiny_llama.loader`. The one-layer path uses randomly
initialized weights. A full checkpoint and full-model accuracy run are separate
measurements.

```python
from workloads.tiny_llama.loader import get_model_and_inputs
from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization

for scheme in ("mx_gemmini_fp8", "mx_gemmini_fp6", "mx_gemmini_fp4"):
    # Rebuild the model for each scheme; quantize_ replaces modules in place.
    model, inputs = get_model_and_inputs()
    captured = apply_quantization(model, QuantizationConfig(scheme=scheme),
                                  example_inputs=inputs)
    print(scheme, captured._m2m_quantization_stats)
    print(captured(*inputs).shape)
```

The local architecture smoke observed eight eligible Linear modules and two
functional attention matmuls for each format, with no skipped contractions and
logits shape `[1, 32, 32000]`. This is one randomly initialized decoder layer,
not a whole-checkpoint accuracy result.

When a caller applies `apply_quantization` before conversion to retain the
quantization census and its own eager reference, pass the returned model to
`m2m.convert` with the same config and `quantization_preapplied=True`. This
records the selected scheme without running the TorchAO transform twice:

```python
import m2m

config = QuantizationConfig(scheme="mx_gemmini_fp8")
captured = apply_quantization(model.eval(), config, example_inputs=inputs)
result = m2m.convert(captured, inputs, quantization=config,
                     quantization_preapplied=True, backend="fx_importer")
```

For a full-checkpoint diagnostic, run the reproducible probe with an explicit
artifact directory:

```sh
python -m workloads.tiny_llama.mx_gemmini_probe \
  --out /configured/out/artifacts/probes/mx-gemmini-tinyllama-full-1
```

It writes `input_ids.npy` and `report.json` with the checkpoint revision and
SHA-256, source-file digests, library versions, fixed token input hash, three
contraction censuses, logit hashes, and
FP32-relative RMSE, maximum absolute error, and cosine similarity. The input
is random tokens, so these metrics are a numerical diagnostic rather than a
task-accuracy result.

TorchAO's handler converts eligible `nn.Linear` modules into
`MXGemminiLinear`. It stores quantized static weights, unsigned element codes,
and E8M0 scale bytes. Its forward quantizes each activation dynamically. An
exported-graph pass inserts operand Q/DQ at eligible functional `linear`,
`matmul`, `mm`, `bmm`, and `addmm` sites, including exposed attention QK and PV
matmuls. The pass reports skipped sites. It rejects fused SDPA because that op
hides both contractions. The current first-pass tile policy requires K divisible
by 32, and M/N divisible by 16 (MXFP8) or 32 (MXFP6/MXFP4).
The capture stats also report the total `nn.Linear` count and list each module
left unquantized because its K or N dimension fails that policy.

Operand conversion begins with BF16, uses a per-32-element scale
`2^floor(log2(max(amax, 2^-23)))`, E8M0 exponent encoding, and RNE element
rounding. FP8 uses OCP E4M3; FP6 uses E3M2; FP4 rounds through E3M1 then
E2M1. The direct Python quantizer refuses nonfinite blocks; exported graphs
currently have no runtime nonfinite guard, so their input contract excludes
them until poisoning is compared against the selected RTL. A randomized local comparison against the
independent `microscaling-quant` block implementation produced equal finite
BF16-dequantized elements and E8M0 scales for all three formats; it is a
software cross-check, not simulator qualification.

## MLIR handoff and boundary

Eager and exported execution computes matrix products in PyTorch after
operand Q/DQ. It does not model the RTL's product truncation, lane schedule,
LUT projection, packing, accumulator behavior, or transfers. The exported
model remains a **capture and operand-numerics prototype**. FXImporter emits
generic MLIR with zero opaque calls for the three-format toy contraction test
and attaches `prov.mx_capture_contract` with the RTL pin, selected configuration,
format, scale rule,
and contraction census. This is a frontend proof; model2MLIR has not yet
established typed MX lowering or simulator conformance. Its stats record
`numeric_status: operand_fake_quant_only`, so this cannot be interpreted as a
functional compiler certificate. The Merlin Phase 0 software spec remains
`unreviewed` until elaborated RTL, Spike, and Verilator agree on the selected
source closure and each format's vectors.
