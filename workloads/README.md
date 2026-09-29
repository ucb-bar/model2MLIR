# Workloads — the standardized model→MLIR capture flow

One command captures any model to MLIR in every datatype/quant format:

```bash
python workloads/capture.py <model>                 # fp32 + int8 + fp8
python workloads/capture.py <model> --formats fp32  # just one
python workloads/capture.py <model> --level high-level   # structured (linalg_ext.*) form
python workloads/capture.py <model> --sections      # also emit per-source-module .mlir
python workloads/capture.py --all                   # every model
python workloads/capture.py --list                  # what's available
```

Output per model/format:
- `workloads/<model>/<model>{,_int8,_fp8}.mlir` — the graph, each asserted to **0 opaque ops**.
- `workloads/<model>/<model>{,_int8,_fp8}.safetensors` (+ `.manifest.json`) — the real
  weights/buffers, keyed by name; the `.mlir` carries `prov.weights_file`. Graph stays small;
  data is fully recoverable. (See `m2m.transforms.externalize`.)
- with `--sections`: `workloads/<model>/sections/<model>{...}.<section>.mlir` — one `func.func`
  per top-level source module (VLM / action expert / ...), with cross-section I/O, so each
  section can compile/run at its own cadence. (See `m2m.transforms.split_by_section`.)

`convert()` options: `level` (linalg-on-tensors | high-level), `quantization`, `preserve_qdq`,
`fully_standard` (pure core MLIR, no `*_ext`), `weights_path` (externalize to safetensors).
`.mlir`/`.safetensors` are gitignored (build artifacts).

## What a model directory contains

```
workloads/<model>/
├── loader.py        # REQUIRED: get_model_and_inputs() -> (nn.Module, tuple[Tensor, ...])
├── capture.toml     # OPTIONAL: venv / deps / env / upstream config
└── <model>*.mlir    # OUTPUT (gitignored)
```

### `loader.py` — the one contract
```python
def get_model_and_inputs() -> tuple[torch.nn.Module, tuple[torch.Tensor, ...]]:
    """Build the model and example inputs such that model(*inputs) runs eagerly."""
```
Rules that make capture stable and scalable:
- **Small config, no big/gated downloads.** Build the real architecture from a *reduced*
  config (fewer layers/hidden/vocab) and random init — capture is about the op set, not weights.
- **Capture ONE unit.** A single `forward` / denoise step — not generation loops or sampling.
- **`sys.path` the upstream repo** (cloned at `/scratch/agustin/projects/<Repo>`, a sibling of
  `model2MLIR`). Stub out heavy side-effect imports if the package `__init__` pulls the world.
- **Eager attention**, `use_cache=False` — keep the traced graph export-friendly.

### `capture.toml` — environment (optional)
```toml
venv   = ".venv"          # default: workloads/<model>/.venv (the STANDARD location).
                          # may be an absolute path to reuse an existing dedicated venv.
python = "3.11"           # python version for a fresh venv
deps   = ["torch==2.7.1", "transformers==4.53.3", ...]   # pip specs for a fresh venv
upstream = "/scratch/agustin/projects/<Repo>"            # documentation
[env]                     # env vars set for the capture (e.g. small-config knobs)
M2M_<MODEL>_LAYERS = "4"
```
The driver always also installs `xdsl structlog ml_dtypes torchao` and the `m2m` package
(editable, `--no-deps`) into the venv.

## How the driver works
1. Reads `capture.toml`, resolves the model's dedicated venv (default
   `workloads/<model>/.venv`; builds it from `deps`/`python` if missing — idempotent).
2. Runs the capture **inside that venv** (subprocess) so each model's conflicting
   torch/transformers/torchao versions stay isolated.
3. For each format, calls `m2m.convert(model, inputs, backend="fx_importer",
   quantization=..., level=...)`, writes the `.mlir`, and checks 0 opaque ops.

For a deferred multi-program loader that implements `write_bundle(out, quant=...)`,
the same generic CLI instead writes `<model>[_format]_session/` with a root
`session_contract.yaml` and separate `stages/<name>/model.mlir` plus
`capture_receipt.json` for each program. It checks the stage roster, byte/hash
binding of every receipt-listed artifact, required bundle members, and opaque-op
count before reporting success. Session captures only support
`linalg-on-tensors`; `bf16`/`fp16` dtype-cast formats and `--sections` are refused
rather than silently captured as FP32. These are capture artifacts, not
source-closure or execution proof; the driver reports
`source_closure_verified: false`.

The two output levels (see `docs/OP_TAXONOMY.md`): `linalg-on-tensors` (default, portable
standard dialects) and `high-level` (opt-in `linalg_ext.*` named ops; `m2m.expand_to_linalg`
lowers it back). Quantization is QDQ-preserved by default (`quant_ext.dequantize` on the
weight); fp8 renders the native `f8E4M3FN` spelling.

## Adding a new model (the whole recipe)
1. `git clone <upstream>` → `/scratch/agustin/projects/<Repo>`.
2. `mkdir workloads/<model>` ; write `loader.py` (the contract above).
3. Write `capture.toml` (deps + any small-config env vars).
4. `python workloads/capture.py <model>` → three `.mlir` files at 0 opaque.

If a new op shows up opaque, add its decomposition (one entry in
`m2m/ir/decompositions.py::DECOMPOSITION_TABLE` + a family in
`m2m/ir/import_fx.py::_FAMILY_OF`) — see `docs/OP_TAXONOMY.md`. Brand-new architectures so
far have needed at most one new op each (RDT=`squeeze`, GR00T=`bitwise_not`, xr0=`repeat`,
BitVLA=`mean.default`).

The four vision/audio/control workloads below broke that pattern and are the honest exception:
they are the first conv-heavy, spectral and recurrent models here, and between them they needed
the conv family generalized (padding, rank-3/Conv1d, groups, transposed), `torch.fft` as real DFT
contractions with a `(re, im)` pair layout for complex tensors, bilinear/nearest resize, inference
BatchNorm, `constant_pad_nd`, and integer `abs`. Those are shared capability rather than per-model
glue — see each workload's README for which one needed what.

## Models & status
Each has a `loader.py` + `capture.toml`; per-model env notes in `workloads/<model>/README.md`.
```bash
python workloads/capture.py --all     # capture/refresh all 10 in fp32+int8+fp8
```

| Model | fp32 | int8 | fp8 | notes |
|---|---|---|---|---|
| tiny_llama | 0 | 0 | 0 | |
| smolvla | 0 | 0 | 0 | |
| rdt | 0 | 0 | 0 | |
| rdt2 | 0 | 0 | 0 | |
| molmoact | 0 | 0 | 0 | |
| openvla | 0 | 0 | 0 | real DINO+SigLIP+Llama stack, no 14 GB download |
| xr0 | 0 | 0 | 0 | |
| bitvla | 0 | 0 | 0 | int8/fp8 quantize lm_head only (BitNet W1.58 stays) |
| groot_n1d7 | 0 | 0 | 0 | |
| pi05 | 0 | 0 | 0 | 3.6B; recursive HOP flattener + SDPA/N-D-matmul/dequant decompositions (was 5193) |
| spectformer | 0 | 0 | — | SpectFormer-Ti; **torch.fft** spectral gating + trained ImageNet weights |
| whisper_tiny | 0 | 0 | — | encoder + ONE decoder step; Conv1d stem |
| lstmnetvit | 0 | 0 | — | ViT+LSTM depth controller; PixelShuffle + bilinear upsample |
| deepjscc | 0 | 0 | — | JSCC codec only (not DiffJSCC's diffusion stage); ConvTranspose + BN |

**All 10 original models lower to 0 opaque ops in fp32, int8, and fp8 (30/30 artifacts).** Each capture
also emits `<model>{,_int8,_fp8}.safetensors` (+ manifest) with the real weights, and (with
`--sections`) per-source-module `.mlir` files. "0" = 0 opaque ops. Re-run `capture.py <model>`
to regenerate after converter changes.
