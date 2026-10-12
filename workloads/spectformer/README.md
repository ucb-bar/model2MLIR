# SpectFormer-Ti

Image classifier whose first blocks mix tokens in the **frequency domain** instead of with
attention. Repo: https://github.com/badripatro/SpectFormers (`vanilla_architecture/spectformer.py`).

Config captured: `img_size=224, patch_size=16, embed_dim=256, depth=12, num_heads=4` — 9,158,376
params, the published SpectFormer-Ti. With `alpha=4`, blocks 0–3 are `SpectralGatingNetwork`
(`rfft2` → multiply by a learned complex weight → `irfft2` over the 14×14 token grid) and blocks
4–11 are ordinary multi-head attention.

Capture unit: ONE forward of the whole classifier on a 224×224 image.

## Weights

Real trained weights: a 300-epoch SpectFormer-Ti checkpoint at **73.14% top-1 / 91.56% top-5**
on ImageNet, loaded with `missing=0, unexpected=0`. Override with `SPECTFORMER_CKPT`. If the
checkpoint is unreadable the loader falls back to random init and **says so on stderr** — the
golden then checks lowering exactness rather than accuracy.

## Venv
The main `model2MLIR/.venv` (torch + timm + torch-mlir).

## Run
```bash
cd /path/to/model2MLIR
python workloads/capture.py spectformer --formats fp32,int8
.venv/bin/python workloads/capture_consistent.py spectformer int8 \
    <merlin>/out/artifacts/recaptures/spectformer_int8_full
```
Env: `SPECTFORMER_CKPT`, `M2M_SPECTFORMER_DEPTH` (truncate blocks), `M2M_SPECTFORMER_DIM`.

## Two upstream quirks the loader works around

Both are in upstream, not in the capture, and neither is patched into the clone:

1. **`from numpy.lib.arraypad import pad`** on line 7 — a path numpy removed, and a name the
   file never uses. The loader stubs the module.
2. **`Block_attention` hardcodes `num_heads = 6`** ("4 for tiny, 6 for small and 12 for base")
   while `SpectFormer.__init__` takes no `num_heads`, so the Ti config **does not run as
   published**: 256 is not divisible by 6. The loader sets `num_heads = 4` and the matching qk
   scale. This changes no parameter shape, so the trained Ti checkpoint still loads exactly.

## Status
- fp32: 1240 `linalg`, **0 opaque**.
- int8: 1256 `linalg`, **0 opaque**.

## Op-set notes

This is the first workload here to use `torch.fft`, and it needed real frontend work rather than
one new op: `aten._fft_r2c` / `aten._fft_c2r` lower to real DFT contractions with complex tensors
carried as a trailing `(re, im)` pair (`m2m/ir/decompositions.py`). Measured against a float64
reference, the lowered f32 spectral block is accurate to 226 ppb where torch's own f32 butterfly
is 181 ppb — a 1.2× round-off gap from doing O(n²) work instead of O(n log n), not an indexing
error. `M2M_DFT_MAX_LEN` bounds how long a transform may be before the quadratic form is refused.
