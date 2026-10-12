# DeepJSCC codec (from DiffJSCC)

Joint source-channel coding: an image is encoded straight to channel symbols, sent over a noisy
channel, and decoded back to an image. Repo: https://github.com/mingyuyng/DiffJSCC
(`model/deepjscc_cnn.py`, `configs/model/deepjscc_cnn.yaml`).

Captured: the JSCC **encoder + decoder** — a reflection-padded 7×7 conv stem, strided
downsampling convs, ResNet blocks with SNR-conditioned feature modulation, a channel projection,
then `ConvTranspose2d` upsampling back to RGB through a sigmoid.

## What this is NOT

**Not the full DiffJSCC pipeline.** That adds a Stable-Diffusion-2.1 UNet (320 channels), an
`AutoencoderKL` and an OpenCLIP text encoder, driven for tens of sampling steps. The paper's
reconstruction quality comes from that diffusion stage, so do not quote those numbers for this
bundle. This is the deployable codec, which is the part that makes sense on an embedded target.

The AWGN channel is also excluded: it multiplies in fresh Gaussian noise, which would make the
golden unreproducible. What is captured is encode → power-normalize → decode, the deterministic
compute; the channel is a runtime input in the real system.

## Weights
Random init. Upstream publishes only full DiffJSCC checkpoints (Stable-Diffusion-merged, several
GB each), so the golden checks lowering exactness rather than rate-distortion.

## Venv
The main `model2MLIR/.venv`. `deepjscc_cnn.py` imports a **training** stack at module scope
(`pytorch_lightning`, LPIPS/PSNR metrics, `pytorch_msssim`, `einops`) for its `LightningModule`;
the captured encoder/decoder are plain `nn.Module`s, so the loader stubs those rather than
installing a training stack, and they are deliberately not listed as deps.

## Run
```bash
cd /path/to/model2MLIR
python workloads/capture.py deepjscc --formats fp32,int8
.venv/bin/python workloads/capture_consistent.py deepjscc int8 \
    <merlin>/out/artifacts/recaptures/deepjscc_int8_full
```
Env: `DIFFJSCC_DIR`, `M2M_JSCC_SIZE` (default 64; the paper trains at 256, which makes the im2col
intermediates much larger — raise deliberately), `M2M_JSCC_NGF` (default 16; upstream uses 64).

## Status
- fp32: 380 `linalg`, **0 opaque**.
- int8: 380 `linalg`, **0 opaque**.

## Op-set notes
- Needed `ConvTranspose2d` (rewritten to a direct conv by the zero-insert / full-pad /
  flipped-kernel identity) and reflection-padded convs.
- Needed **integer `abs`**: torch decomposes reflection padding into index reflection, which
  takes `abs` of int64 index tensors, and `abs` was float-only (`math.absf`), so the emitted IR
  was ill-typed and silently became an opaque call.
- Needed inference `BatchNorm` (`aten._native_batch_norm_legit_no_training`), emitted as
  elementwise arithmetic over the channel axis — the statistics are frozen buffers, so no
  reduction is involved.
