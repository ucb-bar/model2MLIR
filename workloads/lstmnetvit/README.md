# LSTMNetVIT (vitfly)

Vision-based quadrotor obstacle avoidance: a depth image in, a 3-DoF velocity command out.
Repo: https://github.com/anish-bhattacharya/vitfly (`models/model.py`), 3.56 M params.

Structure: a two-stage SegFormer-style Mix-Transformer encoder over a 60×90 depth image
(overlapping patch-embed convs, efficient self-attention with spatial reduction, a
depthwise-conv MixFFN), a `PixelShuffle` + bilinear-upsample fusion of the two feature scales,
a spectral-norm Linear decoder, then a **3-layer LSTM** head.

Capture unit: one recurrent control step. The explicit hidden/cell inputs and updates form a continuous
multi-step session rather than resetting the LSTM on every frame.

## Scope limits, stated up front

- The capture exposes all three LSTM hidden/cell layers and records eager trajectory goldens. A paper
  session feeds every update into the next step.
- Random weights and a deterministic synthetic trajectory are smoke-test inputs only and produce
  `paper_ready: false`. A paper capture loads the complete published checkpoint with `strict=True` and
  requires an attributed external trajectory.

## Input shapes (the published forward is easy to get wrong)

`forward([depth, desired_velocity, quaternion])` with `depth (N,1,60,90)`,
`desired_velocity (N,1)`, `quaternion (N,4)`. The desired velocity is **one** element: the LSTM's
`input_size=517` is `512 + 1 + 4`, so passing a 5-vector makes the concatenation 521 wide and the
model raises.

## Venv
The main `model2MLIR/.venv`. No extra deps.

## Run
```bash
cd /scratch/agustin/projects/model2MLIR
python workloads/capture.py lstmnetvit --formats fp32,int8
.venv/bin/python workloads/capture_consistent.py lstmnetvit int8 \
    <merlin>/out/artifacts/recaptures/lstmnetvit_int8_full
```
Paper capture environment:

```sh
VITFLY_CKPT=/absolute/path/to/state_dict.pt \
VITFLY_SESSION_NPZ=/absolute/path/to/trajectory.npz \
VITFLY_SESSION_SOURCE='stable dataset/split/sequence identifier' \
VITFLY_PAPER_READY=1 \
M2M_SESSION_STEPS=256 \
python workloads/capture.py lstmnetvit --formats fp32,int8
```

The NPZ must contain `frames [steps,1,1,60,90]`, `desired_velocity [steps,1,1]`, and
`quaternions [steps,1,4]`; quaternion norms must already be one. The bundle records checkpoint and
trajectory SHA-256 digests. `VITFLY_DIR` selects the upstream checkout.

## Status
- fp32: 653 `linalg`, **0 opaque**. Verified end to end against the torch golden through
  merlin's host dispatch runtime: **cos = 1.000000000, rel = 2.8e-07**.
- int8: 653 `linalg`, **0 opaque**.

## Op-set notes
- The two `spectral_norm` Linear layers are folded back into plain `weight` before export. In
  eval mode that value is fixed, so folding is exact — and it is required, because torchAO's
  `quantize_` swaps `weight` in place and raises when a reparametrization owns the attribute.
- Needed the padded / depthwise / rank-3 conv work, and bilinear upsample, which lowers as two
  resize contractions against constant weight matrices (every size is static, so the
  interpolation weights are compile-time data).
- `PixelShuffle` and the unrolled LSTM already lowered.
