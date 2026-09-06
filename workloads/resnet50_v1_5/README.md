# ResNet-50 v1.5

The loader uses torchvision's `resnet50` (the v1.5 stride placement) and the
`IMAGENET1K_V2` weights.  A paper capture also requires `M2M_RESNET_INPUT_NPZ` containing normalized
images under `images`, shaped either `[steps,3,224,224]` or `[steps,1,3,224,224]`.  The capture writes
the entire distinct-image trajectory and eager logits goldens. It rejects non-finite values and a step
count different from `M2M_SESSION_STEPS`.

Paper capture is fail-closed:

```sh
M2M_RESNET_INPUT_NPZ=/absolute/path/to/images.npz \
M2M_RESNET_INPUT_SOURCE='stable dataset/split/sample-list identifier' \
M2M_RESNET_PREPROCESSING=IMAGENET1K_V2 \
M2M_RESNET_PAPER_READY=1 \
M2M_SESSION_STEPS=256 \
python workloads/capture.py resnet50_v1_5 --formats fp32,int8
```

The bundle records SHA-256 digests for both the input file and a canonical serialization of the loaded
state dict. The source label must identify the exact corpus split/sample list; the preprocessing string is
not inferred from the input values.

For compiler-only bring-up, set `M2M_RESNET_RANDOM=1`; optionally set
`M2M_RESNET_PRETRAINED=0` to avoid loading weights.  Either choice marks `paper_ready: false` and the
paper freeze must reject it.
