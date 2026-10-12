# OpenVLA-7b (vision-language-action)

OpenVLA: Llama-2 7B language model + DINOv2/SigLIP vision backbone (`openvla/openvla-7b`,
custom HF code). Capture unit: `forward(input_ids, pixel_values) -> logits` (the single
forward; `predict_action` wraps a generation loop that isn't export-friendly).

## Venv (dedicated)
```bash
mkdir -p /path/to/openvla_capture && cd /path/to/openvla_capture
uv venv --python 3.12
uv pip install 'transformers>=4.40' timm tokenizers accelerate pillow numpy
uv pip install xdsl structlog ml_dtypes
uv pip install -e /path/to/model2MLIR --no-deps
# optional torch-mlir wheel (cp312/torch 2.x) for the accelerator path
```

## Run
```bash
cd /path/to/openvla_capture
uv run --no-sync python -c "
import m2m, sys; sys.path.insert(0,'/path/to/model2MLIR/workloads/openvla')
from loader import get_model_and_inputs
m,i=get_model_and_inputs(); r=m2m.convert(m,i,backend='fx_importer')
open('/path/to/model2MLIR/workloads/openvla/openvla.mlir','w').write(r.mlir_text)
print(r.ok, r.mlir_text.count('linalg.'))"
```

## Status
Captured. No 14 GB download: the loader fetches only the hub-side custom code
(`configuration_prismatic.py` / `modeling_prismatic.py` + `config.json`, tens of KB) and
builds the *real* OpenVLA architecture (fused DINO+SigLIP ViT -> 3-layer GELU MLP projector
-> Llama-2 causal LM) from a SMALL RANDOM config: shrunk LLM (2 layers / hidden 128 / vocab
512), two tiny timm ViTs (truncated to 2 blocks), eager attention, 4 text tokens + a 6x64x64
fused image. Capture unit is the single multimodal `forward(input_ids, pixel_values) -> logits`.

Modern-stack fixes baked into the loader (we run timm 1.x / transformers 5.x, not the
pinned 0.9.x / 4.40.1): neutralize the timm version assert; instantiate the custom class
directly (transformers 5 dropped `AutoModelForVision2Seq`); make `tie_weights()` kwarg-tolerant;
set the LLM `pad_token_id=None` (upstream 32000 is out of the small vocab); and re-wrap each
ViT featurizer forward to unwrap timm 1.x's `get_intermediate_layers` *list* return.

Backend: FXImporter (`backend="fx_importer"`). All three formats lower fully (0 opaque ops):

| format | ok | opaque | linalg. | .mlir |
|--------|----|--------|---------|-------|
| fp32   | yes | 0 | 528 | `openvla.mlir` (~214 KB) |
| int8   | yes | 0 | 644 | `openvla_int8.mlir` (~254 KB) — i8 weight tensors preserved |
| fp8 (e4m3) | yes | 0 | 638 | `openvla_fp8.mlir` (~239 KB) — float8 weights dequant to f32 in-graph |

Run: `M2M_OPENVLA_LLM_LAYERS`, `M2M_OPENVLA_HIDDEN`, `M2M_OPENVLA_VOCAB`,
`M2M_OPENVLA_VIT_LAYERS`, `M2M_OPENVLA_IMG`, `M2M_SEQ` override the small config.
