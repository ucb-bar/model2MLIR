# Whisper tiny

Speech-to-text encoder/decoder transformer. Repo: https://github.com/openai/whisper; weights come
from the HF port `openai/whisper-tiny` (same checkpoint, already in the local HF cache), 37.76 M
params, 4 encoder + 4 decoder layers, 80 mel bins, 1500 encoder positions.

Capture unit: the **audio encoder plus ONE cross-attending decoder step** — the graph a decode
loop runs repeatedly, with `use_cache=False` so no KV cache is threaded in or out.

Two scope statements worth being explicit about:

- This does **not** prove autoregressive behaviour. It proves the per-step graph. A real
  transcription runs this graph once per token with a growing cache.
- Log-mel / STFT feature extraction is **outside** the capture. It is host-side preprocessing in
  Whisper, not part of the model graph; including it would measure a spectrogram front end.

## Venv
The main `model2MLIR/.venv` (torch + transformers + torch-mlir).

## Run
```bash
cd /path/to/model2MLIR
python workloads/capture.py whisper_tiny --formats fp32,int8
.venv/bin/python workloads/capture_consistent.py whisper_tiny int8 \
    <merlin>/out/artifacts/recaptures/whisper_tiny_int8_full
```
Env: `M2M_WHISPER_MODEL`, `M2M_WHISPER_LAYERS` (truncate enc+dec, random init),
`M2M_WHISPER_FRAMES` (mel frames; default 3000 = the full 30 s window).

`attn_implementation="eager"` is set deliberately: the SDPA and flash paths export as fused
opaque calls, and the math form is what the frontend decomposes.

## Status
- fp32: 1296 `linalg`, **0 opaque**.
- int8: 1296 `linalg`, **0 opaque**.

## Op-set notes
The encoder stem is two `Conv1d` layers (`k=3 pad=1`, then `k=3 stride=2 pad=1`). Rank-3 and
padded convolutions previously fell to an opaque `func.call`; they now normalize onto the same
im2col + `linalg.matmul` path as every other conv, so the whole model reaches real linalg.
