# TinyLlama cached decoder layer capture

This workload extracts a decoder layer from the real
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` checkpoint. It runs the checkpoint on
captured token IDs, takes the selected layer's hidden states and populated KV
cache, then exports that same `LlamaDecoderLayer` with an explicit causal
mask. The loader checks its output against the full checkpoint before
returning it to model2MLIR. It does not construct a hand-written MLIR layer.

Use the `inputs.npz` from the TinyLlama prefill stage described in the
[full-model guide](https://github.com/ucb-bar/mx-gemmini-mlir/blob/handwritten-implementation/docs/compile_e2e_models.md).
For the pinned MX+VPU source section, use layer 5 and 64 prefix plus 64 query
tokens:

```sh
python workloads/capture.py tiny_llama_layer --formats fp32 \
  --output-root /path/to/captures \
  --env M2M_LLAMA_LAYER_INPUT_IDS=/path/to/prefill/inputs.npz \
  --env M2M_LLAMA_LAYER_INDEX=5 \
  --env M2M_LLAMA_PREFIX_TOKENS=64 \
  --env M2M_LLAMA_QUERY_TOKENS=64
```

The inputs are six tensors: 64 hidden-state rows, the two 64-token KV caches,
RoPE cosine and sine tables, and a 64×128 additive causal mask. The layer
weights are the checkpoint's own tensors. `M2M_LLAMA_LAYER_INDEX`,
`M2M_LLAMA_PREFIX_TOKENS`, and `M2M_LLAMA_QUERY_TOKENS` can select another
section when the captured token file has the corresponding length.

This captures the layer's PyTorch semantics at float32. Target-specific FP4
or FP8 choices belong to the downstream compiler recipe. A cycle comparison
with Nicolas's source requires the same precision, cache representation,
mask, outputs, and Spike timing configuration on both paths.
