# TinyLlama capture

The loader supports the full `TinyLlama/TinyLlama-1.1B-Chat-v1.0` checkpoint
and a reduced, random-weight smoke model. Set `HF_HOME` if the checkpoint is
cached outside the default Hugging Face location. The full session exports
separate prefill and decode programs with explicit KV-cache and position state.

From the model2MLIR repository root:

```sh
python workloads/capture.py tiny_llama --formats fp32 \
  --output-root /path/to/captures \
  --env M2M_LLAMA_LAYERS= \
  --env M2M_LLAMA_SESSION=e2e \
  --env M2M_LLAMA_ATTENTION=eager
```

This writes
`/path/to/captures/tiny_llama/tiny_llama_session/`, including a root
`session_contract.yaml`, `stages/prefill/`, and `stages/decode/`. Each stage
contains MLIR, checkpoint weights, inputs, a PyTorch reference output, and a
capture receipt. The default session uses synthetic token inputs; supply an
attributed token file through `M2M_LLAMA_TOKEN_IDS` and
`M2M_LLAMA_TOKEN_SOURCE` for a workload-specific evaluation.

For a faster frontend smoke capture, pass `--env M2M_LLAMA_LAYERS=2`. This
uses random weights and does not measure pretrained model quality.

For a short compiler diagnostic with the **pretrained checkpoint**, pass
`--env M2M_LLAMA_CHECKPOINT_LAYERS=2` instead. It loads the full checkpoint,
keeps its first two decoder layers, and retains the same embeddings, final
normalization, and output head. Its logits are for the truncated graph, so
they are not a full TinyLlama inference result. The two layer options are
mutually exclusive.

The MX+VPU compilation steps for this session and the other supported models
are in [mx-gemmini-mlir's end-to-end guide](https://github.com/ucb-bar/mx-gemmini-mlir/blob/handwritten-implementation/docs/compile_e2e_models.md).
