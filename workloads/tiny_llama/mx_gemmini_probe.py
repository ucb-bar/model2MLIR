"""Run a full-checkpoint MX operand-capture diagnostic and save exact inputs.

This is a framework comparison, not an accelerator simulator qualification.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchao
import transformers

from m2m.capture.mx_gemmini_quant import RTL_COMMIT
from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization
from workloads.tiny_llama.loader import get_model_and_inputs

SCHEMES = ("mx_gemmini_fp8", "mx_gemmini_fp6", "mx_gemmini_fp4")


def _sha256_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _source_digests() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = (
        "m2m/capture/mx_gemmini_quant.py",
        "m2m/capture/torchao_pipeline.py",
        "m2m/ir/decompositions.py",
        "workloads/tiny_llama/loader.py",
        "workloads/tiny_llama/mx_gemmini_probe.py",
    )
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths}


def run(output: Path, *, seed: int = 812, seq: int = 32, threads: int = 8) -> dict:
    if output.exists():
        raise FileExistsError(f"diagnostic output already exists: {output}")
    if seq % 32:
        raise ValueError("TinyLlama MX smoke requires sequence length divisible by 32")
    output.mkdir(parents=True)
    os.environ.pop("M2M_LLAMA_LAYERS", None)
    os.environ["M2M_SEQ"] = str(seq)
    os.environ["M2M_LLAMA_ATTENTION"] = "eager"
    torch.set_num_threads(threads)
    torch.manual_seed(seed)

    model, _ = get_model_and_inputs()
    from huggingface_hub import hf_hub_download

    checkpoint = Path(hf_hub_download(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "model.safetensors", local_files_only=True
    ))
    with checkpoint.open("rb") as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    model_revision = checkpoint.parent.name
    token_ids = torch.randint(0, model.lm.config.vocab_size, (1, seq), dtype=torch.long)
    np.save(output / "input_ids.npy", token_ids.numpy(), allow_pickle=False)
    with torch.no_grad():
        baseline = model(token_ids).float()
    report = {
        "schema": "m2m.mx_gemmini_tinyllama_probe.v1",
        "qualification": "framework_operand_fake_quant_only",
        "rtl_commit_selected": RTL_COMMIT,
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "model_revision": model_revision,
        "checkpoint_sha256": checkpoint_sha256,
        "full_checkpoint": True,
        "seed": seed,
        "sequence_length": seq,
        "attention_implementation": "eager",
        "torch_version": torch.__version__,
        "torchao_version": torchao.__version__,
        "transformers_version": transformers.__version__,
        "source_sha256": _source_digests(),
        "input_ids_sha256": _sha256_tensor(token_ids),
        "baseline_logits_sha256": _sha256_tensor(baseline),
        "logits_shape": list(baseline.shape),
        "formats": {},
    }
    for index, scheme in enumerate(SCHEMES):
        if index:
            del model, captured
            gc.collect()
            model, _ = get_model_and_inputs()
        captured = apply_quantization(
            model, QuantizationConfig(scheme=scheme), example_inputs=(token_ids,)
        )
        with torch.no_grad():
            result = captured(token_ids).float()
        delta = result - baseline
        report["formats"][scheme] = {
            "capture": captured._m2m_quantization_stats,
            "logits_sha256": _sha256_tensor(result),
            "finite": bool(torch.isfinite(result).all()),
            "rmse_vs_fp32": float(torch.sqrt(torch.mean(delta.square()))),
            "max_abs_vs_fp32": float(delta.abs().max()),
            "cosine_vs_fp32": float(torch.nn.functional.cosine_similarity(
                result.flatten(), baseline.flatten(), dim=0
            )),
        }
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=812)
    parser.add_argument("--seq", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    result = run(args.out, seed=args.seed, seq=args.seq, threads=args.threads)
    print(json.dumps({name: {key: value for key, value in row.items() if key != "capture"}
                      for name, row in result["formats"].items()}, indent=2))


if __name__ == "__main__":
    main()
