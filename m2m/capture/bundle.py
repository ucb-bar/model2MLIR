"""Write a self-contained Merlin capture bundle (mlir + weights + golden + inputs + extra +
input_order) from ONE seeded model instance.

Factored out of ``workloads/capture_consistent.py`` so any capture path (per-model loader, or the
GGUF frontend) produces a byte-compatible bundle for the Merlin RVV runtime. It recovers the two
argument classes m2m elides to the runtime — registered buffers and lifted get_attr constants — by
type, plus quantized subclass inner tensors, exactly as the consistent-capture worker does.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class _LogitsOnly(nn.Module):
    """Wrap a HF causal LM so export sees a clean ``input_ids -> logits`` forward."""

    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        self.lm = lm

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm(input_ids=input_ids, use_cache=False).logits


def capture_hf_bundle(model_dir, out, *, quant_scheme: str | None = None, seq_len: int = 8,
                      dtype: "torch.dtype" = torch.float32) -> dict:
    """Capture a HuggingFace causal-LM (local dir or hub id) into a Merlin bundle via the torch path.

    ``quant_scheme`` is a torchAO scheme name (e.g. ``"int8_weight_only"``) or ``None`` for fp. This
    is the torch/torchAO ingestion arm (the GGUF arm is :func:`m2m.frontends.gguf.capture_gguf_bundle`).
    """
    from transformers import AutoModelForCausalLM

    lm = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype, use_cache=False).eval()
    model = _LogitsOnly(lm).eval()
    vocab = int(lm.config.vocab_size)
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab, (1, seq_len))
    quant = None
    if quant_scheme:
        from m2m.capture.torchao_pipeline import QuantizationConfig
        quant = QuantizationConfig(scheme=quant_scheme)
    summary = write_bundle(model, (input_ids,), out, quant=quant)
    summary["hf"] = str(model_dir)
    summary["quant_scheme"] = quant_scheme
    return summary


def _flatten_subclass(obj: Any, prefix: str, out: dict) -> None:
    """Recursively flatten a tensor-subclass parameter to its leaf tensors (qinner::<attr-path>)."""
    flat = getattr(obj, "__tensor_flatten__", None)
    if callable(flat):
        try:
            names, _ = flat()
        except Exception:  # noqa: BLE001
            names = []
        for nm in names:
            child = getattr(obj, nm, None)
            if child is not None:
                _flatten_subclass(child, f"{prefix}.{nm}", out)
    elif hasattr(obj, "detach"):                       # a leaf torch.Tensor
        t = obj.detach().cpu()
        _dt = str(t.dtype)
        arr = t.float().numpy() if ("float8" in _dt or "bfloat16" in _dt) else t.numpy()
        out[f"qinner::{prefix}"] = arr


def _lifted_constants(mdl, inputs, extra: dict) -> None:
    """Populate c_lifted_tensor_<i> from m2m's own export (graph order matches the importer)."""
    try:
        from m2m.capture.torch_export import capture_frontend_artifact
        from m2m.ir.torchmlir_decomps import torch_mlir_gap_decompositions
        from torch.export.graph_signature import InputKind

        artifact = capture_frontend_artifact(
            mdl, inputs, export_decomposition_table=torch_mlir_gap_decompositions())
        ep = artifact.exported_program or artifact.original_exported_program
        consts = dict(getattr(ep, "constants", {}) or {})
        sd = dict(getattr(ep, "state_dict", {}) or {})
        li = 0
        for spec in ep.graph_signature.input_specs:
            if spec.kind != InputKind.CONSTANT_TENSOR:
                continue
            val = consts.get(str(spec.target), sd.get(str(spec.target)))
            if val is not None and hasattr(val, "detach"):
                extra[f"c_lifted_tensor_{li}"] = val.detach().cpu().numpy()
            li += 1
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"[warn] lifted-constant export failed: {exc}\n{traceback.format_exc()[-800:]}")


def _extract_prov_fqns(mlir_text: str) -> list[str]:
    """The distinct ``prov.fqn`` module paths the export tagged, in first-seen order.

    Structured scan of the emitted MLIR (locate the ``prov.fqn = "..."`` attribute literal and read to
    the closing quote) — deterministic string handling, never a semantic regex.
    """
    marker = 'prov.fqn = "'
    out: list[str] = []
    seen: set[str] = set()
    i = mlir_text.find(marker)
    while i != -1:
        j = mlir_text.find('"', i + len(marker))
        if j == -1:
            break
        fqn = mlir_text[i + len(marker):j]
        if fqn and fqn not in seen:
            seen.add(fqn)
            out.append(fqn)
        i = mlir_text.find(marker, j + 1)
    return out


# npz key convention for region_goldens.npz, SHARED with the Merlin reader
# (merlin.baselines.bundle.CaptureBundle.region_goldens): one entry per (region fqn, slot) where slot
# is ``in<k>`` (the region's k-th boundary input = the upstream region's output) or ``out`` (the
# region's output golden). Flat, self-describing keys so np.load needs no sidecar.
_REGION_KEY = "{fqn}::{slot}"


def _capture_region_goldens(mdl, inputs, fqns) -> dict[str, "np.ndarray"]:
    """Register forward hooks on the nn.Modules named by ``fqns`` and capture, during ONE forward,
    each region's boundary tensors: its positional INPUTS and its OUTPUT. Best-effort — a module that
    is not registered under that exact name, or whose IO is not a plain tensor, is simply skipped."""
    named = dict(mdl.named_modules())
    caught: dict[str, np.ndarray] = {}
    handles = []

    def _mk(fqn: str):
        def hook(_m, args, output):
            for k, a in enumerate(args):
                if isinstance(a, torch.Tensor):
                    caught[_REGION_KEY.format(fqn=fqn, slot=f"in{k}")] = \
                        a.detach().float().cpu().numpy()
            out_t = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(out_t, torch.Tensor):
                caught[_REGION_KEY.format(fqn=fqn, slot="out")] = \
                    out_t.detach().float().cpu().numpy()
        return hook

    for fqn in fqns:
        mod = named.get(fqn)
        if mod is not None:
            handles.append(mod.register_forward_hook(_mk(fqn)))
    try:
        with torch.no_grad():
            g = mdl(*inputs)
    finally:
        for h in handles:
            h.remove()
    return caught, g


def _numpy_unrepresentable() -> frozenset:
    import torch
    out = [torch.bfloat16]
    for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
        dt = getattr(torch, name, None)
        if dt is not None:
            out.append(dt)
    return frozenset(out)


def _numpy_safe(x):
    """A tensor as numpy, converting ONLY the dtypes numpy cannot represent.

    `numpy` has no bfloat16 (nor any float8), so `.numpy()` on such a tensor raises
    `TypeError: Got unsupported ScalarType BFloat16`. Measured on smolvla, whose vision tower takes
    bf16 inputs: the export succeeded, the goldens were written, and the bundle died on `inputs.npz` --
    the one write here that lacked the conversion `golden.npy` and the buffer writes already do.

    ⚠️ NOT a blanket `.float()`. An LLM's `input_ids` are int64, and casting those to float32 corrupts
    the token ids silently -- the input would still load, still have the right shape, and index a
    different embedding row. So the conversion is keyed on the dtype actually being unrepresentable and
    every other dtype passes through untouched.

    bf16 -> f32 is lossless (bf16 is a truncated f32), so no precision is traded for the storage.
    """
    t = x.detach().cpu()
    if t.dtype in _numpy_unrepresentable():
        t = t.float()
    return t.numpy()


def write_bundle(mdl, inputs, out: str | Path, *, quant=None, capture_regions: bool = True) -> dict:
    """Convert ``mdl`` and write the full bundle to ``out``. Returns a summary dict.

    ``quant`` is an m2m ``QuantizationConfig`` (or ``None`` for an unquantized/fp bundle). The golden
    is the forward of the SAME instance on the SAME inputs, so the bundle is self-consistent.

    ``capture_regions`` (default on) additionally records per-region boundary tensors to
    ``region_goldens.npz`` (keyed by the ``prov.fqn`` modules the export tagged) — the shared substrate
    for per-region equivalence + standalone-section profiling. Captured in the SAME golden forward.
    """
    import m2m

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mdl.eval()
    inputs = tuple(inputs)

    weights_path = str(out / "weights.safetensors")
    r = m2m.convert(mdl, inputs, backend="fx_importer", quantization=quant,
                    level="linalg-on-tensors", weights_path=weights_path)
    assert r.ok, "m2m.convert failed"
    (out / "model.mlir").write_text(r.mlir_text)

    n_regions = 0
    if capture_regions:
        fqns = _extract_prov_fqns(r.mlir_text)
        region_goldens, g = _capture_region_goldens(mdl, inputs, fqns)
        if region_goldens:
            np.savez(out / "region_goldens.npz", **region_goldens)
        n_regions = len({k.split("::", 1)[0] for k in region_goldens})
    else:
        with torch.no_grad():
            g = mdl(*inputs)
    golden = g[0] if isinstance(g, (tuple, list)) else g
    np.save(out / "golden.npy", golden.detach().float().cpu().numpy())

    np.savez(out / "inputs.npz",
             **{f"in{i}": _numpy_safe(x) for i, x in enumerate(inputs)})

    extra: dict = {}
    for name, t in mdl.named_buffers():
        extra["buf::" + name] = t.detach().float().cpu().numpy()
    for pname, p in mdl.named_parameters():
        if type(p).__name__ not in ("Parameter", "Tensor") or hasattr(p, "__tensor_flatten__"):
            _flatten_subclass(p, pname, extra)
    _lifted_constants(mdl, inputs, extra)
    np.savez(out / "extra.npz", **extra)

    man = json.loads(Path(weights_path + ".manifest.json").read_text())
    order: dict = {}
    k = 0
    for i in range(len(man)):
        meta = man[str(i)]
        nm = meta.get("name", "") or ""
        if meta["kind"] in ("param", "buffer") or "lifted_tensor" in nm:
            continue
        order[nm] = k
        k += 1
    (out / "input_order.json").write_text(json.dumps(order, indent=2))

    return {
        "out": str(out), "n_inputs": len(inputs),
        "n_buffers": sum(1 for kk in extra if kk.startswith("buf::")),
        "n_lifted": sum(1 for kk in extra if kk.startswith("c_lifted")),
        "n_qinner": sum(1 for kk in extra if kk.startswith("qinner::")),
        "golden_shape": list(golden.shape), "linalg": r.mlir_text.count("linalg."),
        "input_order": order, "n_regions": n_regions,
    }
