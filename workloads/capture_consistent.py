#!/usr/bin/env python3
"""Consistent-capture worker: emit a self-contained bundle (inputs+golden+MLIR+weights
+extra+input_order) from ONE seeded model instance, for the Merlin RVV runtime.

The model loaders are unseeded and (for VLAs) randomly initialized, so a golden is only
valid captured in the same process as the MLIR/weights it checks. This driver, run INSIDE
the model's venv, does exactly that and also recovers the two argument classes m2m elides
to the runtime (it materializes them by type only):

  - registered buffers (rotary inv_freq, ...) via ``model.named_buffers()``;
  - lifted ``get_attr`` constants (attention masks, position tables, ...) via the exported
    program, in graph order so they line up with m2m's ``c_lifted_tensor_<i>`` numbering.

Usage (driver builds the venv, then runs this inside it):
  python workloads/capture_consistent.py <model> <fmt> <out_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
WORKLOADS = REPO / "workloads"


def _perturb_zero_parameters_for_smoke(mdl) -> int:
    """Make random-init smoke goldens nondegenerate without changing paper checkpoints.

    Some smoke-only diffusion loaders intentionally start with an all-zero output head.  Perturbing
    that head is useful for lowering coverage, but doing the same to a ``paper_ready`` pretrained
    model silently changes its checkpoint.  Paper captures therefore preserve every parameter byte.
    """
    if bool(getattr(mdl, "paper_ready", False)):
        return 0
    perturbed = 0
    with torch.no_grad():
        for parameter in mdl.parameters():
            if float(parameter.detach().abs().max()) == 0.0:
                parameter.copy_(torch.randn_like(parameter) * 0.02)
                perturbed += 1
    return perturbed


def _bundle(model: str, fmt: str, out: Path) -> None:
    import m2m
    sys.path.insert(0, str(WORKLOADS))
    from capture import _load_toml, _quant_for                      # reuse the scheme logic

    out.mkdir(parents=True, exist_ok=True)
    cfg = _load_toml(WORKLOADS / model)
    sys.path.insert(0, str(WORKLOADS / model))
    import loader as loader_module                                  # type: ignore

    torch.manual_seed(0)
    np.random.seed(0)
    mdl, inputs = loader_module.get_model_and_inputs()
    q = _quant_for(cfg, fmt)
    write_multi = getattr(mdl, "write_bundle", None)
    if callable(write_multi):
        summary = write_multi(out, quant=q)
        print("__BUNDLE_OK__ " + json.dumps({
            "model": model, "fmt": fmt, "out": str(out), **summary,
        }))
        return
    mdl.eval()
    inputs = tuple(inputs)

    # DiT/flow-matching models zero-init their output head (adaLN-zero), so a random-init
    # forward is all-zeros -- a degenerate golden that doesn't exercise the compute. Perturb
    # exactly-zero parameters with small noise; the captured weights stay self-consistent
    # with the golden, and the test now covers the full numeric path.
    _perturb_zero_parameters_for_smoke(mdl)

    weights_path = str(out / "weights.safetensors")
    r = m2m.convert(mdl, inputs, backend="fx_importer", quantization=q,
                    level="linalg-on-tensors", weights_path=weights_path)
    assert r.ok, "m2m.convert failed"
    (out / "model.mlir").write_text(r.mlir_text)

    # golden from the (now-quantized) instance on the same inputs
    with torch.no_grad():
        g = mdl(*inputs)
    golden = g[0] if isinstance(g, (tuple, list)) else g
    np.save(out / "golden.npy", golden.detach().float().cpu().numpy())

    np.savez(out / "inputs.npz",
             **{f"in{i}": x.detach().cpu().numpy() for i, x in enumerate(inputs)})

    # extra.npz: registered buffers (buf::<dotted>) + lifted get_attr constants (in order)
    extra: dict = {}
    for name, t in mdl.named_buffers():
        extra["buf::" + name] = t.detach().float().cpu().numpy()

    # Quantized weight inner tensors (qinner::<attr-path>). torchao weight-only quant under
    # torch>=2.8 leaves the int_data/scale accessible only via subclass-inner-tensor accesses
    # that m2m can't externalize (it emits an uninitialized empty tagged prov.quant_inner=
    # <attr-path>). Recursively flatten every quantized parameter to its leaf tensors -- the
    # __tensor_flatten__ names match the FX access chain -- so the runtime can bind them.
    def _flatten_subclass(obj, prefix, out):
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
            # numpy has no float8/bfloat16 dtype; decode fp8/bf16 inner data to f32 (the graph
            # casts the inner tensor to f32 before the matmul anyway, so f32 is the bound value).
            _dt = str(t.dtype)
            arr = t.float().numpy() if ("float8" in _dt or "bfloat16" in _dt) else t.numpy()
            out[f"qinner::{prefix}"] = arr
    for pname, p in mdl.named_parameters():
        if type(p).__name__ not in ("Parameter", "Tensor") or hasattr(p, "__tensor_flatten__"):
            _flatten_subclass(p, pname, extra)
    # Lifted get_attr constants: use m2m's OWN export (same decomposition table) so the
    # get_attr graph order matches m2m's c_lifted_tensor_<i> numbering; plain torch.export
    # inlines them. Values come from the exported program's constants / graph-module attrs.
    try:
        from m2m.capture.torch_export import capture_frontend_artifact
        from m2m.ir.torchmlir_decomps import torch_mlir_gap_decompositions
        from torch.export.graph_signature import InputKind
        artifact = capture_frontend_artifact(
            mdl, inputs, export_decomposition_table=torch_mlir_gap_decompositions())
        ep = artifact.exported_program or artifact.original_exported_program
        consts = dict(getattr(ep, "constants", {}) or {})
        sd = dict(getattr(ep, "state_dict", {}) or {})
        # torch-mlir's FxImporter names lifted CONSTANT_TENSOR inputs c_lifted_tensor_<i>
        # in graph-signature input order; values live in ep.constants (or state_dict).
        li = 0
        for spec in ep.graph_signature.input_specs:
            if spec.kind != InputKind.CONSTANT_TENSOR:
                continue
            tgt = str(spec.target)
            val = consts.get(tgt, sd.get(tgt))
            if val is not None and hasattr(val, "detach"):
                extra[f"c_lifted_tensor_{li}"] = val.detach().cpu().numpy()
            li += 1
    except Exception as exc:                                         # noqa: BLE001
        import traceback
        print(f"[warn] lifted-constant export failed: {exc}\n{traceback.format_exc()[-800:]}")
    np.savez(out / "extra.npz", **extra)

    # input_order.json: manifest arg-name -> inputs.npz index (genuine inputs, in order)
    man = json.loads(Path(weights_path + ".manifest.json").read_text())
    order, k = {}, 0
    for i in range(len(man)):
        meta = man[str(i)]
        nm = meta.get("name", "") or ""
        if meta["kind"] in ("param", "buffer") or "lifted_tensor" in nm:
            continue
        order[nm] = k
        k += 1
    (out / "input_order.json").write_text(json.dumps(order, indent=2))

    session_summary = {}
    get_session_spec = getattr(loader_module, "get_session_spec", None)
    if callable(get_session_spec):
        from m2m.capture.bundle import write_session_artifacts
        session = get_session_spec(mdl, inputs)
        if session is not None:
            session_summary = write_session_artifacts(
                mdl, inputs, out, manifest=man, input_order=order, session=session)

    print("__BUNDLE_OK__ " + json.dumps({
        "model": model, "fmt": fmt, "out": str(out), "n_inputs": len(inputs),
        "n_buffers": sum(1 for k in extra if k.startswith("buf::")),
        "n_lifted": sum(1 for k in extra if k.startswith("c_lifted")),
        "golden_shape": list(golden.shape), "linalg": r.mlir_text.count("linalg."),
        "input_order": order, **session_summary,
    }))


if __name__ == "__main__":
    _bundle(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
