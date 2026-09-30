"""Externalize weights/constants to a safetensors file.

Keep the ``.mlir`` small and inspectable while preserving the actual parameter/buffer data:
write them to a widely-supported ``safetensors`` file keyed by their state-dict names, tag the
module with ``prov.weights_file``, and emit a JSON manifest mapping each func argument
(placeholder) to its weight key + dtype + shape. A consumer loads the safetensors and binds
each tensor to the corresponding function argument.
"""

from __future__ import annotations

import json
from typing import Any

from xdsl.dialects.builtin import ModuleOp, StringAttr


def externalize_weights(exported: Any, module: ModuleOp, path: str) -> dict:
    """Write the exported program's parameters/buffers to ``path`` (safetensors) and a
    ``path + '.manifest.json'`` mapping func-arg index -> {weight, kind, dtype, shape}.
    Tags ``module`` with ``prov.weights_file``. Returns a summary dict. Best-effort: tensors
    that can't be serialized (e.g. exotic quant subclasses) are recorded in the manifest
    with an ``error`` and skipped, never aborting capture."""
    import torch
    from safetensors.torch import save_file

    sig = getattr(exported, "graph_signature", None)
    sd = dict(getattr(exported, "state_dict", {}) or {})
    in2param = dict(getattr(sig, "inputs_to_parameters", {})) if sig else {}
    in2buf = dict(getattr(sig, "inputs_to_buffers", {})) if sig else {}
    placeholders = [n for n in exported.graph.nodes if n.op == "placeholder"]

    tensors: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    for idx, ph in enumerate(placeholders):
        wname = in2param.get(ph.name) or in2buf.get(ph.name)
        kind = "param" if ph.name in in2param else ("buffer" if ph.name in in2buf else "input")
        if wname is not None and wname in sd:
            try:
                t = sd[wname]
                t = t.detach() if hasattr(t, "detach") else t
                # Quantized tensor-subclass weights (torchao AffineQuantizedTensor & friends)
                # are NOT serialized as their dequantized fp32: torch.export unfolds them into
                # access_subclass_inner_tensor chains, so the matmul consumes the int8 ``int_data``
                # + ``scale`` inner tensors (bound at runtime via the ``qinner::`` / prov.quant_inner
                # channel from extra.npz), and the original fused weight ARG IS DEAD in the graph.
                # Dequantizing it (the old behavior) wrote a full-size fp32 copy of every quantized
                # weight to the blob -- e.g. pi0.5's 458 quantized Linears bloated weights.bin to
                # 16 GB of never-read fp32. Emit a 1-element zero STUB instead and flag the manifest
                # entry ``stub: true`` so the runtimes synthesize a zero buffer of the arg's true
                # shape/dtype for the (dead) descriptor rather than reading the blob.
                is_subclass = type(t).__name__ not in ("Tensor", "Parameter") or hasattr(
                    t, "__tensor_flatten__")
                if is_subclass:
                    full_shape = list(getattr(t, "shape", []))
                    dtype_str = str(getattr(t, "dtype", torch.float32)).replace("torch.", "")
                    tensors[wname] = torch.zeros(1, dtype=getattr(t, "dtype", torch.float32))
                    manifest[str(idx)] = {"weight": wname, "kind": kind,
                                          "dtype": dtype_str, "shape": full_shape,
                                          "stub": True}
                    continue
                t = t.contiguous().cpu().clone()
                tensors[wname] = t
                manifest[str(idx)] = {"weight": wname, "kind": kind,
                                      "dtype": str(t.dtype).replace("torch.", ""),
                                      "shape": list(t.shape)}
            except Exception as exc:  # noqa: BLE001
                manifest[str(idx)] = {"weight": wname, "kind": kind, "error": str(exc)[:120]}
        else:
            manifest[str(idx)] = {"kind": kind, "name": ph.name}

    # Parameter-free programs are valid bundle stages too. Persist a genuine
    # empty safetensors container so the declared weights path and receipt
    # contract are satisfied without inventing a tensor or weakening hashing.
    save_file(tensors, path)
    with open(path + ".manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    try:
        module.attributes["prov.weights_file"] = StringAttr(path)
    except Exception:  # noqa: BLE001
        pass
    return {"weights": len(tensors), "args": len(placeholders), "path": path}
