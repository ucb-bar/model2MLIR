"""Public conversion API: PyTorch / torchAO model -> MLIR (linalg-on-tensors).

`convert()` is the one call most users want. It runs the torch-mlir bridge
(torch-mlir primary path, decomposition-based FXImporter fallback) and returns
the MLIR text plus diagnostics about which path was taken and what coverage was
needed.

The self-updating coverage loop lives under `m2m.capture.unsupported`;
`coverage_report()` surfaces it for a given model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects.builtin import ModuleOp

from m2m.capture.torch_mlir_bridge import bridge_fx_graph, module_to_text
from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization


@dataclass
class ConversionResult:
    """Outcome of converting a model to MLIR.

    Attributes:
        mlir_text: the emitted MLIR (linalg-on-tensors by default).
        module: the parsed xDSL ModuleOp, or None on failure.
        path_taken: "torch_mlir" | "fx_importer" | "failed".
        output_type: the MLIR output dialect requested.
        diagnostics: human-readable diagnostic strings from the bridge.
    """

    mlir_text: str = ""
    module: ModuleOp | None = None
    path_taken: str = "failed"
    output_type: str = "linalg-on-tensors"
    frontend: str = "torch"
    diagnostics: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        # The MLIR text is the deliverable; the xDSL ModuleOp is best-effort.
        return bool(self.mlir_text) and self.path_taken != "failed"


def convert(
    model: Any,
    example_inputs: tuple[Any, ...],
    *,
    output_type: str = "linalg-on-tensors",
    quantization: QuantizationConfig | None = None,
    func_name: str = "forward",
    allow_fallback: bool = True,
    decompose: bool = True,
    backend: str = "auto",
    level: str = "linalg-on-tensors",
    preserve_qdq: bool = True,
    fully_standard: bool = False,
    weights_path: str | None = None,
    quantization_preapplied: bool = False,
) -> ConversionResult:
    """Convert a PyTorch model to MLIR.

    Args:
        model: a torch.nn.Module (or an ExportedProgram, which torch-mlir accepts).
        example_inputs: tuple of example tensors the artifact will be called with.
        output_type: torch-mlir output dialect. "linalg-on-tensors" is the default
            and the level merlin ingests. Pass "torch" for the raw Torch dialect.
        quantization: optional torchAO QuantizationConfig applied before export, so
            quantization is captured end-to-end.
        func_name: name of the public func in the emitted MLIR.
        allow_fallback: fall back to the decomposition-based FXImporter when
            torch-mlir is unavailable or fails (default True).
        backend: "auto" (try torch-mlir, then FXImporter), "torch_mlir" (torch-mlir
            only, no fallback), or "fx_importer" (skip torch-mlir entirely). Use
            "fx_importer" for models whose torch-mlir lowering OOMs (e.g. the
            vision-heavy VLAs) -- an OOM SIGKILL can't be caught for fallback.
    """
    if backend == "torch_mlir":
        allow_fallback = False

    # Frontend dispatch: a .gguf path -> GGUF frontend (graph reconstructed from GGUF metadata);
    # torch nn.Module / ExportedProgram -> torch path; any other callable -> jax path (StableHLO).
    import torch
    from pathlib import Path as _Path

    if isinstance(model, (str, _Path)) and str(model).endswith(".gguf"):
        from m2m.frontends.gguf import convert_gguf

        return convert_gguf(model, example_inputs, level=level)

    _is_torch = isinstance(model, torch.nn.Module) or type(model).__name__ == "ExportedProgram"
    if not _is_torch:
        try:
            import jax  # noqa: F401

            from m2m.frontends.jax import convert_jax

            return convert_jax(model, example_inputs)
        except ImportError:
            pass  # no jax; fall through and let the torch path try (will error clearly)
    if quantization_preapplied and quantization is None:
        raise ValueError("quantization_preapplied=True requires a quantization config for provenance")
    if quantization is not None and not quantization_preapplied:
        # Quantized (tensor-subclass) weights are swapped by .to()/.eval() during
        # capture; disable swap-on-conversion process-wide so capture uses copy
        # semantics and doesn't trip the weakref guard.
        import torch

        try:
            torch.__future__.set_swap_module_params_on_conversion(False)
        except Exception:  # noqa: BLE001
            pass
        model = apply_quantization(model, quantization)

    # Decompose-first: capture an ExportedProgram and run decompositions so
    # composite ops torch-mlir can't legalize (e.g. aten.diff) are lowered
    # away before torch-mlir sees them. Falls back to letting the bridge
    # re-export the module if capture fails.
    exported_program = None
    if decompose:
        try:
            from m2m.capture.torch_export import capture_frontend_artifact
            from m2m.ir.torchmlir_decomps import inline_set_grad_hops, torch_mlir_gap_decompositions

            artifact = capture_frontend_artifact(
                model,
                tuple(example_inputs),
                export_decomposition_table=torch_mlir_gap_decompositions(),
            )
            exported_program = artifact.exported_program or artifact.original_exported_program
            # Inline torch.no_grad() HOPs (wrap_with_set_grad_enabled) so the FXImporter
            # sees a flat graph (torch 2.7.x quantized exports keep these; modern stacks
            # inline them). Cascades to resolve their downstream ops.
            if exported_program is not None:
                try:
                    for _ in range(8):  # bounded fixpoint (nested HOPs); guards against spin
                        if not inline_set_grad_hops(exported_program.graph_module):
                            break
                except Exception:  # noqa: BLE001 - non-fatal; importer handles residual
                    pass
        except Exception:  # noqa: BLE001 - bridge will re-export as a fallback
            exported_program = None

    # The high-level (structured) form is only produced by the FXImporter path -- torch-mlir
    # always emits standard linalg -- so requesting it forces the FXImporter backend.
    emit_named = level == "high-level"
    result = bridge_fx_graph(
        model,
        tuple(example_inputs),
        func_name=func_name,
        output_type=output_type,
        allow_fallback=allow_fallback,
        exported_program=exported_program,
        use_torch_mlir=(backend != "fx_importer" and not emit_named),
        emit_named_ops=emit_named,
        weights_path=weights_path,
    )
    # Stamp the representation level so downstream/merlin knows which contract it received
    # (default portable standard form vs the opt-in structured high-level form).
    if result.module is not None:
        try:
            from xdsl.dialects.builtin import StringAttr

            result.module.attributes["prov.level"] = StringAttr(level)
        except Exception:  # noqa: BLE001
            pass

    # Record the quantization scheme on the module so the (fp8/int8) quantization
    # is captured in the IR even when fp8 element types render as f32 (xDSL has no
    # builtin Float8). Downstream targets read prov.quantization to recover the scheme.
    if quantization is not None and result.module is not None:
        try:
            from xdsl.dialects.builtin import StringAttr

            scheme = getattr(quantization, "scheme", None) or str(quantization)
            result.module.attributes["prov.quantization"] = StringAttr(str(scheme))
            per_module = getattr(quantization, "per_module", None)
            if per_module:
                # serialize the mixed-precision map as "pattern=scheme;pattern=scheme"
                mixed = ";".join(f"{k}={v}" for k, v in per_module.items())
                result.module.attributes["prov.quantization_mixed"] = StringAttr(mixed)
        except Exception:  # noqa: BLE001
            pass

    # Preserve QDQ: fold the implicit weight-dequant (dtype_cast -> mul scale) into an
    # explicit quant_ext.dequantize op so the quantization is first-class/matchable. Only
    # meaningful for the FXImporter path (we have a module) and when quantizing. Verify-
    # guarded inside fuse_qdq -> never regresses the portable output.
    import os as _os
    def _qcount(tag):
        if _os.environ.get("M2M_DEBUG_QINNER") and result.module is not None:
            from m2m.capture.torch_mlir_bridge import module_to_text as _mt
            try:
                n = _mt(result.module).count("prov.quant_inner")
            except Exception:  # noqa: BLE001
                n = -1
            print(f"[qinner-api] {tag}: prov.quant_inner={n}", flush=True)
    # Propagate quant-inner tags from the (elided, uninitialized) torchao subclass
    # inner-tensor empties onto their CONSUMER ops. xDSL's printer drops attributes on
    # tensor.empty, so the binding key can't survive the text handoff on the empty itself;
    # a consuming linalg.generic (the dtype_cast / mul / dequant) does serialize its attrs.
    # Covers BOTH int8 (dequant) and fp8 (mul x scale) weight-only patterns.
    if quantization is not None and result.module is not None:
        try:
            from collections import deque

            # xDSL's printer drops attributes on tensor.empty AND on layout ops like
            # linalg.transpose / tensor.*_shape, so the binding key must land on the first
            # downstream op that DOES serialize its attrs (a linalg.generic / quant op).
            # BFS forward from each elided inner-tensor empty through the attr-dropping
            # pass-throughs and tag that carrier with prov.quant_inner_<operand-index>.
            _CARRIERS = ("linalg.generic", "builtin.unregistered")
            for _op in list(result.module.walk()):
                if _op.name != "tensor.empty" or "prov.quant_inner" not in _op.attributes:
                    continue
                _tag = _op.attributes["prov.quant_inner"]
                _q, _seen = deque([_op.results[0]]), set()
                while _q:
                    _val = _q.popleft()
                    for _use in list(_val.uses):
                        _c = _use.operation
                        if _c.name in _CARRIERS:
                            _c.attributes[f"prov.quant_inner_{_use.index}"] = _tag
                        elif id(_c) not in _seen:
                            _seen.add(id(_c))
                            for _r in _c.results:
                                _q.append(_r)
        except Exception:  # noqa: BLE001 - best-effort; runtime falls back to empty
            pass
    _qcount("after_bridge")
    if preserve_qdq and quantization is not None and result.module is not None:
        try:
            from m2m.transforms import fuse_qdq

            result.module = fuse_qdq(result.module)
        except Exception:  # noqa: BLE001
            pass
    _qcount("after_fuse_qdq")

    # Fully-standard mode: lower ALL extension ops (linalg_ext + quant_ext) to pure upstream
    # MLIR core dialects (linalg/scf/tensor/arith/math). The artifact then uses no custom
    # dialect at all -- a "purely MLIR supported" version.
    if fully_standard and result.module is not None:
        try:
            from m2m.transforms import to_standard

            result.module = to_standard(result.module)
        except Exception:  # noqa: BLE001
            pass

    mlir_text = result.mlir_text or (module_to_text(result.module) if result.module is not None else "")
    return ConversionResult(
        mlir_text=mlir_text,
        module=result.module,
        path_taken=result.path_taken,
        output_type=output_type,
        diagnostics=list(result.diagnostics),
    )


def coverage_report(
    model: Any,
    example_inputs: tuple[Any, ...],
    *,
    quantization: QuantizationConfig | None = None,
) -> dict[str, Any]:
    """Capture the model and report op-coverage: which ops are unsupported and
    what the self-updating loop resolved for each.

    Returns a dict with the decomposition targets applied and the list of
    unsupported-op resolutions (target + verification status).
    """
    from m2m.capture.torch_export import capture_frontend_artifact

    if quantization is not None:
        model = apply_quantization(model, quantization)

    artifact = capture_frontend_artifact(model, example_inputs, quantization_config=quantization)
    return {
        "valid": artifact.validation.valid,
        "num_ops": getattr(artifact.validation, "num_ops", None),
        "decomposition_targets": list(artifact.decomposition_targets),
        "unsupported": [
            {
                "target": r.target,
                "strategy": getattr(r.classification, "strategy", None),
                "eager_reference_ok": getattr(r.verification, "eager_reference_ok", None),
            }
            for r in artifact.unsupported_resolutions
        ],
    }


__all__ = ["ConversionResult", "convert", "coverage_report"]
