"""torch-mlir bridge: nn.Module / ExportedProgram -> xDSL ModuleOp.

Mirrors the `hexagon-mlir` production pattern exactly (see
`tmp/hexagon-mlir/test/python/torch-mlir/utils.py:28-34` +
`qcom_hexagon_backend/backend/torch_mlir_hexagon_launcher.py`):

    torch_mlir.fx.export_and_import(model, *inputs,
                                    output_type="linalg-on-tensors")

The returned MLIR text is then parsed back into an xDSL ModuleOp so
the rest of CompGen's passes can operate on it.

When torch-mlir is not installed (no `cp312` wheel on PyPI today; it
ships as a source build), the bridge falls back to CompGen's own
``FXImporter``. The caller gets a diagnostic string telling them
which path was taken.

The bridge is deliberately thin: zero business logic, zero op-level
translation. The whole point is to delegate `ATen -> linalg` to
torch-mlir when available (which handles hundreds of ops correctly)
instead of us expanding our own decomposition table.

Usage:

    from m2m.capture.torch_mlir_bridge import bridge_fx_graph
    result = bridge_fx_graph(model, example_inputs)
    if result.module is not None:
        # run downstream passes on result.module
        ...
    else:
        raise RuntimeError(result.diagnostics)
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import structlog
import torch
from xdsl.dialects.builtin import ModuleOp
from xdsl.printer import Printer

log = structlog.get_logger()


@dataclass
class BridgeResult:
    """Outcome of running the FX->MLIR bridge.

    Attributes:
        module: the parsed xDSL ModuleOp, or ``None`` on failure.
        path_taken: ``"torch_mlir"`` when the torch-mlir path succeeded,
            ``"fx_importer"`` when the CompGen FXImporter fallback ran,
            ``"failed"`` when both paths failed.
        output_type: the ``output_type`` the torch-mlir path used
            (``"linalg-on-tensors"`` by default).
        mlir_text: raw MLIR text produced by torch-mlir (empty when
            the fallback was used).
        diagnostics: list of human-readable diagnostic strings.
    """

    module: ModuleOp | None = None
    path_taken: str = "failed"
    output_type: str = ""
    mlir_text: str = ""
    diagnostics: list[str] = field(default_factory=list)
    trace_dispositions: dict[str, Any] = field(default_factory=dict)


def _try_torch_mlir_import() -> Any:
    """Return the torch-mlir ``fx`` module, or ``None`` when unavailable.

    Lazily imports so CompGen packages that never call the bridge pay
    zero import cost.
    """
    try:
        from torch_mlir import fx as _fx  # type: ignore
    except ImportError:
        return None
    return _fx


def _parse_mlir_text_to_xdsl(mlir_text: str) -> ModuleOp | None:
    """Parse linalg-on-tensors MLIR text back into an xDSL ModuleOp.

    xDSL's Parser understands most of builtin/linalg/arith/tensor/func
    out of the box. Returns ``None`` if parsing fails -- typically
    because the MLIR text uses a dialect xDSL doesn't register by
    default.
    """
    from xdsl.context import Context
    from xdsl.dialects.arith import Arith
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.dialects.linalg import Linalg
    from xdsl.dialects.math import Math
    from xdsl.dialects.tensor import Tensor
    from xdsl.parser import Parser

    ctx = Context(allow_unregistered=True)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Arith)
    ctx.load_dialect(Func)
    ctx.load_dialect(Linalg)
    ctx.load_dialect(Math)
    ctx.load_dialect(Tensor)
    # torch-mlir's linalg-on-tensors output may include cf.assert / scf bounds checks.
    for _mod, _cls in (("cf", "Cf"), ("scf", "Scf"), ("ml_program", "MLProgram")):
        try:
            _m = __import__(f"xdsl.dialects.{_mod}", fromlist=[_cls])
            ctx.load_dialect(getattr(_m, _cls))
        except Exception:  # noqa: BLE001 - optional dialects
            pass

    # Register CompGen's own dialects so they round-trip if present.
    try:
        from m2m.ir.linalg_ext import LinalgExt
        from m2m.ir.quant import Quant
        from m2m.ir.tensor_ext import TensorExt

        ctx.load_dialect(LinalgExt)
        ctx.load_dialect(Quant)
        ctx.load_dialect(TensorExt)
    except Exception:
        pass  # Optional — bridge works without them.

    parser = Parser(ctx, mlir_text)
    try:
        module = parser.parse_module()
    except Exception as exc:  # noqa: BLE001
        log.warning("torch_mlir_bridge.parse_failed", error=str(exc))
        return None
    return module


def bridge_fx_graph(
    model: torch.nn.Module | Any,
    example_inputs: tuple[torch.Tensor, ...],
    *,
    func_name: str = "forward",
    output_type: str = "linalg-on-tensors",
    allow_fallback: bool = True,
    exported_program: Any | None = None,
    use_torch_mlir: bool = True,
    emit_named_ops: bool = False,
    weights_path: str | None = None,
    capture_trace: bool = False,
) -> BridgeResult:
    """Convert ``model`` + ``example_inputs`` into an xDSL ModuleOp.

    Args:
        model: a ``torch.nn.Module``. torch-mlir also accepts an
            ``ExportedProgram`` via the same API.
        example_inputs: the tuple of example tensors (same shapes the
            compiled artifact will be called with).
        func_name: name of the public func in the emitted MLIR.
        output_type: torch-mlir output dialect. ``"linalg-on-tensors"``
            is the right choice for CompGen (linalg is our downstream
            substrate). Pass ``"torch"`` for the higher-level Torch
            dialect.
        allow_fallback: when ``True`` (default), fall back to CompGen's
            ``FXImporter`` if torch-mlir is unavailable or its import
            fails. When ``False``, a torch-mlir failure becomes a hard
            error and the returned ``module`` is ``None``.
        exported_program: an already-captured (and ideally decomposed)
            ``torch.export.ExportedProgram``. When provided, it is fed
            directly to torch-mlir (and the FXImporter fallback) instead
            of re-exporting ``model``. Decomposing first lowers composite
            ops (e.g. ``aten.diff``) that torch-mlir would otherwise mark
            illegal and abort on -- this is what lets real LLMs/VLAs take
            the torch-mlir path instead of falling back wholesale.
    """
    result = BridgeResult(output_type=output_type)
    if capture_trace and exported_program is None:
        result.diagnostics.append("capture_trace requires the exact already-captured program")
        return result

    fx_module = _try_torch_mlir_import() if use_torch_mlir else None
    if not use_torch_mlir:
        result.diagnostics.append("torch-mlir path disabled (use_torch_mlir=False); using FXImporter")
    if fx_module is not None:
        try:
            if exported_program is not None:
                mlir_module = fx_module.export_and_import(
                    exported_program,
                    output_type=output_type,
                    func_name=func_name,
                )
            else:
                mlir_module = fx_module.export_and_import(
                    model,
                    *example_inputs,
                    output_type=output_type,
                    func_name=func_name,
                )
            # torch-mlir returns an MlirModule; serialize to text and
            # parse into xDSL.
            # Elide large constant tensors AND resource blobs (model weights) so the
            # emitted MLIR is compact structural IR, not gigabytes of weight data.
            mlir_text = mlir_module.operation.get_asm(
                binary=False,
                large_elements_limit=16,
                large_resource_limit=16,
                enable_debug_info=False,
            )
            # torch-mlir's text IS the authoritative output: it lowers hundreds
            # of ops straight to standard dialects (linalg/tensor/arith/...). Parsing
            # it back into an xDSL ModuleOp is best-effort convenience only -- xDSL's
            # parser doesn't register every standard op (e.g. tensor.extract_slice),
            # and a parse miss must NOT trigger a fallback to the FXImporter.
            if mlir_text:
                result.mlir_text = mlir_text
                result.path_taken = "torch_mlir"
                result.module = _parse_mlir_text_to_xdsl(mlir_text)  # may be None
                note = "parsed into xDSL" if result.module is not None else "xDSL re-parse skipped/failed (text is authoritative)"
                result.diagnostics.append(
                    f"torch-mlir produced {len(mlir_text)} chars of {output_type} MLIR; {note}"
                )
                log.info("torch_mlir_bridge.ok", path="torch_mlir", mlir_bytes=len(mlir_text),
                         xdsl_parsed=result.module is not None)
                return result
            result.diagnostics.append("torch-mlir produced no MLIR text")
        except Exception as exc:  # noqa: BLE001
            result.diagnostics.append(f"torch-mlir path raised: {exc}")
            log.warning("torch_mlir_bridge.torch_mlir_failed", error=str(exc))
    else:
        result.diagnostics.append("torch-mlir not installed; falling back to CompGen FXImporter")

    if not allow_fallback:
        result.diagnostics.append("allow_fallback=False; returning no module")
        return result

    # Fallback: use CompGen's FXImporter via torch.export capture.
    try:
        from m2m.capture.torch_export import capture_model
        from m2m.ir.import_fx import FXImporter

        exported = exported_program if exported_program is not None else capture_model(model, example_inputs)
        # Inline torch.no_grad() HOPs on whatever graph we're about to import (covers the
        # case where api.convert's pre-inlined exported_program was None and we re-exported).
        try:
            from m2m.ir.torchmlir_decomps import inline_set_grad_hops

            gm = getattr(exported, "graph_module", None)
            if gm is not None:
                for _ in range(8):  # bounded fixpoint (nested HOPs); guards against spin
                    if not inline_set_grad_hops(gm):
                        break
        except Exception:  # noqa: BLE001 - non-fatal
            pass
        importer = FXImporter(emit_named_ops=emit_named_ops)
        module = importer.import_graph(exported)
        result.trace_dispositions = dict(importer.trace_dispositions)
        errors = [d for d in importer.diagnostics if d.level == "error"]
        if errors:
            result.diagnostics.append(
                f"FXImporter fallback produced {len(errors)} errors: {[d.message for d in errors[:3]]}"
            )
            return result
        result.module = module
        result.path_taken = "fx_importer"
        # Externalize weights/buffers to safetensors (graph stays inspectable, data recoverable).
        if weights_path:
            try:
                from m2m.transforms.externalize import externalize_weights

                summary = externalize_weights(exported, module, weights_path)
                result.diagnostics.append(f"externalized {summary['weights']} weights -> {weights_path}")
            except Exception as exc:  # noqa: BLE001 - non-fatal
                result.diagnostics.append(f"weight externalization failed: {exc}")
        result.diagnostics.append(
            f"FXImporter fallback succeeded with "
            f"{importer.decomposed_count} decomposed ops, "
            f"{importer.opaque_count} opaque"
        )
        log.info(
            "torch_mlir_bridge.ok",
            path="fx_importer",
            decomposed=importer.decomposed_count,
            opaque=importer.opaque_count,
        )
        return result
    except Exception as exc:  # noqa: BLE001
        result.diagnostics.append(f"FXImporter fallback raised: {exc}")
        log.error("torch_mlir_bridge.both_failed", error=str(exc))
        return result


def bridge_fx_graph_or_raise(
    model: torch.nn.Module | Any,
    example_inputs: tuple[torch.Tensor, ...],
    **kwargs: Any,
) -> ModuleOp:
    """Raise-on-failure wrapper around :func:`bridge_fx_graph`."""
    result = bridge_fx_graph(model, example_inputs, **kwargs)
    if result.module is None:
        raise RuntimeError("FX -> xDSL bridge failed for both paths:\n  " + "\n  ".join(result.diagnostics))
    return result.module


# xDSL has no builtin Float8, so our shim types print as `!builtin_ext.f8E4M3FN`. On TEXT
# emission we render the MLIR-native spelling (`f8E4M3FN`) so the artifact a standard MLIR
# toolchain receives uses native f8 types; the custom type stays internal for IR manipulation.
# When xDSL gains native f8 (or we route fp8 through torch-mlir) this shim is dropped.
_NATIVE_F8 = {
    "!builtin_ext.f8E4M3FN": "f8E4M3FN",
    "!builtin_ext.f8E5M2": "f8E5M2",
    "!builtin_ext.f8E8M0FNU": "f8E8M0FNU",
}


def module_to_text(module: ModuleOp, *, generic: bool = False) -> str:
    """Pretty-print an xDSL ModuleOp as MLIR text, rendering shim fp8 types with their
    MLIR-native spelling (``f8E4M3FN`` etc.) for portability."""
    buf = io.StringIO()
    # Custom printers for e.g. tensor.empty historically discard attributes.
    # Traced artifacts must retain exact source IDs across serialization.
    Printer(stream=buf, print_generic_format=generic).print(module)
    text = buf.getvalue()
    for shim, native in _NATIVE_F8.items():
        if shim in text:
            text = text.replace(shim, native)
    return text


def torch_mlir_available() -> bool:
    """Whether the torch-mlir path can be used."""
    return _try_torch_mlir_import() is not None


__all__ = [
    "BridgeResult",
    "bridge_fx_graph",
    "bridge_fx_graph_or_raise",
    "module_to_text",
    "torch_mlir_available",
]
