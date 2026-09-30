"""m2m: a self-updating PyTorch / torchAO -> MLIR frontend.

Built on torch-mlir (primary lowering path) with a decomposition-based fallback
and a self-updating op-coverage loop that grows support for new and torchAO ops.
Emits linalg-on-tensors by default.

Quick start:

    import m2m
    result = m2m.convert(model, (example_input,))
    print(result.mlir_text)        # linalg-on-tensors MLIR
    print(result.path_taken)       # "torch_mlir" or "fx_importer"

Top-level names are resolved **lazily** (PEP 562): importing ``m2m`` — or any
submodule such as the pure-xDSL ``m2m.ir.quant`` dialect — does not eagerly pull
torch / torch-mlir. The heavy dependency is imported only when a torch-bound
symbol (``convert``, ``coverage_report``, ...) is first accessed. This lets a
consumer that only needs the quantization dialect (e.g. the Merlin frontend
registering ``quant_ext`` for parsing) import it without a torch install.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.0.1"

# Public name -> submodule that defines it (imported on first access).
_LAZY: dict[str, str] = {
    "ConversionResult": "m2m.api",
    "convert": "m2m.api",
    "coverage_report": "m2m.api",
    "capture_frontend_snapshot": "m2m.capture.trace",
    "materialize_frontend_precision": "m2m.capture.trace",
    "torch_mlir_available": "m2m.capture.torch_mlir_bridge",
    "expand_to_linalg": "m2m.transforms",
    "to_standard": "m2m.transforms",
}

__all__ = [
    "ConversionResult",
    "__version__",
    "convert",
    "coverage_report",
    "capture_frontend_snapshot",
    "materialize_frontend_precision",
    "expand_to_linalg",
    "to_standard",
    "torch_mlir_available",
]


def __getattr__(name: str):  # PEP 562
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'm2m' has no attribute {name!r}")
    obj = getattr(importlib.import_module(target), name)
    globals()[name] = obj  # cache so subsequent access is a plain attribute lookup
    return obj


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


if TYPE_CHECKING:  # static analysers / IDEs still see the real symbols
    from m2m.api import ConversionResult, convert, coverage_report
    from m2m.capture.torch_mlir_bridge import torch_mlir_available
    from m2m.capture.trace import capture_frontend_snapshot
    from m2m.transforms import expand_to_linalg, to_standard
