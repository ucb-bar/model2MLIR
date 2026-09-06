"""Model capture: torch.export -> ExportedProgram, torchAO quantization,
unsupported-op recovery (the self-updating coverage loop).

Extracted from CompGen's capture stage; the CompGen-specific inductor harvest
was left behind during extraction.
"""

from __future__ import annotations

from m2m.capture.dynamo_baseline import (
    BaselineReport,
    DynamoReport,
    GuardObservation,
    collect_diagnostics,
    compile_baseline,
)
from m2m.capture.external_runtime import (
    ExternalRuntimeInput,
    ExternalRuntimeInputBinding,
    ExternalRuntimeInvocation,
    ExternalRuntimeOutput,
    ExternalRuntimeProgram,
    ExternalRuntimeRoute,
    ExternalRuntimeSession,
    ExternalRuntimeStream,
    external_runtime_session,
    make_external_runtime_session,
)
from m2m.capture.torch_export import (
    CaptureArtifact,
    ExportValidation,
    RangeConstraint,
    capture_dynamo_partitions,
    capture_frontend,
    capture_frontend_artifact,
    capture_model,
    validate_export,
)
from m2m.capture.torch_mlir_bridge import (
    BridgeResult,
    bridge_fx_graph,
    bridge_fx_graph_or_raise,
    module_to_text,
    torch_mlir_available,
)
from m2m.capture.torchao_pipeline import (
    AccuracyReport,
    QuantizationConfig,
    apply_quantization,
    verify_quant_accuracy,
)

__all__ = [
    "AccuracyReport",
    "BaselineReport",
    "BridgeResult",
    "CaptureArtifact",
    "DynamoReport",
    "ExternalRuntimeInput",
    "ExternalRuntimeInputBinding",
    "ExternalRuntimeInvocation",
    "ExternalRuntimeOutput",
    "ExternalRuntimeProgram",
    "ExternalRuntimeRoute",
    "ExternalRuntimeSession",
    "ExternalRuntimeStream",
    "ExportValidation",
    "GuardObservation",
    "QuantizationConfig",
    "RangeConstraint",
    "apply_quantization",
    "bridge_fx_graph",
    "bridge_fx_graph_or_raise",
    "capture_dynamo_partitions",
    "capture_frontend",
    "capture_frontend_artifact",
    "capture_model",
    "collect_diagnostics",
    "compile_baseline",
    "external_runtime_session",
    "make_external_runtime_session",
    "module_to_text",
    "torch_mlir_available",
    "validate_export",
    "verify_quant_accuracy",
]
