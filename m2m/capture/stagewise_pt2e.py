"""Fail-closed, named PT2E capture for method-based shared-model sessions.

PT2E exports each stage's ``forward`` and returns a separate GraphModule.  The
original Python module cannot be shared between those GraphModules.  This module
instead binds every frozen int8 weight to the *same source Parameter object* via
the pre-quantization export graph, verifies its numeric meaning, and requires
identical frozen encodings wherever that Parameter is used by multiple stages.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from torch import nn

from m2m.capture.external_runtime import (
    ExternalRuntimeProgram,
    ExternalRuntimeSession,
    make_external_runtime_session,
)
from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization


_CONTRACTIONS = {torch.ops.aten.linear.default, torch.ops.aten.conv2d.default}


def _tensor_sha256(value: torch.Tensor) -> str:
    import hashlib

    tensor = value.detach().cpu().contiguous()
    if tensor.layout != torch.strided:
        raise ValueError("stage-wise weight binding requires strided tensors")
    return hashlib.sha256(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).cast("B")).hexdigest()


def _source_weights(program: ExternalRuntimeProgram, shared: nn.Module,
                    source_names: Mapping[int, str]) -> list[tuple[Any, str, torch.Tensor]]:
    exported = torch.export.export(program.module.eval(), program.inputs)
    parameters = exported.graph_signature.inputs_to_parameters
    nodes = []
    for node in exported.graph.nodes:
        if node.op != "call_function" or node.target not in _CONTRACTIONS:
            continue
        if len(node.args) < 2 or getattr(node.args[1], "op", None) != "placeholder":
            raise ValueError(
                f"stage {program.name!r} has a transformed contraction weight; "
                "source-parameter provenance cannot be proved")
        stage_name = parameters.get(str(node.args[1].target))
        if stage_name is None:
            raise ValueError(f"stage {program.name!r} contraction weight is not a parameter")
        parameter = program.module.get_parameter(stage_name)
        canonical = source_names.get(id(parameter))
        if canonical is None or shared.get_parameter(canonical) is not parameter:
            raise ValueError(
                f"stage {program.name!r} weight {stage_name!r} is not the shared source parameter")
        nodes.append((node.target, canonical, parameter))
    if not nodes:
        raise ValueError(f"stage {program.name!r} has no PT2E-supported contractions")
    return nodes


def _frozen_weights(program: str, quantized: nn.Module) -> list[tuple[Any, str, torch.Tensor,
                                                                        torch.Tensor, torch.Tensor, int]]:
    nodes = []
    for node in quantized.graph.nodes:
        if node.op != "call_function" or node.target not in _CONTRACTIONS:
            continue
        weight = node.args[1] if len(node.args) > 1 else None
        if (weight is None or weight.op != "call_function" or
                weight.target != torch.ops.quantized_decomposed.dequantize_per_channel.default or
                len(weight.args) < 4 or any(
                    getattr(arg, "op", None) != "get_attr" for arg in weight.args[:3])):
            raise ValueError(f"stage {program!r} has a contraction without a frozen per-channel int8 weight")
        frozen, scale, zero = (getattr(quantized, str(arg.target)) for arg in weight.args[:3])
        if not isinstance(frozen, torch.Tensor) or frozen.dtype != torch.int8:
            raise ValueError(f"stage {program!r} has a non-int8 frozen weight")
        axis = int(weight.args[3])
        nodes.append((node.target, str(weight.args[0].target), frozen, scale, zero, axis))
    frozen_names = {name for _, name, *_ in nodes}
    all_int8 = {name for name, value in quantized.named_buffers() if value.dtype == torch.int8}
    if frozen_names != all_int8:
        raise ValueError(f"stage {program!r} has unmapped int8 buffers")
    return nodes


def _verified_weight_record(stage: str, source: tuple[Any, str, torch.Tensor],
                            frozen: tuple[Any, str, torch.Tensor, torch.Tensor, torch.Tensor, int]) -> dict:
    source_op, source_name, parameter = source
    frozen_op, frozen_name, codes, scales, zero_points, axis = frozen
    if source_op != frozen_op or axis != 0 or tuple(parameter.shape) != tuple(codes.shape):
        raise ValueError(f"stage {stage!r} frozen weight topology differs from source")
    if (scales.ndim != 1 or zero_points.ndim != 1 or
            scales.numel() != codes.shape[axis] or zero_points.numel() != codes.shape[axis]):
        raise ValueError(f"stage {stage!r} has an invalid per-channel weight encoding")
    scales_cpu = scales.detach().cpu().to(torch.float64)
    zero_cpu = zero_points.detach().cpu().to(torch.float64)
    codes_cpu = codes.detach().cpu()
    source_cpu = parameter.detach().cpu()
    if not torch.isfinite(scales_cpu).all() or (scales_cpu <= 0).any():
        raise ValueError(f"stage {stage!r} has an invalid weight scale")
    if (codes_cpu < -127).any():
        raise ValueError(f"stage {stage!r} uses an out-of-policy int8 weight code")
    # Bound temporary memory even for a large VLA projection: verify at most
    # 64 output channels at a time, not several fp64 copies of a full matrix.
    for start in range(0, codes.shape[0], 64):
        stop = min(start + 64, codes.shape[0])
        shape = [1] * codes.ndim
        shape[axis] = stop - start
        step = scales_cpu[start:stop].reshape(shape)
        zero = zero_cpu[start:stop].reshape(shape)
        source_value = source_cpu[start:stop].to(torch.float64)
        code_values = codes_cpu[start:stop]
        if not torch.isfinite(source_value).all():
            raise ValueError(f"stage {stage!r} has a non-finite source weight")
        decoded = (code_values.to(torch.float64) - zero) * step
        if not torch.all(torch.abs(decoded - source_value) <= (step / 2 + 1e-6)):
            raise ValueError(f"stage {stage!r} frozen weight does not represent source {source_name!r}")
        # Do not insist on bitwise agreement with a second quantize operation:
        # torchao's fp32 kernel can choose the adjacent code at half-step ties.
        # The decoded-value error bound plus exact stage-to-stage encoding hash
        # is the semantic identity proof used here.
    return {
        "stage": stage, "source_parameter": source_name,
        "source_sha256": _tensor_sha256(parameter),
        "frozen_buffer": frozen_name, "frozen_sha256": _tensor_sha256(codes),
        "scale_sha256": _tensor_sha256(scales),
        "zero_point_sha256": _tensor_sha256(zero_points),
        "shape": list(codes.shape), "dtype": "torch.int8", "channel_axis": axis,
    }


def _stage_abis(session: ExternalRuntimeSession) -> tuple[dict, list[dict]]:
    from torch.utils._pytree import tree_flatten

    by_name = {program.name: program for program in session.programs}
    outputs = {}
    abis = {}
    with torch.no_grad():
        for program in session.programs:
            values, _ = tree_flatten(program.module(*program.inputs))
            if not values or any(not isinstance(value, torch.Tensor) for value in values):
                raise ValueError(f"stage {program.name!r} has a non-tensor output ABI")
            outputs[program.name] = values
            abis[program.name] = {
                "inputs": [{"dtype": str(value.dtype), "shape": list(value.shape)}
                           for value in program.inputs],
                "outputs": [{"dtype": str(value.dtype), "shape": list(value.shape)}
                            for value in values],
            }
    routes = []
    for route in session.routes:
        source = outputs[route.source.program]
        if route.source.output_index >= len(source):
            raise ValueError(f"route {route.name!r} selects an absent output")
        value = source[route.source.output_index]
        target = by_name[route.target.program].inputs[route.target.input_index]
        if value.dtype != target.dtype or tuple(value.shape) != tuple(target.shape):
            raise ValueError(f"route {route.name!r} has a post-PT2E dtype/shape mismatch")
        routes.append({
            "name": route.name,
            "from": {"program": route.source.program, "output_index": route.source.output_index},
            "to": {"program": route.target.program, "input_index": route.target.input_index},
            "dtype": str(value.dtype), "shape": list(value.shape),
        })
    return abis, routes


@dataclass(frozen=True)
class StagewisePT2ESession:
    """Verified converted stages and provenance, still only diagnostic source closure."""

    shared_model: nn.Module
    session: ExternalRuntimeSession
    quant: QuantizationConfig
    shared_programs: tuple[str, ...]
    source_inventory: tuple[dict, ...]
    frozen_weights: tuple[dict, ...]
    program_abis: dict
    routes: tuple[dict, ...]
    calibration_counts: dict[str, int]
    source_snapshot: dict[str, Any] | None = None


def quantize_stagewise_pt2e_session(
    shared_model: nn.Module,
    source_session: ExternalRuntimeSession,
    *,
    quant: QuantizationConfig,
    shared_programs: Sequence[str],
    calibration_inputs: Mapping[str, Iterable[tuple[torch.Tensor, ...]]],
    source_snapshot: Mapping[str, Any] | None = None,
) -> StagewisePT2ESession:
    """PT2E-convert named stages with explicit calibration and verified source weights.

    All selected stages must refer to the exact ``shared_model``.  A source
    parameter is considered shared only when converted encodings are byte-equal
    across its stages; separate GraphModules are never claimed to share storage.
    Unsupported weight rewrites (including Conv+BN folding) fail closed.
    """
    from m2m.capture.bundle import _shared_tensor_inventory

    snapshot = None
    if source_snapshot is not None:
        from m2m.capture.source_closure import verify_source_snapshot

        verify_source_snapshot(source_snapshot)
        package = next(owner["root"] for owner in source_snapshot["owners"]
                       if owner["role"] == "m2m_package")
        if Path(package) != Path(__file__).resolve().parents[1]:
            raise ValueError("source snapshot M2M owner is not the executing package")
        snapshot = deepcopy(source_snapshot)

    if not isinstance(shared_model, nn.Module) or not isinstance(source_session, ExternalRuntimeSession):
        raise TypeError("stage-wise PT2E needs a source nn.Module and external-runtime session")
    if source_session.version != 2 or quant.scheme != "int8_static_act_int8_weight":
        raise ValueError("stage-wise PT2E requires a v2 session and static W8A8 policy")
    names = tuple(shared_programs)
    by_name = {program.name: program for program in source_session.programs}
    if not names or len(set(names)) != len(names) or any(name not in by_name for name in names):
        raise ValueError("shared_programs must name distinct source stages")
    if set(calibration_inputs) != set(names):
        raise ValueError("explicit calibration inputs are required for every shared stage")
    inventory = tuple(_shared_tensor_inventory(shared_model))
    source_names = {id(param): name for name, param in shared_model.named_parameters()}
    quality_sessions: dict[str, Mapping[str, Any] | None] = {}
    for program in source_session.programs:
        child = dict(program.session or {})
        quality = dict(child.get("quality", {}) or {})
        if child.get("paper_ready") is True and quality.get("reference_values") is None:
            from m2m.capture.bundle import capture_session_trajectory

            quality["reference_values"] = capture_session_trajectory(
                program.module, program.inputs, child)
            child["quality"] = quality
        quality_sessions[program.name] = child if program.session is not None else None
    converted = []
    records = []
    counts = {}
    for program in source_session.programs:
        if program.name not in names:
            converted.append(ExternalRuntimeProgram(
                program.name, program.module, program.inputs, program.steps,
                quality_sessions[program.name]))
            continue
        if not any(module is shared_model for module in program.module.modules()):
            raise ValueError(f"stage {program.name!r} does not contain the shared source model")
        samples = tuple(islice(iter(calibration_inputs[program.name]),
                              quant.calibration_samples))
        if not samples or any(not isinstance(sample, tuple) or any(
                not isinstance(value, torch.Tensor) for value in sample) for sample in samples):
            raise ValueError(f"stage {program.name!r} needs explicit tensor calibration tuples")
        source_weights = _source_weights(program, shared_model, source_names)
        module = apply_quantization(program.module, quant, example_inputs=program.inputs,
                                    calibration_inputs=samples)
        frozen = _frozen_weights(program.name, module)
        if len(source_weights) != len(frozen):
            raise ValueError(f"stage {program.name!r} changed contraction count during PT2E")
        records.extend(_verified_weight_record(program.name, source, target)
                       for source, target in zip(source_weights, frozen, strict=True))
        counts[program.name] = module._m2m_quantization_stats["calibration_samples"]
        converted.append(ExternalRuntimeProgram(program.name, module, program.inputs,
                                                program.steps, quality_sessions[program.name]))
    if tuple(_shared_tensor_inventory(shared_model)) != inventory:
        raise ValueError("source shared weights changed during stage-wise PT2E")
    by_source: dict[str, tuple[str, str, str]] = {}
    by_buffer: dict[tuple[str, str], str] = {}
    for record in records:
        buffer_key = (record["stage"], record["frozen_buffer"])
        old_source = by_buffer.setdefault(buffer_key, record["source_parameter"])
        if old_source != record["source_parameter"]:
            raise ValueError(f"frozen buffer {buffer_key!r} has ambiguous source provenance")
        identity = (record["frozen_sha256"], record["scale_sha256"],
                    record["zero_point_sha256"])
        old = by_source.setdefault(record["source_parameter"], identity)
        if old != identity:
            raise ValueError(f"shared parameter {record['source_parameter']!r} has unequal stage encodings")
    session = make_external_runtime_session(version=2, programs=tuple(converted),
                                            metadata=source_session.metadata)
    abis, routes = _stage_abis(session)
    if snapshot is not None:
        verify_source_snapshot(snapshot)
    return StagewisePT2ESession(shared_model, session, quant, names, inventory,
                               tuple(records), abis, tuple(routes), counts, snapshot)
