"""In-process export protocol for non-Merlin runtimes.

External runtimes need the exact PyTorch objects used by capture, not enough metadata to try to
reconstruct a model later.  This module deliberately keeps the protocol in-process: a loader returns
its model and example inputs, then :func:`external_runtime_session` exposes the actual ``nn.Module``
objects, their tensor ABI, and the loader-authored semantic-session metadata.

There is no workload-name registry.  Ordinary (v1) captures become one executable program.  Deferred
multi-program captures (v2) implement an ``external_runtime_session()`` capability that returns their
real stages and tuple-index bindings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class ExternalRuntimeProgram:
    """One fixed-shape program offered to an external runtime exporter.

    ``inputs`` are the exact example tensors for ``module.forward``.  ``session`` uses loader tuple
    indices (not Merlin's weight-prefixed ABI indices), so it can drive the same streams and state
    updates for PyTorch-derived runtimes such as ExecuTorch.
    """

    name: str
    module: nn.Module
    inputs: tuple[torch.Tensor, ...]
    steps: int
    session: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("external-runtime program name must be non-empty")
        if not isinstance(self.module, nn.Module):
            raise TypeError(f"external-runtime program {self.name!r} has no nn.Module")
        if not isinstance(self.inputs, tuple) or any(
                not isinstance(value, torch.Tensor) for value in self.inputs):
            raise TypeError(
                f"external-runtime program {self.name!r} inputs must be a tuple of tensors")
        if int(self.steps) < 1:
            raise ValueError(f"external-runtime program {self.name!r} needs positive steps")

    def bundle_record(self) -> dict[str, Any]:
        """Return the legacy record accepted by ``write_multi_program_bundle``."""
        record: dict[str, Any] = {
            "name": self.name, "model": self.module, "inputs": self.inputs,
            "steps": int(self.steps),
        }
        if self.session is not None:
            record["session"] = dict(self.session)
        return record


@dataclass(frozen=True)
class ExternalRuntimeInput:
    program: str
    input_index: int


@dataclass(frozen=True)
class ExternalRuntimeOutput:
    program: str
    output_index: int


@dataclass(frozen=True)
class ExternalRuntimeInputBinding:
    """The reset value and semantic role of one program input."""

    target: ExternalRuntimeInput
    initial: torch.Tensor
    kind: Literal["static", "stream", "state"]
    name: str


@dataclass(frozen=True)
class ExternalRuntimeStream:
    """One value per semantic observation, applied before the target program call."""

    name: str
    target: ExternalRuntimeInput
    values: torch.Tensor


@dataclass(frozen=True)
class ExternalRuntimeRoute:
    """Copy a stage output to an input immediately after the source stage executes."""

    name: str
    source: ExternalRuntimeOutput
    target: ExternalRuntimeInput
    update: Literal["after_source"] = "after_source"


@dataclass(frozen=True)
class ExternalRuntimeInvocation:
    """An ordered program invocation with an explicit session cadence.

    Consecutive ``per_observation`` entries are executed in schedule order inside each observation;
    they are not each run to completion as separate loops.
    """

    program: str
    repeats: int
    cadence: Literal[
        "once_before_observations", "per_observation", "once_after_observations"
    ]
    timed: bool


@dataclass(frozen=True)
class ExternalRuntimeSession:
    """A complete semantic session over one or more actual PyTorch programs.

    ``metadata`` is the un-resolved loader contract.  In particular, cross-program bindings still
    use ``input_index`` because external runtimes expose only user inputs; Merlin resolves those to
    its weight-prefixed numeric ABI separately while writing a capture bundle.
    """

    version: int
    programs: tuple[ExternalRuntimeProgram, ...]
    metadata: Mapping[str, Any]
    observations: int = 0
    execution_schedule: tuple[ExternalRuntimeInvocation, ...] = ()
    streams: tuple[ExternalRuntimeStream, ...] = ()
    routes: tuple[ExternalRuntimeRoute, ...] = ()
    input_bindings: tuple[ExternalRuntimeInputBinding, ...] = ()
    observation_output: ExternalRuntimeOutput | None = None
    final_output: ExternalRuntimeOutput | None = None
    reset: Literal["restore_initial_inputs"] = "restore_initial_inputs"

    def __post_init__(self) -> None:
        if int(self.version) not in (1, 2):
            raise ValueError("external-runtime protocol version must be 1 or 2")
        if not self.programs:
            raise ValueError("external-runtime session needs at least one program")
        names = [program.name for program in self.programs]
        if len(set(names)) != len(names):
            raise ValueError("external-runtime program names must be unique")
        if int(self.version) == 1 and len(self.programs) != 1:
            raise ValueError("v1 external-runtime sessions contain exactly one program")
        if int(self.version) == 2:
            stages = [str(value) for value in self.metadata.get("stages", ())]
            if stages != names:
                raise ValueError("v2 metadata stages must exactly match program order")
            schedule = list(self.metadata.get("stage_schedule", ()) or ())
            expected = [(program.name, int(program.steps)) for program in self.programs]
            actual = [(str(row.get("name")), int(row.get("steps", 0))) for row in schedule]
            if actual != expected:
                raise ValueError("v2 stage schedule must exactly match program names and steps")
            self._validate_bindings()
        self._validate_driver_plan()

    def _validate_bindings(self) -> None:
        programs = {program.name: program for program in self.programs}
        for index, binding in enumerate(self.metadata.get("bindings", ()) or ()):
            source = dict(binding.get("from", {}) or {})
            target = dict(binding.get("to", {}) or {})
            source_name, target_name = str(source.get("program", "")), str(
                target.get("program", ""))
            if source_name not in programs or target_name not in programs:
                raise ValueError(f"external-runtime binding {index} names an unknown program")
            output_index = int(source.get("output_index", -1))
            input_index = int(target.get("input_index", -1))
            if output_index < 0:
                raise ValueError(f"external-runtime binding {index} has invalid output_index")
            if input_index < 0 or input_index >= len(programs[target_name].inputs):
                raise ValueError(f"external-runtime binding {index} has invalid input_index")

    def _validate_driver_plan(self) -> None:
        programs = {program.name: program for program in self.programs}
        if int(self.observations) < 1:
            raise ValueError("external-runtime session needs positive observations")
        if not self.execution_schedule:
            raise ValueError("external-runtime session needs an execution schedule")
        per_observation = 0
        for invocation in self.execution_schedule:
            if invocation.program not in programs or invocation.repeats < 1:
                raise ValueError("external-runtime execution schedule is invalid")
            if invocation.cadence == "per_observation":
                per_observation += 1
                if invocation.repeats != self.observations:
                    raise ValueError("per-observation invocation count differs from observations")
        if per_observation < 1:
            raise ValueError("external-runtime session needs a per-observation program")
        all_inputs = {
            ExternalRuntimeInput(program.name, index)
            for program in self.programs for index in range(len(program.inputs))
        }
        if {binding.target for binding in self.input_bindings} != all_inputs:
            raise ValueError("external-runtime input bindings must cover every program input")
        if len({binding.target for binding in self.input_bindings}) != len(self.input_bindings):
            raise ValueError("external-runtime input bindings contain duplicate targets")
        stream_targets = {stream.target for stream in self.streams}
        route_targets = {route.target for route in self.routes}
        if stream_targets & route_targets:
            raise ValueError("an external-runtime input cannot be both stream and state")
        for stream in self.streams:
            if stream.target not in all_inputs or stream.values.shape[0] != self.observations:
                raise ValueError("external-runtime stream shape/target is invalid")
            initial = programs[stream.target.program].inputs[stream.target.input_index]
            if tuple(stream.values.shape[1:]) != tuple(initial.shape):
                raise ValueError("external-runtime stream element shape differs from target input")
            if stream.values.dtype != initial.dtype:
                raise ValueError("external-runtime stream dtype differs from target input")
        for route in self.routes:
            if route.source.program not in programs or route.target not in all_inputs:
                raise ValueError("external-runtime route endpoint is invalid")
        for selector in (self.observation_output, self.final_output):
            if selector is None or selector.program not in programs or selector.output_index < 0:
                raise ValueError("external-runtime output selector is invalid")

    @property
    def kind(self) -> str:
        return str(self.metadata.get("kind", "single_forward"))

    @property
    def paper_ready(self) -> bool:
        return bool(self.metadata.get("paper_ready", False))

    @property
    def stage_schedule(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.metadata.get("stage_schedule", ()) or ())

    @property
    def bindings(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.metadata.get("bindings", ()) or ())

    @property
    def parameters(self) -> Mapping[str, Any]:
        return dict(self.metadata.get("parameters", {}) or {})

    @property
    def provenance(self) -> Mapping[str, Any]:
        return dict(self.metadata.get("provenance", {}) or {})

    def bundle_programs(self) -> list[dict[str, Any]]:
        """Return legacy multi-program writer records without changing object identity."""
        return [program.bundle_record() for program in self.programs]


def _semantic_steps(session: Mapping[str, Any]) -> int:
    stream_steps = {
        int(stream["values"].shape[0])
        for stream in session.get("streams", ()) or ()
    }
    if stream_steps:
        if len(stream_steps) != 1 or next(iter(stream_steps)) < 1:
            raise ValueError("semantic streams must have one common positive step count")
        return next(iter(stream_steps))
    if int(session.get("steps", 0)) > 0:
        return int(session["steps"])
    schedule_steps = {
        int(row.get("steps", 0)) for row in session.get("stage_schedule", ()) or ()
    }
    if len(schedule_steps) == 1 and next(iter(schedule_steps)) > 0:
        return next(iter(schedule_steps))
    raise ValueError("single-program semantic session has no unambiguous positive step count")


def _driver_plan(version: int, programs: tuple[ExternalRuntimeProgram, ...],
                 metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize v1/v2 authoring metadata into a runtime-executable tensor routing plan."""
    by_name = {program.name: program for program in programs}
    quality_program = (str(metadata.get("quality_program")) if version == 2
                       else programs[0].name)
    if quality_program not in by_name:
        raise ValueError("external-runtime quality program is absent")
    quality_session = by_name[quality_program].session or metadata
    observations = _semantic_steps(quality_session)
    quality = dict(quality_session.get("quality", {}) or {})
    observation_output = ExternalRuntimeOutput(
        quality_program, int(quality.get("output_index", 0)))
    final_output = ExternalRuntimeOutput(programs[-1].name, int(
        metadata.get("final_output_index", 0)))

    invocations: list[ExternalRuntimeInvocation] = []
    if version == 1:
        invocations.append(ExternalRuntimeInvocation(
            programs[0].name, observations, "per_observation", True))
    else:
        quality_position = [program.name for program in programs].index(quality_program)
        schedule = list(metadata.get("stage_schedule", ()) or ())
        for position, (program, row) in enumerate(zip(programs, schedule, strict=True)):
            repeats = int(row["steps"])
            cadence = ("per_observation" if repeats == observations else
                       "once_before_observations" if position < quality_position else
                       "once_after_observations")
            if position == quality_position and cadence != "per_observation":
                raise ValueError("quality stage steps differ from semantic observations")
            invocations.append(ExternalRuntimeInvocation(
                program.name, repeats, cadence, bool(row.get("timed", True))))

    streams: list[ExternalRuntimeStream] = []
    routes: list[ExternalRuntimeRoute] = []
    for program in programs:
        if program.session is None:
            continue
        for index, stream in enumerate(program.session.get("streams", ()) or ()):
            values = stream["values"]
            values = values if isinstance(values, torch.Tensor) else torch.as_tensor(values)
            streams.append(ExternalRuntimeStream(
                str(stream.get("name") or stream.get("key") or f"stream{index}"),
                ExternalRuntimeInput(program.name, int(stream["input_index"])), values))
        for index, state in enumerate(program.session.get("states", ()) or ()):
            routes.append(ExternalRuntimeRoute(
                str(state.get("name") or f"state{index}"),
                ExternalRuntimeOutput(program.name, int(state["output_index"])),
                ExternalRuntimeInput(program.name, int(state["input_index"]))))
    for index, binding in enumerate(metadata.get("bindings", ()) or ()):
        source, target = dict(binding["from"]), dict(binding["to"])
        routes.append(ExternalRuntimeRoute(
            str(binding.get("name") or f"binding{index}"),
            ExternalRuntimeOutput(str(source["program"]), int(source["output_index"])),
            ExternalRuntimeInput(str(target["program"]), int(target["input_index"]))))

    stream_names = {stream.target: stream.name for stream in streams}
    route_names: dict[ExternalRuntimeInput, str] = {}
    for route in routes:
        route_names[route.target] = route.name
    bindings = []
    for program in programs:
        for input_index, initial in enumerate(program.inputs):
            endpoint = ExternalRuntimeInput(program.name, input_index)
            if endpoint in stream_names:
                kind, name = "stream", stream_names[endpoint]
            elif endpoint in route_names:
                kind, name = "state", route_names[endpoint]
            else:
                kind, name = "static", f"{program.name}.input{input_index}"
            bindings.append(ExternalRuntimeInputBinding(endpoint, initial, kind, name))
    return {
        "observations": observations,
        "execution_schedule": tuple(invocations),
        "streams": tuple(streams),
        "routes": tuple(routes),
        "input_bindings": tuple(bindings),
        "observation_output": observation_output,
        "final_output": final_output,
    }


def make_external_runtime_session(*, version: int,
                                  programs: tuple[ExternalRuntimeProgram, ...],
                                  metadata: Mapping[str, Any]) -> ExternalRuntimeSession:
    """Construct and validate a normalized external-runtime session."""
    metadata = dict(metadata)
    return ExternalRuntimeSession(
        version=version, programs=programs, metadata=metadata,
        **_driver_plan(version, programs, metadata))


def external_runtime_session(model: Any, inputs: tuple[torch.Tensor, ...], *,
                             session: Mapping[str, Any] | None = None) \
        -> ExternalRuntimeSession:
    """Expose a loader result to external runtimes without dispatching on workload names.

    Deferred capture objects advertise the zero-argument ``external_runtime_session`` capability.
    An ordinary ``nn.Module`` is represented as one program; its loader's ``get_session_spec``
    result should be passed as ``session``.  A missing session is still useful for one-shot runtime
    smoke tests and is represented as a one-step ``single_forward`` contract.
    """
    provider = getattr(model, "external_runtime_session", None)
    if callable(provider):
        if tuple(inputs):
            raise ValueError("deferred external-runtime captures own their stage inputs")
        if session is not None:
            raise ValueError("deferred external-runtime captures own their session metadata")
        result = provider()
        if not isinstance(result, ExternalRuntimeSession):
            raise TypeError("external_runtime_session capability returned an invalid object")
        return result

    if not isinstance(model, nn.Module):
        raise TypeError("loader result is neither an nn.Module nor a deferred capture provider")
    tensor_inputs = tuple(inputs)
    metadata = dict(session) if session is not None else {
        "kind": "single_forward", "paper_ready": False, "stages": ["forward"],
        "stage_schedule": [
            {"name": "forward", "steps": 1, "execution": "external_runtime", "timed": True}
        ],
        "parameters": {}, "states": [], "streams": [],
        "quality": {"key": "output0", "output_index": 0}, "provenance": {},
    }
    steps = _semantic_steps(metadata)
    program = ExternalRuntimeProgram(
        name="forward", module=model, inputs=tensor_inputs, steps=steps,
        session=metadata if session is not None else None)
    return make_external_runtime_session(
        version=1, programs=(program,), metadata=metadata)
