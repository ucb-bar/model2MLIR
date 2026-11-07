"""Observed Torch CPU facts for source-arithmetic capture provenance.

These are observations of the process producing an artifact, not proof that
every imported dependency or model source file is sealed.
"""

from __future__ import annotations

from collections.abc import Mapping


def observed_torch_cpu() -> dict[str, str] | None:
    """Return the selected wheel's commit and CPU dispatch when both are known."""
    import torch

    commit = getattr(torch.version, "git_version", None)
    capability_fn = getattr(torch.backends.cpu, "get_cpu_capability", None)
    if not isinstance(commit, str) or not commit or not callable(capability_fn):
        return None
    try:
        capability = capability_fn()
    except (RuntimeError, TypeError, ValueError):
        return None
    if not isinstance(capability, str) or not capability:
        return None
    return {
        "torch_version": str(torch.__version__),
        "torch_git_commit": commit,
        "cpu_capability": capability,
    }


def cpu_inputs_and_state(model: object, inputs: object) -> bool:
    """Conservatively identify a capture selected for CPU tensor execution.

    This checks the observed tensor inputs and module state. It does not inspect
    arbitrary Python inside ``forward`` or prove each internal kernel's
    dispatch; the bundle writer also checks the actual golden output device.
    """
    import torch

    if isinstance(model, torch.export.ExportedProgram):
        model = model.module()
    if not isinstance(model, torch.nn.Module):
        return False

    tensors: list[torch.Tensor] = []

    def collect(value: object) -> bool:
        if isinstance(value, torch.Tensor):
            tensors.append(value)
            return True
        if isinstance(value, Mapping):
            return all(collect(part) for part in value.values())
        if isinstance(value, (tuple, list)):
            return all(collect(part) for part in value)
        return value is None or isinstance(value, (bool, int, float, str))

    if not collect(inputs) or not tensors:
        return False
    tensors.extend(model.parameters())
    tensors.extend(model.buffers())
    return all(tensor.device.type == "cpu" for tensor in tensors)


def cpu_outputs(value: object) -> bool:
    """Require every observed top-level golden tensor to reside on CPU."""
    import torch

    outputs = value if isinstance(value, (tuple, list)) else (value,)
    return bool(outputs) and all(
        isinstance(output, torch.Tensor) and output.device.type == "cpu"
        for output in outputs
    )
