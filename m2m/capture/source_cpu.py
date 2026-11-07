"""Observed Torch CPU facts for source-arithmetic capture provenance.

These are observations of the process producing an artifact, not proof that
every imported dependency or model source file is sealed.
"""

from __future__ import annotations


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
