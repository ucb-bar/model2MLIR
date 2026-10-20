"""Session reference provenance follows the actual supplied-reference branch."""

import numpy as np
import pytest
import torch
import yaml

from m2m.capture.bundle import write_session_artifacts


class Step(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return value + 1


def _write(tmp_path, *, provided, paper_ready):
    module = Step()
    session = {
        "kind": "frames",
        "paper_ready": paper_ready,
        "stages": ["step"],
        "states": [],
        "streams": [
            {
                "name": "frame",
                "input_index": 0,
                "key": "frames",
                "values": torch.zeros(2, 1),
            }
        ],
        "quality": {
            "key": "values",
            "output_index": 0,
            "reference": "eager_fp32",
            "observation": {
                "status": "observed"
            },  # Caller labels cannot replace the branch.
        },
    }
    write_session_artifacts(
        module,
        (torch.zeros(1),),
        tmp_path,
        manifest={"0": {"kind": "input", "name": "value"}},
        input_order={"value": 0},
        session=session,
        quality_reference=np.ones((2, 1), dtype=np.float32) if provided else None,
    )
    assert module.calls == 2  # Provenance never triggers another forward.
    return yaml.safe_load((tmp_path / "session_contract.yaml").read_bytes())


@pytest.mark.parametrize("paper_ready", [False, True])
def test_absent_quality_keeps_correctness_but_does_not_claim_fp_reference(
    tmp_path, paper_ready
):
    contract = _write(tmp_path, provided=False, paper_ready=paper_ready)
    quality = contract["quality"]
    assert quality["reference"] == "unavailable"
    assert quality["observation"] == {
        "schema": "m2m.session_quality_observation.v1",
        "status": "unavailable",
        "origin": "correctness_fallback",
    }
    assert contract["correctness"]["reference"] == "eager_same_precision"
    with (
        np.load(tmp_path / "session_goldens.npz") as a,
        np.load(tmp_path / "session_quality_fp32.npz") as b,
    ):
        np.testing.assert_array_equal(a["values"], b["values"])


def test_supplied_quality_can_equal_correctness_and_remains_separate_observation(
    tmp_path,
):
    contract = _write(tmp_path, provided=True, paper_ready=False)
    quality = contract["quality"]
    assert quality["reference"] == "eager_fp32"
    assert quality["reference_sha256"] == contract["correctness"]["reference_sha256"]
    assert quality["observation"] == {
        "schema": "m2m.session_quality_observation.v1",
        "status": "observed",
        "origin": "provided_quality_reference",
    }
