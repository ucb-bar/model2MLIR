"""Explicit shared-model factories preserve one selected source conversion."""

import pytest
from torch import nn

from m2m.capture.external_runtime import (
    ExternalRuntimeSharedModel,
    external_runtime_shared_model,
)


def test_capability_is_explicit_and_preserves_identity():
    shared = nn.Linear(4, 4)
    calls = []

    def make_session(selected):
        calls.append(selected)
        return selected

    class Capture:
        def external_runtime_shared_model(self):
            return ExternalRuntimeSharedModel(shared, make_session, ("first", "step"))

    actual = external_runtime_shared_model(Capture())
    selected = nn.Linear(4, 4)
    assert actual.model is shared
    assert actual.make_session(selected) is selected
    assert calls == [selected]


@pytest.mark.parametrize("names", [(), ("x", "x"), ("",), ["x"]])
def test_ambiguous_or_absent_shared_programs_refuse(names):
    with pytest.raises(ValueError):
        ExternalRuntimeSharedModel(nn.Linear(1, 1), lambda model: model, names)


def test_no_capability_or_wrong_protocol_refuses():
    with pytest.raises(TypeError, match="requires external_runtime_shared_model"):
        external_runtime_shared_model(nn.Linear(1, 1))

    class Incorrect:
        def external_runtime_shared_model(self):
            return {"model": nn.Linear(1, 1)}

    with pytest.raises(TypeError, match="wrong protocol"):
        external_runtime_shared_model(Incorrect())
