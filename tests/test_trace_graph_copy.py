"""Copied graph/metadata is owned; immutable tensor state retains real identity."""

import copy

import pytest
import torch

from m2m.capture.subclass_trace import bind, observe
from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization
from m2m.capture.trace import _graph_modules, snapshot_exported_program
from m2m.capture.trace_copy import copy_trace_program
from m2m.ir.torchmlir_decomps import inline_set_grad_hops

pytest.importorskip("torchao")


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32, bias=False)

    def forward(self, value):
        with torch.no_grad():
            value = value.sin().cos() * 2 + value
        return self.linear(value)


def fixture():
    model = apply_quantization(
        Model().eval().requires_grad_(False),
        QuantizationConfig(scheme="int8_dyn_act_int8_weight"),
    )
    ep = torch.export.export(model, (torch.randn(1, 32),), strict=True)
    raw = snapshot_exported_program(ep, stage="quantized")
    return ep, raw


def test_owned_bodies_metadata_and_signature_keep_actual_state_identity():
    raw, snapshot = fixture()
    owned = copy_trace_program(raw)
    assert owned.graph_signature is not raw.graph_signature
    assert owned.state_dict is not raw.state_dict
    assert all(owned.state_dict[key] is value for key, value in raw.state_dict.items())
    assert snapshot_exported_program(owned, stage="quantized") == snapshot
    for (_, before), (_, after) in zip(
        _graph_modules(raw.graph_module),
        _graph_modules(owned.graph_module),
        strict=True,
    ):
        assert before is not after
        for first, second in zip(before.graph.nodes, after.graph.nodes, strict=True):
            assert first is not second and first.meta is not second.meta
            assert first.meta["custom"] is not second.meta["custom"]
            second.meta["custom"]["m2m_lineage"].append("owned-only")
            assert "owned-only" not in first.meta["custom"]["m2m_lineage"]
    assert snapshot_exported_program(raw, stage="quantized") == snapshot


@pytest.mark.parametrize("change", ["body", "source_write", "field_bytes", "owner"])
def test_source_mutation_after_observation_cannot_bind_from_stale_raw_proof(change):
    raw, snapshot = fixture()
    observation = observe(raw, quantized_frontend_snapshot=snapshot)
    assert observation is not None
    working = copy_trace_program(raw)
    assert inline_set_grad_hops(
        working.graph_module, graph_signature=working.graph_signature
    )
    prepared = working.run_decompositions({})
    if change == "body":
        body = next(
            module
            for scope, module in _graph_modules(raw.graph_module)
            if scope != "root"
        )
        node = next(
            node
            for node in body.graph.nodes
            if node.target is torch.ops.aten.sin.default
        )
        node.target = torch.ops.aten.neg.default
        body.recompile()
    elif change == "source_write":
        value = next(
            node
            for node in raw.graph.nodes
            if node.op == "placeholder" and node.name == "value"
        )
        with raw.graph.inserting_after(value):
            node = raw.graph.call_function(torch.ops.aten.add_.Tensor, (value, 1))
        node.meta["val"] = value.meta["val"]
        raw.graph_module.recompile()
    elif change == "field_bytes":
        observation.fields[0].value.reshape(-1)[0] += 1
    else:
        key = observation.fields[0].source_target
        raw.state_dict[key] = copy.deepcopy(raw.state_dict[key])
    assert bind(observation, prepared) == []


def test_later_external_alias_write_after_context_refuses_raw_observation():
    raw, _ = fixture()
    hop = next(
        node
        for node in raw.graph.nodes
        if node.target is torch.ops.higher_order.wrap_with_set_grad_enabled
    )
    value = next(
        node
        for node in raw.graph.nodes
        if node.op == "placeholder" and node.name == "value"
    )
    with raw.graph.inserting_after(hop):
        write = raw.graph.call_function(torch.ops.aten.add_.Tensor, (value, 1))
    write.meta["val"] = value.meta["val"]
    raw.graph_module.recompile()
    snapshot = snapshot_exported_program(raw, stage="quantized")
    assert observe(raw, quantized_frontend_snapshot=snapshot) is None


def test_two_prepared_storage_owners_do_not_bind_even_with_same_field_bytes():
    from torch.export.graph_signature import InputKind, InputSpec, TensorArgument

    raw, snapshot = fixture()
    observed = observe(raw, quantized_frontend_snapshot=snapshot)
    working = copy_trace_program(raw)
    inline_set_grad_hops(working.graph_module, graph_signature=working.graph_signature)
    prepared = working.run_decompositions({})
    field = observed.fields[0]
    target = next(
        key
        for key, value in prepared.state_dict.items()
        if value.untyped_storage()._cdata == field.storage
    )
    spec = next(
        spec for spec in prepared.graph_signature.input_specs if spec.target == target
    )
    original = next(node for node in prepared.graph.nodes if node.name == spec.arg.name)
    with prepared.graph.inserting_before(next(iter(prepared.graph.nodes))):
        duplicate = prepared.graph.placeholder("duplicate_field_owner")
    duplicate.meta = copy.copy(original.meta)
    prepared.state_dict["duplicate_field_owner"] = prepared.state_dict[target].view_as(
        prepared.state_dict[target]
    )
    prepared.graph_signature.input_specs.insert(
        0,
        InputSpec(
            kind=InputKind.PARAMETER,
            arg=TensorArgument(name=duplicate.name),
            target="duplicate_field_owner",
            persistent=None,
        ),
    )
    before = {
        node.name: copy.deepcopy(node.meta.get("custom"))
        for node in prepared.graph.nodes
    }
    assert bind(observed, prepared) == []
    assert before == {
        node.name: node.meta.get("custom") for node in prepared.graph.nodes
    }


@pytest.mark.parametrize("change", ["hash", "contents"])
def test_unrelated_archived_raw_observation_refuses_before_decomposition(change):
    from m2m.capture.torch_export import _prepare_exported_program

    raw, snapshot = fixture()
    observation = observe(raw, quantized_frontend_snapshot=snapshot)
    other, other_snapshot = fixture()
    altered = copy.deepcopy(other_snapshot)
    if change == "hash":
        altered["sha256"] = "0" * 64
    else:
        altered["nodes"][0]["target"] = "forged"
    with pytest.raises(ValueError, match="archived source"):
        _prepare_exported_program(
            other,
            run_default_decompositions=True,
            export_decomposition_table=None,
            capture_trace=True,
            quantized_frontend_snapshot=altered,
            raw_subclass_observation=observation,
        )


def test_same_raw_graph_hash_with_cloned_state_never_binds_original_storage():
    from m2m.capture.torch_export import _prepare_exported_program

    raw, snapshot = fixture()
    observation = observe(raw, quantized_frontend_snapshot=snapshot)
    other = copy_trace_program(raw)
    for target, value in list(other.state_dict.items()):
        other.state_dict[target] = copy.deepcopy(value)
    assert snapshot_exported_program(other, stage="quantized") == snapshot
    assert inline_set_grad_hops(
        other.graph_module, graph_signature=other.graph_signature
    )
    prepared, _ = _prepare_exported_program(
        other,
        run_default_decompositions=True,
        export_decomposition_table=None,
        capture_trace=True,
        quantized_frontend_snapshot=snapshot,
        raw_subclass_observation=observation,
    )
    assert prepared._m2m_trace_subclass_fields == []
