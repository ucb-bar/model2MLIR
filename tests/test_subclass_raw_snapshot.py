"""Published qIDs describe the archived raw graph, including pre-inline HOPs."""

import copy

import pytest
import torch

from m2m.capture.subclass_trace import bind, observe
from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization
from m2m.capture.trace import snapshot_exported_program
from m2m.ir.torchmlir_decomps import inline_set_grad_hops

pytest.importorskip("torchao")


class WithHop(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32, bias=False)

    def forward(self, value):
        with torch.no_grad():
            hidden = value.sin().cos() * 2 + value
        return self.linear(hidden)


def test_getter_after_inlined_hop_keeps_raw_qids_and_refuses_rebinding():
    model = apply_quantization(
        WithHop().eval().requires_grad_(False),
        QuantizationConfig(scheme="int8_dyn_act_int8_weight"),
    )
    ep = torch.export.export(model, (torch.randn(1, 32),), strict=True)
    hop = next(
        node
        for node in ep.graph.nodes
        if str(node.target) == "wrap_with_set_grad_enabled"
    )
    field = next(
        node
        for node in ep.graph.nodes
        if str(node.target) == "export.access_subclass_inner_tensor.default"
    )
    # A valid, explicitly ordered fixture: the immutable weight field is read
    # after the HOP and before its first use. No arithmetic is changed.
    hop.append(field)
    ep.graph_module.recompile()
    ep.validate()
    raw = snapshot_exported_program(ep, stage="quantized")
    raw_hash = raw["sha256"]
    assert list(ep.graph.nodes).index(field) > list(ep.graph.nodes).index(hop)
    original_id = field.meta["custom"]["m2m_node_id"]
    field_name = field.name
    assert inline_set_grad_hops(ep.graph_module, graph_signature=ep.graph_signature)
    ep.validate()
    # The inliner rebuilds FX nodes; inspect the surviving occurrence, not the
    # old Python Node object which is no longer owned by this graph.
    field = next(node for node in ep.graph.nodes if node.name == field_name)
    before = {
        node.name: copy.deepcopy(node.meta.get("custom") or {})
        for node in ep.graph.nodes
    }
    assert field.meta["custom"]["m2m_node_id"] == original_id
    assert original_id != f"g:quantized:root:n{list(ep.graph.nodes).index(field)}"
    assert observe(ep) is None
    assert before == {
        node.name: node.meta.get("custom") or {} for node in ep.graph.nodes
    }
    assert observe(ep, quantized_frontend_snapshot=raw) is None
    assert raw["sha256"] == raw_hash
    assert before == {
        node.name: node.meta.get("custom") or {} for node in ep.graph.nodes
    }
    assert field.meta["custom"]["m2m_node_id"] == original_id


def test_flat_observation_is_bound_to_the_actual_raw_snapshot_hash():
    class Plain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 32, bias=False)

        def forward(self, value):
            return self.linear(value)

    model = apply_quantization(
        Plain().eval(), QuantizationConfig(scheme="int8_dyn_act_int8_weight")
    )
    ep = torch.export.export(model, (torch.randn(1, 32),))
    raw = snapshot_exported_program(ep, stage="quantized")
    observation = observe(ep, quantized_frontend_snapshot=raw)
    assert observation is not None and observation.source_graph_sha256 == raw["sha256"]
    records = bind(observation, ep.run_decompositions({}))
    assert records and all(
        row["source_graph_sha256"] == raw["sha256"] for row in records
    )
    altered = copy.deepcopy(raw)
    altered["sha256"] = "0" * 64
    assert observe(ep, quantized_frontend_snapshot=altered) is None


def test_stale_bind_does_not_renumber_the_surviving_source_nodes():
    class Plain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 32, bias=False)

        def forward(self, value):
            return self.linear(value)

    model = apply_quantization(
        Plain().eval(), QuantizationConfig(scheme="int8_dyn_act_int8_weight")
    )
    ep = torch.export.export(model, (torch.randn(1, 32),))
    raw = snapshot_exported_program(ep, stage="quantized")
    observation = observe(ep, quantized_frontend_snapshot=raw)
    prepared = ep.run_decompositions({})
    field = next(
        node
        for node in ep.graph.nodes
        if str(node.target) == "export.access_subclass_inner_tensor.default"
    )
    with ep.graph.inserting_after(field):
        extra = ep.graph.call_function(torch.ops.aten.alias.default, (field,))
    extra.meta = {"val": field.meta["val"]}
    before = {
        node.name: copy.deepcopy(node.meta.get("custom") or {})
        for node in ep.graph.nodes
    }
    assert bind(observation, prepared) == []
    assert before == {
        node.name: node.meta.get("custom") or {} for node in ep.graph.nodes
    }


def test_actual_conversion_never_emits_post_inline_fields_as_raw_source_claims():
    import m2m

    quant = QuantizationConfig(scheme="int8_dyn_act_int8_weight")
    model = apply_quantization(WithHop().eval().requires_grad_(False), quant)
    value = torch.randn(1, 32)
    ep = torch.export.export(model, (value,), strict=True)
    hop = next(
        node
        for node in ep.graph.nodes
        if str(node.target) == "wrap_with_set_grad_enabled"
    )
    field = next(
        node
        for node in ep.graph.nodes
        if str(node.target) == "export.access_subclass_inner_tensor.default"
    )
    hop.append(field)
    ep.graph_module.recompile()
    ep.validate()
    result = m2m.convert(
        ep,
        (value,),
        quantization=quant,
        quantization_preapplied=True,
        backend="fx_importer",
        capture_trace=True,
    )
    assert result.ok
    # The source HOP still needs an explicit field-lifting proof. Absence of
    # that proof is diagnostic, rather than a newly numbered source claim.
    assert (
        result.capture_trace["graphs"]["prepared"].get("tensor_subclass_fields", [])
        == []
    )
    assert result.capture_trace["transformations"][1]["status"] == "diagnostic"
    getter = next(
        row
        for row in result.capture_trace["graphs"]["quantized"]["nodes"]
        if row["target"] == "export.access_subclass_inner_tensor.default"
        and row["args"][-1] == "qdata"
    )
    # A rewritten F32 body must not consume the old I8 accessor's source ID.
    assert (
        getter["id"]
        in result.capture_trace["transformations"][1]["unresolved_source_ids"]
    )
    assert not any(
        getter["id"] in row["origin_node_ids"]
        for row in result.capture_trace["graphs"]["prepared"]["nodes"]
    )
