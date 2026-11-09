"""Source attribution cannot authorize changed public or callee contracts."""

import copy

import pytest
import torch

import m2m
from m2m.capture.dynamic_linear_trace import (
    DynamicLinearLineageError,
    LinearInstrumentation,
)
from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16, bias=False)

    def forward(self, value):
        return {"linear": self.linear(value), "skip": value.sin()}


def fixture():
    torch.manual_seed(47)
    model = Model().eval().requires_grad_(False)
    inputs = (torch.randn(1, 16),)
    quant = QuantizationConfig(scheme="int8_dyn_act_int8_weight")
    return model, inputs, quant


def mutate(exported, change):
    from torch.export.graph_signature import TensorArgument
    from torch.utils._pytree import TreeSpec

    root = exported.module_call_graph[0].signature
    leaf = next(entry for entry in exported.module_call_graph if entry.fqn == "linear")
    if change == "output_key":
        root.out_spec = TreeSpec(
            dict, ["changed", "skip"], list(root.out_spec.children_specs)
        )
    elif change == "output_key_order":
        root.out_spec = TreeSpec(
            dict, ["skip", "linear"], list(root.out_spec.children_specs)
        )
    elif change == "tree_count":
        object.__setattr__(root.out_spec, "num_leaves", root.out_spec.num_leaves + 1)
    elif change == "input_tree":
        root.in_spec = TreeSpec(list, None, list(root.in_spec.children_specs))
    elif change == "output_tree":
        root.out_spec = TreeSpec(tuple, None, list(root.out_spec.children_specs))
    elif change in ("module_input_ref", "module_output_ref"):
        slots = (
            leaf.signature.inputs
            if change == "module_input_ref"
            else leaf.signature.outputs
        )
        slots[0] = TensorArgument("sin")
    elif change == "module_path":
        leaf.fqn = "changed_linear"
    elif change == "entry_order":
        exported.module_call_graph.reverse()
    elif change == "forward_names":
        root.forward_arg_names = ["changed"]
    elif change == "signature_extra":
        root.hidden = torch.ones(1)
    elif change == "tree_context_tensor":
        root.out_spec = TreeSpec(
            dict, [torch.ones(1), "skip"], list(root.out_spec.children_specs)
        )
    elif change in ("range_bounds", "range_extra"):
        import sympy
        from torch.utils._sympy.value_ranges import ValueRanges

        bound = ValueRanges(0, 13)
        if change == "range_extra":
            object.__setattr__(bound, "hidden", torch.ones(1))
        exported.range_constraints[sympy.Symbol("extra", integer=True)] = bound
    else:
        raise AssertionError(change)


CHANGES = [
    "output_key",
    "output_key_order",
    "tree_count",
    "input_tree",
    "output_tree",
    "module_input_ref",
    "module_output_ref",
    "module_path",
    "entry_order",
    "forward_names",
    "signature_extra",
    "tree_context_tensor",
    "range_bounds",
    "range_extra",
]


@pytest.mark.parametrize("change", CHANGES)
@pytest.mark.parametrize("owner", ["selected", "original"])
def test_contract_mutation_after_proof_refuses_without_restamping_ids(change, owner):
    model, inputs, quant = fixture()
    with LinearInstrumentation(model) as lineage:
        source = lineage.capture_source(model, inputs)
        apply_quantization(model, quant)
        selected, _ = lineage.export_selected(source, model, inputs, quantization=quant)
    mutate(selected if owner == "selected" else source.exported, change)
    before = {
        node.name: copy.deepcopy(node.meta.get("custom"))
        for node in selected.graph.nodes
    }
    result = m2m.convert(
        selected,
        inputs,
        quantization=quant,
        quantization_preapplied=True,
        original_frontend_snapshot=source.snapshot,
        capture_trace=True,
        backend="fx_importer",
    )
    assert not result.ok and result.capture_trace["status"] == "diagnostic"
    assert any(
        "sealed" in message and "contract" in message for message in result.diagnostics
    )
    assert before == {
        node.name: node.meta.get("custom") for node in selected.graph.nodes
    }


@pytest.mark.parametrize("change", CHANGES)
def test_original_contract_mutation_before_selected_export_refuses(change):
    model, inputs, quant = fixture()
    with LinearInstrumentation(model) as lineage:
        source = lineage.capture_source(model, inputs)
        mutate(source.exported, change)
        apply_quantization(model, quant)
        with pytest.raises(DynamicLinearLineageError, match="sealed.*contract"):
            lineage.export_selected(source, model, inputs, quantization=quant)


def test_unchanged_dict_tree_and_equal_owned_metadata_copy_keep_complete_proof():
    model, inputs, quant = fixture()
    with LinearInstrumentation(model) as lineage:
        source = lineage.capture_source(model, inputs)
        apply_quantization(model, quant)
        selected, _ = lineage.export_selected(source, model, inputs, quantization=quant)
    selected._module_call_graph = copy.deepcopy(selected.module_call_graph)
    result = m2m.convert(
        selected,
        inputs,
        quantization=quant,
        quantization_preapplied=True,
        original_frontend_snapshot=source.snapshot,
        capture_trace=True,
        backend="fx_importer",
    )
    assert result.ok, result.diagnostics
    assert result.capture_trace["status"] == "complete"
    assert list(selected.module()(*inputs)) == ["linear", "skip"]
