"""Disabled dropout retains exact tensor/alias lifetime and bypassed use edges."""

import copy

import pytest
import torch

import m2m
from m2m.capture.trace import _forward_identity_value, _identity_conversion_bypass
from m2m.capture.view_trace import SourceStorageAliases


class Disabled(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, value):
        hidden = value.sin()
        hidden = torch.ops.aten.dropout.default(hidden, self.p, False)
        return hidden.cos() + value


def snapshots(p=0.0):
    converted = m2m.convert(
        Disabled(p).eval(),
        (torch.randn(2, 5),),
        capture_trace=True,
        backend="fx_importer",
    )
    assert converted.ok
    source = converted.capture_trace["graphs"]["quantized"]
    dest = converted.capture_trace["graphs"]["prepared"]
    node = next(
        node for node in source["nodes"] if node["target"] == "aten.dropout.default"
    )
    return converted, source, dest, node


@pytest.mark.parametrize("p", [0.0, 0.25, 1.0])
def test_real_inference_dropout_identity_keeps_bits_alias_and_rng(p):
    value = torch.randn(2, 5)
    state = torch.random.get_rng_state().clone()
    actual = torch.ops.aten.dropout.default(value, p, False)
    assert actual is value and torch.equal(
        actual.view(torch.uint8), value.view(torch.uint8)
    )
    assert torch.equal(state, torch.random.get_rng_state())
    result, source, dest, node = snapshots(p)
    assert result.capture_trace["status"] == "complete"
    checked = _forward_identity_value(source, node)
    assert checked is not None
    aliases = SourceStorageAliases(source)
    assert aliases.root(node) == aliases.root(checked[0])
    relation = _identity_conversion_bypass(source, dest, node)
    assert relation and relation["kind"] == "eliminated"
    assert "inference dropout" in relation["reason"]


@pytest.mark.parametrize(
    "mutation",
    [
        "training",
        "missing_training",
        "nan_p",
        "invalid_p",
        "bool_p",
        "schema",
        "dtype",
        "stride",
    ],
)
def test_inactive_dropout_requires_actual_complete_schema_controls_and_tensor_identity(
    mutation,
):
    _, source, dest, node = snapshots()
    source = copy.deepcopy(source)
    node = next(part for part in source["nodes"] if part["id"] == node["id"])
    if mutation == "training":
        node["args"][2] = True
    elif mutation == "missing_training":
        node["args"] = node["args"][:2]
    elif mutation == "nan_p":
        node["args"][1] = float("nan")
    elif mutation == "invalid_p":
        node["args"][1] = 1.5
    elif mutation == "bool_p":
        node["args"][1] = False
    elif mutation == "schema":
        source["operator_schemas"][node["target"]] = str(
            torch.ops.aten.relu.default._schema
        )
    elif mutation == "dtype":
        node["results"][0]["dtype"] = "float64"
    else:
        node["results"][0]["stride"] = [20, 2]
    assert _forward_identity_value(source, node) is None
    assert _identity_conversion_bypass(source, dest, node) is None


@pytest.mark.parametrize("mutation", ["wrong_input", "wrong_consumer", "wrong_type"])
def test_inference_dropout_bypass_requires_exact_destination_use(mutation):
    _, source, dest, node = snapshots()
    dest = copy.deepcopy(dest)
    checked = _forward_identity_value(source, node)
    owner = next(
        part for part in dest["nodes"] if checked[0]["id"] in part["origin_node_ids"]
    )
    edge = next(
        edge for edge in dest["edges"] if edge["producer_node_id"] == owner["id"]
    )
    if mutation == "wrong_input":
        edge["producer_value_id"] = "wrong-result"
    elif mutation == "wrong_consumer":
        edge["argument_path"] = "args/1"
    else:
        edge["dtype"] = "float64"
    assert _identity_conversion_bypass(source, dest, node) is None


def test_later_source_alias_write_prevents_inference_dropout_bypass():
    class Writing(torch.nn.Module):
        def forward(self, value):
            hidden = value.sin()
            result = torch.ops.aten.dropout.default(hidden, 0.0, False)
            hidden.add_(1)
            return result.cos()

    result = m2m.convert(
        Writing(), (torch.randn(2, 5),), capture_trace=True, backend="fx_importer"
    )
    assert result.ok
    source, dest = (
        result.capture_trace["graphs"][name] for name in ("quantized", "prepared")
    )
    node = next(
        node for node in source["nodes"] if node["target"] == "aten.dropout.default"
    )
    assert _identity_conversion_bypass(source, dest, node) is None


def test_destination_alias_write_before_consumer_refuses_inference_dropout_bypass():
    _, source, dest, node = snapshots()
    dest = copy.deepcopy(dest)
    checked = _forward_identity_value(source, node)
    owner = next(
        part for part in dest["nodes"] if checked[0]["id"] in part["origin_node_ids"]
    )
    consumer = next(
        part for part in dest["nodes"] if part["target"] == "aten.cos.default"
    )
    write = copy.deepcopy(owner)
    write.update(
        id="g:prepared:root:write", target="aten.add_.Tensor", origin_node_ids=[]
    )
    write["args"] = [{"node_id": owner["id"], "value_id": owner["results"][0]["id"]}, 1]
    write["results"][0]["id"] = write["id"] + ":v0"
    dest["nodes"].insert(dest["nodes"].index(consumer), write)
    for index, part in enumerate(dest["nodes"]):
        part["ordinal"] = index
    edge = copy.deepcopy(
        next(edge for edge in dest["edges"] if edge["producer_node_id"] == owner["id"])
    )
    edge.update(consumer_node_id=write["id"], argument_path="args/0")
    dest["edges"].append(edge)
    dest["operator_schemas"]["aten.add_.Tensor"] = str(
        torch.ops.aten.add_.Tensor._schema
    )
    assert _identity_conversion_bypass(source, dest, node) is None
