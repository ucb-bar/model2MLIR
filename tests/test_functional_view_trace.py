"""Duplicated functionalized views require real owners and every typed use."""

import copy
import itertools
import json
import os
import subprocess
import sys

import pytest
import torch

import m2m
from m2m.capture.dynamic_linear_trace import LinearInstrumentation
from m2m.capture.functional_view_trace import _runs, functional_view_relations
from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization
from m2m.capture.trace import snapshot_exported_program


class Attention(torch.nn.Module):
    def __init__(self, *, batch=1, bias=True, tokens=5):
        super().__init__()
        self.qkv = torch.nn.Linear(8, 24, bias=bias)
        self.batch, self.tokens = batch, tokens

    def forward(self, value):
        joined = self.qkv(value.sin()).reshape(self.batch, self.tokens, 3, 2, 4)
        joined = joined.permute(2, 0, 3, 1, 4)
        q, k, v = joined[0], joined[1], joined[2]
        probability = (q @ k.transpose(-2, -1)).softmax(-1)
        probability = torch.nn.functional.dropout(probability, 0.0, training=False)
        return probability @ v


def capture(batch=1, bias=True, tokens=5):
    torch.manual_seed(0)
    model = (
        Attention(batch=batch, bias=bias, tokens=tokens).eval().requires_grad_(False)
    )
    inputs = (torch.linspace(-1, 1, batch * tokens * 8).reshape(batch, tokens, 8),)
    quant = QuantizationConfig("int8_dyn_act_int8_weight")
    with LinearInstrumentation(model) as owner, torch.no_grad():
        source = owner.capture_source(model, inputs)
        apply_quantization(model, quant)
        exported, _ = owner.export_selected(source, model, inputs, quantization=quant)
        before = snapshot_exported_program(exported, stage="quantized")
        calls = copy.deepcopy(exported.module_call_graph)
        expected = exported.module()(*inputs).clone()
        result = m2m.convert(
            exported,
            inputs,
            backend="fx_importer",
            capture_trace=True,
            quantization=quant,
            quantization_preapplied=True,
            original_frontend_snapshot=source.snapshot,
        )
        assert result.ok, result.diagnostics
        assert snapshot_exported_program(exported, stage="quantized") == before
        assert exported.module_call_graph == calls
        torch.testing.assert_close(exported.module()(*inputs), expected, rtol=0, atol=0)
    return result.capture_trace


@pytest.fixture(scope="module")
def actual():
    trace = capture()
    assert trace["status"] == "complete", trace["blockers"]
    source, dest = trace["graphs"]["quantized"], trace["graphs"]["prepared"]
    rows = trace["transformations"][-1]["relations"]
    added = [row for row in rows if row["kind"] == "functionalized_static_view_value"]
    assert len(added) == 7
    premises = [row for row in rows if row not in added]
    consumed = {identity for row in premises for identity in row["source_ids"]}
    return source, dest, premises, consumed, added


def test_real_public_AO_QKV_capture_closes_only_proved_view_values(actual):
    source, dest, premises, consumed, added = actual
    assert functional_view_relations(source, dest, premises, consumed) == added
    assert len({row["proof"]["source_write_id"] for row in added}) == 1
    assert {row["proof"]["relative_element_offset"] for row in added} == {0, 8, 16}
    assert all(row["proof"]["exact_ordered_write_operands"] for row in added)
    # K/V source args/1 is checked at the actual BMM, beyond prefix args/0.
    assert any(
        binding["proof"]["source_argument_slot"] == 1
        for row in added
        for edge in row["proof"]["typed_consumer_edges"]
        for binding in edge["matmul_decomposition_edges"]
    )
    assert any(
        value["source_identity_path"]
        for row in added
        for edge in row["proof"]["typed_consumer_edges"]
        for binding in edge["matmul_decomposition_edges"]
        for paths in binding["proof"]["input_alias_paths"]
        for value in paths
    )


@pytest.mark.parametrize(
    "change",
    [
        "different_bias_owner",
        "select_offset",
        "select_out_of_bounds",
        "source_stride",
        "source_dtype",
        "prepared_stride",
        "prepared_edge",
        "BMM_operands",
        "BMM_result",
        "source_schema",
        "prepared_schema",
        "alpha",
        "unknown_source_consumer",
        "unknown_prepared_consumer",
        "enabled_other_operand_dropout",
        "missing_source_dtype",
    ],
)
def test_wrong_owner_geometry_effect_or_consumer_refuses(change, actual):
    source, dest, premises, consumed, added = copy.deepcopy(actual)
    source_nodes = {row["id"]: row for row in source["nodes"]}
    dest_nodes = {row["id"]: row for row in dest["nodes"]}
    writer = source_nodes[added[0]["proof"]["source_write_id"]]
    anchor = dest_nodes[added[0]["proof"]["prepared_functional_add_id"]]
    selected = next(
        row for row in source["nodes"] if row["target"] == "aten.select.int"
    )
    permutation = next(
        row for row in source["nodes"] if row["target"] == "aten.permute.default"
    )
    target_select = next(
        row for row in dest["nodes"] if row["target"] == "aten.select.int"
    )
    bmm = next(row for row in dest["nodes"] if row["target"] == "aten.bmm.default")
    if change == "different_bias_owner":
        original = dest_nodes[anchor["args"][1]["node_id"]]
        other = copy.deepcopy(original)
        other["id"] += ":unrelated"
        other["results"][0]["id"] = other["id"] + ":v0"
        dest["nodes"].insert(0, other)
        anchor["args"][1] = {
            "node_id": other["id"],
            "value_id": other["results"][0]["id"],
        }
        for edge in dest["edges"]:
            if (
                edge["consumer_node_id"] == anchor["id"]
                and edge["argument_path"] == "args/1"
            ):
                edge.update(
                    producer_node_id=other["id"],
                    producer_value_id=other["results"][0]["id"],
                )
    elif change == "select_offset":
        selected["args"][2] = 1
    elif change == "select_out_of_bounds":
        selected["args"][2] = 3
    elif change == "source_stride":
        permutation["results"][0]["stride"][0] += 1
    elif change == "source_dtype":
        selected["results"][0]["dtype"] = "float16"
        selected["results"][0]["storage_dtype"] = "float16"
    elif change == "prepared_stride":
        target_select["results"][0]["stride"][-1] += 1
    elif change == "prepared_edge":
        edge = next(
            edge
            for edge in dest["edges"]
            if edge["producer_node_id"] == target_select["id"]
        )
        edge["producer_value_id"] = "not-that-result"
    elif change == "BMM_operands":
        bmm["args"].reverse()
    elif change == "BMM_result":
        bmm["results"][0]["shape"][-1] += 1
    elif change == "source_schema":
        source["operator_schemas"][selected["target"]] = str(
            torch.ops.aten.clone.default._schema
        )
    elif change == "prepared_schema":
        dest["operator_schemas"][target_select["target"]] = str(
            torch.ops.aten.clone.default._schema
        )
    elif change == "alpha":
        writer["kwargs"] = {"alpha": 2}
    elif change == "unknown_source_consumer":
        next(row for row in source["nodes"] if row["target"] == "aten.matmul.default")[
            "op"
        ] = "call_module"
    elif change == "unknown_prepared_consumer":
        bmm["op"] = "call_method"
    elif change == "enabled_other_operand_dropout":
        next(row for row in source["nodes"] if row["target"] == "aten.dropout.default")[
            "args"
        ][2] = True
    else:
        del selected["results"][0]["dtype"]
    assert functional_view_relations(source, dest, premises, consumed) == [], change


def insert_write(graph, before, value):
    for node in graph["nodes"]:
        node["ordinal"] *= 2
    writer = copy.deepcopy(value)
    writer.update(
        id=value["id"] + ":write",
        ordinal=before["ordinal"] - 1,
        op="call_function",
        target="aten.copy_.default",
        kwargs={},
    )
    ref = {"node_id": value["id"], "value_id": value["results"][0]["id"]}
    writer["args"] = [copy.deepcopy(ref), copy.deepcopy(ref)]
    writer["results"][0]["id"] = writer["id"] + ":v0"
    graph["nodes"].insert(graph["nodes"].index(before), writer)
    graph["operator_schemas"][writer["target"]] = str(
        torch.ops.aten.copy_.default._schema
    )
    for index in range(2):
        graph["edges"].append(
            {
                "producer_node_id": value["id"],
                "producer_value_id": value["results"][0]["id"],
                "consumer_node_id": writer["id"],
                "argument_path": f"args/{index}",
                "shape": value["results"][0]["shape"],
                "dtype": value["results"][0]["dtype"],
                "value_kind": "tensor",
            }
        )


@pytest.mark.parametrize("side", ["source", "prepared"])
@pytest.mark.parametrize("reader", [0, 2])
def test_alias_write_before_first_or_later_reader_invalidates_group(
    side, reader, actual
):
    source, dest, premises, consumed, _ = copy.deepcopy(actual)
    graph = source if side == "source" else dest
    selections = [
        node for node in graph["nodes"] if node["target"] == "aten.select.int"
    ]
    before = selections[reader]
    owner = next(
        node for node in graph["nodes"] if node["id"] == before["args"][0]["node_id"]
    )
    insert_write(graph, before, owner)
    assert functional_view_relations(source, dest, premises, consumed) == []


def test_unknown_runtime_and_unbound_write_do_not_grant_view_ancestry(actual):
    source, dest, premises, consumed, _ = copy.deepcopy(actual)
    dest["runtime_versions"]["torch"] = "unobserved"
    assert functional_view_relations(source, dest, premises, consumed) == []
    source, dest, premises, consumed, added = copy.deepcopy(actual)
    writer = added[0]["proof"]["source_write_id"]
    premises = [row for row in premises if writer not in row["source_ids"]]
    assert functional_view_relations(source, dest, premises, consumed) == []


@pytest.mark.parametrize(
    "shape,stride,target_shape,target_stride,equal",
    [
        ([2, 3], [3, 1], [3, 2], [2, 1], True),
        ([1, 4, 5, 4], [480, 4, 24, 1], [4, 5, 4], [4, 24, 1], True),
        ([1, 1], [99, 7], [1], [1], True),
        ([2, 3], [1, 2], [3, 2], [2, 1], False),
        ([2, 2], [0, 1], [4], [1], False),
        ([2, 2], [0, 0], [4], [0], True),
    ],
)
def test_reshape_run_proof_matches_complete_logical_element_addresses(
    shape, stride, target_shape, target_stride, equal
):
    def addresses(dims, steps):
        return [
            sum(i * s for i, s in zip(indices, steps, strict=True))
            for indices in itertools.product(*(range(dim) for dim in dims))
        ]

    assert (addresses(shape, stride) == addresses(target_shape, target_stride)) is equal
    assert (_runs(shape, stride) == _runs(target_shape, target_stride)) is equal


def test_multi_batch_copying_decomposition_is_not_silently_admitted():
    trace = capture(batch=2)
    # Current narrow rule proves zero-copy, equal-batch reshape paths only.
    # Flattening these noncontiguous two-batch QKV operands requires a clone.
    assert not any(
        row["kind"] == "functionalized_static_view_value"
        for row in trace["transformations"][-1]["relations"]
    )


@pytest.mark.parametrize("side", ["source", "prepared"])
def test_write_operand_version_must_be_unchanged_since_its_definition(side, actual):
    source, dest, premises, consumed, added = copy.deepcopy(actual)
    graph = source if side == "source" else dest
    key = "source_write_id" if side == "source" else "prepared_functional_add_id"
    write = next(
        node for node in graph["nodes"] if node["id"] == added[0]["proof"][key]
    )
    operand = next(
        node for node in graph["nodes"] if node["id"] == write["args"][0]["node_id"]
    )
    insert_write(graph, write, operand)
    assert functional_view_relations(source, dest, premises, consumed) == []


def test_other_operand_identity_cannot_hide_an_earlier_alias_write(actual):
    source, dest, premises, consumed, _ = copy.deepcopy(actual)
    identity = next(
        node for node in source["nodes"] if node["target"] == "aten.dropout.default"
    )
    producer = next(
        node for node in source["nodes"] if node["id"] == identity["args"][0]["node_id"]
    )
    insert_write(source, identity, producer)
    assert functional_view_relations(source, dest, premises, consumed) == []


def test_square_BMM_input_transpose_is_not_a_batch_flatten():
    trace = capture(tokens=4)
    source, dest = trace["graphs"]["quantized"], trace["graphs"]["prepared"]
    rows = trace["transformations"][-1]["relations"]
    added = [row for row in rows if row["kind"] == "functionalized_static_view_value"]
    premises = [row for row in rows if row not in added]
    consumed = {identity for row in premises for identity in row["source_ids"]}
    bmm = next(row for row in dest["nodes"] if row["target"] == "aten.bmm.default")
    operand = next(
        row for row in dest["nodes"] if row["id"] == bmm["args"][0]["node_id"]
    )
    assert operand["results"][0]["shape"][-2:] == [4, 4]
    for node in dest["nodes"]:
        node["ordinal"] *= 2
    changed = copy.deepcopy(operand)
    changed.update(
        id=operand["id"] + ":transpose",
        ordinal=bmm["ordinal"] - 1,
        target="aten.transpose.int",
        op="call_function",
        kwargs={},
    )
    changed["args"] = [copy.deepcopy(bmm["args"][0]), 1, 2]
    value = changed["results"][0]
    value["id"] = changed["id"] + ":v0"
    value["stride"][-2:] = list(reversed(value["stride"][-2:]))
    dest["nodes"].insert(dest["nodes"].index(bmm), changed)
    dest["operator_schemas"][changed["target"]] = str(
        torch.ops.aten.transpose.int._schema
    )
    edge = next(
        edge
        for edge in dest["edges"]
        if edge["consumer_node_id"] == bmm["id"] and edge["argument_path"] == "args/0"
    )
    dest["edges"].append({**edge, "consumer_node_id": changed["id"]})
    edge.update(producer_node_id=changed["id"], producer_value_id=value["id"])
    bmm["args"][0] = {"node_id": changed["id"], "value_id": value["id"]}
    # This remains a valid square tensor with the same storage span, offset,
    # element count and BMM shape. Its logical element order has changed.
    example = torch.arange(16).reshape(4, 4)
    assert not torch.equal(example, example.transpose(0, 1))
    assert functional_view_relations(source, dest, premises, consumed) == []


def test_proof_serialization_is_stable_across_hash_seeds(actual):
    source, dest, premises, consumed, _ = actual
    payload = json.dumps([source, dest, premises, sorted(consumed)])
    code = (
        "import json,sys; "
        "from m2m.capture.functional_view_trace import functional_view_relations; "
        "s,d,p,c=json.load(sys.stdin); "
        "print(json.dumps(functional_view_relations(s,d,p,set(c)),sort_keys=True))"
    )
    outputs = []
    for seed in ("0", "91"):
        completed = subprocess.run(
            [sys.executable, "-c", code],
            input=payload,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            timeout=30,
            check=True,
        )
        outputs.append(completed.stdout)
    assert outputs[0] == outputs[1]
    assert len(json.loads(outputs[0])) == 7
