"""Coarse source-graph -> transformed graph -> final MLIR trace qualification."""
import hashlib
import json

import pytest
import torch

import m2m
from m2m.capture.trace import finalize_trace, graph_relation
from m2m.capture.torchao_pipeline import QuantizationConfig


def _check_trace(result):
    assert result.ok, result.diagnostics
    trace = json.loads(json.dumps(result.capture_trace, allow_nan=False))
    assert trace["status"] == "complete", trace["blockers"]
    assert trace["counting_unit"] == "static_captured_call_sites"
    assert trace["mlir"]["sha256"] == hashlib.sha256(result.mlir_text.encode()).hexdigest()
    assert trace["mlir"]["bytes"] == len(result.mlir_text.encode())
    for graph in trace["graphs"].values():
        claimed = graph.pop("sha256")
        actual = hashlib.sha256(json.dumps(graph, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
        assert claimed == actual
        graph["sha256"] = claimed
        assert graph["call_count"] == sum(n["op"].startswith("call_") for n in graph["nodes"])
        known = {n["id"] for n in graph["nodes"]}
        assert len(known) == len(graph["nodes"])
        for edge in graph["edges"]:
            assert edge["producer_node_id"] in known
            assert edge["consumer_node_id"] in known
    assert [op["ordinal"] for op in trace["mlir"]["operations"]] == list(
        range(len(list(result.module.walk()))))
    assert all(op["role"] != "unresolved" for op in trace["mlir"]["operations"])
    return trace


def test_introduced_metadata_guard_needs_a_proven_typed_producer():
    import copy
    from m2m.capture.trace import snapshot_exported_program

    class Model(torch.nn.Module):
        def forward(self, value):
            return value.relu()

    exported = torch.export.export(Model().eval(), (torch.randn(2, 4),))
    source = snapshot_exported_program(exported, stage="original")
    dest = snapshot_exported_program(exported, stage="quantized")
    producer = next(node for node in dest["nodes"] if node["target"] == "aten.relu.default")
    guard = {
        "id": "g:quantized:root:guard", "graph_id": "g:quantized:root",
        "op": "call_function", "target": "aten._assert_tensor_metadata.default",
        "args": [{"node_id": producer["id"], "value_id": producer["results"][0]["id"]}],
        "kwargs": {"dtype": {"kind": "dtype", "value": "torch.float32"},
                   "device": {"kind": "device", "value": "cpu"}},
        "results": [{"id": "g:quantized:root:guard:v0", "kind": "unknown",
                     "shape": None, "dtype": None}],
        "origin_node_ids": [],
    }
    dest["nodes"].append(guard)
    relation = graph_relation(source, dest)
    assert relation["status"] == "complete"
    assert any(row["kind"] == "introduced_metadata_guard" and
               row["destination_ids"] == [guard["id"]] and
               row["source_ids"] == producer["origin_node_ids"]
               for row in relation["relations"])

    wrong_type = copy.deepcopy(dest)
    wrong_type["nodes"][-1]["kwargs"]["dtype"]["value"] = "torch.float16"
    assert guard["id"] in graph_relation(source, wrong_type)["unresolved_destination_ids"]
    wrong_size = copy.deepcopy(dest)
    wrong_size["nodes"][-1]["kwargs"]["size"] = [9, 4]
    assert guard["id"] in graph_relation(source, wrong_size)["unresolved_destination_ids"]
    wrong_value = copy.deepcopy(dest)
    wrong_value["nodes"][-1]["args"][0]["value_id"] = "not-a-result"
    assert guard["id"] in graph_relation(source, wrong_value)["unresolved_destination_ids"]
    no_lineage = copy.deepcopy(dest)
    next(node for node in no_lineage["nodes"] if node["id"] == producer["id"])["origin_node_ids"] = []
    assert guard["id"] in graph_relation(source, no_lineage)["unresolved_destination_ids"]


@pytest.mark.parametrize("decomposed_clone", [False, True])
def test_lifted_tensor_constant_keeps_lineage_on_reexport(decomposed_clone):
    from m2m.capture.trace import (prepare_lifted_constant_lineage,
                                   snapshot_exported_program)

    class Model(torch.nn.Module):
        def forward(self, value):
            return value + torch.tensor(1.0, dtype=torch.float32)

    inputs = (torch.randn(2, 4),)
    exported = torch.export.export(Model().eval(), inputs)
    original = snapshot_exported_program(exported, stage="original")
    graph_module = exported.module()
    if decomposed_clone:
        copied = next(node for node in graph_module.graph.nodes
                      if node.target == torch.ops.aten.lift_fresh_copy.default)
        copied.target = torch.ops.aten.clone.default
        graph_module.recompile()
    assert prepare_lifted_constant_lineage(graph_module) == 1
    transformed = torch.export.export(graph_module, inputs)
    quantized = snapshot_exported_program(transformed, stage="quantized")
    relation = graph_relation(original, quantized)
    assert relation["status"] == "complete", relation
    copied = next(node for node in original["nodes"]
                  if node["target"] == "aten.lift_fresh_copy.default")
    assert any(copied["id"] in row["source_ids"] for row in relation["relations"])


def test_reconstructed_tuple_selector_requires_exact_parent_index_and_type():
    import copy
    from m2m.capture.trace import snapshot_exported_program

    class Model(torch.nn.Module):
        def forward(self, value):
            left, right = value.split(4, dim=-1)
            return left + right

    exported = torch.export.export(Model().eval(), (torch.randn(2, 8),))
    source = snapshot_exported_program(exported, stage="original")
    dest = snapshot_exported_program(exported, stage="quantized")
    parent = next(node for node in source["nodes"] if node["target"] == "aten.split.Tensor")
    selectors = [node for node in dest["nodes"]
                 if node["target"] == "<built-in function getitem>"]
    assert len(selectors) == 2
    for selector in selectors:
        selector["origin_node_ids"] = [parent["id"]]
    relation = graph_relation(source, dest)
    assert relation["status"] == "complete", relation
    assert sum(row["kind"] == "tuple_selection" for row in relation["relations"]) == 2

    wrong_index = copy.deepcopy(dest)
    next(node for node in wrong_index["nodes"] if node["id"] == selectors[0]["id"])["args"][1] = 1
    assert graph_relation(source, wrong_index)["status"] == "diagnostic"
    wrong_type = copy.deepcopy(dest)
    next(node for node in wrong_type["nodes"] if node["id"] == selectors[0]["id"])["results"][0]["dtype"] = "float16"
    assert graph_relation(source, wrong_type)["status"] == "diagnostic"


def test_mixed_precision_source_sites_decomposition_and_final_serialization(tmp_path, monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)

        def forward(self, x):
            y = self.fc(x).relu()
            scores = torch.matmul(y.to(torch.float16), y.T.to(torch.float16)).float()
            return scores.softmax(-1), y + y

    import m2m.capture.torch_export as capture
    calls = []
    actual_capture = capture.capture_model
    def counted(*args, **kwargs):
        calls.append(1)
        return actual_capture(*args, **kwargs)
    monkeypatch.setattr(capture, "capture_model", counted)
    model, inputs = Model().eval(), (torch.randn(2, 4),)
    result = m2m.convert(model, inputs, backend="fx_importer", capture_trace=True,
                         weights_path=str(tmp_path / "weights.safetensors"))
    trace = _check_trace(result)
    assert len(calls) == 1
    assert trace["graphs"]["original"]["call_count"] < trace["graphs"]["prepared"]["call_count"]
    assert any(r["kind"] == "decomposition" for r in trace["transformations"][1]["relations"])
    assert {e["dtype"] for e in trace["graphs"]["original"]["edges"]} >= {"float16", "float32"}
    assert any(n["target"] == "aten.linear.default" for n in trace["graphs"]["original"]["nodes"])
    assert '"tensor.empty"' in result.mlir_text
    assert "prov.source_node_ids" in result.mlir_text
    assert (tmp_path / "weights.safetensors").is_file()
    assert m2m.convert(model, inputs, backend="fx_importer").capture_trace is None


def test_original_and_quantized_graphs_pt2e_and_expansion():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)
        def forward(self, x):
            return self.fc(x).relu()

    result = m2m.convert(Model().eval(), (torch.randn(2, 4),), backend="fx_importer",
                         capture_trace=True, fully_standard=True,
                         quantization=QuantizationConfig(scheme="int8_static_act_int8_weight"))
    trace = _check_trace(result)
    assert trace["graphs"]["original"]["call_count"] == 2
    assert trace["graphs"]["quantized"]["call_count"] == 5
    assert any(n["target"].startswith("quantized_decomposed.")
               for n in trace["graphs"]["quantized"]["nodes"])
    assert any(r["kind"] == "introduced" for r in trace["transformations"][0]["relations"])
    assert any(e["dtype"] == "int8" for e in trace["graphs"]["quantized"]["edges"])


def test_pt2e_conv_batch_norm_fold_carries_exact_original_lineage():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
            self.bn = torch.nn.BatchNorm2d(4)

        def forward(self, x):
            return self.bn(self.conv(x)).relu()

    result = m2m.convert(Model().eval(), (torch.randn(1, 3, 8, 8),),
                         backend="fx_importer", capture_trace=True,
                         quantization=QuantizationConfig(scheme="int8_static_act_int8_weight"))
    trace = _check_trace(result)
    original_bn = next(node for node in trace["graphs"]["original"]["nodes"]
                       if node["target"] == "aten.batch_norm.default")
    fused_convs = [node for node in trace["graphs"]["quantized"]["nodes"]
                   if node["target"] == "aten.conv2d.default"
                   and original_bn["id"] in node["origin_node_ids"]]
    assert len(fused_convs) == 1


def test_pt2e_batch_norm_without_direct_conv_edge_has_no_fusion_lineage():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)
            self.bn = torch.nn.BatchNorm2d(3)

        def forward(self, x):
            return self.bn(self.conv(x) + x)

    result = m2m.convert(Model().eval(), (torch.randn(1, 3, 8, 8),),
                         backend="fx_importer", capture_trace=True,
                         quantization=QuantizationConfig(scheme="int8_static_act_int8_weight"))
    assert result.ok, result.diagnostics
    trace = result.capture_trace
    original_bn = next(node for node in trace["graphs"]["original"]["nodes"]
                       if node["target"] == "aten.batch_norm.default")
    quantized_nodes = trace["graphs"]["quantized"]["nodes"]
    assert not any(node["target"] == "aten.conv2d.default"
                   and original_bn["id"] in node["origin_node_ids"] for node in quantized_nodes)


def test_exact_aliases_and_unknown_original_remain_distinct():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.dropout(x, training=False).relu()
    model, inputs = Model().eval(), (torch.randn(2, 4),)
    result = m2m.convert(model, inputs, backend="fx_importer", capture_trace=True)
    trace = _check_trace(result)
    aliases = [r for r in trace["mlir"]["source_correspondence"] if r["status"] == "alias"]
    assert aliases and aliases[0]["alias_of_node_id"]
    absent = {"stage": "original", "status": "unavailable", "reason": "loader already quantized",
              "call_count": None}
    result = m2m.convert(model, inputs, backend="fx_importer", capture_trace=True,
                         original_frontend_snapshot=absent)
    assert result.ok
    assert result.capture_trace["status"] == "diagnostic"
    assert result.capture_trace["graphs"]["original"] == absent
    assert result.capture_trace["graphs"]["original"]["call_count"] is None
    # A chunk returns several values: each selecting frontend call must survive
    # even when decomposition reuses the split result's already-created proxy.
    class TupleModel(torch.nn.Module):
        def forward(self, x):
            a, b, c = x.chunk(3, dim=-1)
            return a + b + c
    from m2m.capture.torch_export import capture_model
    exported = capture_model(TupleModel(), (torch.randn(2, 9),))
    before = str(exported.graph)
    tuple_result = m2m.convert(exported, (), backend="fx_importer", capture_trace=True)
    trace = _check_trace(tuple_result)
    import operator
    assert str(exported.graph) == before
    assert sum(n.target is operator.getitem for n in exported.graph.nodes) == 3
    selectors = [n for n in trace["graphs"]["quantized"]["nodes"]
                 if n["target"] == str(operator.getitem)]
    relations = trace["transformations"][1]["relations"]
    assert all(any(n["id"] in r["source_ids"] for r in relations) for n in selectors)
    assert all(e["producer_value_id"] for e in trace["graphs"]["quantized"]["edges"]
               if e["consumer_node_id"] in {n["id"] for n in selectors} and e["argument_path"] == "args/0")


def test_snapshot_identity_tampering_and_other_backend_are_fail_closed():
    class Model(torch.nn.Module):
        def forward(self, x): return x.relu()
    model, inputs = Model(), (torch.randn(2, 4),)
    original = m2m.capture_frontend_snapshot(model, inputs)
    result = m2m.convert(model, inputs, backend="fx_importer", capture_trace=True,
                         original_frontend_snapshot=original)
    _check_trace(result)
    diagnostic = finalize_trace(result.capture_trace["graphs"], result.mlir_text, path="torch_mlir")
    assert diagnostic["status"] == "diagnostic"
    original["nodes"][0]["module_stack"] = {"tampered": "not identity evidence"}
    failed = m2m.convert(model, inputs, backend="fx_importer", capture_trace=True,
                         original_frontend_snapshot=original)
    assert not failed.ok
    assert "digest" in failed.diagnostics[0]
    # A non-CPU device assertion cannot be certified by the CPU capture.
    from m2m.capture.torch_export import capture_model
    class Cast(torch.nn.Module):
        def forward(self, x): return x.half()
    exported = capture_model(Cast(), inputs)
    guards = [n for n in exported.graph.nodes
              if n.target == torch.ops.aten._assert_tensor_metadata.default]
    assert guards
    for guard in guards:
        guard.kwargs = {**guard.kwargs, "device": torch.device("cuda:0")}
    unknown = m2m.convert(exported, inputs, backend="fx_importer", capture_trace=True, decompose=False)
    assert unknown.capture_trace["status"] == "diagnostic"


def test_traced_failed_capture_is_not_recaptured(monkeypatch):
    import m2m.capture.torch_export as capture
    attempts = []
    def unavailable(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError("unsupported control flow")
    monkeypatch.setattr(capture, "capture_model", unavailable)
    result = m2m.convert(torch.nn.ReLU(), (torch.randn(2, 4),), capture_trace=True)
    assert not result.ok
    assert result.capture_trace["status"] == "diagnostic"
    assert len(attempts) == 1


def test_unused_tuple_selection_requires_exact_typed_dead_result():
    import copy

    def value(identity):
        return {"id": identity, "kind": "tensor", "dtype": "int64", "storage_dtype": "int64",
                "compute_dtype": None, "shape": [1, 1], "device": "cpu",
                "layout": "torch.strided", "stride": [1, 1]}

    source = {"stage": "quantized", "status": "complete", "nodes": [
        {"id": "q:min", "op": "call_function", "target": "aten.min.dim",
         "results": [value("q:min:v0"), value("q:min:v1")]},
        {"id": "q:getitem", "op": "call_function", "target": "<built-in function getitem>",
         "args": [{"node_id": "q:min", "value_id": "q:min:v1"}, 1], "kwargs": {},
         "results": [value("q:getitem:v0")]},
    ], "edges": [{"producer_node_id": "q:min", "producer_value_id": "q:min:v1",
                   "consumer_node_id": "q:getitem", "argument_path": "args/0"}]}
    dest = {"stage": "prepared", "status": "complete", "nodes": [
        {"id": "p:min", "op": "call_function", "target": "aten.min.dim",
         "origin_node_ids": ["q:min"]},
    ], "edges": []}
    relation = graph_relation(source, dest)
    assert relation["status"] == "complete"
    assert any(row.get("proof") == {"tuple_node_id": "q:min", "result_index": 1}
               for row in relation["relations"])

    used = copy.deepcopy(source)
    used["edges"].append({"producer_node_id": "q:getitem", "consumer_node_id": "q:output",
                          "argument_path": "args/0"})
    assert "q:getitem" in graph_relation(used, dest)["unresolved_source_ids"]
    mismatched = copy.deepcopy(source)
    mismatched["nodes"][1]["results"][0]["dtype"] = "int32"
    assert "q:getitem" in graph_relation(mismatched, dest)["unresolved_source_ids"]
    invalid = copy.deepcopy(source)
    invalid["nodes"][1]["args"][1] = 2
    assert "q:getitem" in graph_relation(invalid, dest)["unresolved_source_ids"]


def test_dead_pure_value_chain_cannot_reach_output_or_unknown_effect():
    import copy

    source = {"stage": "quantized", "status": "complete", "nodes": [
        {"id": "q:arange", "op": "call_function", "target": "aten.arange.default"},
        {"id": "q:unsqueeze", "op": "call_function", "target": "aten.unsqueeze.default"},
        {"id": "q:output", "op": "output", "target": "output"},
    ], "edges": [{"producer_node_id": "q:arange", "consumer_node_id": "q:unsqueeze"}]}
    dest = {"stage": "prepared", "status": "complete", "nodes": [], "edges": []}
    relation = graph_relation(source, dest)
    assert relation["status"] == "complete"
    assert {row["source_ids"][0] for row in relation["relations"]} == {"q:arange", "q:unsqueeze"}

    used = copy.deepcopy(source)
    used["edges"].append({"producer_node_id": "q:unsqueeze", "consumer_node_id": "q:output"})
    assert set(graph_relation(used, dest)["unresolved_source_ids"]) == {"q:arange", "q:unsqueeze"}
    effect_unknown = copy.deepcopy(source)
    effect_unknown["nodes"][0]["target"] = "aten.rand.default"
    assert "q:arange" in graph_relation(effect_unknown, dest)["unresolved_source_ids"]


def test_data_dependent_size_guards_survive_preparation_and_mlir():
    import copy

    class Masked(torch.nn.Module):
        def forward(self, x):
            selected = x[x > 0]
            torch.sym_constrain_range_for_size(selected.shape[0], max=4)
            return selected

    result = m2m.convert(Masked(), (torch.tensor([1.0, 0.0, 2.0, 0.0]),),
                         backend="fx_importer", capture_trace=True)
    trace = _check_trace(result)
    assert result.mlir_text.count('"tensor.dim"') == 1
    assert result.mlir_text.count('"cf.assert"') == 3
    relations = trace["transformations"][1]["relations"]
    guards = [row for row in relations if row["kind"] == "structural_call_equivalence"]
    assert len(guards) == 6
    assert any(row["proof"]["diagnostic_text_equated"] is False for row in guards)
    lineage_free = {node["id"] for node in trace["graphs"]["prepared"]["nodes"]
                    if node["target"] in {"aten.sym_size.int", "aten.sym_constrain_range_for_size.default"}}
    assert all(not op["origin_node_ids"] for op in trace["mlir"]["operations"]
               if lineage_free.intersection(op["source_node_ids"]))

    changed = copy.deepcopy(trace["graphs"]["prepared"])
    upper = next(node for node in changed["nodes"] if node["target"] == "<built-in function le>")
    upper["args"][1] = 5
    mismatch = graph_relation(trace["graphs"]["quantized"], changed)
    assert mismatch["status"] == "diagnostic"
    assert any(node["target"] == "<built-in function le>" and node["id"] in mismatch["unresolved_source_ids"]
               for node in trace["graphs"]["quantized"]["nodes"])


def test_grad_disabled_inline_binding_requires_exact_caller_value():
    import copy

    class Wrapped(torch.nn.Module):
        def forward(self, x):
            with torch.set_grad_enabled(False):
                return torch.cos(x.unsqueeze(0))

    result = m2m.convert(Wrapped(), (torch.randn(3),), backend="fx_importer", capture_trace=True)
    assert result.ok
    source, dest = result.capture_trace["graphs"]["quantized"], result.capture_trace["graphs"]["prepared"]
    nested = next(node for node in source["nodes"] if node["target"] == "aten.unsqueeze.default"
                  and node["graph_id"] != "g:quantized:root")
    relation = graph_relation(source, dest)
    assert nested["id"] not in relation["unresolved_source_ids"]
    assert any(row["kind"] == "inlined_parameter_binding" for row in relation["relations"])
    assert any(row["kind"] == "inlined_origin" and row["source_ids"] == [nested["id"]]
               and row["proof"]["exact_caller_bindings"]
               for row in relation["relations"])

    mismatched = copy.deepcopy(source)
    wrapper = next(node for node in mismatched["nodes"] if node["target"] == "wrap_with_set_grad_enabled"
                   and len(node["args"]) == 3)
    wrapper["args"][2] = wrapper["args"][1]
    assert nested["id"] in graph_relation(mismatched, dest)["unresolved_source_ids"]


def test_pt2e_unchanged_nested_graph_requires_exact_body_and_caller():
    import copy

    class Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)

        def forward(self, x):
            with torch.set_grad_enabled(False):
                nested = torch.cos(x.unsqueeze(0))
            return self.fc(nested)

    result = m2m.convert(
        Wrapped().eval(), (torch.randn(4),), backend="fx_importer", capture_trace=True,
        quantization=QuantizationConfig(scheme="int8_static_act_int8_weight"),
    )
    assert result.ok, result.diagnostics
    source, dest = (result.capture_trace["graphs"][name] for name in ("original", "quantized"))
    nested = next(node for node in source["nodes"]
                  if node["target"] == "aten.cos.default" and node["graph_id"] != "g:original:root")
    relation = graph_relation(source, dest)
    assert nested["id"] not in relation["unresolved_source_ids"]
    assert any(row["kind"] == "nested_structural_identity" and nested["id"] in row["source_ids"]
               for row in relation["relations"])

    changed = copy.deepcopy(dest)
    next(node for node in changed["nodes"] if node["target"] == "aten.cos.default"
         and node["graph_id"] != "g:quantized:root")["target"] = "aten.sin.default"
    assert nested["id"] in graph_relation(source, changed)["unresolved_source_ids"]

    wrong_value = copy.deepcopy(dest)
    inner = next(node for node in wrong_value["nodes"] if node["target"] == "aten.cos.default"
                 and node["graph_id"] != "g:quantized:root")
    inner["args"][0]["value_id"] = "not-a-result"
    assert nested["id"] in graph_relation(source, wrong_value)["unresolved_source_ids"]

    wrong_caller = copy.deepcopy(dest)
    wrapper = next(node for node in wrong_caller["nodes"] if node["target"] == "wrap_with_set_grad_enabled")
    wrapper["args"][2] = wrapper["args"][1]
    assert nested["id"] in graph_relation(source, wrong_caller)["unresolved_source_ids"]


def test_nested_matmul_decomposition_retains_the_inner_call_identity():
    class Nested(torch.nn.Module):
        def forward(self, x, y):
            with torch.set_grad_enabled(False):
                value = torch.matmul(x.to(torch.float32), y)
                return value.transpose(0, 1).cos()

    trace = _check_trace(m2m.convert(
        Nested(), (torch.arange(6).reshape(2, 3), torch.randn(3, 4)),
        backend="fx_importer", capture_trace=True))
    source = trace["graphs"]["quantized"]
    inner = next(node for node in source["nodes"]
                 if node["target"] == "aten.matmul.default"
                 and node["graph_id"] != "g:quantized:root")
    relation = trace["transformations"][1]
    assert relation["status"] == "complete"
    assert any(inner["id"] in row["source_ids"] and row["kind"] == "inlined_origin"
               and row["proof"]["exact_caller_bindings"] for row in relation["relations"])


def test_nested_tuple_outputs_keep_outer_getitem_identities():
    class Pair(torch.nn.Module):
        def forward(self, x):
            with torch.set_grad_enabled(False):
                return torch.cos(x).to(x.dtype), torch.sin(x).to(x.dtype)

    trace = _check_trace(m2m.convert(
        Pair(), (torch.randn(1, 3),), backend="fx_importer", capture_trace=True))
    source = trace["graphs"]["quantized"]
    selected = {node["id"] for node in source["nodes"]
                if node["graph_id"] == "g:quantized:root"
                and node["target"] == "<built-in function getitem>"}
    assert len(selected) == 2
    relation = trace["transformations"][1]
    assert relation["status"] == "complete"
    assert selected <= {identity for row in relation["relations"]
                        for identity in row["source_ids"]}


def test_identity_dtype_cast_requires_exact_typed_bypass():
    class IdentityCast(torch.nn.Module):
        def forward(self, x):
            return x.to(torch.float32) + 1

    trace = _check_trace(m2m.convert(
        IdentityCast(), (torch.ones(2, 3),), backend="fx_importer", capture_trace=True))
    source, dest = trace["graphs"]["quantized"], trace["graphs"]["prepared"]
    cast = next(n for n in source["nodes"] if n["target"] == "aten.to.dtype")
    relation = graph_relation(source, dest)
    assert cast["id"] not in relation["unresolved_source_ids"]
    assert any(r["kind"] == "eliminated" and r["source_ids"] == [cast["id"]]
               and r["proof"]["typed_bypass_edges"] for r in relation["relations"])

    import copy
    # One source consumer may expand into several prepared nodes. The exact
    # producer-to-entry edge still proves the no-op cast disappeared, while
    # internal decomposition edges must not be mistaken for that bypass.
    decomposed = copy.deepcopy(dest)
    source_add = next(n for n in source["nodes"] if n["target"] == "aten.add.Tensor")
    prepared_add = next(n for n in decomposed["nodes"] if n["target"] == "aten.add.Tensor")
    tail = copy.deepcopy(prepared_add)
    tail["id"] = "g:prepared:root:decomposition_tail"
    tail["target"] = "aten.relu.default"
    tail["origin_node_ids"] = [source_add["id"]]
    tail["results"][0]["id"] = f"{tail['id']}:v0"
    tail["args"] = [{"node_id": prepared_add["id"],
                      "value_id": prepared_add["results"][0]["id"]}]
    decomposed["nodes"].append(tail)
    decomposed["edges"].append({
        "producer_node_id": prepared_add["id"],
        "producer_value_id": prepared_add["results"][0]["id"],
        "consumer_node_id": tail["id"], "argument_path": "args/0",
        "dtype": prepared_add["results"][0]["dtype"],
        "shape": prepared_add["results"][0]["shape"], "value_kind": "tensor",
    })
    assert cast["id"] not in graph_relation(source, decomposed)["unresolved_source_ids"]

    unproven = copy.deepcopy(dest)
    add = next(n["id"] for n in dest["nodes"] if n["target"] == "aten.add.Tensor")
    edge = next(e for e in unproven["edges"] if e["consumer_node_id"] == add
                and e["argument_path"] == "args/0")
    edge["dtype"] = "float16"
    assert cast["id"] in graph_relation(source, unproven)["unresolved_source_ids"]

    requested_copy = copy.deepcopy(source)
    next(n for n in requested_copy["nodes"] if n["id"] == cast["id"])["kwargs"]["copy"] = True
    assert cast["id"] in graph_relation(requested_copy, dest)["unresolved_source_ids"]

    detached = copy.deepcopy(source)
    detach = next(n for n in detached["nodes"] if n["id"] == cast["id"])
    detach["target"] = "aten.detach_.default"
    detach["args"] = detach["args"][:1]
    bypass = graph_relation(detached, dest)
    proof = next(r["proof"] for r in bypass["relations"] if r["source_ids"] == [cast["id"]])
    assert proof["typed_bypass_edges"]
    assert "autograd metadata is not certified" in proof["scope"]
    next(edge for edge in detached["edges"] if edge["producer_node_id"] == cast["id"])["dtype"] = "float16"
    assert cast["id"] in graph_relation(detached, dest)["unresolved_source_ids"]

    dtype_layout = copy.deepcopy(source)
    changed_cast = next(n for n in dtype_layout["nodes"] if n["id"] == cast["id"])
    changed_cast["target"] = "aten.to.dtype_layout"
    changed_cast["args"] = changed_cast["args"][:1]
    changed_cast["kwargs"] = {
        "dtype": {"kind": "dtype", "value": "torch.float32"},
        "device": {"kind": "device", "value": "cpu"},
        "layout": {"kind": "layout", "value": "torch.strided"},
    }
    assert cast["id"] not in graph_relation(dtype_layout, dest)["unresolved_source_ids"]

    device = copy.deepcopy(source)
    changed_cast = next(n for n in device["nodes"] if n["id"] == cast["id"])
    changed_cast["target"] = "aten.to.device"
    changed_cast["args"] = [changed_cast["args"][0],
                            {"kind": "device", "value": "cpu"},
                            {"kind": "dtype", "value": "torch.float32"}]
    assert cast["id"] not in graph_relation(device, dest)["unresolved_source_ids"]

    class RealCast(torch.nn.Module):
        def forward(self, x):
            return x.to(torch.float16) + 1

    changed = m2m.convert(RealCast(), (torch.ones(2, 3),),
                          backend="fx_importer", capture_trace=True).capture_trace
    source_cast = next(n for n in changed["graphs"]["quantized"]["nodes"]
                       if n["target"] == "aten.to.dtype")
    assert not any(r["kind"] == "eliminated" and source_cast["id"] in r["source_ids"]
                   for r in changed["transformations"][1]["relations"])
