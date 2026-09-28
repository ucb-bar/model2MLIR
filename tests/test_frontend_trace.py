"""Coarse source-graph -> transformed graph -> final MLIR trace qualification."""
import hashlib
import json

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
    unproven = copy.deepcopy(dest)
    add = next(n["id"] for n in dest["nodes"] if n["target"] == "aten.add.Tensor")
    edge = next(e for e in unproven["edges"] if e["consumer_node_id"] == add
                and e["argument_path"] == "args/0")
    edge["dtype"] = "float16"
    assert cast["id"] in graph_relation(source, unproven)["unresolved_source_ids"]

    requested_copy = copy.deepcopy(source)
    next(n for n in requested_copy["nodes"] if n["id"] == cast["id"])["kwargs"]["copy"] = True
    assert cast["id"] in graph_relation(requested_copy, dest)["unresolved_source_ids"]

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
