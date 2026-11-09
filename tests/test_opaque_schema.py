"""Opaque import retains ordered tensor results, operands, and schema literals."""
import json
import operator

import pytest
import torch
from xdsl.dialects.builtin import ArrayAttr
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp

from m2m.capture.trace import snapshot_exported_program
from m2m.ir.import_fx import FXImporter


_LIB = torch.library.Library("m2m_opaque_test", "DEF")
_LIB.define("pair(Tensor x, str mode, int? exponent=None, int[] axes=[], *, Tensor? bias=None, Scalar factor=1., bool flag=False) -> (Tensor, Tensor)")


def _pair(x, mode, exponent=None, axes=(), *, bias=None, factor=1., flag=False):
    return torch.empty_like(x, dtype=torch.uint8), x.new_empty(x.shape[:-1], dtype=torch.int32)


_LIB.impl("pair", _pair, "CompositeExplicitAutograd")
torch.library.register_fake("m2m_opaque_test::pair")(_pair)


class Pair(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bias", torch.ones(3))

    def forward(self, x):
        codes, exponent = torch.ops.m2m_opaque_test.pair.default(
            x, "static", -7, [-1, 0], bias=self.bias, factor=float("inf"), flag=True)
        return exponent, codes


def test_opaque_tuple_results_literals_and_lifted_input_lineage():
    exported = torch.export.export(Pair(), (torch.ones(2, 3),))
    snapshot_exported_program(exported, stage="adapter")
    adapter_ids = {node.name: node.meta["_m2m_node_id"] for node in exported.graph.nodes}
    snapshot_exported_program(exported, stage="prepared")
    importer = FXImporter(opaque_custom_ops=True)
    module = importer.import_graph(exported)
    assert not [d.message for d in importer.diagnostics if d.level == "error"]
    module.verify()
    call, = [op for op in module.walk() if isinstance(op, CallOp)]
    assert [str(result.type) for result in call.results] == ["tensor<2x3xi8>", "tensor<2xi32>"]
    assert [str(value.type) for value in call.arguments] == ["tensor<2x3xf32>", "tensor<3xf32>"]
    assert call.attributes["m2m.schema"].data == str(torch.ops.m2m_opaque_test.pair.default._schema)
    literals = {name.removeprefix("m2m.literal."): json.loads(value.data)
                for name, value in call.attributes.items() if name.startswith("m2m.literal.")}
    assert literals == {"mode": "static", "exponent": -7, "axes": [-1, 0],
                        "factor": {"float": "inf"}, "flag": True}
    node = next(node for node in exported.graph.nodes if node.target == torch.ops.m2m_opaque_test.pair.default)
    assert [value.data for value in call.attributes["prov.source_node_ids"]] == [node.meta["_m2m_node_id"]]
    assert adapter_ids[node.name] in [value.data for value in call.attributes["prov.origin_node_ids"]]
    returned, = [op for op in module.walk() if isinstance(op, ReturnOp)]
    assert tuple(returned.arguments) == (call.results[1], call.results[0])
    main = next(op for op in module.body.block.ops if isinstance(op, FuncOp) and op.sym_name.data == "forward")
    assert isinstance(main.arg_attrs, ArrayAttr)
    bias = next(i for i, node in enumerate(n for n in exported.graph.nodes if n.op == "placeholder")
                if node.name in exported.graph_signature.inputs_to_buffers)
    assert main.arg_attrs.data[bias].data["prov.input_target"].data == "bias"
    assert adapter_ids["b_bias"] in [value.data for value in main.arg_attrs.data[bias].data["prov.origin_node_ids"]]


@pytest.mark.parametrize("opaque_custom_ops", [False, True])
def test_direct_tuple_return_and_defaults(opaque_custom_ops):
    class Direct(torch.nn.Module):
        def forward(self, x):
            return torch.ops.m2m_opaque_test.pair.default(x, "dynamic")

    exported = torch.export.export(Direct(), (torch.ones(2, 3),))
    # FX also permits a tuple producer to be returned directly, with no getitems.
    graph = exported.graph
    producer = next(node for node in graph.nodes if node.target == torch.ops.m2m_opaque_test.pair.default)
    output = next(node for node in graph.nodes if node.op == "output")
    output.args = ((producer,),)
    for node in list(graph.nodes):
        if node.target is operator.getitem:
            graph.erase_node(node)
    module = FXImporter(opaque_custom_ops=opaque_custom_ops).import_graph(exported)
    module.verify()
    call, = [op for op in module.walk() if isinstance(op, CallOp)]
    returned, = [op for op in module.walk() if isinstance(op, ReturnOp)]
    assert tuple(returned.arguments) == tuple(call.results)
    assert len(call.results) == (2 if opaque_custom_ops else 1)
    if not opaque_custom_ops:
        assert "m2m.schema" not in call.attributes
        return
    assert json.loads(call.attributes["m2m.literal.exponent"].data) is None
    assert json.loads(call.attributes["m2m.literal.bias"].data) is None
    assert json.loads(call.attributes["m2m.literal.axes"].data) == []
    assert json.loads(call.attributes["m2m.literal.factor"].data) == 1.
    assert json.loads(call.attributes["m2m.literal.flag"].data) is False


def test_distinct_result_signatures_have_matching_externs():
    class TwoShapes(torch.nn.Module):
        def forward(self, x, y):
            return (*torch.ops.m2m_opaque_test.pair.default(x, "dynamic"),
                    *torch.ops.m2m_opaque_test.pair.default(y, "dynamic"))

    exported = torch.export.export(TwoShapes(), (torch.ones(2, 3), torch.ones(4, 5)))
    module = FXImporter(opaque_custom_ops=True).import_graph(exported)
    module.verify()
    declarations = {op.sym_name.data: op for op in module.body.block.ops
                    if isinstance(op, FuncOp) and not op.body.blocks}
    calls = [op for op in module.walk() if isinstance(op, CallOp)]
    assert len(calls) == len(declarations) == 2
    for call in calls:
        assert list(declarations[call.callee.string_value()].function_type.outputs) == [r.type for r in call.results]


@pytest.mark.parametrize("prepare", [False, True])
@pytest.mark.parametrize("conversion", ["to", "to_copy"])
def test_identity_aliases_retain_precise_ssa_result_indices(prepare, conversion):
    class Aliases(torch.nn.Module):
        def forward(self, x):
            same = (torch.ops.aten.to.dtype(x, torch.float32) if conversion == "to"
                    else torch.ops.aten._to_copy.default(x, dtype=torch.float32))
            codes, exponent = torch.ops.m2m_opaque_test.pair.default(same, "dynamic")
            copied = (torch.ops.aten.to.dtype(exponent, torch.int32) if conversion == "to"
                      else torch.ops.aten._to_copy.default(exponent, dtype=torch.int32))
            return codes, copied

    exported = torch.export.export(Aliases(), (torch.ones(2, 3),))
    snapshot_exported_program(exported, stage="adapter")
    source_ids = [node.meta["_m2m_node_id"] for node in exported.graph.nodes
                  if node.target in {torch.ops.aten._to_copy.default, torch.ops.aten.to.dtype}]
    if prepare:
        from m2m.capture.torch_export import _prepare_exported_program

        exported, _ = _prepare_exported_program(exported, run_default_decompositions=True,
                                               export_decomposition_table=None, capture_trace=True,
                                               opaque_custom_ops=True)
    snapshot_exported_program(exported, stage="prepared")
    module = FXImporter(opaque_custom_ops=True).import_graph(exported)
    module.verify()
    main = next(op for op in module.body.block.ops if isinstance(op, FuncOp) and op.sym_name.data == "forward")
    argument_aliases = json.loads(main.arg_attrs.data[0].data["prov.value_aliases"].data)
    assert argument_aliases[0]["result_index"] == argument_aliases[0]["alias_result_index"] == 0
    assert source_ids[0] in argument_aliases[0]["node_ids"]
    call = next(op for op in module.walk() if isinstance(op, CallOp))
    result_aliases = json.loads(call.attributes["prov.value_aliases"].data)
    assert result_aliases[0]["result_index"] == 1
    assert result_aliases[0]["alias_result_index"] == 0
    assert source_ids[1] in result_aliases[0]["node_ids"]
    assert not any(op.name == "tensor.cast" for op in module.walk())


@pytest.mark.parametrize("case", ["missing", "ambiguous"])
@pytest.mark.parametrize("typed", [False, True])
def test_prepared_alias_refuses_unproved_input_binding(case, typed):
    class Inputs(torch.nn.Module):
        def forward(self, x, y):
            return x, y

    exported = torch.export.export(Inputs(), (torch.ones(2), torch.zeros(2)))
    snapshot_exported_program(exported, stage="prepared")
    inputs = [node.meta["_m2m_node_id"] for node in exported.graph.nodes if node.op == "placeholder"]
    exported._m2m_trace_eliminations = [{
        "source_ids": ["eliminated:alias"], "kind": "eliminated", "reason": "identity decomposition",
        "alias_input_source_ids": ["unrecorded:input"] if case == "missing" else inputs,
        "alias_input_result_index": 0,
    }]
    if typed:
        from m2m.capture.trace import _literal

        exported._m2m_trace_eliminations[0]["alias_input_abi"] = _literal(torch.ones(2))
    with pytest.raises(ValueError, match=f"{case.capitalize()} prepared identity alias input binding"):
        FXImporter(opaque_custom_ops=True).import_graph(exported)


@pytest.mark.parametrize("opaque_custom_ops", [False, True])
def test_schema_abi_requires_explicit_provider_opt_in(opaque_custom_ops):
    exported = torch.export.export(Pair(), (torch.ones(2, 3),))
    module = FXImporter(opaque_custom_ops=opaque_custom_ops).import_graph(exported)
    module.verify()
    call, = [op for op in module.walk() if isinstance(op, CallOp)]
    returned, = [op for op in module.walk() if isinstance(op, ReturnOp)]
    assert len(call.results) == (2 if opaque_custom_ops else 1)
    assert len(call.arguments) == (2 if opaque_custom_ops else 1)
    assert ("m2m.schema" in call.attributes) == opaque_custom_ops
    assert any(name.startswith("m2m.literal.") for name in call.attributes) == opaque_custom_ops
    assert len(returned.arguments) == (2 if opaque_custom_ops else 1)
    if not opaque_custom_ops:
        assert tuple(returned.arguments) == (call.results[0],)


@pytest.mark.parametrize("index", [-1, 9])
def test_strict_tuple_index_handling_requires_provider_opt_in(index):
    exported = torch.export.export(Pair(), (torch.ones(2, 3),))
    node = next(node for node in exported.graph.nodes
                if node.target is operator.getitem and node.args[1] == 1)
    node.args = (node.args[0], index)
    # Default opaque import retains its primary-result representative behavior.
    FXImporter().import_graph(exported).verify()
    if index == 9:
        with pytest.raises(ValueError, match="Invalid tuple result index"):
            FXImporter(opaque_custom_ops=True).import_graph(exported)
    else:
        module = FXImporter(opaque_custom_ops=True).import_graph(exported)
        module.verify()
        call, = [op for op in module.walk() if isinstance(op, CallOp)]
        returned, = [op for op in module.walk() if isinstance(op, ReturnOp)]
        assert returned.arguments[0] is call.results[-1]


@pytest.mark.parametrize("conversion", ["to", "to_copy"])
def test_prepared_reshape_alias_uses_certified_input_abi(conversion):
    from m2m.capture.torch_export import _prepare_exported_program

    class ReshapeAlias(torch.nn.Module):
        def forward(self, x):
            expanded = x.unsqueeze(2).expand(1, 2, 3, 4, 4)
            reshaped = expanded.reshape(1, 6, 4, 4)
            same = (torch.ops.aten.to.dtype(reshaped, torch.float32) if conversion == "to"
                    else torch.ops.aten._to_copy.default(reshaped, dtype=torch.float32))
            return torch.ops.m2m_opaque_test.pair.default(same, "dynamic")

    exported = torch.export.export(ReshapeAlias(), (torch.ones(1, 2, 4, 4),))
    snapshot_exported_program(exported, stage="adapter")
    alias = next(node for node in exported.graph.nodes if node.target in
                 {torch.ops.aten.to.dtype, torch.ops.aten._to_copy.default})
    alias_id = alias.meta["_m2m_node_id"]
    prepared, _ = _prepare_exported_program(exported, run_default_decompositions=True,
                                          export_decomposition_table=None, capture_trace=True,
                                          opaque_custom_ops=True)
    snapshot_exported_program(prepared, stage="prepared")
    module = FXImporter(opaque_custom_ops=True).import_graph(prepared)
    module.verify()
    call, = [op for op in module.walk() if isinstance(op, CallOp)]
    assert str(call.arguments[0].type) == "tensor<1x6x4x4xf32>"
    aliases = json.loads(call.arguments[0].owner.attributes["prov.value_aliases"].data)
    assert any(alias_id in row["node_ids"] for row in aliases)
