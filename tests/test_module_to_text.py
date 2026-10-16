"""Portable module emission preserves the source attributes used by captures."""

import pytest
from xdsl.context import Context
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region
from xdsl.parser import Parser

from m2m.capture.torch_mlir_bridge import module_to_text


@pytest.mark.parametrize("generic", [False, True])
def test_module_emission_roundtrips_source_attributes(generic):
    block = Block()
    block.add_op(func.ReturnOp())
    entry = func.FuncOp("forward", ((), ()), region=Region(block))
    entry.attributes["m2m.source_id"] = builtin.StringAttr("source-7")
    module = builtin.ModuleOp([entry])

    text = module_to_text(module, generic=generic)
    context = Context()
    context.load_dialect(builtin.Builtin)
    context.load_dialect(func.Func)
    restored = Parser(context, text).parse_module()
    restored.verify()
    assert next(iter(restored.ops)).attributes["m2m.source_id"] == builtin.StringAttr("source-7")
