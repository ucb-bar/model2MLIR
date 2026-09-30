"""Expand quant_ext.* ops to pure standard dialects (linalg.generic + arith).

The default output keeps QDQ as `quant_ext.dequantize_*` (the one non-core op family). For a
"purely upstream-MLIR" artifact, this pass lowers those to `linalg.generic` + `arith`
(sitofp/sub/mul) -- no quant dialect, no custom dialect, just core compute dialects. Reuses
``build_dequantize_body`` (single source of truth).
"""

from __future__ import annotations

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.rewriter import InsertPoint, Rewriter

from m2m.ir.decompositions import _attach_region_id, _next_region_id, build_dequantize_body
from m2m.ir.quant.ops import DequantizePerChannelOp, DequantizePerTensorOp


def _expand_dequant(op):
    out_t = op.results[0].type
    out_shape = list(out_t.get_shape())
    axis = op.axis.value.data if getattr(op, "axis", None) is not None else 0
    return build_dequantize_body(op.input, op.scales if hasattr(op, "scales") else op.scale,
                                 op.zero_points if hasattr(op, "zero_points") else op.zero_point,
                                 axis=axis, out_shape=out_shape, out_elem=out_t.element_type)


QUANT_EXPANDERS = {
    DequantizePerChannelOp: _expand_dequant,
    DequantizePerTensorOp: _expand_dequant,
}


def expand_quant_to_standard(module: ModuleOp) -> ModuleOp:
    """Lower every quant_ext.dequantize op to standard linalg+arith, in place. Returns the
    module. Ops it can't lower (dynamic shapes) are left untouched."""
    for op in list(module.walk()):
        handler = QUANT_EXPANDERS.get(type(op))
        if handler is None:
            continue
        built = handler(op)
        if built is None:
            continue
        ops, result = built
        rid = _next_region_id("dequantize")
        for o in ops:
            from m2m.capture.trace import copy_mlir_sources

            copy_mlir_sources(o, op, role="lowering")
            _attach_region_id(o, rid)
            o.attributes["prov.op"] = StringAttr("dequantize")
            o.attributes["prov.family"] = StringAttr("quantize")
        Rewriter.insert_op(ops, InsertPoint.before(op))
        op.results[0].replace_all_uses_with(result)
        Rewriter.erase_op(op)
    return module
