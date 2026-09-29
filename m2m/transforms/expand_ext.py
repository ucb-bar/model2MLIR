"""Expansion pass: high-level ``*_ext`` named ops -> standard linalg-on-tensors.

This is the deterministic legalization that turns the opt-in structured form into the
default portable form. It is the *single source of truth*'s second caller: each handler
reuses the same ``build_*_body`` emitters the importer uses (in
``m2m.ir.decompositions``), so the two paths can never drift.

Always run this before exporting a high-level module to a non-xDSL MLIR toolchain
(``linalg_ext``/``tensor_ext``/``quant_ext`` are unregistered there).
"""

from __future__ import annotations

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.rewriter import InsertPoint, Rewriter

from m2m.ir.decompositions import (
    _attach_region_id,
    _next_region_id,
    build_layer_norm_body,
    build_softmax_body,
)
from m2m.ir.linalg_ext.ops import LayerNormOp, SoftmaxOp


def _expand_softmax(op: SoftmaxOp):
    return build_softmax_body(op.input, dim=op.dim.value.data), "softmax", "normalization"


def _expand_layer_norm(op: LayerNormOp):
    rank = len(op.input.type.get_shape())
    axis = op.axis.value.data if op.axis is not None else rank - 1
    k = max(1, rank - axis)
    weight = op.weight if op.weight is not None else None
    bias = op.bias if op.bias is not None else None
    return build_layer_norm_body(op.input, weight, bias, eps=op.eps.value.data, k=k), "layer_norm", "normalization"


# op type -> handler. The set of keys MUST equal the set of named ops the high-level form
# can emit (enforced by tests/test_transforms.py::test_every_named_op_has_expander).
EXPANDERS = {
    SoftmaxOp: _expand_softmax,
    LayerNormOp: _expand_layer_norm,
}


def expand_to_linalg(module: ModuleOp) -> ModuleOp:
    """Lower every ``*_ext`` named op in ``module`` to standard dialects, in place.

    Each named op is replaced by its standard body (built via the shared ``build_*_body``
    emitters), the body ops are tagged with the same taxonomy (``prov.op`` / ``prov.family``
    / ``prov.region_id``) as the importer would produce, and the module's ``prov.level`` is
    set to ``linalg-on-tensors``. Returns the same module for chaining."""
    for op in list(module.walk()):
        handler = EXPANDERS.get(type(op))
        if handler is None:
            continue
        built, op_kind, family = handler(op)
        if built is None:
            continue  # dynamic shapes -- leave the named op (caller must handle)
        ops, result = built
        rid = _next_region_id(op_kind)
        for o in ops:
            from m2m.capture.trace import copy_mlir_sources

            copy_mlir_sources(o, op, role="lowering")
            _attach_region_id(o, rid)
            o.attributes["prov.op"] = StringAttr(op_kind)
            o.attributes["prov.family"] = StringAttr(family)
        Rewriter.insert_op(ops, InsertPoint.before(op))
        op.results[0].replace_all_uses_with(result)
        Rewriter.erase_op(op)

    module.attributes["prov.level"] = StringAttr("linalg-on-tensors")
    return module
