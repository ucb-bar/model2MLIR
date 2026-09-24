"""Turn supported PT2E per-tensor Q/DQ linear regions into integer contractions.

This operates on the converted PT2E ``GraphModule`` before capture.  The same
rewritten module must be used for the host golden: integer accumulation followed
by scaling is a different floating-point evaluation order from PT2E's portable
dequantize-then-matmul form.  Unsupported qparams remain unchanged and are
reported, so callers can enforce their own coverage gate.
"""

from __future__ import annotations

import math
from typing import Any


def integerize_pt2e_linear(model: Any, example_inputs: tuple[Any, ...]) -> tuple[Any, dict[str, Any]]:
    """Rewrite symmetric int8 per-tensor ``aten.linear`` regions in place.

    Each matched pair of dequantize nodes supplies the actual frozen activation
    and weight scales and integer tensors.  We preserve the bias in f32 after
    i32 accumulation and apply activation scale before weight scale.  Dynamic
    dimensions, nonzero zero points, non-int8 storage and unsupported graph
    shapes are left in portable PT2E form with an explicit refusal.
    """
    import torch
    from torch.fx import GraphModule, Node
    from torch.fx.passes.shape_prop import ShapeProp

    if not isinstance(model, GraphModule):
        raise TypeError("PT2E integerization requires a converted GraphModule")
    # Converted PT2E Q/DQ nodes commonly lack ``meta['val']`` even though their
    # adjacent linear node has it. Propagate the exact capture inputs once so
    # matching is based on observed shapes, never inferred from a name.
    ShapeProp(model).propagate(*example_inputs)

    dequant = torch.ops.quantized_decomposed.dequantize_per_tensor.default
    linear = torch.ops.aten.linear.default
    refusals: list[dict[str, str]] = []
    converted = 0
    seen = 0

    def _shape(node: Node) -> tuple[int, ...]:
        value = node.meta.get("tensor_meta") or node.meta.get("val")
        dims = getattr(value, "shape", None)
        if dims is None or not all(isinstance(dim, int) and dim > 0 for dim in dims):
            raise ValueError("static positive tensor shape is unavailable")
        return tuple(dims)

    def _quant_operand(node: Any) -> tuple[Node, float]:
        if not isinstance(node, Node) or node.op != "call_function" or node.target != dequant:
            raise ValueError("operand is not a per-tensor PT2E dequantize")
        if len(node.args) < 6 or not isinstance(node.args[0], Node):
            raise ValueError("dequantize has no integer tensor or complete qparams")
        _, scale, zero_point, qmin, qmax, dtype = node.args[:6]
        if (
            not isinstance(scale, (int, float))
            or not math.isfinite(float(scale))
            or float(scale) <= 0
            or zero_point != 0
            or not isinstance(qmin, int)
            or not isinstance(qmax, int)
            or qmin < -128
            or qmax > 127
            or dtype != torch.int8
        ):
            raise ValueError("requires finite positive scalar scale, zero point 0 and int8 range")
        return node.args[0], float(scale)

    graph = model.graph
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target != linear:
            continue
        seen += 1
        try:
            if len(node.args) < 2:
                raise ValueError("linear has fewer than two operands")
            activation, activation_scale = _quant_operand(node.args[0])
            weight, weight_scale = _quant_operand(node.args[1])
            activation_shape = _shape(node.args[0])
            weight_shape = _shape(node.args[1])
            result_shape = _shape(node)
            if (
                len(activation_shape) < 2
                or len(weight_shape) != 2
                or activation_shape[-1] != weight_shape[-1]
                or result_shape != (*activation_shape[:-1], weight_shape[0])
            ):
                raise ValueError("linear has unsupported activation, weight or output geometry")
            bias = node.args[2] if len(node.args) > 2 else None
        except ValueError as exc:
            refusals.append({"node": node.name, "reason": str(exc)})
            continue

        with graph.inserting_before(node):
            lhs = graph.call_function(
                torch.ops.aten.reshape.default,
                args=(activation, [-1, activation_shape[-1]]),
            )
            rhs = graph.call_function(torch.ops.aten.transpose.int, args=(weight, 0, 1))
            accumulator = graph.call_function(torch.ops.aten._int_mm.default, args=(lhs, rhs))
            value = graph.call_function(
                torch.ops.aten._to_copy.default,
                args=(accumulator,),
                kwargs={"dtype": torch.float32},
            )
            value = graph.call_function(torch.ops.aten.mul.Tensor, args=(value, activation_scale))
            value = graph.call_function(torch.ops.aten.mul.Tensor, args=(value, weight_scale))
            value = graph.call_function(torch.ops.aten.reshape.default, args=(value, list(result_shape)))
            if bias is not None:
                value = graph.call_function(torch.ops.aten.add.Tensor, args=(value, bias))
        node.replace_all_uses_with(value)
        graph.erase_node(node)
        converted += 1

    if converted:
        graph.eliminate_dead_code()
        graph.lint()
        model.recompile()
    return model, {
        "schema": "m2m.pt2e-integerize.v1",
        "linear_seen": seen,
        "linear_integerized": converted,
        "refusals": refusals,
    }
