"""Turn supported PT2E per-tensor Q/DQ contractions into integer contractions.

This operates on the converted PT2E ``GraphModule`` before capture.  The same
rewritten module must be used for the host golden: integer accumulation followed
by scaling is a different floating-point evaluation order from PT2E's portable
dequantize-then-matmul form.  Unsupported qparams remain unchanged and are
reported, so callers can enforce their own coverage gate.
"""

from __future__ import annotations

import math
from itertools import product
from typing import Any


def integerize_pt2e(model: Any, example_inputs: tuple[Any, ...]) -> tuple[Any, dict[str, Any]]:
    """Rewrite symmetric int8 per-tensor ``aten.linear`` and ``aten.conv2d`` in place.

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
    conv2d = torch.ops.aten.conv2d.default
    matmuls = {
        torch.ops.aten.matmul.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
    }
    unsupported_contractions = {
        torch.ops.aten.addmm.default,
        torch.ops.aten.convolution.default,
    }
    refusals: list[dict[str, str]] = []
    quantized_by_kind = {
        kind: {"seen": 0, "integerized": 0, "remaining": 0}
        for kind in ("linear", "conv2d", "matmul", "unsupported")
    }
    counts = {
        "linear_seen": 0, "linear_integerized": 0,
        "conv2d_seen": 0, "conv2d_integerized": 0,
        "matmul_seen": 0, "matmul_integerized": 0,
        "quantized_contractions_seen": 0,
        "quantized_contractions_integerized": 0,
        "integer_mm_emitted": 0,
        "max_reduction_k": 0,
    }

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
        storage_meta = node.args[0].meta.get("tensor_meta")
        if getattr(storage_meta, "dtype", None) != torch.int8:
            raise ValueError("dequantize source is not an observed int8 tensor")
        return node.args[0], float(scale)

    def _pair(raw: Any, name: str, *, allow_zero: bool = False) -> tuple[int, int]:
        values = (raw, raw) if isinstance(raw, int) else tuple(raw) if isinstance(raw, (list, tuple)) else ()
        minimum = 0 if allow_zero else 1
        if len(values) != 2 or not all(isinstance(v, int) and v >= minimum for v in values):
            raise ValueError(f"unsupported {name}: {raw!r}")
        return values[0], values[1]

    graph = model.graph
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in ({linear, conv2d} | matmuls | unsupported_contractions):
            continue
        quantized = any(
            isinstance(arg, Node) and arg.op == "call_function" and arg.target == dequant
            for arg in node.args
        )
        if quantized:
            counts["quantized_contractions_seen"] += 1
        if node.target in unsupported_contractions:
            if quantized:
                quantized_by_kind["unsupported"]["seen"] += 1
                refusals.append({
                    "kind": str(node.target), "node": node.name,
                    "reason": "PT2E Q/DQ contraction has no verified integer rewrite",
                })
            continue
        kind = "linear" if node.target == linear else "conv2d" if node.target == conv2d else "matmul"
        counts[f"{kind}_seen"] += 1
        if not quantized:
            # Plain floating-point islands were not selected by PT2E and do
            # not belong to the integer coverage claim.
            continue
        quantized_by_kind[kind]["seen"] += 1
        try:
            if len(node.args) < 2:
                raise ValueError(f"{kind} has fewer than two operands")
            activation, activation_scale = _quant_operand(node.args[0])
            weight, weight_scale = _quant_operand(node.args[1])
            activation_shape = _shape(node.args[0])
            weight_shape = _shape(node.args[1])
            result_shape = _shape(node)
            bias = node.args[2] if kind != "matmul" and len(node.args) > 2 else None
            if kind == "linear":
                if (
                    len(activation_shape) < 2
                    or len(weight_shape) != 2
                    or activation_shape[-1] != weight_shape[-1]
                    or result_shape != (*activation_shape[:-1], weight_shape[0])
                ):
                    raise ValueError("linear has unsupported activation, weight or output geometry")
            elif kind == "conv2d":
                stride = _pair(node.args[3] if len(node.args) > 3 else 1, "stride")
                padding = _pair(node.args[4] if len(node.args) > 4 else 0, "padding", allow_zero=True)
                dilation = _pair(node.args[5] if len(node.args) > 5 else 1, "dilation")
                groups = node.args[6] if len(node.args) > 6 else 1
                if groups != 1:
                    raise ValueError("grouped convolution needs a separate integer layout")
                if len(activation_shape) != 4 or len(weight_shape) != 4 or len(result_shape) != 4:
                    raise ValueError("conv2d requires static NCHW/OIHW rank-4 geometry")
                n, cin, h, w = activation_shape
                cout, wcin, kh, kw = weight_shape
                on, oc, oh, ow = result_shape
                sh, sw = stride
                ph, pw = padding
                dh, dw = dilation
                if (
                    n != on or cin != wcin or cout != oc
                    or oh != (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
                    or ow != (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
                ):
                    raise ValueError("conv2d output disagrees with input, weight or window geometry")
            else:
                if (
                    len(activation_shape) != len(weight_shape)
                    or len(activation_shape) < 2
                    or activation_shape[:-2] != weight_shape[:-2]
                    or activation_shape[-1] != weight_shape[-2]
                    or result_shape != (*activation_shape[:-2], activation_shape[-2], weight_shape[-1])
                ):
                    raise ValueError("matmul has unsupported broadcasting or output geometry")
                batch_shape = activation_shape[:-2]
                batch_count = math.prod(batch_shape)
                if batch_count > 256:
                    raise ValueError(f"batched integer matmul would unroll {batch_count} instances (>256)")
            reduction = activation_shape[-1] if kind != "conv2d" else cin * kh * kw
            # Each product of signed int8 values is accumulated into i32.
            # Refuse shapes where the declared quant ranges could overflow it.
            # This check is conservative and independent of observed samples.
            if reduction * 128 * 128 > (1 << 31) - 1:
                raise ValueError(f"i32 accumulator may overflow for reduction K={reduction}")
            counts["max_reduction_k"] = max(counts["max_reduction_k"], reduction)
        except ValueError as exc:
            refusals.append({"kind": kind, "node": node.name, "reason": str(exc)})
            continue

        with graph.inserting_before(node):
            if kind == "linear":
                lhs = graph.call_function(
                    torch.ops.aten.reshape.default,
                    args=(activation, [-1, activation_shape[-1]]),
                )
                rhs = graph.call_function(torch.ops.aten.transpose.int, args=(weight, 0, 1))
            elif kind == "conv2d":
                # Every int8 value is exactly representable as f32, but no float
                # im2col is needed: zero-pad, slice and concatenate on int8 itself.
                padded = activation
                if ph or pw:
                    padded = graph.call_function(
                        torch.ops.aten.constant_pad_nd.default,
                        args=(activation, [pw, pw, ph, ph], 0),
                    )
                patches = []
                for ih in range(kh):
                    for iw in range(kw):
                        patch = graph.call_function(
                            torch.ops.aten.slice.Tensor,
                            args=(padded, 2, ih * dh, ih * dh + oh * sh, sh),
                        )
                        patch = graph.call_function(
                            torch.ops.aten.slice.Tensor,
                            args=(patch, 3, iw * dw, iw * dw + ow * sw, sw),
                        )
                        patch = graph.call_function(
                            torch.ops.aten.permute.default,
                            args=(patch, [0, 2, 3, 1]),
                        )
                        patches.append(graph.call_function(
                            torch.ops.aten.reshape.default,
                            args=(patch, [n * oh * ow, cin]),
                        ))
                lhs = patches[0] if len(patches) == 1 else graph.call_function(
                    torch.ops.aten.cat.default, args=(patches, 1)
                )
                rhs = graph.call_function(
                    torch.ops.aten.permute.default,
                    args=(weight, [0, 2, 3, 1]),
                )
                rhs = graph.call_function(
                    torch.ops.aten.reshape.default,
                    args=(rhs, [cout, cin * kh * kw]),
                )
                rhs = graph.call_function(torch.ops.aten.transpose.int, args=(rhs, 0, 1))
            if kind == "matmul":
                matrices = []
                for batch_indices in product(*(range(extent) for extent in batch_shape)):
                    lhs = activation
                    rhs = weight
                    for index in batch_indices:
                        lhs = graph.call_function(torch.ops.aten.select.int, args=(lhs, 0, index))
                        rhs = graph.call_function(torch.ops.aten.select.int, args=(rhs, 0, index))
                    matrices.append(graph.call_function(torch.ops.aten._int_mm.default, args=(lhs, rhs)))
                accumulator = matrices[0] if not batch_shape else graph.call_function(
                    torch.ops.aten.reshape.default,
                    args=(graph.call_function(torch.ops.aten.stack.default, args=(matrices, 0)), list(result_shape)),
                )
                counts["integer_mm_emitted"] += len(matrices)
            else:
                accumulator = graph.call_function(torch.ops.aten._int_mm.default, args=(lhs, rhs))
                counts["integer_mm_emitted"] += 1
            value = graph.call_function(
                torch.ops.aten._to_copy.default,
                args=(accumulator,),
                kwargs={"dtype": torch.float32},
            )
            value = graph.call_function(torch.ops.aten.mul.Tensor, args=(value, activation_scale))
            value = graph.call_function(torch.ops.aten.mul.Tensor, args=(value, weight_scale))
            if kind == "linear":
                value = graph.call_function(torch.ops.aten.reshape.default, args=(value, list(result_shape)))
            elif kind == "conv2d":
                value = graph.call_function(torch.ops.aten.reshape.default, args=(value, [n, oh, ow, cout]))
                value = graph.call_function(torch.ops.aten.permute.default, args=(value, [0, 3, 1, 2]))
            if bias is not None:
                if kind == "conv2d":
                    bias = graph.call_function(torch.ops.aten.reshape.default, args=(bias, [1, cout, 1, 1]))
                value = graph.call_function(torch.ops.aten.add.Tensor, args=(value, bias))
        value.meta["pt2e_integerized_from"] = node.name
        node.replace_all_uses_with(value)
        graph.erase_node(node)
        counts[f"{kind}_integerized"] += 1
        counts["quantized_contractions_integerized"] += 1
        quantized_by_kind[kind]["integerized"] += 1

    if counts["integer_mm_emitted"]:
        graph.eliminate_dead_code()
        graph.lint()
        model.recompile()
    remaining_dequant = sum(
        1 for node in graph.nodes if node.op == "call_function" and node.target == dequant
    )
    for row in quantized_by_kind.values():
        row["remaining"] = row["seen"] - row["integerized"]
    return model, {
        "schema": "m2m.pt2e-integerize.v1",
        **counts,
        "quantized_contractions_remaining": (
            counts["quantized_contractions_seen"] - counts["quantized_contractions_integerized"]
        ),
        "remaining_dequant_count": remaining_dequant,
        "quantized_by_kind": quantized_by_kind,
        "accumulator_bound_checked": True,
        "refusals": refusals,
    }
