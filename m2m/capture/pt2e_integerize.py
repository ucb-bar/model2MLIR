"""Turn supported PT2E Q/DQ contractions into integer contractions.

This operates on the converted PT2E ``GraphModule`` before capture.  The same
rewritten module must be used for the host golden: integer accumulation followed
by scaling is a different floating-point evaluation order from PT2E's portable
dequantize-then-matmul form.  Unsupported qparams remain unchanged and are
reported, so callers can enforce their own coverage gate.
"""

from __future__ import annotations

import math
import hashlib
from itertools import product
from typing import Any


def integerize_pt2e(model: Any, example_inputs: tuple[Any, ...]) -> tuple[Any, dict[str, Any]]:
    """Rewrite symmetric int8 linear, conv2d and static matmul contractions in place.

    Each matched pair of dequantize nodes supplies the actual frozen activation
    and weight scales and integer tensors.  We preserve the bias in f32 after
    i32 accumulation and apply activation scale before weight scale.  Dynamic
    dimensions, nonzero zero points, non-int8 storage and unsupported graph
    shapes are left in portable PT2E form with an explicit refusal. Activation
    scales must be per tensor. Frozen per-output-channel weight scales are
    supported; a scale along the reduction dimension cannot be factored out.
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
    channel_dequant = torch.ops.quantized_decomposed.dequantize_per_channel.default
    dequants = {dequant, channel_dequant}
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
    qparam_receipts: list[dict[str, Any]] = []
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

    def _frozen(node: Any, name: str) -> Any:
        if not isinstance(node, Node) or node.op != "get_attr":
            raise ValueError(f"per-channel {name} must be a frozen tensor attribute")
        value = model
        for component in str(node.target).split("."):
            value = getattr(value, component)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"per-channel {name} is not a frozen tensor")
        return value.detach()

    def _tensor_digest(value: Any) -> str:
        return hashlib.sha256(value.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()

    def _quant_operand(node: Any) -> tuple[Node, Any, int | None, dict[str, Any]]:
        if not isinstance(node, Node) or node.op != "call_function" or node.target not in dequants:
            raise ValueError("operand is not a supported PT2E dequantize")
        per_channel = node.target == channel_dequant
        if len(node.args) < (7 if per_channel else 6) or not isinstance(node.args[0], Node):
            raise ValueError("dequantize has no integer tensor or complete qparams")
        if node.kwargs.get("out_dtype", torch.float32) not in (None, torch.float32):
            raise ValueError("dequantize output must be float32")
        if per_channel:
            _, scale, zero_point, axis, qmin, qmax, dtype = node.args[:7]
        else:
            _, scale, zero_point, qmin, qmax, dtype = node.args[:6]
            axis = None
        if (
            not isinstance(qmin, int)
            or not isinstance(qmax, int)
            or qmin < -128
            or qmax > 127
            or qmin >= qmax
            or dtype != torch.int8
        ):
            raise ValueError("requires a valid signed int8 quantization range")
        storage_meta = node.args[0].meta.get("tensor_meta")
        if getattr(storage_meta, "dtype", None) != torch.int8:
            raise ValueError("dequantize source is not an observed int8 tensor")
        receipt: dict[str, Any] = {"granularity": "channel" if per_channel else "tensor",
                                   "quant_min": qmin, "quant_max": qmax, "zero_point": 0}
        if not per_channel:
            if (not isinstance(scale, (int, float)) or not math.isfinite(float(scale))
                    or float(scale) <= 0 or zero_point != 0):
                raise ValueError("requires finite positive scalar scale and zero point 0")
            receipt["scale"] = float(scale)
            return node.args[0], float(scale), None, receipt
        shape = _shape(node.args[0])
        if not isinstance(axis, int) or not 0 <= axis < len(shape):
            raise ValueError("per-channel axis must be a static nonnegative tensor dimension")
        scales = _frozen(scale, "scales")
        if (scales.dtype not in (torch.float32, torch.float64)
                or scales.ndim != 1 or scales.numel() != shape[axis]
                or not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all())):
            raise ValueError("per-channel scales must be finite positive frozen f32/f64 values matching the axis")
        if zero_point is not None:
            zeros = _frozen(zero_point, "zero points")
            if (zeros.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
                    or zeros.ndim != 1 or zeros.numel() != shape[axis] or bool((zeros != 0).any())):
                raise ValueError("per-channel weights require frozen integer zero point 0 for every channel")
            receipt["zero_points_sha256"] = _tensor_digest(zeros)
        receipt.update(axis=axis, channels=shape[axis], scale_dtype=str(scales.dtype),
                       scales_sha256=_tensor_digest(scales))
        return node.args[0], scale, axis, receipt

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
            isinstance(arg, Node) and arg.op == "call_function" and arg.target in dequants
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
            activation, activation_scale, activation_axis, activation_qparams = _quant_operand(node.args[0])
            weight, weight_scale, weight_axis, weight_qparams = _quant_operand(node.args[1])
            if activation_axis is not None:
                raise ValueError("activation per-channel scaling has no verified integer layout; requires per-tensor activation")
            activation_shape = _shape(node.args[0])
            weight_shape = _shape(node.args[1])
            result_shape = _shape(node)
            bias = (node.args[2] if len(node.args) > 2 else node.kwargs.get("bias")) if kind != "matmul" else None
            if kind == "linear":
                if (
                    len(activation_shape) < 2
                    or len(weight_shape) != 2
                    or activation_shape[-1] != weight_shape[-1]
                    or result_shape != (*activation_shape[:-1], weight_shape[0])
                ):
                    raise ValueError("linear has unsupported activation, weight or output geometry")
            elif kind == "conv2d":
                stride = _pair(node.args[3] if len(node.args) > 3 else node.kwargs.get("stride", 1), "stride")
                padding = _pair(node.args[4] if len(node.args) > 4 else node.kwargs.get("padding", 0), "padding", allow_zero=True)
                dilation = _pair(node.args[5] if len(node.args) > 5 else node.kwargs.get("dilation", 1), "dilation")
                groups = node.args[6] if len(node.args) > 6 else node.kwargs.get("groups", 1)
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
            output_axis = len(weight_shape) - 1 if kind == "matmul" else 0
            if weight_axis is not None and weight_axis != output_axis:
                raise ValueError("weight channel axis must be the output dimension, not a reduction or batch dimension")
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

        preceding = set(graph.nodes)
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
            if weight_axis is not None:
                # Keep the actual scale tensor and its dtype. PT2E's per-channel
                # dequantize explicitly returns f32 even when its scales are f64.
                value = graph.call_function(torch.ops.aten._to_copy.default, args=(value,),
                                            kwargs={"dtype": torch.float32})
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
        # Every generated operation has an exact rewritten source consumer.
        # Preserve its carried source identities through the subsequent export;
        # names remain a human-readable debug label, never identity evidence.
        custom = dict(node.meta.get("custom") or {})
        origins = set(custom.get("m2m_lineage") or ())
        for operand in node.args[:2]:
            if isinstance(operand, Node):
                origins.update((operand.meta.get("custom") or {}).get("m2m_lineage") or ())
        if origins:
            for created in graph.nodes:
                if created not in preceding:
                    metadata = dict(created.meta.get("custom") or {})
                    metadata["m2m_lineage"] = sorted(set([*metadata.get("m2m_lineage", ()), *origins]))
                    metadata["m2m_transform"] = "pt2e_integerization"
                    created.meta["custom"] = metadata
        node.replace_all_uses_with(value)
        graph.erase_node(node)
        counts[f"{kind}_integerized"] += 1
        counts["quantized_contractions_integerized"] += 1
        quantized_by_kind[kind]["integerized"] += 1
        qparam_receipts.append({"kind": kind, "activation": activation_qparams,
                               "weight": weight_qparams, "reduction_k": reduction,
                               "accumulator_dtype": "torch.int32", "output_dtype": "torch.float32"})

    if counts["integer_mm_emitted"]:
        graph.eliminate_dead_code()
        graph.lint()
        model.recompile()
    remaining_dequant = sum(
        1 for node in graph.nodes if node.op == "call_function" and node.target in dequants
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
        "qparams": qparam_receipts,
        "refusals": refusals,
    }
