"""Turn supported PT2E Q/DQ contractions into integer contractions.

This operates on the converted PT2E ``GraphModule`` before export. The default
canonicalizes frozen scales to fp32 and evaluates integer reductions and exact
rational readouts, with only the final required rounding. The reference must
use the same canonical graph. Unsupported qparams are reported so callers can
enforce their coverage gate. Explicit legacy float readout is host-only.
"""

from __future__ import annotations

import math
import hashlib
from fractions import Fraction
from functools import partial
from itertools import product
from typing import Any


def _geometry_fake_mode(model: Any, example_inputs: tuple[Any, ...]):
    """Use the capture's symbol owner for fake-only, static-input propagation."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    from torch.utils._pytree import tree_leaves

    environments = {}
    values = [node.meta.get("val") for module in model.modules()
              if hasattr(module, "graph") for node in module.graph.nodes]
    for value in tree_leaves([values, example_inputs]):
        dimensions = [value]
        if isinstance(value, torch.Tensor):
            dimensions = [*value.shape, *value.stride(), value.storage_offset()]
        for dimension in dimensions:
            if isinstance(dimension, (torch.SymInt, torch.SymFloat, torch.SymBool)):
                environment = dimension.node.shape_env
                environments[id(environment)] = environment
    if len(environments) > 1:
        raise ValueError("captured symbolic dimensions have incompatible ShapeEnv owners")
    environment = next(iter(environments.values()), None)
    # A fresh environment can reuse u0 and collide with rebind_unbacked's
    # existing bindings. Keep the capture's owner, but use a fresh fake mode.
    return FakeTensorMode(shape_env=environment if environment is not None else ShapeEnv(),
                          static_shapes=True)


def integerize_pt2e(
    model: Any,
    example_inputs: tuple[Any, ...],
    *,
    numerics: str = "integer_readout",
    restore_result_layout: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Rewrite byte-quantized contractions (linear, conv, addmm, matmul) in place.

    Each matched pair of dequantize nodes supplies the actual frozen activation
    and weight scales and integer tensors. Dynamic dimensions, non-byte
    storage and unsupported graph shapes are left in portable
    PT2E form with an explicit refusal. Activation scales must be per tensor.
    Frozen per-output-channel weight scales and zero points are supported;
    a scale along the reduction dimension cannot be factored out.

    Covered contractions: ``aten.linear``, ``aten.conv{1,2,3}d`` and non-transposed
    ``aten.convolution`` (any groups, as one integer GEMM per group), ``aten.addmm``
    (alpha and beta folded into the readout and bias) and ``aten.mm``/``bmm``/
    ``matmul``. Census kinds stay ``linear``/``conv2d``/``matmul``/``unsupported``;
    each decision and qparam receipt also names its ``source_op``.

    ``numerics`` selects the readout:

    * ``"float_readout"`` (explicit, approximate, host-only): i32 accumulation, then the
      activation scale and the weight scale as two f32 multiplies, then the f32
      bias (scaled by ``beta`` for addmm).
    * ``"integer_readout"`` (default): exact rational evaluation of the fp32-canonical
      activation/weight scales, alpha, beta and bias, with a single final
      half-to-even rounding and clamp for a fused byte quantize, or one IEEE
      rounding for an f32 result. Bias becomes an i32 seed only when its exact
      scaled value is integral and the seed plus the entire reduction is
      statically safe. Other finite f32 bias stays in the exact readout.

    Eager float Q/DQ is informational, not the exactness oracle.
    ``float_readout`` retains its legacy floating evaluation order and makes
    no exact-arithmetic claim. Neither mode grants device-placement permission.

    ``restore_result_layout`` materializes each convolution's source result
    layout (contiguous or channels-last) after the readout, instead of exposing
    the patch-ordered NHWC-backed view. Merlin's ``integerize_static_pt2e``
    performs the same restoration for ``aten.conv2d`` sources itself.
    """
    import torch
    from torch.fx import GraphModule, Node
    from torch.fx.passes.shape_prop import ShapeProp

    if numerics not in ("float_readout", "integer_readout"):
        raise ValueError(f"unknown integerization numerics {numerics!r}")
    integer_readout = numerics == "integer_readout"
    arithmetic = "exact" if integer_readout else "eager_float_reassociated"
    if integer_readout:
        from m2m.capture import integer_readout as readout_ops
    if not isinstance(model, GraphModule):
        raise TypeError("PT2E integerization requires a converted GraphModule")
    scale_changes = []
    if integer_readout:
        from m2m.capture.quant_scales import canonicalize_pt2e_scales
        scale_changes = canonicalize_pt2e_scales(model)
    # Converted PT2E Q/DQ nodes commonly lack ``meta['val']`` even though their
    # adjacent linear node has it. Propagate input geometry and dtype once.
    # Exact readout must not execute the source's floating arithmetic here:
    # e.g. a finite binary64 alpha can overflow addmm's eager f32 scalar cast
    # while its integer/rational rewrite is entirely well-defined.
    if integer_readout:
        ShapeProp(model, fake_mode=_geometry_fake_mode(model, example_inputs)).propagate(*example_inputs)
    else:
        ShapeProp(model).propagate(*example_inputs)

    dequant = torch.ops.quantized_decomposed.dequantize_per_tensor.default
    channel_dequant = torch.ops.quantized_decomposed.dequantize_per_channel.default
    quantize = torch.ops.quantized_decomposed.quantize_per_tensor.default
    dequants = {dequant, channel_dequant}
    linear = torch.ops.aten.linear.default
    addmm = torch.ops.aten.addmm.default
    convolution = torch.ops.aten.convolution.default
    convs = {
        torch.ops.aten.conv1d.default: 1,
        torch.ops.aten.conv2d.default: 2,
        torch.ops.aten.conv3d.default: 3,
        convolution: None,
    }
    matmuls = {
        torch.ops.aten.matmul.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
    }
    contractions = {linear, addmm, *convs, *matmuls}
    refusals: list[dict[str, str]] = []
    precision_decisions: list[dict[str, str]] = []
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
        "integer_bmm_emitted": 0,
        "max_reduction_k": 0,
        "fused_requantizations": 0,
        "int32_bias_seeds": 0,
    }
    source_ops: dict[str, dict[str, int]] = {}
    class _IncompatibleDequantPrecision(ValueError):
        def __init__(self, declared: Any, observed: Any) -> None:
            self.source_dtype = str(observed if observed is not None else declared)
            super().__init__(
                f"dequantize output is declared {declared} and observed {observed}; "
                "this i8xi8->i32 rewrite assumes float32 Q/DQ operands. Non-float32 "
                "dequantization changes per-operand rounding, so it needs an explicit "
                "precision conversion and a separate numerical qualification"
            )

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
        declared_output = node.kwargs.get("out_dtype", torch.float32) or torch.float32
        observed_output = getattr(node.meta.get("tensor_meta") or node.meta.get("val"), "dtype", None)
        if observed_output is None:
            raise ValueError("dequantize output dtype is unobserved")
        if declared_output != observed_output:
            raise ValueError(f"dequantize output dtype mismatch: declared {declared_output}, observed {observed_output}")
        if observed_output != torch.float32:
            raise _IncompatibleDequantPrecision(declared_output, observed_output)
        if per_channel:
            _, scale, zero_point, axis, qmin, qmax, dtype = node.args[:7]
        else:
            _, scale, zero_point, qmin, qmax, dtype = node.args[:6]
            axis = None
        low, high = (-128, 127) if dtype == torch.int8 else (0, 255)
        if (
            not isinstance(qmin, int)
            or not isinstance(qmax, int)
            or qmin < low
            or qmax > high
            or qmin >= qmax
            or dtype not in (torch.int8, torch.uint8)
        ):
            raise ValueError("requires a valid int8/uint8 quantization range")
        storage_meta = node.args[0].meta.get("tensor_meta")
        if getattr(storage_meta, "dtype", None) != dtype:
            raise ValueError("dequantize source storage dtype disagrees with its declaration")
        receipt: dict[str, Any] = {"granularity": "channel" if per_channel else "tensor",
                                   "quant_min": qmin, "quant_max": qmax,
                                   "storage_dtype": str(dtype), "zero_point": 0}
        if not per_channel:
            if (not isinstance(scale, (int, float)) or not math.isfinite(float(scale))
                    or float(scale) <= 0 or not isinstance(zero_point, int)
                    or not low <= zero_point <= high):
                raise ValueError("requires finite positive scalar scale and storage-representable integer zero point")
            receipt["zero_point"] = zero_point
            receipt["centered_bound"] = max(abs(low - zero_point), abs(high - zero_point))
            receipt["scale"] = float(scale)
            return node.args[0], float(scale), None, receipt
        shape = _shape(node.args[0])
        if not isinstance(axis, int) or not -len(shape) <= axis < len(shape):
            raise ValueError("per-channel axis must be a static tensor dimension")
        axis %= len(shape)
        scales = _frozen(scale, "scales")
        if (scales.dtype not in (torch.float32, torch.float64)
                or scales.ndim != 1 or scales.numel() != shape[axis]
                or not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all())):
            raise ValueError("per-channel scales must be finite positive frozen f32/f64 values matching the axis")
        if zero_point is not None:
            zeros = _frozen(zero_point, "zero points")
            if (zeros.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
                    or zeros.ndim != 1 or zeros.numel() != shape[axis]
                    or bool(((zeros.to(torch.int64) < low) | (zeros.to(torch.int64) > high)).any())):
                raise ValueError("per-channel weights require frozen storage-representable integer zero points")
            receipt["zero_points_sha256"] = _tensor_digest(zeros)
            receipt["zero_point"] = zeros.to(torch.int64).tolist()
        else:
            receipt["zero_point"] = [0] * shape[axis]
        receipt["centered_bound"] = max(max(abs(low - z), abs(high - z)) for z in receipt["zero_point"])
        receipt.update(axis=axis, channels=shape[axis], scale_dtype=str(scales.dtype),
                       scales_sha256=_tensor_digest(scales))
        return node.args[0], scale, axis, receipt

    def _ints(raw: Any, rank: int, name: str, *, minimum: int) -> tuple[int, ...]:
        values = (raw,) * rank if isinstance(raw, int) else tuple(raw) if isinstance(raw, (list, tuple)) else ()
        if len(values) == 1 and rank > 1:
            values = values * rank
        if len(values) != rank or not all(isinstance(v, int) and v >= minimum for v in values):
            raise ValueError(f"unsupported {name}: {raw!r}")
        return values

    def _arg(node: Node, index: int, name: str, default: Any) -> Any:
        return node.args[index] if len(node.args) > index else node.kwargs.get(name, default)

    def _buffer(value: Any, stem: str) -> Node:
        name, suffix = stem, 0
        while hasattr(model, name):
            suffix += 1
            name = f"{stem}_{suffix}"
        model.register_buffer(name, value)
        return graph.get_attr(name)

    def _exact_seed(bias, scales, beta, bound):
        if not isinstance(bias, Node) or bias.op != "get_attr":
            return None
        value = _frozen(bias, "bias")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("exact integer readout requires finite bias")
        values = []
        for index, v in enumerate(value.reshape(-1).tolist()):
            seed = Fraction(float(v)) * Fraction(float(beta)) / scales[index % len(scales)]
            # Keep alpha's sign in the divisor: (acc + beta*bias/scale)*scale
            # must reproduce the signed affine expression without rounding.
            # Bias may be seeded ONLY when exact; no pre-rounding or saturation.
            if seed.denominator != 1 or abs(seed) + bound > (1 << 31) - 1:
                return None
            values.append(int(seed))
        return torch.tensor(values, dtype=torch.int32, device=value.device).reshape(value.shape)

    def _needs_centering(params):
        zeros = params["zero_point"]
        return params["storage_dtype"] != str(torch.int8) or (
            any(zeros) if isinstance(zeros, list) else zeros != 0)

    def _center(call, source, params, shape):
        """Extend bytes before subtracting zero points; never wrap in i8."""
        wide = call(torch.ops.aten._to_copy.default, args=(source,), kwargs={"dtype": torch.int32})
        zeros = params["zero_point"]
        if isinstance(zeros, list):
            dims = [1] * len(shape)
            dims[params["axis"]] = len(zeros)
            zero = _buffer(torch.tensor(zeros, dtype=torch.int32).reshape(dims), "_m2m_zero_points")
        else:
            zero = zeros
        return call(torch.ops.aten.sub.Tensor, args=(wide, zero))

    def _call(roles, role, target, args=(), kwargs=None):
        created = graph.call_function(target, args=args, kwargs=kwargs or {})
        roles[created] = role
        return created

    def _stamp(preceding, sources, roles, quants):
        """Give each node created after ``preceding`` the sources' lineage, its role and quant record."""
        origins = set()
        for source_node in sources:
            if isinstance(source_node, Node):
                origins.update((source_node.meta.get("custom") or {}).get("m2m_lineage") or ())
        for created in graph.nodes:
            if created in preceding:
                continue
            metadata = dict(created.meta.get("custom") or {})
            if origins:
                metadata["m2m_lineage"] = sorted(set([*metadata.get("m2m_lineage", ()), *origins]))
                metadata["m2m_transform"] = "pt2e_integerization"
            if roles.get(created):
                metadata["m2m_role"] = roles[created]
            if created in quants:
                metadata["m2m_quant"] = quants[created]
            if metadata:
                created.meta["custom"] = metadata

    graph = model.graph
    convolution_backward = torch.ops.aten.convolution_backward.default

    def _integerize_backward(node: Node) -> None:
        """2-D convolution_backward over int8 grad / input / weight -> i8 GEMMs + int32 sums.

        grad_input: per group, the backward GEMM columns W^T[Cg*K, Fg] x dY[Fg, N*O]
        (i32), accumulated tap by tap (ascending) into strided windows of an int32
        zero accumulator (exact; no structural zero products). grad_weight: per group
        dY[Fg, N*O] x im2col(X)[N*O, K*Cg] (i32, K ordered taps then channel).
        grad_bias: an int32 sum of dY over (N, H, W). Each result is read out with its
        scale product (s_g*s_w, s_g*s_x, s_g): two f32 multiplies (float_readout) or
        one exact rational readout (integer_readout).
        """
        source = str(node.target)
        operands = tuple(node.args[:3])
        quantized = any(isinstance(a, Node) and a.op == "call_function" and a.target in dequants for a in operands)
        if not quantized:
            return
        counts["quantized_contractions_seen"] += 1
        counts["conv2d_seen"] += 1
        quantized_by_kind["conv2d"]["seen"] += 1
        by_source = source_ops.setdefault(source, {"seen": 0, "integerized": 0})
        by_source["seen"] += 1
        decision = {"kind": "conv2d", "node": node.name, "source_op": source, "decision": "unresolved"}
        precision_decisions.append(decision)
        try:
            quantized_operands = [_quant_operand(a) for a in operands]
            if any(axis is not None for _, _, axis, _ in quantized_operands):
                raise ValueError("convolution_backward integerization requires per-tensor scales")
            (grad, s_g, _, grad_q), (inp, s_x, _, inp_q), (weight, s_w, _, weight_q) = quantized_operands
            grad_shape, in_shape, w_shape = (_shape(a) for a in operands)
            if not len(grad_shape) == len(in_shape) == len(w_shape) == 4:
                raise ValueError("convolution_backward integerization requires 2-D NCHW geometry")
            stride = _ints(_arg(node, 4, "stride", 1), 2, "stride", minimum=1)
            padding = _ints(_arg(node, 5, "padding", 0), 2, "padding", minimum=0)
            dilation = _ints(_arg(node, 6, "dilation", 1), 2, "dilation", minimum=1)
            if bool(_arg(node, 7, "transposed", False)) or any(_arg(node, 8, "output_padding", [0, 0])):
                raise ValueError("transposed convolution_backward has no verified integer rewrite")
            groups = _arg(node, 9, "groups", 1)
            mask = list(_arg(node, 10, "output_mask", [True, True, True]))
            if len(mask) != 3 or not all(isinstance(v, bool) for v in mask):
                raise ValueError("convolution_backward requires three boolean output_mask entries")
            n, cin, h, w = in_shape
            _, fout, oh, ow = grad_shape
            _, cg, kh, kw = w_shape
            if (not isinstance(groups, int) or groups < 1 or cin % groups or fout % groups or cg * groups != cin
                    or [(h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1,
                        (w + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1] != [oh, ow]):
                raise ValueError("convolution_backward geometry is inconsistent")
            fg = fout // groups
            positions = n * oh * ow
            bound = max(fg * kh * kw if mask[0] else 0,
                        positions if (mask[1] or mask[2]) else 0)
            centered = any(_needs_centering(q) for q in (grad_q, inp_q, weight_q))
            bg, bx, bw = (q["centered_bound"] for q in (grad_q, inp_q, weight_q))
            if max(fg * kh * kw * bg * bw if mask[0] else 0,
                   positions * bg * bx if mask[1] else 0,
                   positions * bg if mask[2] else 0) > (1 << 31) - 1:
                raise ValueError(f"i32 accumulator may overflow for reduction K={bound}")
            counts["max_reduction_k"] = max(counts["max_reduction_k"], bound)
        except ValueError as exc:
            refusals.append({"kind": "conv2d", "node": node.name, "reason": str(exc)})
            decision["reason"] = str(exc)
            return

        preceding = set(graph.nodes)
        aten = torch.ops.aten
        roles, quants = {}, {}
        call = partial(_call, roles, None)

        def contract(lhs, rhs, gradient):
            created = _call(roles, "contraction", integer_mm, (lhs, rhs))
            lhs_scale, rhs_scale = (s_w, s_g) if gradient == "input" else (s_g, s_x)
            quants[created] = {"lhs_scale": lhs_scale, "rhs_scale": rhs_scale, "accumulator": "i32",
                               "numerics": numerics, "arithmetic": arithmetic,
                               "host_only": not integer_readout, "gradient": gradient}
            if integer_readout:
                scale = Fraction(lhs_scale) * Fraction(rhs_scale)
                quants[created].update(scale_numerator=[str(scale.numerator)],
                                       scale_denominator=[str(scale.denominator)])
            counts["integer_mm_emitted"] += 1
            return created

        def readout(acc, scales):
            if integer_readout:
                return call(readout_ops.exact_readout, (acc, scales[0], [scales[1] if len(scales) > 1 else 1.0]))
            value = call(aten._to_copy.default, (acc,), {"dtype": torch.float32})
            for scale in scales:
                value = call(aten.mul.Tensor, (value, scale))
            return value

        def group_slice(value, dim, group, size):
            if groups == 1:
                return value
            return call(aten.slice.Tensor, (value, dim, group * size, (group + 1) * size, 1))

        results = [None, None, None]
        with graph.inserting_before(node):
            integer_mm = aten.mm.default if centered else aten._int_mm.default
            if centered:
                grad = _center(call, grad, grad_q, grad_shape)
                inp = _center(call, inp, inp_q, in_shape)
                weight = _center(call, weight, weight_q, w_shape)
            grad_rows = []
            for group in range(groups):
                rows = call(aten.permute.default, (group_slice(grad, 1, group, fg), [1, 0, 2, 3]))
                grad_rows.append(call(aten.reshape.default, (rows, [fg, positions])))
            if mask[0]:
                pieces = []
                ph, pw = padding
                for group in range(groups):
                    wt = call(aten.reshape.default, (group_slice(weight, 0, group, fg), [fg, cg * kh * kw]))
                    wt = call(aten.transpose.int, (wt, 0, 1))
                    columns = contract(wt, grad_rows[group], "input")
                    columns = call(aten.reshape.default, (columns, [cg, kh, kw, n, oh, ow]))
                    acc = call(aten.zeros.default, ([n, cg, h + 2 * ph, w + 2 * pw],), {"dtype": torch.int32})
                    for ky in range(kh):
                        for kx in range(kw):
                            tap = call(aten.select.int, (call(aten.select.int, (columns, 1, ky)), 1, kx))
                            tap = call(aten.permute.default, (tap, [1, 0, 2, 3]))
                            y0, x0 = ky * dilation[0], kx * dilation[1]
                            rows_view = call(aten.slice.Tensor, (acc, 2, y0, y0 + (oh - 1) * stride[0] + 1,
                                                                 stride[0]))
                            window = call(aten.slice.Tensor, (rows_view, 3, x0, x0 + (ow - 1) * stride[1] + 1,
                                                              stride[1]))
                            updated = call(aten.add.Tensor, (window, tap))
                            rows_view = call(aten.slice_scatter.default, (rows_view, updated, 3, x0,
                                                                          x0 + (ow - 1) * stride[1] + 1, stride[1]))
                            acc = call(aten.slice_scatter.default, (acc, rows_view, 2, y0,
                                                                    y0 + (oh - 1) * stride[0] + 1, stride[0]))
                    acc = call(aten.slice.Tensor, (acc, 2, ph, ph + h, 1))
                    pieces.append(call(aten.slice.Tensor, (acc, 3, pw, pw + w, 1)))
                acc = pieces[0] if groups == 1 else call(aten.cat.default, (pieces, 1))
                results[0] = readout(acc, [s_g, s_w])
            if mask[1]:
                padded = inp if not any(padding) else call(
                    aten.constant_pad_nd.default, (inp, [padding[1], padding[1], padding[0], padding[0]], 0))
                pieces = []
                for group in range(groups):
                    source_x = group_slice(padded, 1, group, cg)
                    patches = []
                    for ky in range(kh):
                        for kx in range(kw):
                            patch = call(aten.slice.Tensor, (source_x, 2, ky * dilation[0],
                                                             ky * dilation[0] + oh * stride[0], stride[0]))
                            patch = call(aten.slice.Tensor, (patch, 3, kx * dilation[1],
                                                             kx * dilation[1] + ow * stride[1], stride[1]))
                            patch = call(aten.permute.default, (patch, [0, 2, 3, 1]))
                            patches.append(call(aten.reshape.default, (patch, [positions, cg])))
                    columns = patches[0] if len(patches) == 1 else call(aten.cat.default, (patches, 1))
                    acc = contract(grad_rows[group], columns, "weight")
                    acc = call(aten.reshape.default, (acc, [fg, kh, kw, cg]))
                    pieces.append(call(aten.permute.default, (acc, [0, 3, 1, 2])))
                acc = pieces[0] if groups == 1 else call(aten.cat.default, (pieces, 0))
                results[1] = readout(acc, [s_g, s_x])
            if mask[2]:
                wide = call(aten._to_copy.default, (grad,), {"dtype": torch.int32})
                acc = call(aten.sum.dim_IntList, (wide, [0, 2, 3]), {"dtype": torch.int32})
                results[2] = readout(acc, [s_g])
        _stamp(preceding, (node, *operands), roles, quants)
        from m2m.capture.trace import is_tuple_selection

        for user in list(node.users):
            if user.op == "call_function" and is_tuple_selection(user.target):
                ordinal = user.args[1]
                result = results[ordinal]
                if result is not None:
                    result.meta.update({k: v for k, v in user.meta.items() if k in ("tensor_meta", "val")})
                    result.meta["pt2e_integerized_from"] = node.name
                    user.replace_all_uses_with(result)
                else:
                    from torch.fx.node import map_arg
                    for consumer in list(user.users):
                        consumer.args = map_arg(consumer.args, lambda arg: None if arg is user else arg)
                        consumer.kwargs = map_arg(consumer.kwargs, lambda arg: None if arg is user else arg)
                graph.erase_node(user)
        if node.users:
            raise ValueError("convolution_backward has a consumer other than its result slots: "
                             + ", ".join(f"{u.name}={u.target}{tuple(u.args[1:])}" for u in node.users)
                             + f"; computed slots {[r is not None for r in results]}")
        graph.erase_node(node)
        counts["conv2d_integerized"] += 1
        counts["quantized_contractions_integerized"] += 1
        quantized_by_kind["conv2d"]["integerized"] += 1
        by_source["integerized"] += 1
        readout_name = "single_scale_f32" if integer_readout else "two_scale_f32"
        decision.update(decision="integerized_i32", source_dtype="torch.float32",
                        required_numeric_semantics="exact-arithmetic" if integer_readout else "integer-accumulate-then-scale", readout=readout_name)
        qparam_receipts.append({"kind": "conv2d", "source_op": source, "grad_output": grad_q, "input": inp_q,
                                "weight": weight_q, "reduction_k": bound, "accumulator_dtype": "torch.int32",
                                "output_dtype": "torch.float32", "numerics": numerics, "readout": readout_name,
                                "groups": groups, "output_mask": mask})

    for node in list(graph.nodes):
        if node.op == "call_function" and node.target == convolution_backward:
            _integerize_backward(node)
            continue
        if node.op != "call_function" or node.target not in contractions:
            continue
        source = str(node.target)
        operands = tuple(node.args[1:3] if node.target == addmm else node.args[:2])
        quantized = any(
            isinstance(arg, Node) and arg.op == "call_function" and arg.target in dequants
            for arg in operands
        )
        if quantized:
            counts["quantized_contractions_seen"] += 1
        if node.target == convolution and bool(_arg(node, 6, "transposed", False)):
            if quantized:
                quantized_by_kind["unsupported"]["seen"] += 1
                precision_decisions.append({"kind": "unsupported", "node": node.name, "source_op": source,
                                            "decision": "unresolved",
                                            "reason": "transposed convolution has no verified integer rewrite"})
                refusals.append({
                    "kind": source, "node": node.name,
                    "reason": "PT2E Q/DQ transposed convolution has no verified integer rewrite",
                })
            continue
        kind = "linear" if node.target == linear else "conv2d" if node.target in convs else "matmul"
        counts[f"{kind}_seen"] += 1
        if not quantized:
            # Plain floating-point islands were not selected by PT2E and do
            # not belong to the integer coverage claim.
            continue
        quantized_by_kind[kind]["seen"] += 1
        by_source = source_ops.setdefault(source, {"seen": 0, "integerized": 0})
        by_source["seen"] += 1
        decision = {"kind": kind, "node": node.name, "source_op": source, "decision": "unresolved"}
        precision_decisions.append(decision)
        alpha: float = 1
        beta: float = 1
        bias = None
        batch_shape: tuple[int, ...] = ()
        groups = 1
        try:
            if len(operands) < 2:
                raise ValueError(f"{kind} has fewer than two operands")
            activation, activation_scale, activation_axis, activation_qparams = _quant_operand(operands[0])
            weight, weight_scale, weight_axis, weight_qparams = _quant_operand(operands[1])
            if activation_axis is not None:
                raise ValueError("activation per-channel scaling has no verified integer layout; requires per-tensor activation")
            activation_shape = _shape(operands[0])
            weight_shape = _shape(operands[1])
            result_shape = _shape(node)
            if kind in ("linear", "conv2d"):
                bias = _arg(node, 2, "bias", None)
            elif node.target == addmm:
                bias = node.args[0]
                alpha, beta = node.kwargs.get("alpha", 1), node.kwargs.get("beta", 1)
                if (not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (alpha, beta))
                        or not math.isfinite(alpha) or not math.isfinite(beta)):
                    raise ValueError("addmm alpha and beta must be finite literals")
                if integer_readout and any(isinstance(v, int) and v != int(float(v)) for v in (alpha, beta)):
                    raise ValueError("exact readout requires alpha and beta to be exactly representable as binary64 scalars")
                if alpha == 0:
                    raise ValueError("addmm with alpha 0 has no contraction to integerize")
                if beta == 0:
                    bias = None  # beta == 0 ignores self, including NaN/inf.
            if kind == "linear":
                if (
                    len(activation_shape) < 2
                    or len(weight_shape) != 2
                    or activation_shape[-1] != weight_shape[-1]
                    or result_shape != (*activation_shape[:-1], weight_shape[0])
                ):
                    raise ValueError("linear has unsupported activation, weight or output geometry")
            elif kind == "conv2d":
                rank = convs[node.target] or len(activation_shape) - 2
                if rank not in (1, 2, 3) or {len(activation_shape), len(weight_shape), len(result_shape)} != {rank + 2}:
                    raise ValueError("convolution requires static N, C, spatial geometry of rank 3, 4 or 5")
                stride = _ints(_arg(node, 3, "stride", 1), rank, "stride", minimum=1)
                padding = _ints(_arg(node, 4, "padding", 0), rank, "padding", minimum=0)
                dilation = _ints(_arg(node, 5, "dilation", 1), rank, "dilation", minimum=1)
                groups = _arg(node, 8 if node.target == convolution else 6, "groups", 1)
                n, cin, *spatial = activation_shape
                cout, wcin, *kernel = weight_shape
                on, oc, *out_spatial = result_shape
                if not isinstance(groups, int) or groups < 1 or cin % groups or cout % groups or wcin * groups != cin:
                    raise ValueError(f"convolution channels {cin}->{cout} do not divide into {groups!r} groups")
                if groups > 256:
                    raise ValueError(f"grouped convolution would unroll {groups} integer GEMMs (>256)")
                expected = [
                    (spatial[i] + 2 * padding[i] - dilation[i] * (kernel[i] - 1) - 1) // stride[i] + 1
                    for i in range(rank)
                ]
                if n != on or cout != oc or out_spatial != expected:
                    raise ValueError("convolution output disagrees with input, weight or window geometry")
            elif node.target == addmm:
                if (len(activation_shape) != 2 or len(weight_shape) != 2 or activation_shape[1] != weight_shape[0]
                        or result_shape != (activation_shape[0], weight_shape[1])):
                    raise ValueError("addmm has unsupported matrix geometry")
                if bias is not None:
                    bias_shape = _shape(bias) if isinstance(bias, Node) else ()
                    if tuple(torch.broadcast_shapes(bias_shape, result_shape)) != result_shape:
                        raise ValueError("addmm bias does not broadcast to the result")
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
            reduction = activation_shape[-1] if kind != "conv2d" else wcin * math.prod(kernel)
            # Centered byte products accumulate into i32. Use the full storage
            # domain, independently of observed samples and DQ subrange metadata.
            accumulator_bound = reduction * activation_qparams["centered_bound"] * weight_qparams["centered_bound"]
            centered = _needs_centering(activation_qparams) or _needs_centering(weight_qparams)
            if accumulator_bound > (1 << 31) - 1:
                raise ValueError(f"i32 accumulator may overflow for reduction K={reduction}")
            weight_scales = ([float(weight_scale)] if weight_axis is None
                             else [float(v) for v in _frozen(operands[1].args[1], "scales")])
            exact_scales = [Fraction(float(alpha)) * Fraction(activation_scale) * Fraction(value)
                            for value in weight_scales]
            acc_scale = ([float(alpha) * activation_scale * value for value in weight_scales]
                         if not integer_readout else None)
            requant = fusable_requantize(node) if integer_readout else None
            exact_seed = None
            if integer_readout:
                from m2m.ir.exact_quant import readout_plan
                if bias is not None:
                    exact_seed = None if centered else _exact_seed(bias, exact_scales, beta, accumulator_bound)
                from xdsl.dialects.builtin import Float32Type
                bias_type = None if bias is None or exact_seed is not None else Float32Type()
                if bias is not None and getattr((bias.meta.get("tensor_meta") or bias.meta.get("val")), "dtype", None) != torch.float32:
                    raise ValueError("exact integer readout requires float32 bias")
                readout_plan(exact_scales, beta, bias_type, requant[2] if requant is not None else None)
            counts["max_reduction_k"] = max(counts["max_reduction_k"], reduction)
        except ValueError as exc:
            refusals.append({"kind": kind, "node": node.name, "reason": str(exc)})
            if isinstance(exc, _IncompatibleDequantPrecision):
                decision.update(decision="preserve_float_qdq", source_dtype=exc.source_dtype,
                                required_numeric_semantics="dequantize-before-floating-contraction")
            else:
                decision["reason"] = str(exc)
            continue

        preceding = set(graph.nodes)
        seeded = False
        # Every created node records the part it plays (prov.role after import).
        roles: dict[Node, str] = {}
        call = partial(_call, roles)

        with graph.inserting_before(node):
            pairs = []
            if centered:
                zero_point = partial(call, "zero_point_correction")
                activation = _center(zero_point, activation, activation_qparams, activation_shape)
                weight = _center(zero_point, weight, weight_qparams, weight_shape)
            integer_mm = torch.ops.aten.mm.default if centered else torch.ops.aten._int_mm.default
            if kind == "linear":
                lhs = call("lhs_layout",
                    torch.ops.aten.reshape.default,
                    args=(activation, [-1, activation_shape[-1]]),
                )
                rhs = call("weight_layout", torch.ops.aten.transpose.int, args=(weight, 0, 1))
                pairs.append((lhs, rhs))
            elif node.target == addmm:
                pairs.append((activation, weight))
            elif kind == "conv2d":
                # No floating im2col is needed: pad, slice and concatenate the
                # integer operand (centered i32 or symmetric signed i8).
                # Each group is one GEMM with K ordered (taps..., channel).
                padded = activation
                if any(padding):
                    padded = call("im2col",
                        torch.ops.aten.constant_pad_nd.default,
                        args=(activation, [p for pad in reversed(padding) for p in (pad, pad)], 0),
                    )
                positions = n * math.prod(out_spatial)
                cin_group, cout_group = cin // groups, cout // groups
                to_channels_last = [0, *range(2, rank + 2), 1]
                for group in range(groups):
                    grouped = padded if groups == 1 else call(
                        "im2col", torch.ops.aten.slice.Tensor,
                        args=(padded, 1, group * cin_group, (group + 1) * cin_group, 1),
                    )
                    patches = []
                    for tap in product(*(range(extent) for extent in kernel)):
                        patch = grouped
                        for axis, offset in enumerate(tap):
                            start = offset * dilation[axis]
                            patch = call(
                                "im2col", torch.ops.aten.slice.Tensor,
                                args=(patch, axis + 2, start, start + out_spatial[axis] * stride[axis], stride[axis]),
                            )
                        patch = call("im2col", torch.ops.aten.permute.default, args=(patch, to_channels_last))
                        patches.append(call("im2col",
                            torch.ops.aten.reshape.default,
                            args=(patch, [positions, cin_group]),
                        ))
                    lhs = patches[0] if len(patches) == 1 else call("im2col", torch.ops.aten.cat.default, args=(patches, 1))
                    rhs = weight if groups == 1 else call(
                        "weight_layout", torch.ops.aten.slice.Tensor,
                        args=(weight, 0, group * cout_group, (group + 1) * cout_group, 1),
                    )
                    rhs = call("weight_layout", torch.ops.aten.permute.default, args=(rhs, to_channels_last))
                    rhs = call("weight_layout", torch.ops.aten.reshape.default,
                               args=(rhs, [cout_group, cin_group * math.prod(kernel)]))
                    rhs = call("weight_layout", torch.ops.aten.transpose.int, args=(rhs, 0, 1))
                    pairs.append((lhs, rhs))
            seed = None
            if integer_readout and exact_seed is not None:
                seed = _buffer(exact_seed, f"_m2m_int32_seed_{node.name}")
                if node.target == addmm:
                    seed_shape = tuple(_shape(bias))
                    if seed_shape not in ((result_shape[1],), result_shape):
                        seed = call("acc_seed", torch.ops.aten.expand.default, args=(seed, list(result_shape)))
                seeded = True
            batched_readout = kind == "matmul" and len(batch_shape) == 1
            if batched_readout:
                # CPU has no i8/i8 -> i32 bmm overload. Exact sign extensions
                # followed by an ordinary i32 bmm retain the batch axes; the
                # importer can fold these proven extensions into the MAC body.
                lhs = call("lhs_layout", torch.ops.aten._to_copy.default, args=(activation,), kwargs={"dtype": torch.int32})
                rhs = call("lhs_layout", torch.ops.aten._to_copy.default, args=(weight,), kwargs={"dtype": torch.int32})
                accumulator = call("contraction", torch.ops.aten.bmm.default, args=(lhs, rhs))
                counts["integer_bmm_emitted"] += 1
            elif kind == "matmul" and node.target != addmm:
                matrices = []
                for batch_indices in product(*(range(extent) for extent in batch_shape)):
                    lhs = activation
                    rhs = weight
                    for index in batch_indices:
                        lhs = call("lhs_layout", torch.ops.aten.select.int, args=(lhs, 0, index))
                        rhs = call("lhs_layout", torch.ops.aten.select.int, args=(rhs, 0, index))
                    matrices.append(call("contraction", integer_mm, args=(lhs, rhs)))
                accumulator = matrices[0] if not batch_shape else call("lhs_layout",
                    torch.ops.aten.reshape.default,
                    args=(call("lhs_layout", torch.ops.aten.stack.default, args=(matrices, 0)), list(result_shape)),
                )
                counts["integer_mm_emitted"] += len(matrices)
            else:
                accumulators = []
                for group, (lhs, rhs) in enumerate(pairs):
                    if seed is None:
                        accumulators.append(call("contraction", integer_mm, args=(lhs, rhs)))
                        continue
                    group_seed = seed if len(pairs) == 1 else call(
                        "lhs_layout", torch.ops.aten.slice.Tensor,
                        args=(seed, 0, group * cout_group, (group + 1) * cout_group, 1),
                    )
                    accumulators.append(call("contraction", readout_ops.int_mm_seeded, args=(lhs, rhs, group_seed)))
                    counts["int32_bias_seeds"] += 1
                accumulator = accumulators[0] if len(accumulators) == 1 else call(
                    "lhs_layout", torch.ops.aten.cat.default, args=(accumulators, 1))
                counts["integer_mm_emitted"] += len(accumulators)
            contraction_nodes = [created for created, part in roles.items() if part == "contraction"]
            if integer_readout:
                out_params = ((requant[2], requant[3], requant[4], requant[5], requant[1] is not None,
                               requant[0].args[5] == torch.uint8)
                              if requant is not None else (None, 0, -128, 127, False, False))
                value = call("readout", readout_ops.exact_readout, args=(
                    accumulator, activation_scale, weight_scales, float(alpha),
                    bias if not seeded else None, float(beta), *out_params))
                if requant is not None:
                    counts["fused_requantizations"] += 1
            else:
                cast_input = accumulator
                if batched_readout:
                    # Only host readout is flattened: i32 -> f32 and the two
                    # separately rounded scale multiplies keep row-major order.
                    cast_input = call("readout", torch.ops.aten.reshape.default,
                                      args=(accumulator, [math.prod(result_shape[:-1]), result_shape[-1]]))
                value = call("readout", torch.ops.aten._to_copy.default, args=(cast_input,),
                             kwargs={"dtype": torch.float32})
                value = call("readout", torch.ops.aten.mul.Tensor, args=(value, activation_scale))
                value = call("readout", torch.ops.aten.mul.Tensor, args=(value, weight_scale))
                if weight_axis is not None:
                    # Keep the actual scale tensor and its dtype. PT2E's per-channel
                    # dequantize explicitly returns f32 even when its scales are f64.
                    value = call("readout", torch.ops.aten._to_copy.default, args=(value,),
                                 kwargs={"dtype": torch.float32})
                if batched_readout:
                    value = call("readout", torch.ops.aten.reshape.default, args=(value, list(result_shape)))
                if alpha != 1:
                    value = call("alpha_scale", torch.ops.aten.mul.Tensor, args=(value, alpha))
            if kind == "linear":
                value = call("result_layout", torch.ops.aten.reshape.default, args=(value, list(result_shape)))
            elif kind == "conv2d":
                value = call("result_layout", torch.ops.aten.reshape.default, args=(value, [n, *out_spatial, cout]))
                value = call("result_layout", torch.ops.aten.permute.default,
                             args=(value, [0, rank + 1, *range(1, rank + 1)]))
            if bias is not None and not integer_readout:
                if kind == "conv2d":
                    bias = call("bias_add", torch.ops.aten.reshape.default, args=(bias, [1, cout, *([1] * rank)]))
                elif beta != 1:
                    bias = call("bias_add", torch.ops.aten.mul.Tensor, args=(bias, beta))
                value = call("bias_add", torch.ops.aten.add.Tensor, args=(value, bias))
            if restore_result_layout and kind == "conv2d":
                source_value = node.meta.get("val")
                formats = [torch.contiguous_format, torch.channels_last, torch.channels_last_3d]
                layout = next((fmt for fmt in formats if isinstance(source_value, torch.Tensor)
                               and (rank + 2 == 4 or fmt is not torch.channels_last)
                               and (rank + 2 == 5 or fmt is not torch.channels_last_3d)
                               and source_value.is_contiguous(memory_format=fmt)), None)
                if layout is not None:
                    value = call("result_layout", torch.ops.aten.clone.default, args=(value,),
                                 kwargs={"memory_format": layout})
            if requant is not None:
                # Replay the pure layout operations between the source and its
                # quantize on the requantized int8 result.
                for layout_node in requant[6]:
                    value = call("result_layout", layout_node.target, args=(value, *layout_node.args[1:]),
                                 kwargs=dict(layout_node.kwargs))
        value.meta["pt2e_integerized_from"] = node.name
        replaced = [node]
        # The replacement has the replaced value's observed shape and dtype; later
        # contractions read them from a fused int8 result's consumer dequantize.
        for key in ("tensor_meta", "val"):
            observed = (requant[0] if requant is not None else node).meta.get(key)
            if observed is not None:
                value.meta[key] = observed
        if requant is not None:
            replaced = [requant[0], *reversed(requant[6]), *([requant[1]] if requant[1] is not None else []), node]
        # Every generated operation has an exact rewritten source consumer.
        # Preserve its carried source identities through the subsequent export;
        # names remain a human-readable debug label, never identity evidence.
        quant = {"lhs_scale": activation_scale,
                 "rhs_scale": weight_scales[0] if len(weight_scales) == 1 else weight_scales,
                 "accumulator": "i32", "numerics": numerics, "host_only": not integer_readout,
                 "arithmetic": arithmetic}
        if integer_readout:
            quant.update(alpha=float(alpha), scale_numerator=[str(s.numerator) for s in exact_scales],
                         scale_denominator=[str(s.denominator) for s in exact_scales])
        else:
            quant["acc_scale"] = acc_scale[0] if len(acc_scale) == 1 else acc_scale
        if requant is not None:
            quant["output_scale"] = requant[2]
        contraction_quant = {created: dict(quant) for created in contraction_nodes}
        if kind == "conv2d" and groups > 1 and len(weight_scales) > 1:
            # Each grouped GEMM has only its own output channels. The readout
            # after concatenation uses all scales, but copying that full array
            # onto each GEMM misdescribes its exact accumulator coefficient.
            for group, created in enumerate(contraction_nodes):
                for key in ("rhs_scale", "scale_numerator", "scale_denominator", "acc_scale"):
                    values = quant.get(key)
                    if isinstance(values, list):
                        contraction_quant[created][key] = values[group * cout_group:(group + 1) * cout_group]
        _stamp(preceding, [*replaced, *operands], roles, contraction_quant)
        replaced[0].replace_all_uses_with(value)
        for stale in replaced:
            graph.erase_node(stale)
        counts[f"{kind}_integerized"] += 1
        counts["quantized_contractions_integerized"] += 1
        quantized_by_kind[kind]["integerized"] += 1
        by_source["integerized"] += 1
        readout = ("fused_requantize" if requant is not None else "single_scale_f32") if integer_readout \
            else "two_scale_f32"
        decision.update(decision="integerized_i32", source_dtype="torch.float32",
                        required_numeric_semantics="exact-arithmetic" if integer_readout else "integer-accumulate-then-scale", readout=readout)
        qparam_receipts.append({"kind": kind, "source_op": source, "activation": activation_qparams,
                               "weight": weight_qparams, "reduction_k": reduction,
                               "accumulator_dtype": "torch.int32",
                               "output_dtype": str(requant[0].args[5]) if requant is not None else "torch.float32",
                               "numerics": numerics, "readout": readout, "groups": groups,
                               "bias_application": ("int32_accumulator_seed" if seeded
                                                    else "exact_readout_bias" if bias is not None and integer_readout
                                                    else "post_scale_f32" if bias is not None else "none"),
                               **({"alpha": alpha, "beta": beta} if node.target == addmm else {}),
                               **({"output": {"scale": requant[2], "zero_point": requant[3],
                                              "quant_min": requant[4], "quant_max": requant[5],
                                              "relu": requant[1] is not None}} if requant is not None else {})})

    if integer_readout:
        # Byte quantizes become exact ones; the dtype argument becomes an is-unsigned flag.
        exact_quantizers = {quantize: (5, readout_ops.quantize),
                            torch.ops.quantized_decomposed.quantize_per_channel.default: (6, readout_ops.quantize_channel)}
        for qnode in graph.nodes:
            at, exact = exact_quantizers.get(qnode.target, (0, None)) if qnode.op == "call_function" else (0, None)
            if exact is not None and len(qnode.args) > at and qnode.args[at] in (torch.int8, torch.uint8):
                qnode.target = exact
                qnode.args = (*qnode.args[:at], qnode.args[at] == torch.uint8, *qnode.args[at + 1:])
    if counts["quantized_contractions_integerized"] or integer_readout:
        graph.eliminate_dead_code()
        graph.lint()
        model.recompile()
    remaining_dequant = sum(
        1 for node in graph.nodes if node.op == "call_function" and node.target in dequants
    )
    for row in quantized_by_kind.values():
        row["remaining"] = row["seen"] - row["integerized"]
    decision_counts = {label: sum(row["decision"] == label for row in precision_decisions)
                       for label in ("integerized_i32", "preserve_float_qdq", "unresolved")}
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
        # Numeric decisions over every PT2E-selected contraction. A target must
        # separately admit each lane and implement any host/device partition;
        # this frontend receipt is never a device-placement certificate.
        "precision_decisions": precision_decisions,
        "precision_decision_counts": decision_counts,
        "numerics": numerics,
        "host_only": not integer_readout,
        "scale_canonicalization": {"applied": integer_readout, "dtype": "float32" if integer_readout else None,
                                   "semantics_changed": bool(scale_changes),
                                   "changes": scale_changes},
        "arithmetic": arithmetic,
        "source_ops": source_ops,
    }


def fusable_requantize(node: Any) -> tuple[Any, ...] | None:
    """The int8 per-tensor quantize that is a contraction's only consumer.

    Returns ``(quantize, relu_or_None, scale, zero_point, quant_min, quant_max,
    layout_chain)`` when ``node`` feeds exactly one ``quantize_per_tensor`` to
    int8, directly or through one ``aten.relu`` and then any single-use pure
    layout operations (view, reshape, flatten, permute, transpose, squeeze,
    unsqueeze); otherwise None. Requantization is elementwise, so it commutes
    with those layout operations, which are replayed on the int8 result.
    Integer readout fuses the pair into one requantization, and the integer
    reference uses the same rule.
    """
    import torch

    aten = torch.ops.aten
    quantize = torch.ops.quantized_decomposed.quantize_per_tensor.default
    layout_ops = {aten.view.default, aten.reshape.default, aten._unsafe_view.default, aten.flatten.using_ints,
                  aten.permute.default, aten.transpose.int, aten.squeeze.dim, aten.squeeze.dims,
                  aten.unsqueeze.default}

    def only_user(value):
        users = list(value.users)
        if len(users) != 1 or users[0].op != "call_function" or not users[0].args or users[0].args[0] is not value:
            return None
        return users[0]

    relu = None
    chain = []
    current = only_user(node)
    if current is not None and current.target == aten.relu.default:
        relu, current = current, only_user(current)
    while current is not None and current.target in layout_ops:
        chain.append(current)
        current = only_user(current)
    if current is None or current.target != quantize or len(current.args) < 6:
        return None
    _, scale, zero_point, quant_min, quant_max, dtype = current.args[:6]
    low, high = (-128, 127) if dtype == torch.int8 else (0, 255)
    if (
        dtype not in (torch.int8, torch.uint8)
        or not isinstance(scale, (int, float)) or isinstance(scale, bool)
        or not math.isfinite(float(scale)) or float(scale) <= 0
        or not all(isinstance(v, int) and not isinstance(v, bool) for v in (zero_point, quant_min, quant_max))
        or not low <= quant_min < quant_max <= high or not low <= zero_point <= high
    ):
        return None
    return current, relu, float(scale), zero_point, quant_min, quant_max, tuple(chain)
