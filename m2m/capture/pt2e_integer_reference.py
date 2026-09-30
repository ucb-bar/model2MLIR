"""Independent integer-contraction reference for TorchAO PT2E Q/DQ graphs.

PT2E's portable converted form expresses quantized Conv2d/Linear as dequantize ->
floating-point contraction.  Integer targets instead accumulate the frozen qvalues in
i32 and apply the two scales after the reduction.  Those expressions are algebraically
equivalent over the reals but not bit-identical after a long floating-point network.

This executor keeps every non-contraction operation in Torch and replaces only a
Conv2d/Linear whose two operands are the corresponding PT2E dequantize operations.  It
therefore provides a framework-side integer reference without consuming compiler IR,
target code, or target output.  Unsupported layouts fail closed rather than silently
falling back while claiming integer coverage.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class IntegerReferenceResult:
    output: Any
    conv2d_count: int
    linear_count: int
    matmul_count: int = 0
    partial_sum_evidence: tuple[dict[str, Any], ...] = ()

    @property
    def contraction_count(self) -> int:
        return self.conv2d_count + self.linear_count + self.matmul_count


def _pair(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    values = tuple(int(v) for v in value)
    if len(values) != 2:
        raise ValueError(f"expected a 2-D convolution parameter, got {value!r}")
    return values


def _operand_identity(value: Any) -> dict[str, Any]:
    """Bind a proof to the exact centered, signed-i8 operand bytes and shape."""
    import torch

    if value.numel() < 1 or bool(((value < -128) | (value > 127)).any()):
        raise ValueError("partial-sum evidence requires nonempty signed-i8 operands")
    raw = value.detach().to(device="cpu", dtype=torch.int8).contiguous().numpy().tobytes()
    return {"dtype": "i8", "shape": [int(dim) for dim in value.shape],
            "sha256": hashlib.sha256(raw).hexdigest()}


def _max_partial_sum(lhs: Any, rhs: Any) -> int:
    """Exact maximum over every prefix of every output reduction, initial sum zero.

    Inputs are rank-2 [output positions, K] and [output channels, K]. Tiles
    bound memory independently of model size; int64 prevents an intermediate
    overflow while checking the i32-safe PT2E sites.
    """
    import torch

    if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape[1] != rhs.shape[1] or min(*lhs.shape, *rhs.shape) < 1:
        raise ValueError("partial-sum evidence has invalid contraction geometry")
    lhs, rhs = lhs.to(device="cpu", dtype=torch.int64), rhs.to(device="cpu", dtype=torch.int64)
    maximum = 0
    for row in range(0, lhs.shape[0], 64):
        left = lhs[row:row + 64]
        for col in range(0, rhs.shape[0], 32):
            right = rhs[col:col + 32]
            carry = torch.zeros((left.shape[0], right.shape[0]), dtype=torch.int64)
            for start in range(0, left.shape[1], 128):
                products = left[:, None, start:start + 128] * right[None, :, start:start + 128]
                prefixes = products.cumsum(dim=-1) + carry[..., None]
                maximum = max(maximum, int(prefixes.abs().amax()))
                carry = prefixes[..., -1]
    return maximum


def run_pt2e_integer_reference(
    model: Any,
    inputs: Iterable[Any],
    *,
    expected_contractions: int | None = None,
) -> IntegerReferenceResult:
    """Execute a converted TorchAO PT2E graph with explicit i32 contractions.

    The quantize/dequantize nodes themselves remain TorchAO operations, preserving its
    exact reciprocal-first rounding behavior.  Only the contraction is reassociated:
    centered integer operands -> i32 reduction -> activation scale -> weight scale ->
    f32 bias.  This is the contract implemented by ordinary int8 accelerator datapaths.
    """
    import torch
    import torch.nn.functional as F
    from torch.fx import Interpreter, Node

    conv2d = torch.ops.aten.conv2d.default
    linear = torch.ops.aten.linear.default
    matmuls = (
        torch.ops.aten.matmul.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
    )
    dequant_tensor = torch.ops.quantized_decomposed.dequantize_per_tensor.default
    dequant_channel = torch.ops.quantized_decomposed.dequantize_per_channel.default

    class _IntegerInterpreter(Interpreter):
        def __init__(self, graph_module: Any) -> None:
            # Raw qvalues/scales are read through their dequant nodes at contraction time.
            super().__init__(graph_module, garbage_collect_values=False)
            self.conv2d_count = 0
            self.linear_count = 0
            self.matmul_count = 0
            self.partial_sum_evidence: list[dict[str, Any]] = []

        def record(self, node: Any, kind: str, xq: Any, wq: Any, maximum: int,
                   *, reduction_k: int, reduction_order: str, bias: Any,
                   geometry: dict[str, Any] | None = None) -> None:
            self.partial_sum_evidence.append({
                "node": node.name, "kind": kind, "reduction_k": reduction_k,
                "initial_sum": 0, "max_abs_partial_sum": maximum,
                "reduction_order": reduction_order,
                "bias_application": "post_scale_f32" if bias is not None else "none",
                "activation": _operand_identity(xq), "weight": _operand_identity(wq),
                "geometry": geometry or {},
            })

        def value(self, item: Any) -> Any:
            return self.env[item] if isinstance(item, Node) else item

        def qparams(self, node: Any, *, weight_axis: int | None) -> tuple[Any, Any, Any, int | None]:
            if not isinstance(node, Node) or node.op != "call_function" or node.target not in (
                (dequant_tensor, dequant_channel) if weight_axis is not None else (dequant_tensor,)
            ):
                raise ValueError(
                    "integer PT2E contraction operand is not produced by a supported "
                    f"dequantize op: {node!r}"
                )
            per_channel = node.target == dequant_channel
            if len(node.args) < (7 if per_channel else 6):
                raise ValueError(f"PT2E dequantize node has incomplete qparams: {node}")
            if node.kwargs.get("out_dtype", torch.float32) not in (None, torch.float32):
                raise ValueError("integer reference requires float32 dequantize output")
            qvalue, scale, zero_point = (self.value(arg) for arg in node.args[:3])
            axis = node.args[3] if per_channel else None
            qmin, qmax, dtype = node.args[4:7] if per_channel else node.args[3:6]
            if (
                not isinstance(qvalue, torch.Tensor)
                or qvalue.dtype != torch.int8
                or not isinstance(qmin, int)
                or not isinstance(qmax, int)
                or qmin < -128
                or qmax > 127
                or qmin >= qmax
                or dtype != torch.int8
            ):
                raise ValueError("integer reference requires a valid signed-int8 dequantize operand")
            if bool(((qvalue < qmin) | (qvalue > qmax)).any()):
                raise ValueError("integer reference qvalues exceed the declared int8 range")
            if per_channel:
                if axis != weight_axis or qvalue.ndim < 1:
                    raise ValueError(f"integer reference requires weight channel axis {weight_axis}")
                if not isinstance(node.args[1], Node) or node.args[1].op != "get_attr":
                    raise ValueError("integer reference requires frozen weight channel scales")
                if (
                    not isinstance(scale, torch.Tensor)
                    or scale.dtype not in (torch.float32, torch.float64)
                    or scale.ndim != 1
                        or scale.numel() != qvalue.shape[axis]
                    or not bool(torch.isfinite(scale).all())
                    or not bool((scale > 0).all())
                ):
                    raise ValueError("integer reference requires positive weight channel scales")
                if zero_point is not None:
                    if not isinstance(node.args[2], Node) or node.args[2].op != "get_attr":
                        raise ValueError("integer reference requires frozen weight channel zero points")
                    if (
                        not isinstance(zero_point, torch.Tensor)
                        or zero_point.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
                        or zero_point.ndim != 1
                        or zero_point.numel() != qvalue.shape[axis]
                        or bool((zero_point != 0).any())
                    ):
                        raise ValueError("integer reference requires zero weight channel zero points")
                return qvalue, scale, zero_point, axis
            if (
                not isinstance(node.args[1], (int, float))
                or not math.isfinite(float(scale))
                or float(scale) <= 0
                or not isinstance(node.args[2], int)
                or zero_point != 0
            ):
                raise ValueError("integer reference requires a positive scalar scale and zero point 0")
            return qvalue, scale, zero_point, None

        def centered(self, value: Any, zero_point: Any, *, channel_axis: int | None) -> Any:
            q = value.to(torch.int32)
            zp = torch.as_tensor(zero_point, dtype=torch.int32, device=q.device)
            if channel_axis is not None:
                zp = zp.reshape((1,) * channel_axis + (-1,) + (1,) * (q.ndim - channel_axis - 1))
            return q - zp

        def run_node(self, node: Any) -> Any:
            if node.op != "call_function" or node.target not in (conv2d, linear, *matmuls):
                return super().run_node(node)
            # The integerizer only rewrites contractions whose operands are
            # direct PT2E dequantize nodes.  An attention matmul over the
            # results of integerized linears (often reshaped/transposed) is a
            # floating-point contraction, not another int8 contraction.
            if len(node.args) < 2 or not all(
                isinstance(operand, Node)
                and operand.op == "call_function"
                and operand.target in (dequant_tensor, dequant_channel)
                for operand in node.args[:2]
            ):
                return super().run_node(node)

            args, kwargs = self.fetch_args_kwargs_from_env(node)
            output_axis = args[1].ndim - 1 if node.target in matmuls else 0
            qx, sx, zx, _ = self.qparams(node.args[0], weight_axis=None)
            qw, sw, zw, weight_channel = self.qparams(node.args[1], weight_axis=output_axis)
            xq = self.centered(qx, zx, channel_axis=None)
            wq = self.centered(qw, 0 if zw is None else zw, channel_axis=weight_channel)
            bias = args[2] if len(args) > 2 else kwargs.get("bias")

            if node.target == linear:
                if xq.ndim < 2 or wq.ndim != 2 or xq.shape[-1] != wq.shape[-1]:
                    raise ValueError(f"invalid integer Linear geometry: {xq.shape}/{wq.shape}")
                if xq.shape[-1] * 128 * 128 > (1 << 31) - 1:
                    raise ValueError("integer Linear may overflow its i32 accumulator")
                acc = torch.matmul(xq, wq.transpose(-1, -2))
                out = acc.to(torch.float32) * torch.as_tensor(
                    sx, dtype=torch.float32, device=acc.device)
                out = out * torch.as_tensor(sw, device=acc.device)
                if weight_channel is not None:
                    out = out.to(torch.float32)
                if bias is not None:
                    out = out + bias
                left = xq.reshape(-1, xq.shape[-1])
                self.record(node, "linear", xq, wq, _max_partial_sum(left, wq),
                            reduction_k=int(xq.shape[-1]), reduction_order="k_ascending", bias=bias)
                self.linear_count += 1
                return out

            if node.target in matmuls:
                if (
                    xq.ndim < 2
                    or xq.ndim != wq.ndim
                    or xq.shape[:-2] != wq.shape[:-2]
                    or xq.shape[-1] != wq.shape[-2]
                ):
                    raise ValueError(f"invalid integer Matmul geometry: {xq.shape}/{wq.shape}")
                if xq.shape[-1] * 128 * 128 > (1 << 31) - 1:
                    raise ValueError("integer Matmul may overflow its i32 accumulator")
                acc = torch.matmul(xq, wq)
                out = acc.to(torch.float32) * torch.as_tensor(sx, dtype=torch.float32, device=acc.device)
                out = out * torch.as_tensor(sw, device=acc.device)
                if weight_channel is not None:
                    out = out.to(torch.float32)
                maximum = 0
                for index in range(math.prod(xq.shape[:-2])):
                    left = xq.reshape(-1, *xq.shape[-2:])[index]
                    right = wq.reshape(-1, *wq.shape[-2:])[index].transpose(0, 1)
                    maximum = max(maximum, _max_partial_sum(left, right))
                self.record(node, "matmul", xq, wq, maximum,
                            reduction_k=int(xq.shape[-1]), reduction_order="k_ascending", bias=None)
                self.matmul_count += 1
                return out

            stride = _pair(args[3] if len(args) > 3 else kwargs.get("stride", 1))
            padding = _pair(args[4] if len(args) > 4 else kwargs.get("padding", 0))
            dilation = _pair(args[5] if len(args) > 5 else kwargs.get("dilation", 1))
            groups = int(args[6] if len(args) > 6 else kwargs.get("groups", 1))
            if xq.ndim != 4 or wq.ndim != 4:
                raise ValueError(f"integer Conv2d requires rank-4 NCHW/OIHW, got {xq.shape}/{wq.shape}")
            if groups < 1 or xq.shape[1] % groups or wq.shape[0] % groups:
                raise ValueError(f"invalid grouped Conv2d geometry: {xq.shape}/{wq.shape}, groups={groups}")
            kh, kw = int(wq.shape[-2]), int(wq.shape[-1])
            if int(wq.shape[1]) * kh * kw * 128 * 128 > (1 << 31) - 1:
                raise ValueError("integer Conv2d may overflow its i32 accumulator")
            cols = F.unfold(
                xq.to(torch.float32), (kh, kw), dilation=dilation,
                padding=padding, stride=stride).to(torch.int32)
            batch, _, positions = cols.shape
            cin_group = int(xq.shape[1]) // groups
            cout_group = int(wq.shape[0]) // groups
            cols = cols.reshape(batch, groups, cin_group * kh * kw, positions)
            weights = wq.reshape(groups, cout_group, cin_group * kh * kw)
            acc = torch.matmul(weights.unsqueeze(0), cols).reshape(batch, wq.shape[0], positions)
            oh = (int(xq.shape[2]) + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
            ow = (int(xq.shape[3]) + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
            if oh * ow != positions:
                raise ValueError(f"unfold produced {positions} columns, expected {oh}x{ow}")
            out = acc.reshape(batch, wq.shape[0], oh, ow).to(torch.float32)
            out = out * torch.as_tensor(sx, dtype=torch.float32, device=acc.device)
            weight_scale = torch.as_tensor(sw, device=acc.device)
            if weight_channel is not None:
                weight_scale = weight_scale.reshape(1, -1, 1, 1)
            out = out * weight_scale
            if weight_channel is not None:
                out = out.to(torch.float32)
            if bias is not None:
                out = out + bias.reshape(1, -1, 1, 1)
            maximum = 0
            # The integerized graph's _int_mm assembles each patch with
            # (kernel_h, kernel_w, channel) as its K order. F.unfold above uses
            # (channel, kernel_h, kernel_w); final sums agree but intermediate
            # MAC prefixes need the *emitted* order.
            prefix_cols = cols.reshape(batch, groups, cin_group, kh, kw, positions)
            prefix_cols = prefix_cols.permute(0, 1, 3, 4, 2, 5).reshape(batch, groups, cin_group * kh * kw, positions)
            prefix_weights = weights.reshape(groups, cout_group, cin_group, kh, kw)
            prefix_weights = prefix_weights.permute(0, 1, 3, 4, 2).reshape(groups, cout_group, cin_group * kh * kw)
            for batch_index in range(batch):
                for group_index in range(groups):
                    left = prefix_cols[batch_index, group_index].transpose(0, 1)
                    maximum = max(maximum, _max_partial_sum(left, prefix_weights[group_index]))
            self.record(node, "conv2d", xq, wq, maximum,
                        reduction_k=cin_group * kh * kw,
                        reduction_order="kh_kw_channel", bias=bias,
                        geometry={"stride": list(stride), "padding": list(padding),
                                  "dilation": list(dilation), "groups": groups})
            self.conv2d_count += 1
            return out

    runner = _IntegerInterpreter(model)
    with torch.no_grad():
        output = runner.run(*tuple(inputs))
    result = IntegerReferenceResult(output, runner.conv2d_count, runner.linear_count,
                                    runner.matmul_count, tuple(runner.partial_sum_evidence))
    if expected_contractions is not None and result.contraction_count != expected_contractions:
        raise ValueError(
            f"integer reference executed {result.contraction_count} contractions, "
            f"expected {expected_contractions}"
        )
    return result
