"""Independent integer-contraction reference for TorchAO PT2E Q/DQ graphs.

PT2E's portable converted form expresses quantized Conv2d/Linear as dequantize ->
floating-point contraction.  Integer targets instead accumulate the frozen qvalues in
i32 and apply the two scales after the reduction.  Those expressions are algebraically
equivalent over the reals but not bit-identical after a long floating-point network.

This executor keeps every non-contraction operation in Torch and replaces only a
Conv2d/Linear/matmul whose two operands are PT2E dequantize operations.  It
therefore provides a framework-side integer reference without consuming compiler IR,
target code, or target output.  Unsupported layouts fail closed rather than silently
falling back while claiming integer coverage.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IntegerReferenceResult:
    output: Any
    conv2d_count: int
    linear_count: int
    matmul_count: int = 0

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
    matmuls = {
        torch.ops.aten.matmul.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
    }
    dequant_tensor = torch.ops.quantized_decomposed.dequantize_per_tensor.default
    dequant_channel = torch.ops.quantized_decomposed.dequantize_per_channel.default

    class _IntegerInterpreter(Interpreter):
        def __init__(self, graph_module: Any) -> None:
            # Raw qvalues/scales are read through their dequant nodes at contraction time.
            super().__init__(graph_module, garbage_collect_values=False)
            self.conv2d_count = 0
            self.linear_count = 0
            self.matmul_count = 0

        def value(self, item: Any) -> Any:
            return self.env[item] if isinstance(item, Node) else item

        def qparams(self, node: Any, *, activation: bool) -> tuple[Any, Any, Any, int | None]:
            if not isinstance(node, Node) or node.target not in (
                {dequant_tensor} if activation else {dequant_tensor, dequant_channel}
            ):
                raise ValueError(
                    f"integer PT2E contraction operand has unsupported dequantize op: {node!r}"
                )
            channel = node.target == dequant_channel
            if len(node.args) < (7 if channel else 6):
                raise ValueError(f"PT2E dequantize node has incomplete qparams: {node}")
            q, scale, zero = (self.value(arg) for arg in node.args[:3])
            axis = int(node.args[3]) if channel else None
            qmin, qmax, dtype = node.args[4:7] if channel else node.args[3:6]
            if q.dtype != torch.int8 or dtype != torch.int8 or qmin < -128 or qmax > 127:
                raise ValueError("integer reference requires signed int8 qvalues and range")
            if not bool(torch.all(torch.as_tensor(0 if zero is None else zero) == 0)):
                raise ValueError("integer reference requires symmetric zero point 0")
            scales = torch.as_tensor(scale)
            if channel and scales.dtype != torch.float32:
                raise ValueError("integer reference requires frozen f32 per-channel scales")
            if not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
                raise ValueError("integer reference requires finite positive scales")
            return q, scale, 0 if zero is None else zero, axis

        def centered(self, value: Any, zero_point: Any, *, channel_axis: bool) -> Any:
            q = value.to(torch.int32)
            zp = torch.as_tensor(zero_point, dtype=torch.int32, device=q.device)
            if channel_axis:
                zp = zp.reshape((-1,) + (1,) * (q.ndim - 1))
            return q - zp

        def run_node(self, node: Any) -> Any:
            if node.op != "call_function" or node.target not in ({conv2d, linear} | matmuls):
                return super().run_node(node)

            args, kwargs = self.fetch_args_kwargs_from_env(node)
            qx, sx, zx, activation_axis = self.qparams(node.args[0], activation=True)
            qw, sw, zw, weight_axis = self.qparams(node.args[1], activation=False)
            expected_weight_axis = qw.ndim - 1 if node.target in matmuls else 0
            if activation_axis is not None or weight_axis not in (None, expected_weight_axis):
                raise ValueError(
                    f"unsupported integer qparam axes: activation={activation_axis}, "
                    f"weight={weight_axis}, expected weight={expected_weight_axis}"
                )
            xq = self.centered(qx, zx, channel_axis=False)
            if weight_axis is None:
                wq = self.centered(qw, zw, channel_axis=False)
            else:
                wq = qw.to(torch.int32) - torch.as_tensor(
                    zw, dtype=torch.int32, device=qw.device
                ).reshape(tuple(-1 if i == weight_axis else 1 for i in range(qw.ndim)))
            reduction = xq.shape[-1] if node.target != conv2d else qw.shape[1] * qw.shape[2] * qw.shape[3]
            if reduction * 128 * 128 > (1 << 31) - 1:
                raise ValueError(f"integer reference i32 accumulator may overflow for K={reduction}")
            bias = (args[2] if len(args) > 2 else kwargs.get("bias")) if node.target not in matmuls else None

            if node.target in matmuls:
                if (xq.ndim != wq.ndim or xq.ndim < 2
                        or xq.shape[:-2] != wq.shape[:-2] or xq.shape[-1] != wq.shape[-2]):
                    raise ValueError("integer matmul requires non-broadcast matching batch dimensions")
                acc = torch.matmul(xq, wq)
                out = acc.to(torch.float32) * torch.as_tensor(sx, dtype=torch.float32, device=acc.device)
                scales = torch.as_tensor(sw, dtype=torch.float32, device=acc.device)
                out = out * scales
                self.matmul_count += 1
                return out

            if node.target == linear:
                acc = torch.matmul(xq, wq.transpose(-1, -2))
                out = acc.to(torch.float32) * torch.as_tensor(
                    sx, dtype=torch.float32, device=acc.device)
                out = out * torch.as_tensor(sw, dtype=torch.float32, device=acc.device)
                if bias is not None:
                    out = out + bias
                self.linear_count += 1
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
            scales = torch.as_tensor(sw, dtype=torch.float32, device=acc.device)
            out = out * (scales.reshape(1, -1, 1, 1) if weight_axis is not None else scales)
            if bias is not None:
                out = out + bias.reshape(1, -1, 1, 1)
            self.conv2d_count += 1
            return out

    runner = _IntegerInterpreter(model)
    with torch.no_grad():
        output = runner.run(*tuple(inputs))
    result = IntegerReferenceResult(
        output, runner.conv2d_count, runner.linear_count, runner.matmul_count
    )
    if expected_contractions is not None and result.contraction_count != expected_contractions:
        raise ValueError(
            f"integer reference executed {result.contraction_count} contractions, "
            f"expected {expected_contractions}"
        )
    return result
