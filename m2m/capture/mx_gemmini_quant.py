"""MX Gemmini operand preparation and a TorchAO module extension.

The selected RTL revision is f0167390b56fb315deea90ac1fc3983772e92d82.
This module models its BF16-to-MX *operand* conversion. It does not model the
mesh product, 16-lane reduction, packing/LUT projection, or host transfers.
Those require separate accelerator-oracle comparison before qualification.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

GROUP = 32
RTL_COMMIT = "f0167390b56fb315deea90ac1fc3983772e92d82"
RTL_CONFIG = "GemminiMxFPConfigs.standaloneMxFPConfig"
RTL_CONFIG_CLASS = "GemminiMxFPStandaloneConfig"
# (exponent bits, fraction bits, exponent bias, highest finite positive code)
_FORMATS = {
    "mxfp8": (4, 3, 7, 0x7E),  # OCP E4M3, max 448
    "mxfp6": (3, 2, 3, 0x1F),  # E3M2, max 28
    "mxfp4": (2, 1, 1, 0x07),  # E2M1, max 6
}


def _positive_grid(fmt: str, device: torch.device) -> Tensor:
    exp_bits, fraction_bits, bias, highest_code = _FORMATS[fmt]
    codes = torch.arange(highest_code + 1, device=device, dtype=torch.int32)
    exponent = codes >> fraction_bits
    fraction = codes & ((1 << fraction_bits) - 1)
    normal = torch.ldexp(1.0 + fraction.to(torch.float32) / (1 << fraction_bits), exponent - bias)
    subnormal = torch.ldexp(fraction.to(torch.float32), torch.full_like(exponent, 1 - bias - fraction_bits))
    return torch.where(exponent == 0, subnormal, normal)


def _round_grid(x: Tensor, fmt: str) -> tuple[Tensor, Tensor]:
    """Nearest representable number, with even low code on an exact tie."""
    grid = _positive_grid(fmt, x.device)
    mag = x.abs().to(torch.float32)
    upper = torch.searchsorted(grid.contiguous(), mag.contiguous()).clamp(max=grid.numel() - 1)
    lower = (upper - 1).clamp(min=0)
    lower_distance = mag - grid[lower]
    upper_distance = grid[upper] - mag
    choose_upper = (upper_distance < lower_distance) | (
        (upper_distance == lower_distance) & ((lower & 1) != 0)
    )
    code = torch.where(choose_upper, upper, lower).to(torch.uint8)
    code = code | (torch.signbit(x).to(torch.uint8) << (_FORMATS[fmt][0] + _FORMATS[fmt][1]))
    unsigned_code = code.to(torch.int32) & ((1 << (_FORMATS[fmt][0] + _FORMATS[fmt][1])) - 1)
    if fmt == "mxfp4":
        # E3M1Tofp4 emits code 0 for underflow, independent of the input sign.
        code = torch.where(unsigned_code == 0, torch.zeros_like(code), code)
    magnitude = grid[unsigned_code]
    sign = torch.signbit(x) & ((unsigned_code != 0) if fmt == "mxfp4" else True)
    return torch.where(sign, -magnitude, magnitude), code


def quantize_mx_gemmini(value: Tensor, format: str = "mxfp8", axis: int = -1) -> tuple[Tensor, Tensor, Tensor]:
    """Quantize each contraction-axis group of 32 BF16 values.

    Returns dequantized BF16 values, unsigned element codes, and E8M0 exponent
    bytes. The exponent tensor replaces the selected dimension with K/32.
    Nonfinite input is refused until its block-poisoning behavior is verified.
    """
    if format not in _FORMATS:
        raise ValueError(f"unsupported MX format: {format}")
    if (value.ndim < 1 or not -value.ndim <= axis < value.ndim
            or value.shape[axis] < GROUP or value.shape[axis] % GROUP):
        raise ValueError("MX contraction K must be a nonempty multiple of 32")
    bf16 = value.to(torch.bfloat16).movedim(axis, -1)
    if not torch.compiler.is_compiling() and not bool(torch.isfinite(bf16).all()):
        raise ValueError("nonfinite MX blocks require RTL-verified poisoning")
    grouped = bf16.float().reshape(*bf16.shape[:-1], bf16.shape[-1] // GROUP, GROUP)
    maximum = grouped.abs().amax(dim=-1)
    # MxRequantizer uses a 2^-23 floor before floor(log2(max)).
    exponent = torch.floor(torch.log2(maximum.clamp(min=2.0**-23))).to(torch.int32)
    # E8M0 code 255 is reserved; finite BF16 maxima cannot exceed code 254.
    exponent = exponent.clamp(-23, 127)
    scale = torch.exp2(exponent.to(torch.float32)).unsqueeze(-1)
    normalized = grouped / scale
    if format == "mxfp4":
        # RTL rounds via E3M1 before packing E2M1.
        normalized, _ = _round_grid(normalized, "e3m1")
    rounded, codes = _round_grid(normalized, format)
    dequant = (rounded * scale).reshape(bf16.shape).to(torch.bfloat16).movedim(-1, axis)
    codes = codes.reshape(bf16.shape).movedim(-1, axis)
    return dequant, codes, (exponent + 127).to(torch.uint8).movedim(-1, axis)


# E3M1 is the FP4 intermediate used by BF16ScaleRoundToTiny.
_FORMATS["e3m1"] = (3, 1, 3, 0x0F)
_MAX_MAGNITUDE = {"mxfp8": 448.0, "mxfp6": 28.0, "mxfp4": 6.0, "e3m1": 24.0}


class MXGemminiLinear(nn.Module):
    """Static MX weights and dynamic MX activations for a Linear module.

    Eager output uses PyTorch's BF16 matmul, so it is a fake-quantization
    diagnostic and cannot serve as an accelerator arithmetic golden.
    """

    def __init__(self, source: nn.Linear, format: str) -> None:
        super().__init__()
        self.format = format
        with torch.no_grad():
            weight, codes, scales = quantize_mx_gemmini(source.weight.detach(), format)
        self.register_buffer("weight", weight)
        self.register_buffer("weight_codes", codes)
        self.register_buffer("weight_scale_e8m0", scales)
        self.bias = source.bias

    def forward(self, x: Tensor) -> Tensor:
        tile = 16 if self.format == "mxfp8" else 32
        if x.ndim < 2 or x.shape[-2] < tile or x.shape[-2] % tile:
            raise ValueError(f"MX {self.format} Linear M must contain a full tile of {tile}")
        activation = _dequant_operand_for_graph(x, self.format, -1).to(torch.bfloat16)
        bias = self.bias.to(torch.bfloat16) if self.bias is not None else None
        return F.linear(activation, self.weight.to(torch.bfloat16), bias).to(x.dtype)


@dataclass(frozen=True)
class MXGemminiContractionOperands:
    """Logical matrices and E8M0 scales before target-specific packing.

    ``activation_codes`` is A[..., M, K], ``weight_codes`` is B[..., K, N],
    ``activation_scales`` is [..., M, K/32], and ``weight_scales`` is
    [..., N, K/32]. The leading batch axes, if any, match exactly.
    This is an operand handoff, not an executable or numerically certified
    accelerator contraction. Bias is intentionally outside the payload.
    """

    format: str
    activation_codes: Tensor
    weight_codes: Tensor
    activation_scales: Tensor
    weight_scales: Tensor


def linear_contraction_operands(module: MXGemminiLinear, activation: Tensor) -> MXGemminiContractionOperands:
    """Prepare one rank-2 Linear site for an out-of-tree MX layout compiler.

    A Linear stores weights as [N, K]. A contraction compiler consumes B[K, N],
    while E8M0 weight scales remain grouped by output channel [N, K/32].
    Dynamic activations are quantized from the caller's actual input.
    """
    if not isinstance(module, MXGemminiLinear):
        raise TypeError("expected an MXGemminiLinear module")
    tile = 16 if module.format == "mxfp8" else 32
    if activation.ndim != 2 or activation.shape[0] < tile or activation.shape[0] % tile:
        raise ValueError(f"MX {module.format} Linear activation M must contain a full tile of {tile}")
    if activation.shape[1] != module.weight_codes.shape[1]:
        raise ValueError("Linear activation K differs from the static weight K")
    _, activation_codes, activation_scales = quantize_mx_gemmini(activation, module.format)
    return MXGemminiContractionOperands(
        module.format,
        activation_codes,
        module.weight_codes.transpose(0, 1),
        activation_scales,
        module.weight_scale_e8m0,
    )


def functional_contraction_operands(
    lhs: Tensor, rhs: Tensor, format: str = "mxfp8"
) -> MXGemminiContractionOperands:
    """Prepare visible matmul operands with independent rank-2 to rank-4 batches.

    The LHS groups along its last axis and the RHS groups along its K axis.
    This exposes element codes and E8M0 bytes only; it does not identify the
    site in an exported graph or schedule the hardware contraction.
    """
    if format not in ("mxfp8", "mxfp6", "mxfp4"):
        raise ValueError(f"unsupported MX format: {format}")
    if lhs.ndim not in (2, 3, 4) or rhs.ndim != lhs.ndim or lhs.shape[:-2] != rhs.shape[:-2]:
        raise ValueError("MX functional matmul needs matching rank-2 to rank-4 batch axes")
    m, k = lhs.shape[-2:]
    rhs_k, n = rhs.shape[-2:]
    tile = 16 if format == "mxfp8" else 32
    if k != rhs_k or k < GROUP or k % GROUP or m < tile or n < tile or m % tile or n % tile:
        raise ValueError(f"MX {format} matmul needs matching K/32 and M/N tile {tile}")
    _, activation_codes, activation_scales = quantize_mx_gemmini(lhs, format, axis=-1)
    _, weight_codes, weight_scales = quantize_mx_gemmini(rhs, format, axis=-2)
    return MXGemminiContractionOperands(
        format, activation_codes, weight_codes,
        activation_scales, weight_scales.transpose(-2, -1),
    )


try:
    from torchao.core.config import AOBaseConfig
    from torchao.quantization.transform_module import register_quantize_module_handler
except ImportError:
    AOBaseConfig = object  # type: ignore[assignment,misc]
    def register_quantize_module_handler(_config):  # type: ignore[no-redef]
        return lambda function: function


@dataclass(frozen=True)
class MXGemminiFakeQuantConfig(AOBaseConfig):
    """TorchAO ``quantize_`` config for static weights and dynamic activations."""

    format: str = "mxfp8"

    def __post_init__(self) -> None:
        if self.format not in ("mxfp8", "mxfp6", "mxfp4"):
            raise ValueError(f"unsupported MX format: {self.format}")


@register_quantize_module_handler(MXGemminiFakeQuantConfig)
def _mx_gemmini_transform(module: nn.Module, config: MXGemminiFakeQuantConfig) -> nn.Module:
    if not isinstance(module, nn.Linear):
        raise TypeError("MXGemminiFakeQuantConfig applies only to nn.Linear")
    if module.in_features % GROUP:
        raise ValueError(f"{module.in_features} is not divisible by MX block size {GROUP}")
    tile = 16 if config.format == "mxfp8" else 32
    if module.out_features % tile:
        raise ValueError(f"{module.out_features} is not divisible by MX N tile {tile}")
    return MXGemminiLinear(module, config.format)


def apply_mx_gemmini_(model: nn.Module, format: str = "mxfp8") -> nn.Module:
    """Apply the registered TorchAO transform, preserving its native API."""
    try:
        from torchao.quantization import quantize_
    except ImportError as exc:
        raise RuntimeError("torchao is required for the MX module transform") from exc
    quantize_(model, MXGemminiFakeQuantConfig(format=format))
    return model


def _dequant_operand_for_graph(value: Tensor, format: str, axis: int) -> Tensor:
    """Exportable arithmetic Q/DQ, equivalent to finite BF16 grid lookup."""
    if not torch.compiler.is_compiling() and not bool(torch.isfinite(value).all()):
        raise ValueError("nonfinite MX blocks require RTL-verified poisoning")
    bf16 = value.to(torch.bfloat16).movedim(axis, -1)
    grouped = bf16.float().reshape(*bf16.shape[:-1], bf16.shape[-1] // GROUP, GROUP)
    exponent = torch.floor(torch.log2(grouped.abs().amax(dim=-1).clamp(min=2.0**-23)))
    scale = torch.exp2(exponent.clamp(-23, 127)).unsqueeze(-1)
    normalized = grouped / scale
    if format == "mxfp4":
        normalized = _round_grid_arithmetic(normalized, "e3m1")
    rounded = _round_grid_arithmetic(normalized, format)
    return (rounded * scale).reshape(bf16.shape).to(torch.bfloat16).movedim(-1, axis).to(value.dtype)


def _round_grid_arithmetic(value: Tensor, format: str) -> Tensor:
    _exp_bits, fraction_bits, bias, _highest_code = _FORMATS[format]
    emin = 1 - bias
    smallest_normal = 2.0 ** emin
    magnitude = value.abs()
    normal_exponent = torch.floor(torch.log2(magnitude.clamp(min=2.0**-126)))
    exponent = torch.where(magnitude < smallest_normal,
                           torch.full_like(normal_exponent, float(emin)), normal_exponent)
    quantum = torch.exp2(exponent - fraction_bits)
    rounded = (torch.round(magnitude / quantum) * quantum).clamp(max=_MAX_MAGNITUDE[format])
    signed = torch.where(value < 0, -rounded, rounded)
    if format == "mxfp4":
        # The RTL's E3M1-to-E2M1 conversion canonicalizes every zero to +0.
        signed = torch.where(rounded == 0, torch.zeros_like(signed), signed)
    else:
        # A comparison treats -0 as zero. Multiplication by +0 retains its sign
        # and uses ops that the generic MLIR importer can lower.
        signed = torch.where(value == 0, value * 0.0, signed)
    return signed


def quantize_functional_contractions_(graph_module: torch.fx.GraphModule, format: str) -> dict:
    """Insert MX Q/DQ on eligible functional matmul edges of an exported graph.

    TorchAO's module handler covers ``nn.Linear``. This pass covers matmul and
    functional linear sites visible after ``torch.export``. Unsupported or
    unknown shapes are reported; a fused SDPA node raises rather than silently
    claiming that attention was quantized.
    """
    if format not in ("mxfp8", "mxfp6", "mxfp4"):
        raise ValueError(f"unsupported MX format: {format}")
    tile = 16 if format == "mxfp8" else 32
    import torch.fx as fx

    targets = {
        torch.ops.aten.matmul.default: (0, 1, False),
        torch.ops.aten.mm.default: (0, 1, False),
        torch.ops.aten.bmm.default: (0, 1, False),
        torch.ops.aten.linear.default: (0, 1, True),
        torch.ops.aten.addmm.default: (1, 2, False),
    }
    rewritten: list[str] = []
    skipped: list[dict] = []
    for node in tuple(graph_module.graph.nodes):
        if "scaled_dot_product_attention" in str(node.target):
            raise ValueError("fused SDPA hides attention contractions; export eager attention for MX")
        if node.target not in targets:
            continue
        # The TorchAO handler already inserted dynamic activation quantization
        # and materialized static weight codes for this Linear.
        stack = node.meta.get("nn_module_stack") or {}
        if any("MXGemminiLinear" in str(value) for value in stack.values()):
            continue
        lhs_index, rhs_index, is_linear = targets[node.target]
        lhs, rhs = node.args[lhs_index], node.args[rhs_index]
        if not isinstance(lhs, fx.Node) or not isinstance(rhs, fx.Node):
            skipped.append({"node": node.name, "reason": "unobserved operand"})
            continue
        lhs_val, rhs_val = lhs.meta.get("val"), rhs.meta.get("val")
        if not isinstance(lhs_val, Tensor) or not isinstance(rhs_val, Tensor):
            skipped.append({"node": node.name, "reason": "unobserved shape"})
            continue
        a, b = tuple(lhs_val.shape), tuple(rhs_val.shape)
        if len(a) < 2 or len(b) != 2 and len(b) != len(a):
            skipped.append({"node": node.name, "reason": "rank or broadcasting"})
            continue
        m, k = a[-2:]
        n = b[-2] if is_linear else b[-1]
        rhs_k = b[-1] if is_linear else b[-2]
        if not all(isinstance(dim, int) for dim in (m, n, k, rhs_k)):
            skipped.append({"node": node.name, "reason": "symbolic dimensions"})
            continue
        if k != rhs_k or k % GROUP or m % tile or n % tile:
            skipped.append({"node": node.name, "reason": f"shape M={m} N={n} K={k} outside MX tile"})
            continue
        if len(a) > 2 and not is_linear and a[:-2] != b[:-2]:
            skipped.append({"node": node.name, "reason": "batch broadcasting"})
            continue
        with graph_module.graph.inserting_before(node):
            qlhs = graph_module.graph.call_function(_dequant_operand_for_graph, args=(lhs, format, -1))
            qrhs = graph_module.graph.call_function(_dequant_operand_for_graph,
                                                    args=(rhs, format, -1 if is_linear else -2))
        arguments = list(node.args)
        arguments[lhs_index] = qlhs
        arguments[rhs_index] = qrhs
        node.args = tuple(arguments)
        rewritten.append(node.name)
    graph_module.graph.lint()
    graph_module.recompile()
    return {"functional_contractions_quantized": rewritten, "functional_contractions_skipped": skipped}
