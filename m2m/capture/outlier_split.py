"""Floating-point-equivalent outlier channel splitting for gated linear units (static int8 recipes).

A static per-tensor int8 activation scale is set by the tensor's largest value. In some decoders one
input channel of a few gated-MLP output projections carries values hundreds of times larger than every
other channel (the "massive activations" of the first token and of delimiters). One scale then either
clips that channel -- and the model stops behaving like itself -- or spends the int8 range on it.
Smoothing (SmoothQuant) can only move such a factor into the weight, where it becomes a weight outlier.

Splitting removes it instead. In ``out(act(gate(x)) * up(x))`` channel ``j`` of the product depends
only on row ``j`` of ``gate`` and ``up``, so repeating row ``j`` of ``gate`` ``k`` times, dividing row
``j`` of ``up`` by ``k`` and repeating it, and repeating column ``j`` of ``out`` leaves the result
unchanged (to rounding) while channel ``j``'s range drops by ``k``. The contractions stay ordinary
int8 contractions with one scale per tensor; the intermediate width grows by ``sum(k - 1)``.

Splitting gives each unit its own intermediate width, so each becomes a distinct contraction shape.
``pad_gated_widths`` (applied last, after any smoothing) rounds those widths up to a few shared
buckets with all-zero channels -- a zero row of ``gate`` and ``up`` and a zero column of ``out``
contribute exactly nothing, and zeros move no per-tensor or per-channel range -- choosing the bucket
widths that minimise the total padding (optionally aligned to a multiple the caller supplies).

RULE (statistics only, no layer or model names): over the calibration samples, ``m_j`` is the absmax
of channel ``j`` of the output projection's input; with ``t = ratio * quantile_q(m)`` (default: 32 x
the median channel), every channel with ``m_j > t`` is split into ``k_j = ceil(m_j / t)`` parts, so
no channel's range exceeds ``t`` afterwards. The gated units are found by tracing each
submodule's forward (``consumer(act(producer_a(x)) * producer_b(x))`` with ``act`` checked to act per
channel), not by attribute names.
"""
from __future__ import annotations

import math
import operator
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class GatedUnit:
    path: str          # the owning submodule
    owner: nn.Module
    gate: str          # child Linear behind the per-channel activation
    up: str            # child Linear multiplied in directly (the branch that is divided)
    down: str          # child Linear consuming the product


def _channelwise(function: Callable[[torch.Tensor], torch.Tensor]) -> bool:
    """Whether ``function`` maps each channel of the last axis independently (probed, not assumed)."""
    generator = torch.Generator().manual_seed(0)
    probe = torch.randn(3, 8, generator=generator) * 3
    permutation = torch.randperm(8, generator=generator)
    try:
        with torch.no_grad():
            out, permuted = function(probe), function(probe[:, permutation])
            widened = function(torch.cat([probe, probe[:, :2]], dim=1))
    except Exception:  # noqa: BLE001 -- anything that cannot be probed is not admitted
        return False
    return (
        isinstance(out, torch.Tensor)
        and out.shape == probe.shape
        and torch.equal(out[:, permutation], permuted)
        and torch.equal(widened[:, :8], out)
    )


def gated_units(model: nn.Module) -> list[GatedUnit]:
    """Every submodule computing ``down(act(gate(x)) * up(x))`` over three child ``nn.Linear``s."""
    found = []
    for path, owner in model.named_modules():
        linears = {name for name, child in owner.named_children() if isinstance(child, nn.Linear)}
        if len(linears) < 3:
            continue
        try:
            graph = torch.fx.symbolic_trace(owner).graph
        except Exception:  # noqa: BLE001 -- an untraceable forward is simply not a candidate
            continue

        def linear(node) -> str | None:
            ok = isinstance(node, torch.fx.Node) and node.op == "call_module" and node.target in linears
            return node.target if ok and len(node.users) == 1 else None

        def activated(node) -> str | None:
            if not isinstance(node, torch.fx.Node) or len(node.users) != 1 or len(node.args) != 1:
                return None
            if node.op == "call_module" and _channelwise(owner.get_submodule(node.target)):
                return linear(node.args[0])
            if node.op == "call_function" and _channelwise(node.target):
                return linear(node.args[0])
            return None

        for node in graph.nodes:
            if node.op != "call_module" or node.target not in linears or len(node.args) != 1:
                continue
            product = node.args[0]
            if not (
                isinstance(product, torch.fx.Node)
                and product.op == "call_function"
                and product.target in (operator.mul, torch.mul)
                and len(product.args) == 2
                and len(product.users) == 1
            ):
                continue
            left, right = product.args
            for gated, plain in ((left, right), (right, left)):
                gate, up = activated(gated), linear(plain)
                if gate and up and len({gate, up, node.target}) == 3:
                    found.append(GatedUnit(path, owner, gate, up, node.target))
                    break
    return found


def _replace(owner: nn.Module, name: str, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
    old = getattr(owner, name)
    new = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None,
                    dtype=old.weight.dtype, device=old.weight.device)
    new.weight = nn.Parameter(weight.contiguous(), requires_grad=False)
    if bias is not None:
        new.bias = nn.Parameter(bias.contiguous(), requires_grad=False)
    setattr(owner, name, new)


def select_splits(maxima: torch.Tensor, *, ratio: float, quantile: float) -> tuple[float, dict[int, int]]:
    """``(t, {channel: parts})`` for one output projection's per-channel input absmax ``maxima``."""
    threshold = float(torch.quantile(maxima.float(), quantile)) * ratio
    chosen = torch.nonzero(maxima > threshold).flatten().tolist()
    return threshold, {int(c): int(math.ceil(float(maxima[c]) / threshold)) for c in chosen}


@torch.no_grad()
def split_outlier_channels(model: nn.Module, observe: Callable[[], Any], *, ratio: float = 32.0,
                           quantile: float = 0.5) -> dict[str, Any]:
    """Split the outlier channels of every gated unit in ``model`` in place; returns the record.

    ``observe()`` runs the calibration samples through ``model`` (floating point); the per-channel
    statistics are taken from those runs only.
    """
    if not ratio > 1 or not 0 < quantile < 1:
        raise ValueError("ratio must exceed 1 and quantile lie in (0, 1)")
    units = gated_units(model)
    stats: dict[int, torch.Tensor] = {}
    hooks = []
    for index, unit in enumerate(units):
        def record(_module, args, index=index):
            value = args[0].detach().reshape(-1, args[0].shape[-1]).abs().amax(0).float()
            stats[index] = torch.maximum(stats[index], value) if index in stats else value

        hooks.append(getattr(unit.owner, unit.down).register_forward_pre_hook(record))
    try:
        observe()
    finally:
        for handle in hooks:
            handle.remove()
    rows, added_columns, added_macs_per_row = [], 0, 0
    for index, unit in enumerate(units):
        if index not in stats:
            continue
        threshold, parts = select_splits(stats[index], ratio=ratio, quantile=quantile)
        if not parts:
            continue
        gate, up, down = (getattr(unit.owner, n) for n in (unit.gate, unit.up, unit.down))
        up_weight = up.weight.clone()
        up_bias = up.bias.clone() if up.bias is not None else None
        rows_gate, rows_up, columns_down, bias_gate, bias_up = [], [], [], [], []
        for channel, k in parts.items():
            up_weight[channel] /= k
            if up_bias is not None:
                up_bias[channel] /= k
            for _ in range(k - 1):
                rows_gate.append(gate.weight[channel])
                rows_up.append(up_weight[channel])
                columns_down.append(down.weight[:, channel])
                if gate.bias is not None:
                    bias_gate.append(gate.bias[channel])
                if up_bias is not None:
                    bias_up.append(up_bias[channel])
        extra = len(rows_up)
        _replace(unit.owner, unit.gate, torch.cat([gate.weight, torch.stack(rows_gate)]),
                 torch.cat([gate.bias, torch.stack(bias_gate)]) if gate.bias is not None else None)
        _replace(unit.owner, unit.up, torch.cat([up_weight, torch.stack(rows_up)]),
                 torch.cat([up_bias, torch.stack(bias_up)]) if up_bias is not None else None)
        _replace(unit.owner, unit.down, torch.cat([down.weight, torch.stack(columns_down, dim=1)], dim=1),
                 down.bias.clone() if down.bias is not None else None)
        added_columns += extra
        # Per token: gate and up gain `extra` outputs of width in_features; down gains `extra` inputs.
        added_macs_per_row += extra * (2 * gate.weight.shape[1] + down.weight.shape[0])
        rows.append({
            "unit": unit.path, "gate": unit.gate, "up": unit.up, "down": unit.down,
            "threshold": threshold,
            "split": [{"channel": c, "parts": k, "absmax": float(stats[index][c])} for c, k in parts.items()],
            "width": int(getattr(unit.owner, unit.down).weight.shape[1]),
        })
    return {
        "method": "outlier_channel_split_fp_equivalent",
        "rule": "split channel j of a gated unit's output-projection input into ceil(m_j / t) parts "
                "when m_j > t = ratio * quantile_q(m) (m: per-channel absmax over the calibration samples)",
        "ratio": float(ratio),
        "quantile": float(quantile),
        "gated_units": len(units),
        "units_split": rows,
        "added_intermediate_columns": added_columns,
        "added_macs_per_token": added_macs_per_row,
    }


def bucket_widths(widths: list[int], buckets: int, *, align: int = 1) -> list[int]:
    """At most ``buckets`` widths, each a multiple of ``align``, minimising sum(bucket(w) - w) when every
    width ``w`` rounds up to the least bucket >= ``w`` (exact dynamic program over the sorted widths)."""
    if buckets < 1 or align < 1:
        raise ValueError("buckets and align must be positive")
    count = Counter(widths)
    values = sorted(count)
    up = [-(-w // align) * align for w in values]
    n = len(values)
    # cost[i][j]: widths values[i..j] all padded to up[j]
    def cost(i: int, j: int) -> int:
        return sum(count[values[t]] * (up[j] - values[t]) for t in range(i, j + 1))

    best = {(0, j): (cost(0, j), [j]) for j in range(n)}
    for b in range(1, min(buckets, n)):
        for j in range(n):
            options = [(best[(b - 1, i)][0] + cost(i + 1, j), best[(b - 1, i)][1] + [j])
                       for i in range(j) if (b - 1, i) in best]
            if options:
                best[(b, j)] = min(options)
    final = min((best[(b, n - 1)] for b in range(min(buckets, n)) if (b, n - 1) in best), key=lambda o: o[0])
    return sorted({up[j] for j in final[1]})


@torch.no_grad()
def pad_gated_widths(model: nn.Module, *, buckets: int, align: int = 1) -> dict[str, Any]:
    """Pad every gated unit's intermediate width up to one of ``buckets`` shared widths with zero
    channels (exact: zero gate/up rows and zero out columns add nothing); returns the record."""
    units = gated_units(model)
    widths = [int(getattr(u.owner, u.down).weight.shape[1]) for u in units]
    if not units:
        return {"method": "gated_width_padding", "gated_units": 0, "buckets": [], "added_intermediate_columns": 0}
    chosen = bucket_widths(widths, buckets, align=align)
    added_columns, added_macs = 0, 0
    rows = []
    for unit, width in zip(units, widths):
        target = min(b for b in chosen if b >= width)
        extra = target - width
        if extra:
            gate, up, down = (getattr(unit.owner, n) for n in (unit.gate, unit.up, unit.down))
            def grow(linear: nn.Linear, rows_added: int) -> tuple[torch.Tensor, torch.Tensor | None]:
                weight = torch.cat([linear.weight, linear.weight.new_zeros(rows_added, linear.weight.shape[1])])
                bias = None if linear.bias is None else torch.cat([linear.bias, linear.bias.new_zeros(rows_added)])
                return weight, bias
            _replace(unit.owner, unit.gate, *grow(gate, extra))
            _replace(unit.owner, unit.up, *grow(up, extra))
            _replace(unit.owner, unit.down,
                     torch.cat([down.weight, down.weight.new_zeros(down.weight.shape[0], extra)], dim=1),
                     down.bias.clone() if down.bias is not None else None)
            added_columns += extra
            added_macs += extra * (2 * gate.weight.shape[1] + down.weight.shape[0])
        rows.append({"unit": unit.path, "width": width, "padded_to": target})
    return {
        "method": "gated_width_padding",
        "rule": "pad each gated unit's intermediate width with zero channels up to the least of the bucket "
                "widths that minimise total padding (each a multiple of align)",
        "buckets": chosen,
        "align": int(align),
        "gated_units": len(units),
        "units": rows,
        "added_intermediate_columns": added_columns,
        "added_macs_per_token": added_macs,
    }
