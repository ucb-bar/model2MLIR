"""Torch-MLIR gap-op decompositions.

torch-mlir lowers hundreds of aten ops to linalg, but a handful have no converter
and torch-mlir marks them *illegal*, aborting the whole module. Several of these are
not even in the original graph -- torch's own decompositions introduce them (e.g.
``aten.empty_permuted``). The fix (not a monkeypatch): provide torch-level
decompositions that rewrite each gap op into ops torch-mlir *does* support, and feed
them into ``run_decompositions`` so the ExportedProgram handed to torch-mlir is free
of gap ops.

This is the growable registry: when torch-mlir reports a new "failed to legalize
operation 'torch.aten.<op>'", add a decomposition here.
"""

from __future__ import annotations

from typing import Any

import torch

_aten = torch.ops.aten


def _empty_permuted(size, physical_layout, *, dtype=None, layout=None, device=None, pin_memory=None):
    """aten.empty_permuted -> aten.empty.

    Contents are uninitialized, so the physical (permuted) layout is irrelevant to the
    computation; the logical shape ``size`` is preserved. torch-mlir has no
    empty_permuted converter but lowers empty fine.
    """
    return torch.empty(
        size,
        dtype=dtype,
        device=device,
        pin_memory=bool(pin_memory) if pin_memory is not None else False,
    )


def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
          scale=None, enable_gqa=False):
    """aten.scaled_dot_product_attention -> the math reference, so export expands it into
    matmul/softmax/(mask) that m2m lowers. Quantized graphs retain SDPA where fp32 does
    not; expanding it here covers both."""
    import math

    if dropout_p != 0.0:
        raise NotImplementedError("SDPA decomposition does not support nonzero dropout_p")
    if enable_gqa:
        raise NotImplementedError("SDPA decomposition does not support enable_gqa=True")
    if is_causal and attn_mask is not None:
        raise ValueError("SDPA cannot combine is_causal=True with an explicit attn_mask")
    output_dtype = query.dtype
    if output_dtype in (torch.float16, torch.bfloat16):
        query, key, value = query.float(), key.float(), value.float()
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.float()
    L, S = query.size(-2), key.size(-2)
    scale_factor = (1.0 / math.sqrt(query.size(-1))) if scale is None else scale
    attn = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    if is_causal:
        mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn = attn.masked_fill(~mask, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn = attn.masked_fill(~attn_mask, float("-inf"))
        else:
            attn = attn + attn_mask
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, value).to(output_dtype)


def _dropout(input, p=0.5, train=False):
    """aten.dropout in inference (train=False) is the identity."""
    return input


def _chunk(input, chunks, dim=0):
    """aten.chunk -> tensor_split (slices); export lowers the slices to ops m2m handles."""
    return torch.tensor_split(input, chunks, dim=dim)


def inline_set_grad_hops(gm) -> bool:
    """Flatten ALL ``wrap_with_*`` higher-order ops (torch.no_grad / torch.autocast regions,
    arbitrarily nested) by recursively rebuilding the FX graph.

    These HOPs wrap a sub-GraphModule -- ``wrap_with_set_grad_enabled(enabled, submod, *ops)``
    / ``wrap_with_autocast(device, dtype, enabled, cache, submod, *ops)`` -- with
    ``getitem(hop, i)`` extracting outputs. The FXImporter can't recurse into them, so their
    bodies (e.g. RoPE precompute) never lower. Rather than fragile in-place surgery (which
    can't converge on nested HOPs), we copy the whole graph into a fresh one, splicing each
    HOP's subgraph inline (recursively) with its placeholders bound to the HOP's operands.
    get_attr'd attributes (tensor constants + nested submodules) are transferred to ``gm``.
    Returns True if any HOP was flattened. One call fully flattens (recursion handles nesting).
    """
    import operator

    import torch.fx as fx

    if not any(n.op == "call_function" and "wrap_with_" in str(n.target) for n in gm.graph.nodes):
        return False

    new_g = fx.Graph()

    def _same_tensor_value(left, right) -> bool:
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return False
        return (left.dtype == right.dtype
                and tuple(map(str, left.shape)) == tuple(map(str, right.shape))
                and left.device == right.device and left.layout == right.layout
                and (left.layout != torch.strided or
                     tuple(map(str, left.stride())) == tuple(map(str, right.stride()))))

    def _carry_inlined_origin(source, result, *, value=None) -> bool:
        """Record an actual, exactly typed FX value substitution for trace mode."""
        source_value = source.meta.get("val") if value is None else value
        if not isinstance(result, fx.Node) or not _same_tensor_value(
                source_value, result.meta.get("val")):
            return False
        source_id = (source.meta.get("custom") or {}).get("m2m_node_id")
        if not source_id:
            return False
        carriers = [result]
        # A default decomposition may erase a same-dtype, no-copy `to`. Its
        # forward tensor value is the input's value, so bind the substitution
        # to that exact typed input too; do not infer this for a real cast.
        alias_proven = (result.op == "call_function" and str(result.target) == "aten.to.dtype"
                and len(result.args) == 2 and isinstance(result.args[0], fx.Node)
                and result.args[1] == getattr(result.args[0].meta.get("val"), "dtype", None)
                and set(result.kwargs) <= {"non_blocking", "copy", "memory_format"}
                and result.kwargs.get("non_blocking", False) is False
                and result.kwargs.get("copy", False) is False
                and result.kwargs.get("memory_format") is None
                and _same_tensor_value(result.meta.get("val"), result.args[0].meta.get("val")))
        if alias_proven:
            carriers.append(result.args[0])
        for carrier in carriers:
            custom = dict(carrier.meta.get("custom") or {})
            lineage = list(custom.get("m2m_lineage") or ())
            if source_id not in lineage:
                lineage.append(source_id)
            custom["m2m_lineage"] = lineage
            carrier.meta["custom"] = custom
        return True

    def _submod_idx(owner, n):
        return next((k for k, a in enumerate(n.args)
                     if isinstance(a, fx.Node) and a.op == "get_attr"
                     and hasattr(getattr(owner, a.target, None), "graph")), None)

    def emit(owner, node_list, local_env):
        """Copy a (sub)graph's nodes into new_g using local_env (sub node -> new value).
        A HOP node maps to a LIST of its output values; getitem on it indexes that list;
        a direct single-output use is unwrapped in _remap. Returns the output value list."""
        def _remap(a):
            v = local_env[a]
            return v[0] if isinstance(v, list) and len(v) == 1 else v

        outputs = None
        for sn in node_list:
            if sn.op == "placeholder":
                continue  # bound by caller in local_env
            if sn.op == "output":
                o = sn.args[0]
                outputs = [_remap(x) if isinstance(x, fx.Node) else x
                           for x in (o if isinstance(o, (tuple, list)) else [o])]
                continue
            if sn.op == "get_attr":
                if not hasattr(gm, sn.target):
                    try:
                        obj = getattr(owner, sn.target)
                        gm.add_module(sn.target, obj) if isinstance(obj, torch.nn.Module) \
                            else setattr(gm, sn.target, obj)
                    except Exception:  # noqa: BLE001
                        pass
                ga = new_g.get_attr(sn.target)
                ga.meta = dict(sn.meta)
                local_env[sn] = ga
                continue
            if sn.op == "call_function":
                # getitem on a HOP's output list
                if (sn.target is operator.getitem and isinstance(sn.args[0], fx.Node)
                        and isinstance(local_env.get(sn.args[0]), list)):
                    selected = local_env[sn.args[0]][sn.args[1]]
                    _carry_inlined_origin(sn, selected)
                    local_env[sn] = selected
                    continue
                if "wrap_with_" in str(sn.target):
                    si = _submod_idx(owner, sn)
                    if si is not None:
                        sub = getattr(owner, sn.args[si].target)
                        operands = [_remap(a) for a in sn.args[si + 1:] if isinstance(a, fx.Node)]
                        phs = [p for p in sub.graph.nodes if p.op == "placeholder"]
                        child = dict(zip(phs, operands))
                        outputs = emit(sub, list(sub.graph.nodes), child) or []
                        values = sn.meta.get("val")
                        values = list(values) if isinstance(values, (tuple, list)) else [values]
                        if len(values) == len(outputs):
                            for value, output in zip(values, outputs, strict=True):
                                if isinstance(output, fx.Node):
                                    _carry_inlined_origin(sn, output, value=value)
                        if not outputs and all(n.op in {"placeholder", "output", "get_attr"}
                                               for n in sub.graph.nodes):
                            lineage = list((sn.meta.get("custom") or {}).get("m2m_lineage") or ())
                            if lineage:
                                gm._m2m_trace_eliminations = [
                                    *getattr(gm, "_m2m_trace_eliminations", ()),
                                    {"source_ids": lineage, "kind": "eliminated",
                                     "reason": "inlined empty higher-order region with no call nodes",
                                     "proof": {"submodule": sn.args[si].target,
                                               "body_nodes": len(list(sub.graph.nodes))}},
                                ]
                        local_env[sn] = outputs
                        continue
            # ordinary node (or unhandled HOP): copy with remapped args
            new_node = new_g.node_copy(sn, _remap)
            local_env[sn] = new_node
        return outputs

    top_env = {}
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            p = new_g.placeholder(n.name)
            p.meta = dict(n.meta)
            top_env[n] = p
    outs = emit(gm, [n for n in gm.graph.nodes if n.op != "placeholder"], top_env)
    new_g.output(tuple(outs) if outs else ())
    new_g.lint()
    gm.graph = new_g
    gm.recompile()
    return True


# op-overload -> decomposition function. Extend as new torch-mlir gap ops surface.
_GAP_DECOMPS: dict[Any, Any] = {
    _aten.empty_permuted.default: _empty_permuted,
    _aten.scaled_dot_product_attention.default: _sdpa,
    _aten.dropout.default: _dropout,
    _aten.chunk.default: _chunk,
}


def torch_mlir_gap_decompositions() -> dict[Any, Any]:
    """Decompositions to merge into the export decomposition table so torch-mlir's
    pipeline does not abort on unsupported ops."""
    return dict(_GAP_DECOMPS)
