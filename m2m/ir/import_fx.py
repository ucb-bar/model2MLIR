"""FX graph to xDSL/MLIR conversion.

Converts PyTorch FX graphs (from torch.export) into CompGen's canonical
Payload IR using real xDSL linalg/arith/tensor ops where decompositions
exist, and opaque func.call for ops without known decompositions.

Invariants:
    - Every FX node maps to at least one xDSL op (or a diagnostic).
    - Decomposed ops get ``compgen.region_id`` attributes for Recipe IR targeting.
    - Unsupported ops fall back to ``func.call`` (flagged as opaque).
    - The output module passes the xDSL verifier.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any

import torch


# Strip Python object memory addresses from built-in callable names so
# region/dispatch ids are deterministic across reruns. dynamo records
# certain targets as ``<built-in method tanh of type object at 0x...>``;
# the address moves heap-randomly and would leak into the MLIR func.call
# callee name and every downstream identifier (region_id, candidate_id,
# region_dossier filenames). Mirrors lower.py::_canonicalize_target_string
# but applied at the IR-emission step rather than only on diagnostics.
_HEX_ADDR_RE = re.compile(r" at 0x[0-9a-fA-F]+>")


def _canonicalize_fx_target_str(s: str) -> str:
    """Return a canonical, address-free string for an FX target.

    For ``<built-in method tanh of type object at 0x70adfe129b40>``
    returns ``<built-in method tanh of type object>`` (stable across
    reruns / machines). Other forms pass through unchanged.
    """
    return _HEX_ADDR_RE.sub(">", s)
from xdsl.dialects.builtin import (
    BFloat16Type,
    FlatSymbolRefAttr,
    Float16Type,
    Float32Type,
    Float64Type,
    FunctionType,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp


def FlatSymbolRefAttr_ref(name: str) -> FlatSymbolRefAttr:
    """Helper: build a FlatSymbolRefAttr from a plain string name."""
    return FlatSymbolRefAttr(name)


from xdsl.ir import Attribute, Block, Operation, Region, SSAValue
from xdsl.printer import Printer

from m2m.ir.decompositions import (
    DECOMPOSITION_TABLE,
    DecompFn,
    reset_region_counters,
)
from m2m.ir.types import Float8E4M3FNType, Float8E5M2Type

# Tags the FX-side graph passes (in ``compgen.transforms.graph_passes``) set
# on ``node.meta``. ``FXImporter`` forwards each onto the emitted xDSL ops
# so downstream Recipe-IR passes don't have to re-detect patterns the FX
# stage already recognized.
_FX_META_FORWARD_KEYS = (
    "_compgen_pattern",
    "_compgen_transpose_absorbed",
    "_compgen_fuse_dequant",
)


# ---------------------------------------------------------------------------
# Canonical op taxonomy (the matchable vocabulary for downstream passes).
#
# Every emitted op is stamped with TWO attributes, centrally and automatically:
#   prov.op     -- the fine canonical op-kind  (e.g. "add", "matmul", "softmax")
#   prov.family -- the coarse family it belongs to (a small, fixed set below)
# so a pass matches on an attribute, never by inspecting linalg.generic bodies.
# Adding a new aten op means slotting its hint into one of these families --
# NOT inventing a new pattern. Keep this vocabulary small; only add a family
# when an op is *fundamentally* different from every existing one.
# ---------------------------------------------------------------------------
_FAMILY_OF: dict[str, str] = {}
def _reg(family: str, *hints: str) -> None:
    for h in hints:
        _FAMILY_OF[h] = family

_reg("elementwise", "add", "mul", "sub", "div", "neg", "reciprocal", "pow",
     "pow_tensor_scalar", "exp", "sqrt", "rsqrt", "tanh", "abs", "floor", "ceil",
     "log", "cos", "sin", "erf", "round", "sigmoid", "silu", "gelu", "relu", "clamp",
     "copy", "clone", "contiguous", "identity")
_reg("cast", "dtype_cast", "to_dtype")
_reg("fill", "fill", "empty", "scalar_tensor")
_reg("iota", "arange")
_reg("compare", "compare")
_reg("select", "where")
_reg("minmax", "maximum", "minimum", "minmax")
_reg("logical", "logical_not")
_reg("bitwise", "bitwise_and", "bitwise_not")
_reg("reduce", "reduce", "reduce_sum", "reduce_mean", "sum", "mean", "any", "bool_reduce")
_reg("arg_reduce", "aten_min_dim", "aten_max_dim", "aten_argmin", "aten_argmax")
_reg("contraction", "matmul", "batch_matmul", "int_matmul", "addmm", "linear",
     "conv2d", "convolution", "convolution_im2col_matmul",
     "weight_int8pack_mm", "weight_int4pack_mm", "weight_int4pack_qm")
_reg("normalization", "softmax", "layer_norm", "batch_norm")
_reg("spectral", "fft_rfft2", "fft_irfft2")
_reg("resize", "resize")
_reg("attention", "sdpa")
_reg("layout", "view", "reshape", "unsqueeze", "squeeze", "flatten", "permute",
     "transpose", "expand", "slice", "select", "slice_scatter", "split", "repeat",
     "pad")
_reg("concat", "cat", "concat")
_reg("gather_scatter", "gather", "embedding", "embedding_lookup", "index_gather",
     "mask_gather", "index_put", "mask_scatter")
_reg("scan", "cumsum")
_reg("search", "bucketize")
_reg("quantize", "quantize_per_tensor", "quantize_per_channel", "quantize_per_group",
     "dequantize_per_tensor", "dequantize_per_channel", "dequantize_per_group",
     "choose_qparams_per_tensor", "choose_qparams_per_channel", "dequantize", "quant_param")


def family_of(hint: str | None) -> str | None:
    """Coarse family for a fine op-kind hint (the matchable vocabulary). None if unknown."""
    return _FAMILY_OF.get(hint) if hint else None


def high_level_named_ops() -> set:
    """The named ``*_ext`` op types the high-level form (``emit_named_ops``) can emit.
    Every one MUST have an expander in ``m2m.transforms.expand_ext.EXPANDERS`` (enforced by
    tests/test_transforms.py). Lazily imported to avoid a dialect import at module load."""
    from m2m.ir.linalg_ext.ops import LayerNormOp, SoftmaxOp
    return {SoftmaxOp, LayerNormOp}


# Fallback when a decomposition emitted a named op but set no pattern_hint: infer the
# (op-kind, family) straight from the MLIR op name, so tagging is 100% regardless of
# whether each decomposition remembered a hint. Maps op.name -> (prov.op, prov.family).
_OP_TYPE_TAXONOMY: dict[str, tuple[str, str]] = {
    "linalg.matmul": ("matmul", "contraction"),
    "linalg.batch_matmul": ("batch_matmul", "contraction"),
    "linalg.quantized_matmul": ("matmul", "contraction"),
    "linalg.transpose": ("transpose", "layout"),
    "linalg.broadcast": ("broadcast", "layout"),
    "linalg.reduce": ("reduce", "reduce"),
    "linalg.fill": ("fill", "fill"),
    "linalg.softmax": ("softmax", "normalization"),
    "linalg.conv_2d": ("conv2d", "contraction"),
    "tensor.concat": ("concat", "concat"),
    "tensor.extract_slice": ("slice", "layout"),
    "tensor.insert_slice": ("slice_scatter", "layout"),
    "tensor.collapse_shape": ("reshape", "layout"),
    "tensor.expand_shape": ("reshape", "layout"),
}


def _forward_fx_meta(
    op: Operation,
    fx_meta: dict[str, Any],
    decomp_hint: str | None = None,
) -> None:
    """Copy FX node meta + DecompResult.pattern_hint onto ``op.attributes`` and stamp the
    canonical taxonomy (``prov.op`` fine kind + ``prov.family`` coarse family) on EVERY op.

    - ``_compgen_pattern`` (FX-level tag) -> ``prov._pattern_hint``
    - ``decomp_hint`` (decomp-side explicit tag) wins when FX didn't set one.
    - ``prov.op`` <- effective hint; ``prov.family`` <- family_of(hint), backfilled only
      when a decomposition didn't already set an (authoritative) family. This makes
      family tagging 100% consistent regardless of whether each decomposition remembered.

    Idempotent: won't overwrite an existing attribute.
    """
    fx_hint = fx_meta.get("_compgen_pattern") if isinstance(fx_meta, dict) else None
    effective_hint = fx_hint or decomp_hint
    if effective_hint and "prov._pattern_hint" not in op.attributes:
        op.attributes["prov._pattern_hint"] = StringAttr(str(effective_hint))
    # canonical two-level taxonomy (matchable by downstream passes):
    #   prov.op     = fine op-kind (the hint)            -- preserved as-is
    #   prov.family = coarse family from the fixed map   -- AUTHORITATIVE: overwrite any
    #     ad-hoc family a decomposition set, so the whole module uses ONE vocabulary.
    fam = family_of(effective_hint)
    op_kind = str(effective_hint) if effective_hint else None
    if fam is None:
        # no usable hint -> infer from the op's own type (named ops only)
        type_tax = _OP_TYPE_TAXONOMY.get(getattr(op, "name", ""))
        if type_tax is not None:
            op_kind, fam = type_tax
    if op_kind and "prov.op" not in op.attributes:
        op.attributes["prov.op"] = StringAttr(op_kind)
    if fam is not None:
        op.attributes["prov.family"] = StringAttr(fam)  # authoritative coarse family

    if isinstance(fx_meta, dict):
        if fx_meta.get("_compgen_transpose_absorbed") and "prov.transpose_absorbed" not in op.attributes:
            op.attributes["prov.transpose_absorbed"] = StringAttr("true")
        if fx_meta.get("_compgen_fuse_dequant") and "prov.fuse_dequant" not in op.attributes:
            op.attributes["prov.fuse_dequant"] = StringAttr("true")
        # provenance: trace each op back to its PyTorch origin (the source aten op) and the
        # original torch dtype before any coercion (e.g. fp8 -> f32). Lets downstream/merlin
        # recover "what this represents in PyTorch" and the true low-precision dtype.
        aten = fx_meta.get("_aten_target")
        if aten and "prov.aten" not in op.attributes:
            op.attributes["prov.aten"] = StringAttr(str(aten))
        if "prov.orig_dtype" not in op.attributes:
            val = fx_meta.get("val")
            dt = getattr(val[0] if isinstance(val, (tuple, list)) and val else val, "dtype", None)
            if dt is not None:
                op.attributes["prov.orig_dtype"] = StringAttr(str(dt).replace("torch.", ""))
        # module provenance: the top-level source nn.Module (VLM / action expert / vision /
        # ...) this op came from -- the basis for per-section partitioning and per-frequency
        # scheduling. Derived from the FX nn_module_stack (first non-empty path component).
        if "prov.module" not in op.attributes or "prov.fqn" not in op.attributes:
            stack = fx_meta.get("nn_module_stack")
            if isinstance(stack, dict) and stack:
                paths = [v[0] for v in stack.values() if isinstance(v, (tuple, list)) and v]
                nonempty = [p for p in paths if p]
                if nonempty:
                    # prov.module: first path component -- the (usually monolithic) top-level
                    # source module; the basis for the existing per-section partitioning.
                    if "prov.module" not in op.attributes:
                        op.attributes["prov.module"] = StringAttr(str(nonempty[0]).split(".")[0])
                    # prov.fqn: the DEEPEST (most specific) module path for this op, e.g.
                    # "model.vision_backbone.layers.3.attn" vs "model.action_expert.denoise.2".
                    # This is the signal that lets a downstream tool distinguish backbone from
                    # action head -- prov.module is usually just the wrapper and cannot. The
                    # full nn_module_stack is available here; we no longer discard the depth.
                    if "prov.fqn" not in op.attributes:
                        deepest = max(nonempty, key=lambda p: str(p).count("."))
                        op.attributes["prov.fqn"] = StringAttr(str(deepest))


def _torch_dtype_to_xdsl(dtype: torch.dtype) -> Attribute:
    """Convert a torch dtype to an xDSL element type."""
    mapping: dict[torch.dtype, type] = {
        torch.float32: Float32Type,
        torch.float64: Float64Type,
        torch.float16: Float16Type,
        torch.bfloat16: BFloat16Type,
    }
    # FP8: torch's float8 dtypes (e4m3fn / e5m2) have MLIR-native equivalents
    # (f8E4M3FN, f8E5M2) but xDSL 0.65 ships no builtin Float8 type and its
    # `AnyFloat` is a closed union, so linalg body-builders (transpose/generic)
    # reject our custom Float8 type with an AnyFloat assertion. Since we can't
    # patch xDSL (and fp8 here is for representation, not execution), fp8 tensors
    # flow through the compute path as f32; the exact torchAO fp8 scheme is recorded
    # as the module attribute `prov.quantization` so the quantization is still captured.
    # `Float8E4M3FNType`/`Float8E5M2Type` remain available for a future xDSL that
    # supports f8 natively (or a torch-mlir round-trip).
    _f8 = {getattr(torch, n, None) for n in ("float8_e4m3fn", "float8_e5m2",
                                             "float8_e4m3fnuz", "float8_e5m2fnuz", "float8_e8m0fnu")}
    if dtype in _f8:
        return Float32Type()
    # Bool -> i1 (so arith.select conds / comparison results verify) and integers ->
    # their bitwidth, instead of silently defaulting to f32.
    from xdsl.dialects.builtin import IntegerType

    if dtype == torch.bool:
        return IntegerType(1)
    int_bits = {torch.int8: 8, torch.uint8: 8, torch.int16: 16, torch.int32: 32, torch.int64: 64}
    if dtype in int_bits:
        return IntegerType(int_bits[dtype])
    # Complex: no complex element type exists here, so a complex tensor is carried as a REAL
    # tensor with a trailing size-2 (re, im) axis -- torch's own view_as_complex layout. This
    # returns the element type of that PAIR; the trailing axis itself is added centrally by
    # `_tensor_type_from_meta`, so shape and element type never disagree.
    if dtype == torch.complex64:
        return Float32Type()
    if dtype == torch.complex128:
        return Float64Type()
    if dtype == getattr(torch, "complex32", None):
        return Float16Type()
    factory = mapping.get(dtype, Float32Type)
    return factory()  # type: ignore[abstract]


def _coerce_static_dim(dim: Any) -> int:
    """Concrete dim → ``int(dim)``; symbolic / data-dependent dim → ``-1``.

    xDSL's ``TensorType`` rejects ``SymInt`` (``"u6 should be of base
    attribute builtin.int"``). For models with dynamic shapes (SmolVLA's
    image-tile counts, etc.) we emit ``-1`` (xDSL's dynamic-dim convention)
    so capture continues; downstream passes that need static shapes will
    short-circuit through their own dynamic-shape paths.
    """
    try:
        return int(dim)
    except Exception:
        return -1


#: The only aten targets whose decompositions are written against the trailing-(re, im) pair
#: layout used for complex tensors. Anything else that touches a complex value falls back to an
#: opaque call (see the guard in ``import_graph``) rather than indexing a logical dim that is
#: off by one in the pair layout. Extend this ONLY together with the decomposition.
_COMPLEX_AWARE_TARGETS = frozenset({
    "aten._fft_r2c.default",
    "aten._fft_c2r.default",
    "aten.view_as_complex.default",
    "aten.view_as_real.default",
    "aten.view_as_real_copy.default",
    "aten.mul.Tensor",
})


def _is_complex_meta(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, (tuple, list)):
        return any(_is_complex_meta(v) for v in val)
    dt = getattr(val, "dtype", None)
    return dt is not None and "complex" in str(dt)


def _node_touches_complex(node: Any, value_map: dict) -> bool:
    """True if the node's result or any tensor argument is complex."""
    if _is_complex_meta(node.meta.get("val")):
        return True
    for arg in node.args:
        cands = arg if isinstance(arg, (list, tuple)) else [arg]
        for c in cands:
            if hasattr(c, "meta") and _is_complex_meta(c.meta.get("val")):
                return True
    return False


def _tensor_type_from_meta(val: Any) -> TensorType | None:
    """Extract a TensorType from an FX node's meta['val'].

    Tuple/list-returning ops (``aten.native_layer_norm`` →
    ``(out, mean, rstd)``, ``aten.var_mean`` → ``(mean, var)``, …)
    surface as Python tuples in ``meta['val']``. We use the *primary*
    element (index 0) as the node's representative type; downstream
    ``operator.getitem(node, 0)`` consumers then resolve to the
    decomposition's single-tensor result rather than crashing as
    orphan opaque calls (REQ-020).
    """
    if val is None:
        return None
    if hasattr(val, "shape") and hasattr(val, "dtype"):
        elem = _torch_dtype_to_xdsl(val.dtype)
        shape = [_coerce_static_dim(d) for d in val.shape]
        # A complex tensor is represented as a real tensor with a trailing size-2 (re, im)
        # axis -- the same layout torch uses, so view_as_complex/view_as_real are identities.
        # Adding the axis HERE keeps every consumer (decompositions and the opaque fallback
        # alike) agreeing on the shape; doing it per-op is how the two drift apart.
        if "complex" in str(val.dtype):
            shape = [*shape, 2]
        return TensorType(elem, shape)
    if isinstance(val, (tuple, list)) and val:
        # Tuple-returning op — recurse on the primary element.
        return _tensor_type_from_meta(val[0])
    return None


@dataclass
class ImportDiagnostic:
    """Diagnostic from an import operation.

    Attributes:
        fx_node: Name of the FX node that produced this diagnostic.
        level: "error", "warning", or "info".
        message: Human-readable description.
    """

    fx_node: str
    level: str
    message: str


@dataclass
class FXImporter:
    """Converts a PyTorch FX graph to an xDSL module.

    Uses the decomposition table from ``decompositions.py`` to produce
    real xDSL ops (linalg.matmul, linalg.transpose, etc.) where possible.
    Falls back to opaque func.call for undecomposed ops.
    """

    diagnostics: list[ImportDiagnostic] = field(default_factory=list)
    decomposed_count: int = 0
    opaque_count: int = 0
    allow_opaque_fallback: bool = True
    explicit_blackboxes: set[str] = field(default_factory=set)
    dynamic_decompositions: dict[str, DecompFn] = field(default_factory=dict)
    emit_named_ops: bool = False  # high-level form: emit linalg_ext.* composites

    @property
    def decomposition_coverage(self) -> float:
        """Fraction of ops that were decomposed to real xDSL ops."""
        total = self.decomposed_count + self.opaque_count
        return self.decomposed_count / total if total > 0 else 1.0

    @staticmethod
    def _while_loop_bound(cond_gm):
        """Extract the constant K from a while_loop cond subgraph whose body is ``i < K``."""
        try:
            for n in cond_gm.graph.nodes:
                if n.op == "call_function" and ("lt" in str(n.target) or "ge" in str(n.target)
                                                or "<" in str(n.target)):
                    for arg in list(n.args)[1:]:
                        if isinstance(arg, (int, float)):
                            return int(arg)
                        v = getattr(arg, "meta", {}).get("val") if hasattr(arg, "meta") else None
                        try:
                            return int(v.item())
                        except Exception:  # noqa: BLE001
                            pass
        except Exception:  # noqa: BLE001
            return None
        return None

    def _lower_while_loop(self, node, value_map, multi_results, block, exported_program) -> bool:
        """Lower a ``torch.while_loop`` HOP to ``scf.for(0, K, 1)`` with carried state as iter_args (P21).

        The cond is ``i < K`` with K a captured constant, so scf.for fits. The loop-body subgraph is
        imported recursively (a fresh ``import_graph``) and its ops are transplanted into the for-region;
        closed-over additional inputs are referenced directly from the region (legal in scf -> avoids the
        torch-mlir additional_inputs defect). Fully additive: only fires on while_loop nodes (never present
        before), and any failure returns False without touching existing-capture behavior."""
        from types import SimpleNamespace

        from xdsl.dialects.arith import ConstantOp, IndexCastOp
        from xdsl.dialects.builtin import IndexType, IntegerAttr, TensorType, i64
        from xdsl.dialects.scf import ForOp
        from xdsl.dialects.scf import YieldOp as ScfYield
        from xdsl.dialects.tensor import FromElementsOp
        from xdsl.ir import Block, Region
        try:
            root = exported_program.graph_module
            a = list(node.args)

            def _gm(x):
                return getattr(root, x.target) if hasattr(x, "op") and x.op == "get_attr" else x
            cond_gm, body_gm = _gm(a[0]), _gm(a[1])
            carried = list(a[2]) if isinstance(a[2], (list, tuple)) else [a[2]]
            additional = (list(a[3]) if isinstance(a[3], (list, tuple)) else [a[3]]) if len(a) > 3 else []
            K = self._while_loop_bound(cond_gm)
            carried_ssa = [value_map[c.name] for c in carried if hasattr(c, "name") and c.name in value_map]
            add_ssa = [value_map[x.name] for x in additional if hasattr(x, "name") and x.name in value_map]
            if K is None or len(carried_ssa) != len(carried):
                return False
            carried_types = [v.type for v in carried_ssa]

            sub = FXImporter(allow_opaque_fallback=self.allow_opaque_fallback,
                             emit_named_ops=self.emit_named_ops)
            fake_ep = SimpleNamespace(graph=body_gm.graph, graph_module=body_gm,
                                      graph_signature=getattr(body_gm, "graph_signature", None))
            sub_mod = sub.import_graph(fake_ep)
            sub_func = next(o for o in sub_mod.body.block.ops
                            if isinstance(o, FuncOp) and list(o.body.blocks)
                            and list(o.body.block.ops))
            sub_block = sub_func.body.block
            if not hasattr(self, "_pending_extern_funcs"):
                self._pending_extern_funcs = []
            # torch.while_loop body_fn(*carried, *additional); carried[0] is the
            # loop counter `i` (the cond is `i < K`) -> it is a normal iter_arg,
            # NOT the scf.for induction variable. The induction var (block arg 0)
            # is left unused; the counter is threaded through iter_args so the
            # body's own `i + 1` is what gets yielded.
            sub_args = list(sub_block.args)              # [*carried, *additional]
            nc = len(carried_ssa)

            c0 = ConstantOp(IntegerAttr(0, IndexType()), IndexType())
            cK = ConstantOp(IntegerAttr(int(K), IndexType()), IndexType())
            c1 = ConstantOp(IntegerAttr(1, IndexType()), IndexType())
            for_blk = Block(arg_types=[IndexType()] + carried_types)
            iter_args = list(for_blk.args[1:])           # nc carried iter_args

            mapping = iter_args + add_ssa                # carried -> iter_args, additional -> closed-over SSA
            for arg, repl in zip(sub_args, mapping):
                arg.replace_by(repl)
            ret_op = None
            for op in list(sub_block.ops):
                if isinstance(op, ReturnOp):
                    ret_op = op
                    continue
                op.detach()
                for_blk.add_op(op)
            ret_vals = list(ret_op.operands) if ret_op is not None else []
            # the body returns the new carry tuple (nc values); yield them all
            new_carried = ret_vals[:nc] if len(ret_vals) >= nc else ret_vals
            for_blk.add_op(ScfYield(*new_carried))

            # The body sub-import emitted opaque extern FuncOps into its own
            # (discarded) module. After remapping placeholders to iter_args /
            # closed-over SSA, some call-operand types change (e.g. a carried
            # token inferred as 1xi64 outside the body but 1x1xi64 inside), so the
            # original decls no longer match. Rebuild every opaque extern fresh
            # from its LIVE call-site types (deduped, globally-unique names) and
            # repoint the call; merge the rebuilt externs into the main module.
            from xdsl.dialects.builtin import SymbolRefAttr
            local_sigs: dict = {}
            for op in for_blk.walk():
                callee = op.properties.get("callee") if hasattr(op, "properties") else None
                if callee is None or not hasattr(callee, "root_reference"):
                    continue
                base_name = callee.root_reference.data
                opnd_types = [o.type for o in op.operands]
                res_types = [r.type for r in op.results]
                sig_key = (base_name,
                           tuple(str(t) for t in opnd_types),
                           tuple(str(t) for t in res_types))
                if sig_key not in local_sigs:
                    uniq = f"{base_name}_wl{len(self._pending_extern_funcs)}"
                    self._pending_extern_funcs.append(
                        FuncOp.external(uniq, opnd_types, res_types))
                    local_sigs[sig_key] = uniq
                op.properties["callee"] = SymbolRefAttr(local_sigs[sig_key])

            forop = ForOp(c0.results[0], cK.results[0], c1.results[0], carried_ssa, Region(for_blk))
            for op in (c0, cK, c1, forop):
                op.attributes["prov.op"] = StringAttr("while_loop")
                op.attributes["prov.family"] = StringAttr("loop")
            block.add_ops([c0, cK, c1, forop])
            # while_loop returns the final carry tuple; getitem(node, k) -> results[k]
            multi_results[node.name] = list(forop.results)
            if forop.results:
                value_map[node.name] = forop.results[0]
            self.decomposed_count += 1
            self.diagnostics.append(ImportDiagnostic(
                fx_node=node.name, level="info",
                message=f"Lowered while_loop -> scf.for(0,{int(K)},1); {nc} iter_args, {len(add_ssa)} additional"))
            return True
        except Exception as e:  # noqa: BLE001
            self.diagnostics.append(ImportDiagnostic(
                fx_node=node.name, level="warning", message=f"while_loop->scf.for failed: {e}"))
            return False

    def import_graph(self, exported_program: Any) -> ModuleOp:
        """Convert an ExportedProgram's FX graph to an xDSL module."""
        reset_region_counters()
        self._pending_extern_funcs = []
        graph = exported_program.graph
        nodes = list(graph.nodes)

        placeholders = [n for n in nodes if n.op == "placeholder"]
        call_nodes = [n for n in nodes if n.op == "call_function"]
        output_nodes = [n for n in nodes if n.op == "output"]

        # Resolve torchao subclass-inner-tensor access chains to a full attribute path.
        # torch>=2.8 + torchao leave weight-only quant unfolded as
        #   w_placeholder -> access("tensor_impl") -> access("int_data"/"scale")
        # whose inner tensors are never externalized; decompose_access_subclass_inner_tensor
        # would emit an uninitialized tensor.empty for them (silent garbage). Tag each access
        # node with the model attribute path of the value it reads (e.g.
        # "model...qkv_proj.weight.tensor_impl.int_data") so a consumer can bind the real data.
        sig = getattr(exported_program, "graph_signature", None)
        _in2param = dict(getattr(sig, "inputs_to_parameters", {})) if sig else {}
        _in2buf = dict(getattr(sig, "inputs_to_buffers", {})) if sig else {}
        _ACCESS = "access_subclass_inner_tensor"

        def _resolve_subclass_path(node):
            args = getattr(node, "args", ())
            if len(args) < 2 or not isinstance(args[1], str):
                return None
            base, inner = args[0], args[1]
            bop = getattr(base, "op", None)
            if bop == "placeholder":
                root = _in2param.get(base.name) or _in2buf.get(base.name) or base.name
                return f"{root}.{inner}"
            if bop == "get_attr":
                return f"{base.target}.{inner}"           # constant/lifted weight FQN
            if bop == "call_function" and _ACCESS in str(base.target):
                parent = _resolve_subclass_path(base)
                return f"{parent}.{inner}" if parent else None
            return None

        _quant_inner_key: dict[str, str] = {}
        for n in nodes:
            if n.op == "call_function" and _ACCESS in str(n.target):
                k = _resolve_subclass_path(n)
                if k:
                    _quant_inner_key[n.name] = k
        import os as _os
        if _os.environ.get("M2M_DEBUG_QINNER"):
            _acc = [n for n in nodes if n.op == "call_function" and _ACCESS in str(n.target)]
            print(f"[qinner] access nodes={len(_acc)} resolved={len(_quant_inner_key)} "
                  f"in2param={len(_in2param)}", flush=True)
            for n in _acc[:4]:
                b = n.args[0] if n.args else None
                print(f"[qinner]   {n.name} base.op={getattr(b,'op',None)} "
                      f"base.name={getattr(b,'name',None)} inner={n.args[1] if len(n.args)>1 else None} "
                      f"-> {_quant_inner_key.get(n.name)}", flush=True)

        # Build xDSL types for each node from meta
        node_types: dict[str, TensorType] = {}
        for node in nodes:
            val = node.meta.get("val")
            tt = _tensor_type_from_meta(val)
            if tt is not None:
                node_types[node.name] = tt

        # All placeholders become func args
        arg_types: list[Attribute] = []
        for p in placeholders:
            tt = node_types.get(p.name)
            if tt is None:
                self.diagnostics.append(
                    ImportDiagnostic(
                        fx_node=p.name,
                        level="warning",
                        message=f"No type info for placeholder {p.name}, using f32[1]",
                    )
                )
                tt = TensorType(Float32Type(), [1])
            arg_types.append(tt)

        # Determine return types
        ret_types: list[Attribute] = []
        if output_nodes:
            out_args = output_nodes[0].args[0] if output_nodes[0].args else ()
            if not isinstance(out_args, (tuple, list)):
                out_args = (out_args,)
            for a in out_args:
                if hasattr(a, "name") and a.name in node_types:
                    ret_types.append(node_types[a.name])
        if not ret_types:
            ret_types = [TensorType(Float32Type(), [1])]

        func_type = FunctionType.from_lists(arg_types, ret_types)

        # Build the function body
        block = Block(arg_types=arg_types)
        value_map: dict[str, SSAValue] = {}
        multi_results: dict[str, list[SSAValue]] = {}  # split/unbind: node -> per-output SSA
        for i, p in enumerate(placeholders):
            value_map[p.name] = block.args[i]

        # Materialize get_attr lifted tensor constants as typed values so ops that consume
        # them resolve. We materialize by TYPE from meta (data elided, like weights) -- this
        # also handles get_attr nodes lifted in from inlined no_grad/autocast subgraphs, whose
        # backing attribute lives on a child module (we never read it, only its meta type).
        from xdsl.dialects.tensor import EmptyOp as _EmptyOp

        for node in nodes:
            if node.op != "get_attr":
                continue
            tt = node_types.get(node.name)
            if tt is None:
                continue
            op = _EmptyOp([], tt)
            op.attributes["prov.op"] = StringAttr("const")
            op.attributes["prov.family"] = StringAttr("fill")
            op.attributes["prov.get_attr"] = StringAttr(str(node.target))
            block.add_op(op)
            value_map[node.name] = op.results[0]

        # Track external function declarations for opaque fallback
        extern_funcs: list[FuncOp] = []
        declared_sigs: dict[str, str] = {}
        name_counters: dict[str, int] = {}

        declared_callee_sig: dict[str, str] = {}

        def ensure_external_decl(call: CallOp) -> None:
            callee = call.callee.string_value()
            operand_types = tuple(str(value.type) for value in call.operands)
            result_types = tuple(str(value.type) for value in call.results)
            sig_key = f"{callee}:{operand_types}:{result_types}"
            if sig_key in declared_sigs:
                # Rewrite the call to use the canonical (possibly
                # disambiguated) name for this signature.
                call.properties["callee"] = FlatSymbolRefAttr_ref(declared_sigs[sig_key])
                return
            # If the callee name already exists with a different
            # signature, generate a unique suffixed name.
            chosen_name = callee
            if callee in declared_callee_sig:
                count = name_counters.get(callee, 1)
                while f"{callee}_{count}" in declared_callee_sig:
                    count += 1
                chosen_name = f"{callee}_{count}"
                name_counters[callee] = count + 1
            declared_callee_sig[chosen_name] = sig_key
            declared_sigs[sig_key] = chosen_name
            if chosen_name != callee:
                # Rewrite the existing CallOp to point at the new name.
                call.properties["callee"] = FlatSymbolRefAttr_ref(chosen_name)
            extern_funcs.append(
                FuncOp.external(
                    chosen_name,
                    [value.type for value in call.operands],
                    [value.type for value in call.results],
                )
            )

        # Process call_function nodes
        for node in call_nodes:
            target_str = _canonicalize_fx_target_str(str(node.target))
            # P21: lower a torch.while_loop HOP to scf.for (loop-preserving capture). Additive — only
            # fires for while_loop nodes; on any failure falls through (no loop emitted) without affecting
            # the existing per-node path below.
            if "while_loop" in target_str:
                if self._lower_while_loop(node, value_map, multi_results, block, exported_program):
                    continue
            result_type = node_types.get(node.name)
            if result_type is None:
                self.diagnostics.append(
                    ImportDiagnostic(
                        fx_node=node.name,
                        level="warning",
                        message=f"No type info for {node.name}, skipping",
                    )
                )
                continue

            # REQ-020: ``operator.getitem`` on a tuple-producing aten op
            # (``native_layer_norm`` returns ``(out, mean, rstd)``,
            # ``var_mean`` returns ``(mean, var)``, etc.) shows up as
            # ``getitem(producer_node, idx)`` after FX export. The
            # producer was decomposed to a single-tensor representative
            # (the primary output, index 0), so:
            # - ``getitem(_, 0)`` → resolve to the producer's tensor.
            # - ``getitem(_, k)`` for ``k > 0`` → drop the node entirely
            #   (auxiliary outputs that user code typically doesn't
            #   reference; if anything DOES reference them, it'll fail
            #   loudly downstream — preferable to emitting an opaque
            #   ``<built-in function getitem>`` op no provider can match).
            # Acceptance per REQ-020: ``<built-in function getitem>``
            # never appears in payload.mlir.
            # Multi-output ops (split/unbind): getitem(node, i) -> the i-th real output.
            if (
                target_str == "<built-in function getitem>"
                and len(node.args) >= 2
                and hasattr(node.args[0], "name")
                and node.args[0].name in multi_results
            ):
                outs = multi_results[node.args[0].name]
                idx = node.args[1]
                if isinstance(idx, int) and 0 <= idx < len(outs):
                    value_map[node.name] = outs[idx]
                    continue

            if (
                target_str == "<built-in function getitem>"
                and len(node.args) >= 2
                and hasattr(node.args[0], "name")
                and node.args[0].name in value_map
            ):
                idx = node.args[1]
                if idx == 0:
                    value_map[node.name] = value_map[node.args[0].name]
                    self.diagnostics.append(
                        ImportDiagnostic(
                            fx_node=node.name,
                            level="info",
                            message=(
                                f"Resolved getitem({node.args[0].name}, 0) → primary result of decomposed tuple op"
                            ),
                        )
                    )
                else:
                    # Drop the auxiliary getitem; no IR emission, no
                    # value_map entry. Consumers that reference this
                    # node will surface as missing operands downstream.
                    self.diagnostics.append(
                        ImportDiagnostic(
                            fx_node=node.name,
                            level="info",
                            message=(
                                f"Dropped getitem({node.args[0].name}, {idx}) — "
                                f"auxiliary output of decomposed tuple op (REQ-020)"
                            ),
                        )
                    )
                continue

            # Resolve operands. Flatten list/tuple args (e.g. cat/stack take a tensor
            # list as their first arg) so the decomposition sees all tensor operands.
            operands: list[SSAValue] = []
            for arg in node.args:
                if hasattr(arg, "name") and arg.name in value_map:
                    operands.append(value_map[arg.name])
                elif isinstance(arg, (list, tuple)):
                    for sub in arg:
                        if hasattr(sub, "name") and sub.name in value_map:
                            operands.append(value_map[sub.name])

            # Try decomposition table first
            decomp_fn = self.dynamic_decompositions.get(target_str, DECOMPOSITION_TABLE.get(target_str))
            # A complex tensor is carried as a real trailing-(re, im) pair, so its xDSL value
            # has ONE MORE axis than torch's logical shape. Any decomposition that indexes,
            # reshapes or concatenates by logical dim would therefore act on the wrong axis --
            # silently, with plausible-looking output. Only the ops written against the pair
            # layout may see a complex value; everything else falls back to an opaque call that
            # is visible in the coverage report.
            if decomp_fn is not None and target_str not in _COMPLEX_AWARE_TARGETS:
                if _node_touches_complex(node, value_map):
                    self.diagnostics.append(
                        ImportDiagnostic(
                            fx_node=node.name,
                            level="warning",
                            message=(f"{target_str} has a complex operand/result and is not "
                                     f"written against the (re, im) pair layout; falling back "
                                     f"to opaque rather than indexing the wrong axis"),
                        )
                    )
                    decomp_fn = None
            if decomp_fn is not None:
                meta = dict(node.meta)
                # Forward FX-level args / kwargs to the decomposition so it
                # can extract scalar properties (group_size, axis, quant_min,
                # quant_max, etc.) that don't show up as SSA operands.
                meta["_fx_args"] = tuple(node.args)
                meta["_fx_kwargs"] = dict(node.kwargs)
                # Keep each SSA value at its original FX argument position. PT2E
                # passes calibrated scalar qparams as literals and weight qparams
                # as graph nodes; the flattened operands alone lose that distinction.
                meta["_fx_ssa_args"] = {
                    i: value_map[arg.name]
                    for i, arg in enumerate(node.args)
                    if hasattr(arg, "name") and arg.name in value_map
                }
                meta["_aten_target"] = target_str  # provenance: source aten op
                meta["_emit_named_ops"] = self.emit_named_ops  # high-level form toggle
                meta["_quant_inner"] = _quant_inner_key.get(node.name)  # subclass attr path
                try:
                    result = decomp_fn(operands, meta, node.name)
                except (IndexError, KeyError, TypeError) as decomp_err:
                    # Decomposition failed (e.g. missing operands from scalar constants).
                    # Fall through to opaque fallback instead of crashing.
                    self.diagnostics.append(
                        ImportDiagnostic(
                            fx_node=node.name,
                            level="warning",
                            message=f"Decomposition failed for {target_str}: {decomp_err}; falling back to opaque call",
                        )
                    )
                else:
                    # Verify the emitted ops before committing them. A single invalid op
                    # (e.g. a decomposition that hardcoded f32 for a bf16 tensor) would
                    # fail whole-module verification and lose the entire lowering, so on
                    # any verification failure we discard and fall through to the opaque
                    # path (typed from meta, which is always consistent downstream).
                    verify_err = None
                    try:
                        for op in result.ops:
                            # CallOp placeholders are opaque-by-design; their external
                            # decl is added below (ensure_external_decl), so don't verify
                            # them here -- verifying an undeclared callee would wrongly
                            # reject a decomposition that mixes real ops + opaque calls.
                            if isinstance(op, CallOp):
                                continue
                            op.verify()
                    except Exception as verr:  # noqa: BLE001
                        verify_err = verr
                    if verify_err is None:
                        for op in result.ops:
                            if isinstance(op, CallOp):
                                ensure_external_decl(op)
                            _forward_fx_meta(op, meta, result.pattern_hint)
                            block.add_op(op)

                        if result.result is not None:
                            value_map[node.name] = result.result
                        if result.results:
                            multi_results[node.name] = list(result.results)

                        self.decomposed_count += 1
                        hint_suffix = f", hint: {result.pattern_hint}" if result.pattern_hint else ""
                        self.diagnostics.append(
                            ImportDiagnostic(
                                fx_node=node.name,
                                level="info",
                                message=(
                                    f"Decomposed {target_str} -> {len(result.ops)} ops "
                                    f"(regions: {result.region_ids}{hint_suffix})"
                                ),
                            )
                        )
                        continue
                    self.diagnostics.append(
                        ImportDiagnostic(
                            fx_node=node.name,
                            level="warning",
                            message=f"Decomposition for {target_str} produced invalid IR ({verify_err}); opaque fallback",
                        )
                    )

            if not self.allow_opaque_fallback and target_str not in self.explicit_blackboxes:
                self.diagnostics.append(
                    ImportDiagnostic(
                        fx_node=node.name,
                        level="error",
                        message=f"Unsupported without explicit blackbox approval: {target_str}",
                    )
                )
                continue

            # Fallback: opaque func.call
            base_name = target_str.replace(".", "_")
            operand_types = tuple(str(v.type) for v in operands)
            sig_key = f"{base_name}:{operand_types}:{result_type}"

            if sig_key not in declared_sigs:
                count = name_counters.get(base_name, 0)
                unique_name = base_name if count == 0 else f"{base_name}_{count}"
                name_counters[base_name] = count + 1
                declared_sigs[sig_key] = unique_name
                real_operand_types = [v.type for v in operands]
                ext_func = FuncOp.external(unique_name, real_operand_types, [result_type])
                extern_funcs.append(ext_func)

            func_name = declared_sigs[sig_key]
            call_op = CallOp(func_name, operands, [result_type])
            block.add_op(call_op)
            value_map[node.name] = call_op.res[0]

            self.opaque_count += 1
            level = "warning" if target_str in self.explicit_blackboxes else "info"
            self.diagnostics.append(
                ImportDiagnostic(
                    fx_node=node.name,
                    level=level,
                    message=f"Opaque: {target_str} -> func.call @{func_name}",
                )
            )

        # Add return
        ret_values: list[SSAValue] = []
        if output_nodes:
            out_args = output_nodes[0].args[0] if output_nodes[0].args else ()
            if not isinstance(out_args, (tuple, list)):
                out_args = (out_args,)
            for a in out_args:
                if hasattr(a, "name") and a.name in value_map:
                    ret_values.append(value_map[a.name])

        if ret_values:
            # Emit ALL outputs, not just the first: a single-output collapse here
            # silently drops the extra results of any multi-output graph (e.g. a
            # while_loop body returning the full carry tuple (i+1, latent, ...)).
            block.add_op(ReturnOp(*ret_values))

        # Reconcile the func signature with the actual return-value types.
        # The original ret_types snapshot (line ~189) was taken from the
        # FX output-node metadata, which can disagree with what the body
        # actually produces — e.g. HF Llama checkpoints declare a bf16
        # output but the attention math upcasts to f32, leaving the
        # declared func.return type at bf16 and the live SSA value at f32.
        # Without this, xDSL's verifier rejects the module on real-scale
        # transformer captures with: "Expected arguments to have the same
        # types as the function output types".
        if ret_values:
            actual_ret_types: list[Attribute] = [v.type for v in ret_values]
            if list(func_type.outputs) != actual_ret_types:
                func_type = FunctionType.from_lists(arg_types, actual_ret_types)

        region = Region([block])
        main_func = FuncOp("forward", func_type, region)

        # Externs merged from while_loop body sub-imports (renamed, collision-free).
        pending = getattr(self, "_pending_extern_funcs", [])
        all_ops = list(extern_funcs) + list(pending) + [main_func]
        module = ModuleOp(all_ops)

        # REQ-023 (generalised): every ``linalg.transpose`` gets a
        # ``compgen.region_id`` + ``dispatch_id`` so the dispatch
        # graph can resolve consumers' operands. When a transpose
        # with permutation ``[1, 0]`` feeds a ``linalg.matmul``'s
        # B operand, the matmul is also tagged with
        # ``compgen.transposed_b="true"`` so providers can short-
        # circuit by emitting a B^T kernel.
        _annotate_transposes_and_matmuls(module)

        # linalg.matmul/quantized_matmul accumulate into their `outs` operand
        # (out += A·B), so the accumulator MUST be zero-initialized. The decompose
        # helpers feed a bare tensor.empty (undefined) as outs -- correct only when
        # the backend happens to give zeroed memory. Insert an explicit
        # `linalg.fill 0` so the contraction is always correct.
        _zero_fill_contraction_accumulators(module)

        # Taxonomy backfill: a final sweep that stamps prov.op / prov.family on every named
        # compute op still missing them (ops emitted by helpers outside any DecompResult --
        # e.g. transpose/reduce built inside matmul/layer-norm helpers -- never pass through
        # _forward_fx_meta). Guarantees the matchable vocabulary covers 100% of named ops.
        for op in module.walk():
            if "prov.family" in op.attributes:
                continue
            tax = _OP_TYPE_TAXONOMY.get(getattr(op, "name", ""))
            if tax is not None:
                kind, fam = tax
                if "prov.op" not in op.attributes:
                    op.attributes["prov.op"] = StringAttr(kind)
                op.attributes["prov.family"] = StringAttr(fam)

        try:
            module.verify()
        except Exception as e:
            self.diagnostics.append(
                ImportDiagnostic(
                    fx_node="<module>",
                    level="error",
                    message=f"Module verification failed: {e}",
                )
            )

        return module

    def get_ir_text(self, module: ModuleOp) -> str:
        """Get the IR text representation of a module."""
        stream = io.StringIO()
        Printer(stream=stream).print(module)
        return stream.getvalue()


def _zero_fill_contraction_accumulators(module: ModuleOp) -> None:
    """Ensure every matmul-family op's `outs` accumulator is zero-initialized.

    linalg contraction ops compute ``out += A·B``; an unfilled tensor.empty as outs
    reads undefined memory. For each MatmulOp/QuantizedMatmulOp whose outs is a bare
    tensor.empty, insert ``%c0 = arith.constant 0 ; %f = linalg.fill ins(%c0) outs(empty)``
    and rewrite the contraction to accumulate into %f.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr, IntegerType
    from xdsl.dialects.linalg.ops import FillOp, MatmulOp, QuantizedMatmulOp
    from xdsl.dialects.tensor import EmptyOp

    for op in list(module.walk()):
        if not isinstance(op, (MatmulOp, QuantizedMatmulOp)):
            continue
        outs = op.outputs[0]
        empty = outs.owner
        if not isinstance(empty, EmptyOp):
            continue
        res_t = outs.type
        elem = res_t.element_type
        zero = (arith.ConstantOp(FloatAttr(0.0, elem)) if not isinstance(elem, IntegerType)
                else arith.ConstantOp(IntegerAttr(0, elem)))
        fill = FillOp(inputs=[zero.result], outputs=[empty.results[0]], res=[res_t])
        fill.attributes["prov.op"] = StringAttr("fill")
        fill.attributes["prov.family"] = StringAttr("fill")
        # Inherit the contraction's source-module tag so the inserted init ops section with
        # their matmul (split_by_section buckets prov.module-less, operand-less ops as 'shared').
        mod = op.attributes.get("prov.module")
        if mod is not None:
            zero.attributes["prov.module"] = mod
            fill.attributes["prov.module"] = mod
        block = op.parent_block()
        block.insert_op_before(zero, op)
        block.insert_op_before(fill, op)
        op.operands[len(op.inputs)] = fill.results[0]


def _annotate_transposes_and_matmuls(module: ModuleOp) -> None:
    """Generalised dispatch-region annotation (REQ-023 + REQ-026).

    Walks the module once and stamps:

    - Every ``linalg.transpose`` with ``compgen.region_id`` +
      ``compgen.dispatch_id`` (idempotent — respects pre-existing
      tags from in-tree decompositions).
    - Every ``linalg.matmul`` whose B operand is a permutation-
      ``[1, 0]`` transpose with ``compgen.transposed_b="true"``.
    - Every ``func.call`` op (the opaque-fallback shape
      ``func.call @aten_relu_default`` for unmapped ATen ops) with
      ``compgen.region_id`` + ``compgen.dispatch_id``. Without this
      tag, codegen-fallback's contract extractor can't surface them
      as kernel boundaries (REQ-026's blocker for any opaque-call op).

    Counters are kept distinct per-op-family so synthesised ids
    don't collide with what decompositions assign.
    """
    from xdsl.dialects.func import CallOp
    from xdsl.dialects.linalg import MatmulOp, TransposeOp

    transpose_counter = 0
    for op in module.walk():
        if isinstance(op, TransposeOp):
            existing_rid = op.attributes.get("prov.region_id")
            if existing_rid is None:
                rid = f"transpose_{transpose_counter}"
                transpose_counter += 1
                op.attributes["prov.region_id"] = StringAttr(rid)
            else:
                rid = existing_rid.data if isinstance(existing_rid, StringAttr) else None
            if rid and "prov.dispatch_id" not in op.attributes:
                op.attributes["prov.dispatch_id"] = StringAttr(rid)

    # Opaque func.call annotation (REQ-026). Per-callee counters so
    # ids stay readable: ``aten_relu_default_0``, ``aten_add_1``, etc.
    callee_counters: dict[str, int] = {}
    for op in module.walk():
        if not isinstance(op, CallOp):
            continue
        if not (hasattr(op, "callee") and hasattr(op.callee, "string_value")):
            continue
        # External func declarations also surface as CallOps elsewhere
        # but those have no `results`; skip them defensively.
        if not op.results:
            continue
        existing_rid = op.attributes.get("prov.region_id")
        if existing_rid is None:
            callee = op.callee.string_value()
            stem = callee.lstrip("@") if callee else "call"
            count = callee_counters.get(stem, 0)
            callee_counters[stem] = count + 1
            rid = f"{stem}_{count}"
            op.attributes["prov.region_id"] = StringAttr(rid)
        else:
            rid = existing_rid.data if isinstance(existing_rid, StringAttr) else None
        if rid and "prov.dispatch_id" not in op.attributes:
            op.attributes["prov.dispatch_id"] = StringAttr(rid)

    for op in module.walk():
        if not isinstance(op, MatmulOp):
            continue
        if "prov.transposed_b" in op.attributes:
            continue  # already tagged by a decomposition
        if len(op.operands) < 2:
            continue
        b_operand = op.operands[1]
        producer = b_operand.owner
        if not isinstance(producer, TransposeOp):
            continue
        # Check the permutation is [1, 0] — the only shape that's
        # safe to advertise as "transposed B" for a 2D matmul.
        perm_attr = producer.permutation
        try:
            perm = list(perm_attr.get_values()) if hasattr(perm_attr, "get_values") else None
        except Exception:
            perm = None
        if perm == [1, 0]:
            op.attributes["prov.transposed_b"] = StringAttr("true")


def fx_to_xdsl(
    exported_program: Any,
    *,
    allow_opaque_fallback: bool = True,
    explicit_blackboxes: set[str] | None = None,
    dynamic_decompositions: dict[str, DecompFn] | None = None,
) -> tuple[ModuleOp, list[ImportDiagnostic]]:
    """Convenience function: export -> xDSL in one call.

    Returns:
        Tuple of (xDSL ModuleOp, list of diagnostics).
    """
    importer = FXImporter(
        allow_opaque_fallback=allow_opaque_fallback,
        explicit_blackboxes=set(explicit_blackboxes or ()),
        dynamic_decompositions=dict(dynamic_decompositions or {}),
    )
    module = importer.import_graph(exported_program)
    return module, importer.diagnostics


__all__ = ["FXImporter", "ImportDiagnostic", "fx_to_xdsl"]
