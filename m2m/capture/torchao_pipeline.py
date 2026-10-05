"""TorchAO quantization pipeline integration.

Integrates TorchAO's quantization and sparsity workflows into the CompGen
capture pipeline. Quantization decisions affect kernel contracts, layout
requirements, and verification (quantized paths need separate golden outputs).

Invariants:
    - Quantization config must be serializable (part of the recipe).
    - Accuracy degradation from quantization is measured and reported.
    - Quantized models produce separate golden outputs for verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable
from typing import Any

import torch

from m2m.capture.external_quantization import ExternalQuantizationConfig, apply_external_quantization


@dataclass(frozen=True)
class QuantizationConfig:
    """Quantization configuration.

    Attributes:
        scheme: Default quantization scheme applied to every quantizable module
            (e.g., "int8_weight_only", "int4_weight_only", "fp8"). Used as a fallback
            for modules not matched by ``per_module``.
        calibration_samples: Number of calibration samples.
        group_size: Group size for grouped quantization.
        extra_args: Additional scheme-specific arguments. Static W8A8 accepts
            ``weight_granularity="per_channel"`` (default) or ``"per_tensor"``.
            This changes quantization numerics and requires fresh calibration.
        per_module: Optional mixed-precision map ``{name_substring_or_regex: scheme}``.
            Each rule quantizes modules whose fully-qualified name matches the key with
            the given scheme -- so one network can mix fp8 + int8 (and leave the rest in
            full precision). Rules are applied in iteration order; first match wins per
            module. If ``per_module`` is set and ``scheme`` is left as the sentinel
            ``"none"``, unmatched modules stay unquantized.
    """

    scheme: str
    calibration_samples: int = 100
    group_size: int | None = None
    extra_args: dict[str, Any] = field(default_factory=dict)
    per_module: dict[str, str] | None = None


@dataclass(frozen=True)
class AccuracyReport:
    """Quantization accuracy report.

    Attributes:
        l2_error: L2 norm of output difference.
        max_abs_error: Maximum absolute output difference.
        cosine_similarity: Cosine similarity between original and quantized outputs.
        within_tolerance: Whether errors are within acceptable bounds.
        tolerance: The tolerance threshold used.
    """

    l2_error: float
    max_abs_error: float
    cosine_similarity: float
    within_tolerance: bool
    tolerance: float


def _rematerialize_parameters(model: Any) -> None:
    """Replace every parameter with a fresh nn.Parameter holding a cloned tensor.

    This drops any outstanding weakrefs on the original parameter objects so
    torchao's swap_tensors-based quantize_ can proceed on HuggingFace models.
    """
    import torch.nn as nn

    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if param is not None:
                module._parameters[name] = nn.Parameter(
                    param.detach().clone(), requires_grad=param.requires_grad
                )


def _as_input_tuple(sample: Any) -> tuple[Any, ...]:
    """Normalize one calibration sample to the positional-input tuple torch expects."""
    if isinstance(sample, tuple):
        return sample
    if isinstance(sample, list):
        return tuple(sample)
    return (sample,)


def _drop_unused_graph_state(graph_module: Any) -> int:
    """Remove parameters/buffers that no live FX ``get_attr`` reads.

    PT2E's frozen-weight conversion adds an int8 ``_frozen_param*`` but leaves
    the original fp32 module parameter registered.  A subsequent torch.export
    lifts every registered parameter into its signature even when the graph no
    longer reads it, causing deployment bundles to contain both copies.  This
    is graph DCE for state: only attributes absent from the live graph are
    removed, so it cannot alter execution.
    """
    live = {
        str(node.target)
        for node in graph_module.graph.nodes
        if node.op == "get_attr"
    }
    removed = 0
    for kind in ("_parameters", "_buffers"):
        names = [name for name, _ in (
            graph_module.named_parameters(recurse=True)
            if kind == "_parameters"
            else graph_module.named_buffers(recurse=True)
        )]
        for qualified_name in names:
            if qualified_name in live:
                continue
            parent = graph_module
            parts = qualified_name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            registry = getattr(parent, kind)
            if parts[-1] in registry:
                del registry[parts[-1]]
                removed += 1
    return removed


def _apply_pt2e_static_w8a8(
    model: Any,
    config: QuantizationConfig,
    *,
    example_inputs: tuple[Any, ...] | None,
    calibration_inputs: Iterable[Any] | None,
    original_frontend_snapshot: dict[str, Any] | None = None,
) -> Any:
    """Calibrate and freeze a portable Conv/Linear W8A8 graph with TorchAO PT2E.

    TorchAO's unified ``quantize_`` API intentionally defaults to ``nn.Linear``.
    Applying ``Int8StaticActivationInt8WeightConfig`` through that API therefore
    leaves every Conv2d in a CNN at f32 while still allowing callers to stamp the
    artifact "int8". PT2E is the graph-level TorchAO path: it can observe tensor
    edges, fold inference BatchNorm into Conv, and freeze weights as actual int8
    tensors. The emitted Q/DQ graph is backend-neutral and is deliberately not
    tied to Gemmini (or any other target).

    The small quantizer below annotates aten Conv2d and Linear directly instead of
    importing a CPU backend recipe. This keeps the capture contract about numeric
    format, not about the host used to perform capture.
    """
    if not example_inputs:
        raise ValueError(
            "int8_static_act_int8_weight requires example_inputs for PT2E export"
        )

    try:
        from torchao.quantization.pt2e import HistogramObserver, MinMaxObserver, PerChannelMinMaxObserver
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
        from torchao.quantization.pt2e.quantizer import (
            QuantizationAnnotation,
            QuantizationSpec,
            Quantizer,
        )
    except ImportError as exc:
        raise RuntimeError("TorchAO PT2E quantization is unavailable") from exc

    weight_granularity = config.extra_args.get("weight_granularity", "per_channel")
    if weight_granularity not in ("per_channel", "per_tensor"):
        raise ValueError("static W8A8 weight_granularity must be per_channel or per_tensor")

    class _PortableW8A8Quantizer(Quantizer):
        def __init__(self, fold_candidates: list | None = None) -> None:
            eps = float(config.extra_args.get("eps", 2**-12))
            self.activation = QuantizationSpec(
                dtype=torch.int8,
                observer_or_fake_quant_ctr=HistogramObserver.with_args(eps=eps),
                quant_min=-128,
                quant_max=127,
                qscheme=torch.per_tensor_symmetric,
            )
            per_channel = weight_granularity == "per_channel"
            weight_observer = PerChannelMinMaxObserver if per_channel else MinMaxObserver
            self.weight = QuantizationSpec(
                dtype=torch.int8,
                observer_or_fake_quant_ctr=weight_observer.with_args(eps=eps),
                quant_min=-127,
                quant_max=127,
                qscheme=torch.per_channel_symmetric if per_channel else torch.per_tensor_symmetric,
                ch_axis=0 if per_channel else None,
            )
            self.annotated = 0
            self.fold_candidates = fold_candidates or []

        def transform_for_annotation(self, graph_module: Any) -> Any:
            # torchao invokes this immediately after its Conv+BN fold, while
            # the original FX node objects and rewired users are observable.
            if self.fold_candidates:
                from m2m.capture.trace import attach_pt2e_conv_bn_folds

                attach_pt2e_conv_bn_folds(graph_module, self.fold_candidates)
            return graph_module

        def annotate(self, graph_module: Any) -> Any:
            supported = {torch.ops.aten.conv2d.default, torch.ops.aten.linear.default}
            for node in graph_module.graph.nodes:
                if node.target not in supported or len(node.args) < 2:
                    continue
                activation, weight = node.args[:2]
                # Quantize each contraction's two inputs. Its result stays f32;
                # a later contraction observes/quantizes its own input edge. This
                # is a portable W8A8 core with an explicit requant/dequant boundary,
                # and does not assume a target can keep arbitrary epilogues in i8.
                node.meta["quantization_annotation"] = QuantizationAnnotation(
                    input_qspec_map={activation: self.activation, weight: self.weight},
                    _annotated=True,
                )
                self.annotated += 1
            return graph_module

        def validate(self, graph_module: Any) -> None:
            del graph_module
            if self.annotated == 0:
                raise ValueError(
                    "int8_static_act_int8_weight found no supported Conv2d/Linear ops"
                )

    # PT2E consumes an exported aten graph. prepare_pt2e also performs the
    # inference Conv+BatchNorm fold before observer insertion.
    exported = torch.export.export(model.eval(), tuple(example_inputs))
    if original_frontend_snapshot is not None:
        from m2m.capture.trace import attach_original_identity, snapshot_exported_program

        actual = snapshot_exported_program(exported, stage="quantization_input")
        attach_original_identity(exported, original_frontend_snapshot, actual)
    exported_module = exported.module()
    fold_candidates = []
    if original_frontend_snapshot is not None:
        from m2m.capture.trace import pt2e_conv_bn_fold_candidates

        fold_candidates = pt2e_conv_bn_fold_candidates(exported_module, original_frontend_snapshot)
    quantizer = _PortableW8A8Quantizer(fold_candidates)
    prepared = prepare_pt2e(exported_module, quantizer)

    samples: Iterable[Any]
    if calibration_inputs is None:
        samples = (tuple(example_inputs),)
    else:
        samples = calibration_inputs
    calibrated = 0
    with torch.no_grad():
        for sample in samples:
            if calibrated >= config.calibration_samples:
                break
            prepared(*_as_input_tuple(sample))
            calibrated += 1
    if calibrated == 0:
        raise ValueError("static W8A8 calibration requires at least one input sample")

    # fold_quantize freezes weight Q nodes to int8 buffers. Keeping float
    # originals in an unused module attribute is harmless: export/externalize
    # follows the live graph and stores the frozen int8 parameter.
    quantized = convert_pt2e(prepared, use_reference_representation=False, fold_quantize=True)
    if original_frontend_snapshot is not None:
        from m2m.capture.trace import attach_quantization_boundaries

        attach_quantization_boundaries(quantized, {torch.ops.aten.conv2d.default,
                                                  torch.ops.aten.linear.default})
    pruned_state = _drop_unused_graph_state(quantized)
    try:
        # Exported graph modules reject ordinary eval()/train() unless this shim
        # is installed; model2MLIR's capture boundary calls eval() defensively.
        from torch.ao.quantization import allow_exported_model_train_eval

        allow_exported_model_train_eval(quantized)
    except (ImportError, AttributeError):  # pragma: no cover - version-dependent API
        pass
    quantized._m2m_quantization_stats = {  # type: ignore[attr-defined]
        "scheme": config.scheme,
        "weight_granularity": weight_granularity,
        "annotated_contractions": quantizer.annotated,
        "calibration_samples": calibrated,
        "pruned_dead_state_tensors": pruned_state,
    }
    return quantized


def apply_quantization(
    model: Any,
    config: QuantizationConfig | ExternalQuantizationConfig,
    *,
    example_inputs: tuple[Any, ...] | None = None,
    calibration_inputs: Iterable[Any] | None = None,
    original_frontend_snapshot: dict[str, Any] | None = None,
) -> Any:
    """Apply TorchAO quantization to a model.

    Args:
        model: PyTorch nn.Module.
        config: Quantization configuration.

    Returns:
        Quantized model (modified in-place or new module).

    """
    if isinstance(config, ExternalQuantizationConfig):
        return apply_external_quantization(
            model, tuple(example_inputs or ()), config,
            original_frontend_snapshot=original_frontend_snapshot,
        )
    if config.scheme == "int8_static_act_int8_weight" and not config.per_module:
        return _apply_pt2e_static_w8a8(
            model,
            config,
            example_inputs=example_inputs,
            calibration_inputs=calibration_inputs,
            original_frontend_snapshot=original_frontend_snapshot,
        )

    # NOTE: CompGen's NPU-custom FP8 schemes ("fp8_e4m3_po2[_npu]") were dropped
    # during extraction (they depended on NPU-specific modules). Reintroduce them
    # as a m2m.quant extension if needed.

    # torchao's quantize_ (and the later .to()/.eval() during capture) swap weight
    # tensors via torch.utils.swap_tensors, which refuses to swap a tensor that has
    # an outstanding weakref -- HuggingFace models leave such weakrefs, producing
    # "Cannot swap ... because it has weakref associated with it". Two mitigations:
    #   1. re-materialize every parameter as a fresh object (drops weakrefs), and
    #   2. disable swap-on-conversion process-wide so the post-quantization capture
    #      (.to()/.eval() on quantized-subclass weights) uses copy semantics.
    _rematerialize_parameters(model)
    try:
        torch.__future__.set_swap_module_params_on_conversion(False)
    except Exception:  # noqa: BLE001 - older torch without the toggle
        pass

    # Legacy TorchAO schemes -- handle both pre-0.17 (function-based) and
    # 0.17+ (Config-class-based) APIs.
    try:
        from torchao.quantization import quantize_
    except ImportError as exc:
        raise RuntimeError("torchao is not installed") from exc

    scheme_map: dict[str, Any] = {}

    # torchao 0.17+: Config classes (preferred)
    try:
        from torchao.quantization import Int8WeightOnlyConfig

        scheme_map["int8_weight_only"] = Int8WeightOnlyConfig
    except ImportError:
        pass
    try:
        from torchao.quantization import Int4WeightOnlyConfig

        scheme_map["int4_weight_only"] = Int4WeightOnlyConfig
    except ImportError:
        pass
    try:
        from torchao.quantization import Float8WeightOnlyConfig

        scheme_map["fp8"] = Float8WeightOnlyConfig
    except ImportError:
        pass
    # int8 W + int8 A for systolic / RVV int8 datapaths (Gemmini, Saturn/OPU).
    # Per-channel weights, per-tensor dynamic activations.
    try:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig

        scheme_map["int8_dyn_act_int8_weight"] = Int8DynamicActivationInt8WeightConfig
    except ImportError:
        pass
    try:
        from torchao.quantization import Int8StaticActivationInt8WeightConfig

        scheme_map["int8_static_act_int8_weight"] = Int8StaticActivationInt8WeightConfig
    except ImportError:
        pass

    # EMBEDDING tables. torchao's `quantize_` default filter matches nn.Linear only, so an
    # `nn.Embedding` stays fp32 no matter which weight-only scheme is asked for -- which is how an
    # "int8" bundle ends up dominated by one fp32 matrix. Measured on whisper-tiny: 76.0 MB of a
    # 116.8 MB int8 bundle was `decoder.embed_tokens.weight` [51865, 384] in fp32, while the TIED
    # `proj_out.weight` (same tensor, verified by data_ptr) was already stored int8 at 19.0 MB -- the
    # same matrix shipped twice, once at 4x the size. On a board loaded over a serial link that is not
    # a footprint nicety: the loader transmits MemSiz, so it was 13 extra minutes per upload.
    #
    # Reach it with a per-axis intx config, which does support nn.Embedding, and apply it by FQN
    # through `per_module` (whose filter matches on name, not module class).
    try:
        import torch as _torch
        from torchao.quantization import IntxWeightOnlyConfig
        from torchao.quantization.granularity import PerAxis

        def _int8_embedding_weight_only() -> Any:
            # Per-axis(0) = one scale per vocabulary row, so a gathered row dequantizes with its own
            # scale. Per-tensor over a 51865-row table would let one outlier row set the scale for
            # every token.
            return IntxWeightOnlyConfig(weight_dtype=_torch.int8, granularity=PerAxis(0))

        scheme_map["int8_embedding_weight_only"] = _int8_embedding_weight_only
    except ImportError:
        pass

    # torchao <0.17: function-based API (fallback)
    if "int8_weight_only" not in scheme_map:
        try:
            from torchao.quantization import int8_weight_only

            scheme_map["int8_weight_only"] = int8_weight_only
        except ImportError:
            pass
    if "int4_weight_only" not in scheme_map:
        try:
            from torchao.quantization import int4_weight_only

            scheme_map["int4_weight_only"] = int4_weight_only
        except ImportError:
            pass
    if "fp8" not in scheme_map:
        try:
            from torchao.quantization import float8_weight_only

            scheme_map["fp8"] = float8_weight_only
        except ImportError:
            pass

    def _resolve_factory(scheme: str) -> Any:
        f = scheme_map.get(scheme)
        if f is None:
            # Fall back to the scheme catalog (covers fp8 variants, mx, nvfp4, etc.):
            # resolve the scheme's config_class_path to a TorchAO Config class.
            from m2m.capture.torchao_schemes import TORCHAO_SCHEMES, resolve_config

            scheme_obj = TORCHAO_SCHEMES.get(scheme)
            if scheme_obj is not None:
                f = resolve_config(scheme_obj)
        if f is None:
            raise ValueError(f"Unsupported TorchAO scheme: {scheme}")
        return f

    def _reset_swap() -> None:
        try:
            torch.__future__.set_swap_module_params_on_conversion(False)
        except Exception:  # noqa: BLE001
            pass

    # Mixed-precision: apply each per-module rule to the modules whose fully-qualified
    # name matches its pattern (regex search). One quantize_ call per rule with a
    # filter_fn; first matching rule wins, so later rules don't re-quantize a module.
    if config.per_module:
        import re

        rules = list(config.per_module.items())

        def _matched_earlier(fqn: str, upto: int) -> bool:
            return any(re.search(pat, fqn) for pat, _ in rules[:upto])

        for i, (pattern, scheme) in enumerate(rules):
            factory = _resolve_factory(scheme)

            def filter_fn(module: Any, fqn: str, _pat=pattern, _i=i) -> bool:
                return bool(re.search(_pat, fqn)) and not _matched_earlier(fqn, _i)

            quantize_(model, factory(), filter_fn=filter_fn)
            _reset_swap()
        # default scheme (if not the "none" sentinel) covers everything still unmatched
        if config.scheme and config.scheme != "none":
            default_factory = _resolve_factory(config.scheme)
            # Passing ANY filter_fn replaces torchao's own, which is `_is_linear` -- so a
            # name-based filter alone offers every module in the tree to a Linear-only transform,
            # and the first one without a `weight` (a LayerNorm, a container, the root) asserts:
            #   "applying int8 weight only quant requires module to have weight attribute"
            # Keep torchao's structural test and AND the name rule onto it. Without this, mixed
            # precision only worked when the rules happened to cover every non-Linear module.
            try:
                from torchao.quantization.quant_api import _is_linear
            except ImportError:                                 # pragma: no cover - old torchao
                def _is_linear(mod: Any, *_a: Any) -> bool:
                    return isinstance(mod, torch.nn.Linear) and hasattr(mod, "weight")

            def default_filter(module: Any, fqn: str) -> bool:
                return (_is_linear(module, fqn)
                        and not any(re.search(pat, fqn) for pat, _ in rules))

            quantize_(model, default_factory(), filter_fn=default_filter)
            _reset_swap()
        return model

    factory = _resolve_factory(config.scheme)
    quantize_(model, factory())
    # torchao's quantize_ re-enables swap-on-conversion; turn it back off so the
    # subsequent capture (.to()/.eval() on quantized-subclass weights) uses copy
    # semantics and doesn't trip the weakref guard.
    _reset_swap()
    return model


def verify_quant_accuracy(
    original_model: Any,
    quantized_model: Any,
    test_inputs: Any,
    tolerance: float = 0.01,
) -> AccuracyReport:
    """Verify quantization accuracy against the original model."""
    with torch.no_grad():
        reference = original_model(*test_inputs)
        candidate = quantized_model(*test_inputs)

    diff = (reference - candidate).float()
    ref_norm = torch.linalg.vector_norm(reference.float()).item()
    cand_norm = torch.linalg.vector_norm(candidate.float()).item()
    l2_error = torch.linalg.vector_norm(diff).item()
    max_abs_error = diff.abs().max().item()

    denom = max(ref_norm * cand_norm, 1e-12)
    cosine_similarity = torch.sum(reference.float() * candidate.float()).item() / denom
    within_tolerance = max_abs_error <= tolerance

    return AccuracyReport(
        l2_error=l2_error,
        max_abs_error=max_abs_error,
        cosine_similarity=float(cosine_similarity),
        within_tolerance=within_tolerance,
        tolerance=tolerance,
    )


__all__ = ["AccuracyReport", "QuantizationConfig", "apply_quantization", "verify_quant_accuracy"]
