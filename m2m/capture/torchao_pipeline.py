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
from typing import Any

import torch


@dataclass(frozen=True)
class QuantizationConfig:
    """Quantization configuration.

    Attributes:
        scheme: Default quantization scheme applied to every quantizable module
            (e.g., "int8_weight_only", "int4_weight_only", "fp8"). Used as a fallback
            for modules not matched by ``per_module``.
        calibration_samples: Number of calibration samples.
        group_size: Group size for grouped quantization.
        extra_args: Additional scheme-specific arguments.
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


def apply_quantization(
    model: Any, config: QuantizationConfig, *, example_inputs: tuple[Any, ...] | None = None
) -> Any:
    """Apply TorchAO quantization to a model.

    Args:
        model: PyTorch nn.Module.
        config: Quantization configuration.

    Returns:
        Quantized model (modified in-place or new module).

    """
    mx_formats = {
        "mx_gemmini_fp8": "mxfp8",
        "mx_gemmini_fp6": "mxfp6",
        "mx_gemmini_fp4": "mxfp4",
    }
    if config.per_module and (
        config.scheme in mx_formats or any(scheme in mx_formats for scheme in config.per_module.values())
    ):
        raise ValueError("per_module MX selection needs a whole-graph contraction policy")
    if config.scheme in mx_formats and not config.per_module:
        if not example_inputs:
            raise ValueError("MX Gemmini graph quantization requires example_inputs")
        from m2m.capture.mx_gemmini_quant import (
            MXGemminiFakeQuantConfig, quantize_functional_contractions_,
        )
        from torchao.quantization import quantize_

        format = mx_formats[config.scheme]
        tile = 16 if format == "mxfp8" else 32
        eligible = [module for module in model.modules()
                    if isinstance(module, torch.nn.Linear)
                    and module.in_features % 32 == 0
                    and module.out_features % tile == 0]
        quantize_(model, MXGemminiFakeQuantConfig(format=format),
                  filter_fn=lambda module, _fqn: module in eligible)
        exported = torch.export.export(model.eval(), tuple(example_inputs))
        graph_module = exported.module()
        coverage = quantize_functional_contractions_(graph_module, format)
        try:
            from torch.ao.quantization import allow_exported_model_train_eval

            allow_exported_model_train_eval(graph_module)
        except (ImportError, AttributeError):
            # PyTorch versions that removed the exported-model shim still
            # install throwing train()/eval() methods on ExportedProgram.module().
            # Capture calls eval() defensively; this graph is already frozen.
            import types

            def _set_frozen_training(self, mode=True):
                if mode:
                    raise ValueError("MX exported graph is inference-only")
                self.training = False
                return self

            graph_module.train = types.MethodType(_set_frozen_training, graph_module)
            graph_module.eval = types.MethodType(lambda self: self.train(False), graph_module)
        graph_module._m2m_quantization_stats = {
            "scheme": config.scheme,
            "torchao_linear_modules": len(eligible),
            **coverage,
            "numeric_status": "operand_fake_quant_only",
        }
        return graph_module

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
