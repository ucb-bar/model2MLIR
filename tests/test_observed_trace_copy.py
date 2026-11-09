"""Known quantization configuration retains owned fields without opaque copies."""

import functools

import pytest
import torch

from m2m.capture.trace_copy import _metadata


def test_observed_graph_metadata_copies_typed_configuration_and_tensor_owners():
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
    from torch.ao.quantization.fx.utils import ObservedGraphModuleAttrs

    config = get_default_qconfig_mapping("x86")
    source = ObservedGraphModuleAttrs(
        node_name_to_qconfig={"linear": config.global_qconfig},
        node_name_to_scope={"linear": ("linear", torch.nn.Linear)},
        prepare_custom_config=PrepareCustomConfig(),
        equalization_node_name_to_qconfig={},
        qconfig_mapping=config,
        is_qat=False,
        observed_node_names={"linear"},
    )
    owned = _metadata(source)
    assert type(owned) is type(source) and owned is not source
    assert owned.node_name_to_qconfig is not source.node_name_to_qconfig
    assert owned.qconfig_mapping is not source.qconfig_mapping
    assert owned.prepare_custom_config is not source.prepare_custom_config
    owned.observed_node_names.add("owned_only")
    assert "owned_only" not in source.observed_node_names
    owned.prepare_custom_config.input_quantized_indexes.append(0)
    assert not source.prepare_custom_config.input_quantized_indexes
    # Binding a callback never invokes it or clones a tensor captured as an arg.
    tensor = torch.ones(4)
    invoked = []

    def callback(value, *, marker):
        invoked.append(marker)
        return value

    original = functools.partial(callback, tensor, marker="never called")
    copied = _metadata(original)
    assert copied is not original and copied.func is callback
    assert copied.args[0] is tensor and copied.keywords is not original.keywords
    assert not invoked


def test_unknown_quantization_extra_or_nested_opaque_metadata_refuses():
    from torch.ao.quantization.qconfig_mapping import QConfigMapping

    source = QConfigMapping()
    source.extra_unproved = object()
    with pytest.raises(ValueError, match="extra quantization"):
        _metadata(source)
    del source.extra_unproved
    source.global_qconfig = object()
    with pytest.raises(ValueError, match="opaque trace metadata"):
        _metadata(source)


def test_symbolic_metadata_preserves_guard_owner_without_concretization():
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    env = ShapeEnv()
    source = env.create_unbacked_symint()
    owned = _metadata({"guard_value": source})
    assert owned["guard_value"] is source
    assert str(source) == str(owned["guard_value"])


def test_symbolic_expression_metadata_preserves_exact_guard_without_evaluation():
    from sympy import Symbol

    size = Symbol("size", integer=True, positive=True)
    expression = 2 * size + 1
    source = {"symbol": size, "expression": expression, "guard": expression <= 9}
    owned = _metadata(source)
    assert owned is not source
    assert all(owned[key] is value for key, value in source.items())


def test_symbolic_path_descriptors_are_copied_without_following_or_evaluating_them():
    from torch.fx.experimental.symbolic_shapes import CallMethodKey, DivideByKey
    from torch.utils._pytree import GetAttrKey, MappingKey, SequenceKey

    source = [
        CallMethodKey("size"),
        SequenceKey(0),
        MappingKey("argument"),
        GetAttrKey("shape"),
        DivideByKey(2),
    ]
    copied = _metadata(source)
    assert copied is not source
    assert all(
        type(before) is type(after) and before == after and before is not after
        for before, after in zip(source, copied, strict=True)
    )


def test_value_ranges_preserve_all_bounds_and_flags_without_rechecking_them(
    monkeypatch,
):
    from torch.utils._sympy.value_ranges import ValueRanges

    source = ValueRanges(0, 4)

    def forbidden(*args, **kwargs):
        raise AssertionError("metadata copy must not evaluate a range constructor")

    monkeypatch.setattr(ValueRanges, "__init__", forbidden)
    owned = _metadata(source)
    assert owned is not source and type(owned) is type(source)
    assert owned.lower is source.lower and owned.upper is source.upper
    assert (owned.is_bool, owned.is_int, owned.is_float) == (
        source.is_bool,
        source.is_int,
        source.is_float,
    )
