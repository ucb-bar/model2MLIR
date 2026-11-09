"""Real FP8 exports require an explicit, separately qualified numeric policy."""

import json

import pytest
import torch

import m2m
from m2m.capture.torch_mlir_bridge import _parse_mlir_text_to_xdsl
from m2m.ir.scaled_mm import OrderedFP8ScaledMMPolicy


def policy():
    return OrderedFP8ScaledMMPolicy(True, True, "rne", "nontrapping_unobserved")


class Scaled(torch.nn.Module):
    def __init__(self, *, shape=(3, 7, 17), dtype=torch.float8_e4m3fn, option=None):
        super().__init__()
        _m, n, k = shape
        weight = torch.linspace(-2.0, 2.0, n * k).reshape(n, k).to(dtype).t()
        if option == "row_major_b":
            weight = weight.contiguous()
        self.register_buffer("weight", weight)
        self.register_buffer("scale_b", torch.linspace(0.03, 0.17, n).reshape(1, n))
        self.register_buffer("bias", torch.linspace(-1.0, 1.0, n))
        self.register_buffer("scale_result", torch.tensor(0.5))
        self.option = option

    def forward(self, a, scale_a):
        kwargs = {"out_dtype": torch.float32}
        if self.option == "bias":
            kwargs["bias"] = self.bias
        elif self.option == "scale_result":
            kwargs["scale_result"] = self.scale_result
        elif self.option == "fast":
            kwargs["use_fast_accum"] = True
        elif self.option == "default_output":
            kwargs.clear()
        return torch._scaled_mm(a, self.weight, scale_a, self.scale_b, **kwargs)


def source(model, shape=(3, 7, 17), dtype=torch.float8_e4m3fn):
    m, _, k = shape
    a = torch.linspace(-1.0, 1.0, m * k).reshape(m, k).to(dtype)
    sa = torch.linspace(0.02, 0.05, m).reshape(m, 1)
    return torch.export.export(model.eval(), (a, sa)), (a, sa)


def convert(exported, inputs, **kwargs):
    return m2m.convert(
        exported,
        inputs,
        backend="fx_importer",
        decompose=kwargs.pop("decompose", False),
        scaled_mm_policy=policy(),
        capture_trace=True,
        **kwargs,
    )


@pytest.mark.parametrize("shape", [(1, 16, 16), (3, 7, 17), (8, 32, 64)])
def test_actual_export_preserves_four_tensor_operands_guards_and_scale_order(shape):
    exported, inputs = source(Scaled(shape=shape), shape)
    original_code = exported.graph_module.code
    raw_inputs = inputs[0].view(torch.uint8).numpy().tobytes()
    result = convert(exported, inputs)
    assert result.ok, result.diagnostics
    result.module.verify()
    assert not any(op.name == "func.call" for op in result.module.walk())
    assert sum(op.name == "cf.assert" for op in result.module.walk()) == 4
    stages = [
        op for op in result.module.walk() if "prov.scaled_mm_stage" in op.attributes
    ]
    assert [op.attributes["prov.scaled_mm_stage"].data for op in stages] == [
        "raw_fp8_dot",
        "row_scale",
        "column_scale",
    ]
    dot, row, column = stages
    assert [str(value.type.element_type) for value in dot.inputs] == [
        "f8E4M3FN",
        "f8E4M3FN",
    ]
    assert dot.results[0] is row.inputs[0]
    assert row.results[0] is column.inputs[0]
    assert [op.name for op in dot.body.block.ops] == [
        "arith.extf",
        "arith.extf",
        "arith.mulf",
        "arith.addf",
        "linalg.yield",
    ]
    assert [kind.data.value for kind in dot.iterator_types] == [
        "parallel",
        "parallel",
        "reduction",
    ]
    assert dot.outputs[0].owner.name == "linalg.fill"
    seed = dot.outputs[0].owner.inputs[0].owner.properties["value"]
    assert seed.value.data == 0.0
    import math

    assert math.copysign(1.0, seed.value.data) == 1.0
    for op in stages:
        assert "prov.source_node_ids" in op.attributes
        assert op.attributes["prov.explicit_numeric_policy"].data
    layouts = json.loads(dot.attributes["prov.scaled_mm_source_layouts"].data)
    assert [layout["shape"] for layout in layouts] == [
        [shape[0], shape[2]],
        [shape[2], shape[1]],
        [shape[0], 1],
        [1, shape[1]],
    ]
    assert result.capture_trace["status"] == "diagnostic"
    assert result.capture_trace["precision"]["arithmetic_qualified"] is False
    assert result.numerical_policy["source_bit_equivalence_claimed"] is False
    assert exported.graph_module.code == original_code
    assert inputs[0].view(torch.uint8).numpy().tobytes() == raw_inputs
    parsed = _parse_mlir_text_to_xdsl(result.mlir_text)
    assert parsed is not None
    parsed.verify()


def test_default_path_stays_opaque_even_after_an_explicit_conversion():
    exported, inputs = source(Scaled())
    assert convert(exported, inputs).ok
    result = m2m.convert(exported, inputs, backend="fx_importer", decompose=False)
    assert result.ok
    assert any(op.name == "func.call" for op in result.module.walk())
    assert not any(
        "prov.scaled_mm_stage" in op.attributes for op in result.module.walk()
    )
    assert result.numerical_policy is None


def test_recorded_metadata_cannot_grant_numeric_permission():
    exported, inputs = source(Scaled())
    with pytest.raises(ValueError, match="explicit OrderedFP8ScaledMMPolicy"):
        m2m.convert(
            exported, inputs, backend="fx_importer", scaled_mm_policy=policy().record()
        )


def test_bridge_cannot_ignore_policy_by_taking_torch_mlir(monkeypatch):
    from m2m.capture import torch_mlir_bridge

    exported, inputs = source(Scaled())

    def forbidden():
        raise AssertionError("an unqualified frontend was attempted")

    monkeypatch.setattr(torch_mlir_bridge, "_try_torch_mlir_import", forbidden)
    with pytest.raises(ValueError, match="selected FXImporter"):
        torch_mlir_bridge.bridge_fx_graph(exported, inputs, scaled_mm_policy=policy())


def test_policy_does_not_demote_an_ordinary_f32_matmul_to_fp8():
    class Ordinary(torch.nn.Module):
        def forward(self, a, b):
            return a @ b

    inputs = (torch.ones(3, 17), torch.ones(17, 7))
    exported = torch.export.export(Ordinary(), inputs)
    result = convert(exported, inputs)
    assert not result.ok
    assert any(
        "actual supported source call" in message for message in result.diagnostics
    )


@pytest.mark.parametrize("bad", ["A_nan", "B_nan", "scale_a_inf", "scale_b_nan"])
def test_finite_domain_is_runtime_guarded_not_inferred_from_example_values(bad):
    model = Scaled()
    exported, inputs = source(model)
    good = convert(exported, inputs)
    assert good.ok
    changed_a, changed_sa = inputs[0].clone(), inputs[1].clone()
    if bad == "A_nan":
        changed_a.view(torch.uint8)[0, 0] = 0x7F
    elif bad == "B_nan":
        model.weight.view(torch.uint8)[0, 0] = 0xFF
    elif bad == "scale_a_inf":
        changed_sa[0, 0] = float("inf")
    else:
        model.scale_b[0, 0] = float("nan")
    changed = torch.export.export(model, (changed_a, changed_sa))
    guarded = convert(changed, (changed_a, changed_sa))
    assert guarded.ok, guarded.diagnostics
    assert sum(op.name == "cf.assert" for op in guarded.module.walk()) == 4
    assert guarded.mlir_text == good.mlir_text


@pytest.mark.parametrize(
    "option", ["bias", "scale_result", "fast", "default_output", "row_major_b"]
)
def test_unproved_actual_source_options_refuse_instead_of_falling_back(option):
    exported, inputs = source(Scaled(option=option))
    result = convert(exported, inputs)
    assert not result.ok and result.module is None
    assert any("ordered FP8" in message for message in result.diagnostics)


def test_e5m2_storage_does_not_grant_arithmetic_capability():
    exported, inputs = source(Scaled(dtype=torch.float8_e5m2), dtype=torch.float8_e5m2)
    result = convert(exported, inputs)
    assert not result.ok
    assert any("E4M3FN operands" in message for message in result.diagnostics)


def test_symbolic_input_hint_does_not_grant_a_static_shape_contract():
    model = Scaled()
    _, inputs = source(model)
    batch = torch.export.Dim("batch", min=2, max=6)
    exported = torch.export.export(
        model, inputs, dynamic_shapes=({0: batch}, {0: batch})
    )
    result = convert(exported, inputs)
    assert not result.ok
    assert any("ordered FP8" in message for message in result.diagnostics)


@pytest.mark.parametrize(
    "fields",
    [
        (False, True, "rne", "nontrapping_unobserved"),
        (True, False, "rne", "nontrapping_unobserved"),
        (1, True, "rne", "nontrapping_unobserved"),
        (True, True, "rtz", "nontrapping_unobserved"),
        (True, True, "rne", "observable"),
    ],
)
def test_numeric_permission_is_exact_and_explicit(fields):
    with pytest.raises(ValueError, match="explicit scale/reduction permission"):
        OrderedFP8ScaledMMPolicy(*fields)


@pytest.mark.parametrize(
    "change", ["default", "argument_alias", "result_alias", "order", "extra"]
)
def test_actual_schema_defaults_and_readonly_effects_are_not_assumed(change):
    from types import SimpleNamespace

    from m2m.ir.scaled_mm import _schema_defaults

    actual = torch.ops.aten._scaled_mm.default._schema
    arguments = [
        SimpleNamespace(
            name=arg.name,
            type=arg.type,
            default_value=arg.default_value,
            has_default_value=arg.has_default_value,
            kwarg_only=arg.kwarg_only,
            alias_info=arg.alias_info,
        )
        for arg in actual.arguments
    ]
    result = SimpleNamespace(type=actual.returns[0].type, alias_info=None)
    schema = SimpleNamespace(
        name=actual.name,
        overload_name=actual.overload_name,
        arguments=arguments,
        returns=[result],
    )
    if change == "default":
        arguments[-1].default_value = True
    elif change == "argument_alias":
        arguments[0].alias_info = object()
    elif change == "result_alias":
        result.alias_info = object()
    elif change == "order":
        arguments[0], arguments[1] = arguments[1], arguments[0]
    else:
        arguments.append(arguments[-1])
    with pytest.raises(ValueError, match="unproved scaled-mm source"):
        _schema_defaults(schema)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"backend": "fx_importer", "decompose": "yes"},
        {"backend": "torch_mlir", "decompose": False},
    ],
)
def test_policy_never_silently_selects_an_unqualified_frontend_or_preparation(kwargs):
    exported, inputs = source(Scaled())
    with pytest.raises(ValueError, match="fx_importer, boolean decomposition"):
        m2m.convert(exported, inputs, scaled_mm_policy=policy(), **kwargs)


def test_actual_public_torchao_source_retains_weight_bytes_and_selected_call(tmp_path):
    from safetensors.torch import load_file
    from torchao.float8.inference import Float8MMConfig
    from torchao.quantization import Float8Tensor
    from torchao.quantization.granularity import PerRow
    from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
    from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
        QuantizeTensorToFloat8Kwargs,
    )

    class ActualAO(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(16, 16, bias=False)
            self.linear.weight.detach().copy_(
                torch.linspace(-1.0, 1.0, 256).reshape(16, 16)
            )
            mm = Float8MMConfig(
                emulate=False, use_fast_accum=False, pad_inner_dim=False
            )
            act = QuantizeTensorToFloat8Kwargs(
                torch.float8_e4m3fn,
                PerRow(),
                mm_config=mm,
                kernel_preference=KernelPreference.TORCH,
            )
            weight = Float8Tensor.from_hp(
                self.linear.weight.detach(),
                float8_dtype=torch.float8_e4m3fn,
                granularity=PerRow(),
                mm_config=mm,
                kernel_preference=KernelPreference.TORCH,
                act_quant_kwargs=act,
            )
            self.linear.weight = torch.nn.Parameter(weight, requires_grad=False)

        def forward(self, x):
            return self.linear(x)

    model = ActualAO().eval()
    inputs = (torch.linspace(-0.7, 0.8, 48).reshape(3, 16),)
    exported = torch.export.export(model, inputs)
    selected = [
        node
        for node in exported.graph.nodes
        if str(node.target) == "aten._scaled_mm.default"
    ]
    assert len(selected) == 1
    before = model.linear.weight.qdata.view(torch.uint8).numpy().tobytes()
    result = convert(
        exported,
        inputs,
        decompose=True,
        weights_path=str(tmp_path / "weights.safetensors"),
    )
    assert result.ok, result.diagnostics
    assert not any(op.name == "func.call" for op in result.module.walk())
    actual = load_file(tmp_path / "weights.safetensors")
    packed = [value for value in actual.values() if value.dtype is torch.float8_e4m3fn]
    assert len(packed) == 1 and packed[0].view(torch.uint8).numpy().tobytes() == before
    assert model.linear.weight.qdata.view(torch.uint8).numpy().tobytes() == before
    assert result.capture_trace["status"] == "diagnostic"
    assert result.numerical_policy["quantizer_or_recipe_replaced"] is False


@pytest.mark.parametrize(
    "boundary", ["factory_to_import", "during_lower", "after_lower", "after_bridge"]
)
def test_policy_mutation_at_each_conversion_boundary_refuses(monkeypatch, boundary):
    import m2m.api as public_api
    from m2m.ir import scaled_mm
    from m2m.ir.import_fx import FXImporter

    exported, inputs = source(Scaled())
    permission = policy()
    if boundary == "factory_to_import":
        owner, name = FXImporter, "import_graph"
    elif boundary == "during_lower":
        owner, name = scaled_mm, "_signature"
    elif boundary == "after_lower":
        owner, name = scaled_mm, "_lower"
    else:
        owner, name = public_api, "bridge_fx_graph"
    original = getattr(owner, name)

    def changed(*args, **kwargs):
        if boundary in ("factory_to_import", "during_lower"):
            object.__setattr__(permission, "accept_scale_reordering", False)
        result = original(*args, **kwargs)
        if boundary in ("after_lower", "after_bridge"):
            object.__setattr__(permission, "accept_scale_reordering", False)
        return result

    monkeypatch.setattr(owner, name, changed)
    try:
        result = m2m.convert(
            exported,
            inputs,
            backend="fx_importer",
            decompose=False,
            capture_trace=True,
            scaled_mm_policy=permission,
        )
    except ValueError:
        return
    assert not result.ok, result.numerical_policy
    assert result.numerical_policy is None


def test_policy_record_is_bound_to_immutable_permission_snapshot(monkeypatch):
    from m2m.ir import scaled_mm

    permission = policy()
    snapshot = scaled_mm.snapshot_policy(permission)
    record = scaled_mm.sealed_policy_record(permission, snapshot)
    assert record["accept_scale_reordering"] is True
    object.__setattr__(permission, "accept_reduction_order_difference", False)
    with pytest.raises(ValueError):
        scaled_mm.sealed_policy_record(permission, snapshot)
    with pytest.raises(ValueError):
        permission.record()


def test_policy_change_during_record_serialization_refuses(monkeypatch):
    from m2m.ir import scaled_mm

    permission = policy()
    snapshot = scaled_mm.snapshot_policy(permission)
    original = OrderedFP8ScaledMMPolicy.record

    def changed(self):
        result = original(self)
        object.__setattr__(permission, "accept_scale_reordering", False)
        return result

    monkeypatch.setattr(OrderedFP8ScaledMMPolicy, "record", changed)
    with pytest.raises(ValueError):
        scaled_mm.sealed_policy_record(permission, snapshot)
