"""Actual public TorchAO FP8 source-pairs, not selected-as-original claims."""

from __future__ import annotations

import pytest
import torch

from m2m.capture.dynamic_linear_trace import (
    DynamicLinearLineageError,
    LinearInstrumentation,
    validate_selected_source,
)
from m2m.capture.float8_linear_trace import (
    RECIPE_ID,
    FiniteFloat8LinearPolicy,
    expected_weight,
)
from m2m.capture.torchao_pipeline import QuantizationConfig

AO = pytest.importorskip("torchao.quantization")


class Model(torch.nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.first = torch.nn.Linear(16, 16, bias=bias)
        self.second = torch.nn.Linear(16, 8, bias=bias)

    def forward(self, x):
        return self.second(self.first(x).relu())


def setup(bias=True):
    torch.manual_seed(7)
    m = Model(bias).eval().requires_grad_(False)
    x = torch.randn(2, 16)
    x[0] = 0
    q = QuantizationConfig(
        "none", per_module={"first": RECIPE_ID, "second": "int8_dyn_act_int8_weight"}
    )
    return m, (x,), q


def apply(m, owner):
    site = owner._sites["first"]
    expected = expected_weight(site)
    AO.quantize_(m, AO.Int8DynamicActivationInt8WeightConfig())
    m.first.weight = torch.nn.Parameter(expected, requires_grad=False)


@pytest.mark.parametrize("bias", [False, True])
def test_actual_mixed_source_pair(bias):
    m, x, q = setup(bias)
    original = m(*x).clone()
    with LinearInstrumentation(
        m, float8_policy=FiniteFloat8LinearPolicy(), float8_modules=(m.first,)
    ) as owner:
        assert torch.equal(m(*x), original)
        source = owner.capture_source(m, x)
        apply(m, owner)
        exported, proof = owner.export_selected(source, m, x, quantization=q)
        assert proof["finite_float8_linear_sites"] == 1
        assert proof["selected_export_exact_eager_output"]
        assert (
            len(
                [
                    n
                    for n in source.exported.graph.nodes
                    if n.target is torch.ops.aten._assert_async.msg
                ]
            )
            == 1
        )
        assert validate_selected_source(exported, x)


@pytest.mark.parametrize(
    "change",
    [
        "qdata",
        "scale",
        "floor",
        "activation_floor",
        "kernel",
        "config",
        "extra",
        "source",
        "bias",
        "graph",
        "policy",
        "removed_policy",
        "module_roster",
        "per_module",
    ],
)
def test_changed_recipe_source_or_permission_refuses(change):
    m, x, q = setup()
    with LinearInstrumentation(
        m, float8_policy=FiniteFloat8LinearPolicy(), float8_modules=(m.first,)
    ) as owner:
        source = owner.capture_source(m, x)
        apply(m, owner)
        w = m.first.weight
        if change == "qdata":
            w.qdata.view(torch.uint8).view(-1)[0] ^= 1
        elif change == "scale":
            w.scale.view(-1)[0] *= 2
        elif change == "floor":
            w.act_quant_kwargs.hp_value_lb = None
        elif change == "activation_floor":
            w.act_quant_kwargs.hp_value_lb = 2e-12
        elif change == "kernel":
            from torchao.quantization.quantize_.common.kernel_preference import (
                KernelPreference,
            )

            w.kernel_preference = KernelPreference.AUTO
        elif change == "config":
            from torchao.float8.inference import Float8MMConfig

            w.mm_config = Float8MMConfig(emulate=True)
        elif change == "extra":
            w.act_quant_kwargs.extra = torch.tensor(1.0)
        elif change == "source":
            owner._sites["first"]["weight"].view(-1)[0] += 1
        elif change == "bias":
            m.first.bias.view(-1)[0] += 1
        elif change == "graph":
            next(
                n
                for n in source.exported.graph.nodes
                if n.target is torch.ops.aten.relu.default
            ).target = torch.ops.aten.neg.default
        elif change == "policy":
            owner._sites["first"]["float8_policy"] = FiniteFloat8LinearPolicy()
        elif change == "removed_policy":
            del owner._sites["first"]["float8_policy"]
        elif change == "module_roster":
            owner._float8_modules = (m.second,)
        elif change == "per_module":
            q.per_module["first"] = "int8_dyn_act_int8_weight"
        with pytest.raises(
            (DynamicLinearLineageError, ValueError, TypeError, RuntimeError)
        ):
            owner.export_selected(source, m, x, quantization=q)


@pytest.mark.parametrize("what", ["qdata", "scale", "activation_floor", "policy"])
def test_late_selected_state_and_permission_mutation_refuses(what):
    m, x, q = setup()
    with LinearInstrumentation(
        m, float8_policy=FiniteFloat8LinearPolicy(), float8_modules=(m.first,)
    ) as owner:
        source = owner.capture_source(m, x)
        apply(m, owner)
        exported, _ = owner.export_selected(source, m, x, quantization=q)
        w = m.first.weight
        if what == "qdata":
            w.qdata.view(torch.uint8).view(-1)[0] ^= 1
        elif what == "scale":
            w.scale.view(-1)[0] *= 2
        elif what == "activation_floor":
            w.act_quant_kwargs.hp_value_lb = 2e-12
        else:
            owner._sites["first"]["float8_policy"] = FiniteFloat8LinearPolicy()
        with pytest.raises(DynamicLinearLineageError):
            validate_selected_source(exported, x)


@pytest.mark.parametrize("nonfinite", [float("inf"), float("-inf"), float("nan")])
def test_original_and_selected_finite_guard_reject(nonfinite):
    m, x, q = setup()
    with LinearInstrumentation(
        m, float8_policy=FiniteFloat8LinearPolicy(), float8_modules=(m.first,)
    ) as owner:
        source = owner.capture_source(m, x)
        bad = x[0].clone()
        bad[0, 0] = nonfinite
        with pytest.raises(RuntimeError, match="finite FP8"):
            source.exported.module()(bad)
        apply(m, owner)
        selected, _ = owner.export_selected(source, m, x, quantization=q)
        with pytest.raises(RuntimeError, match="finite FP8"):
            selected.module()(bad)


def test_actual_mixed_conversion_preserves_guards_and_complete_source_trace():
    import m2m
    from m2m.coverage import opaque_report
    from m2m.ir.scaled_mm import OrderedFP8ScaledMMPolicy

    m, x, q = setup()
    with LinearInstrumentation(
        m, float8_policy=FiniteFloat8LinearPolicy(), float8_modules=(m.first,)
    ) as owner:
        source = owner.capture_source(m, x)
        apply(m, owner)
        selected, _ = owner.export_selected(source, m, x, quantization=q)
        result = m2m.convert(
            selected,
            x,
            quantization=q,
            quantization_preapplied=True,
            original_frontend_snapshot=source.snapshot,
            capture_trace=True,
            backend="fx_importer",
            scaled_mm_policy=OrderedFP8ScaledMMPolicy(
                True, True, "rne", "nontrapping_unobserved"
            ),
        )
        assert result.ok, result.diagnostics
        assert result.capture_trace["blockers"] == [
            "explicit scaled-mm scale/reduction policy differs from original CPU arithmetic"
        ]
        assert not opaque_report(result.mlir_text)
        assert "finite FP8 source activation required" in result.mlir_text
        assert "cf.assert" in result.mlir_text


def test_distinct_storage_owner_but_overlapping_source_bytes_refuses():
    import numpy as np

    m, x, q = setup()
    m.first.weight.zero_()
    with LinearInstrumentation(
        m, float8_policy=FiniteFloat8LinearPolicy(), float8_modules=(m.first,)
    ) as owner:
        source = owner.capture_source(m, x)
        apply(m, owner)
        w = m.first.weight
        original = owner._sites["first"]["weight"]
        raw = np.frombuffer(
            original.detach().numpy().data, dtype=np.uint8, count=w.qdata.numel()
        ).reshape(w.qdata.shape)
        w.qdata = torch.from_numpy(raw).view(torch.float8_e4m3fn)
        assert w.qdata.untyped_storage()._cdata != original.untyped_storage()._cdata
        assert (
            w.qdata.untyped_storage().data_ptr()
            == original.untyped_storage().data_ptr()
        )
        with pytest.raises(DynamicLinearLineageError, match="alias"):
            owner.export_selected(source, m, x, quantization=q)


def test_unrelated_original_fp8_weight_read_refuses():
    class Extra(Model):
        def forward(self, x):
            return super().forward(x), self.first.weight + 1

    m = Extra().eval().requires_grad_(False)
    with (
        LinearInstrumentation(
            m, float8_policy=FiniteFloat8LinearPolicy(), float8_modules=(m.first,)
        ) as owner,
        pytest.raises(DynamicLinearLineageError, match="unclosed unrelated use"),
    ):
        owner.capture_source(m, (torch.zeros(2, 16),))


@pytest.mark.parametrize(
    "bad", ["none", "untyped", "list", "duplicate", "outside", "wrong_type"]
)
def test_capability_requires_actual_unique_owned_module_tuple(bad):
    m, _x, _q = setup()
    policy = FiniteFloat8LinearPolicy()
    selected = (m.first,)
    if bad == "none":
        policy = None
    elif bad == "untyped":
        policy = object()
    elif bad == "list":
        selected = [m.first]
    elif bad == "duplicate":
        selected = (m.first, m.first)
    elif bad == "outside":
        selected = (torch.nn.Linear(16, 16),)
    else:
        selected = (m,)
    with pytest.raises(DynamicLinearLineageError):
        LinearInstrumentation(m, float8_policy=policy, float8_modules=selected)
    assert "forward" not in m.first.__dict__


@pytest.mark.parametrize(
    "change",
    [
        "size",
        "stride",
        "dtype",
        "layout",
        "device",
        "result",
        "consumer",
        "wrong_input",
    ],
)
def test_unannotated_metadata_assertion_requires_exact_input_abi(change):
    from m2m.capture.dynamic_linear_trace import _boundaries, snapshot_exported_program
    from m2m.capture.float8_linear_trace import input_metadata_assertions

    m, x, _q = setup()
    with LinearInstrumentation(
        m, float8_policy=FiniteFloat8LinearPolicy(), float8_modules=(m.first,)
    ) as owner:
        owner.capture_source(m, x)
        apply(m, owner)
        selected = torch.export.export(
            m, x, strict=True, preserve_module_call_signature=tuple(owner._sites)
        )
        snap = snapshot_exported_program(selected, stage="quantized")
        boundaries = _boundaries(selected, owner._sites, snap, source=False)
        assertion = next(
            n
            for n in selected.graph.nodes
            if n.target is torch.ops.aten._assert_tensor_metadata.default
            and not (n.meta.get("custom") or {}).get("m2m_dynamic_site")
        )
        kwargs = dict(assertion.kwargs)
        if change in ("size", "stride"):
            kwargs[change] = [2, 16]
        elif change == "dtype":
            kwargs["dtype"] = torch.bfloat16
        elif change == "layout":
            kwargs["layout"] = torch.sparse_coo
        elif change == "device":
            kwargs["device"] = torch.device("meta")
        elif change == "result":
            assertion.meta["val"] = torch.tensor(True)
        elif change == "consumer":
            with selected.graph.inserting_after(assertion):
                selected.graph.call_function(torch.ops.aten.alias.default, (assertion,))
        else:
            assertion.args = (
                next(
                    n
                    for n in selected.graph.nodes
                    if n.target is torch.ops.aten.relu.default
                ),
            )
        assertion.kwargs = kwargs
        # A different-input assertion remains unclaimed; the complete caller
        # proof must refuse it rather than treating it as a source-input check.
        if change == "wrong_input":
            assert not input_metadata_assertions(selected, owner._sites, boundaries)
        else:
            with pytest.raises(DynamicLinearLineageError):
                input_metadata_assertions(selected, owner._sites, boundaries)


def test_guard_deletion_after_proof_refuses_without_repair():
    m, x, q = setup()
    with LinearInstrumentation(
        m, float8_policy=FiniteFloat8LinearPolicy(), float8_modules=(m.first,)
    ) as owner:
        source = owner.capture_source(m, x)
        apply(m, owner)
        selected, _ = owner.export_selected(source, m, x, quantization=q)
        node = next(
            n
            for n in selected.graph.nodes
            if n.target is torch.ops.aten._assert_async.msg
        )
        selected.graph.erase_node(node)
        with pytest.raises((DynamicLinearLineageError, ValueError)):
            validate_selected_source(selected, x)


class WithPrelude(Model):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(16)
        self.changed_caller = False
        self.reverse_prelude = False

    def forward(self, x):
        normalized = self.norm(x)
        if self.reverse_prelude:
            b = normalized * 2
            a = normalized + 1
        else:
            a = normalized + 1
            b = normalized * 2
        selected = b if self.changed_caller else a
        return self.second(self.first(selected).relu()), b


def test_metadata_assertion_on_exact_unchanged_caller_result_binds_after_ordered_DAG():
    m = WithPrelude().eval().requires_grad_(False)
    x = (torch.randn(1, 7, 16),)
    q = QuantizationConfig(
        "none", per_module={"first": RECIPE_ID, "second": "int8_dyn_act_int8_weight"}
    )
    with LinearInstrumentation(
        m, float8_policy=FiniteFloat8LinearPolicy(), float8_modules=(m.first,)
    ) as owner:
        source = owner.capture_source(m, x)
        apply(m, owner)
        selected, proof = owner.export_selected(source, m, x, quantization=q)
        assert proof["selected_export_exact_eager_output"]
        assert proof["typed_input_metadata_assertions_verified"]
        assert validate_selected_source(selected, x)


@pytest.mark.parametrize("change", ["caller", "reorder"])
def test_same_type_prelude_operand_changes_never_gain_metadata_claim(change):
    m = WithPrelude().eval().requires_grad_(False)
    x = (torch.randn(2, 16),)
    q = QuantizationConfig(
        "none", per_module={"first": RECIPE_ID, "second": "int8_dyn_act_int8_weight"}
    )
    with LinearInstrumentation(
        m, float8_policy=FiniteFloat8LinearPolicy(), float8_modules=(m.first,)
    ) as owner:
        source = owner.capture_source(m, x)
        apply(m, owner)
        before = m(*x)
        if change == "caller":
            m.changed_caller = True
        else:
            m.reverse_prelude = True
            assert all(
                torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                for a, b in zip(before, m(*x), strict=True)
            )
        with pytest.raises(DynamicLinearLineageError):
            owner.export_selected(source, m, x, quantization=q)


class _ScaledEmbedding(torch.nn.Embedding):
    def __init__(self):
        super().__init__(17, 16)
        self.output_scale = 7.25

    def forward(self, indices):
        return super().forward(indices) * self.output_scale


class _EmbeddingMixedModel(Model):
    def __init__(self, scaled):
        super().__init__()
        self.embedding = (
            _ScaledEmbedding() if scaled else torch.nn.Embedding(17, 16)
        )

    def forward(self, indices):
        values = self.embedding(indices)
        return super().forward(values)


@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize(
    "change", [None, "embedding_qdata", "embedding_scale", "embedding_policy", "fp8_qdata"]
)
def test_embedding_and_mixed_Linear_capabilities_compose_without_state_waivers(
    scaled, change
):
    from m2m.capture.embedding_trace import Int8WeightOnlyEmbeddingPolicy

    model = _EmbeddingMixedModel(scaled).eval().requires_grad_(False)
    inputs = (torch.tensor([0, 16, 1, 0], dtype=torch.int64),)
    quant = QuantizationConfig(
        "none", per_module={"first": RECIPE_ID, "second": "int8_dyn_act_int8_weight"}
    )
    with LinearInstrumentation(
        model,
        embedding_policy=Int8WeightOnlyEmbeddingPolicy(),
        float8_policy=FiniteFloat8LinearPolicy(),
        float8_modules=(model.first,),
    ) as owner:
        source = owner.capture_source(model, inputs)
        encoded_fp8 = expected_weight(owner._sites["first"])
        AO.quantize_(
            model,
            AO.Int8DynamicActivationInt8WeightConfig(),
            filter_fn=lambda module, path: isinstance(module, torch.nn.Linear),
        )
        AO.quantize_(
            model,
            AO.Int8WeightOnlyConfig(set_inductor_config=False),
            filter_fn=lambda module, path: isinstance(module, torch.nn.Embedding),
        )
        model.first.weight = torch.nn.Parameter(encoded_fp8, requires_grad=False)
        selected, proof = owner.export_selected(
            source, model, inputs, quantization=quant
        )
        assert proof["finite_float8_linear_sites"] == 1
        assert proof["embedding_weight_only_sites"] == 1
        assert proof["embedding_module_post_arithmetic_unchanged"]
        assert proof["selected_export_exact_eager_output"]
        assert len(proof["site_bindings"]) == 3
        if change == "embedding_qdata":
            model.embedding.weight.qdata[0, 0] ^= 1
        elif change == "embedding_scale":
            model.embedding.weight.scale.mul_(2)
        elif change == "embedding_policy":
            owner._sites["embedding"]["embedding_policy"] = (
                Int8WeightOnlyEmbeddingPolicy()
            )
        elif change == "fp8_qdata":
            model.first.weight.qdata.view(torch.uint8)[0, 0] ^= 1
        if change is not None:
            with pytest.raises(DynamicLinearLineageError):
                validate_selected_source(selected, inputs)
        else:
            import m2m
            from m2m.coverage import opaque_report
            from m2m.ir.scaled_mm import OrderedFP8ScaledMMPolicy

            assert validate_selected_source(selected, inputs)
            converted = m2m.convert(
                selected,
                inputs,
                backend="fx_importer",
                quantization=quant,
                quantization_preapplied=True,
                original_frontend_snapshot=source.snapshot,
                capture_trace=True,
                scaled_mm_policy=OrderedFP8ScaledMMPolicy(
                    True, True, "rne", "nontrapping_unobserved"
                ),
            )
            assert converted.ok, converted.diagnostics
            assert not opaque_report(converted.mlir_text)
            assert converted.capture_trace["blockers"] == [
                "explicit scaled-mm scale/reduction policy differs from original CPU arithmetic"
            ]


@pytest.mark.parametrize("scaled", [False, True])
def test_unmapped_exposed_embedding_intermediate_stays_unadmitted(scaled):
    from m2m.capture.embedding_trace import Int8WeightOnlyEmbeddingPolicy

    class Exposed(_EmbeddingMixedModel):
        def forward(self, indices):
            values = self.embedding(indices)
            return Model.forward(self, values), values

    model = Exposed(scaled).eval().requires_grad_(False)
    inputs = (torch.tensor([0, 16, 1, 0], dtype=torch.int64),)
    quant = QuantizationConfig(
        "none", per_module={"first": RECIPE_ID, "second": "int8_dyn_act_int8_weight"}
    )
    with LinearInstrumentation(
        model,
        embedding_policy=Int8WeightOnlyEmbeddingPolicy(),
        float8_policy=FiniteFloat8LinearPolicy(),
        float8_modules=(model.first,),
    ) as owner:
        source = owner.capture_source(model, inputs)
        encoded_fp8 = expected_weight(owner._sites["first"])
        AO.quantize_(
            model,
            AO.Int8DynamicActivationInt8WeightConfig(),
            filter_fn=lambda module, path: isinstance(module, torch.nn.Linear),
        )
        AO.quantize_(
            model,
            AO.Int8WeightOnlyConfig(set_inductor_config=False),
            filter_fn=lambda module, path: isinstance(module, torch.nn.Embedding),
        )
        model.first.weight = torch.nn.Parameter(encoded_fp8, requires_grad=False)
        # This extra live output is a conservative coverage limit. Neither
        # capability can manufacture a missing exact caller/output mapping.
        with pytest.raises(
            (DynamicLinearLineageError, KeyError), match="unbound exact operand"
        ):
            owner.export_selected(source, model, inputs, quantization=quant)
