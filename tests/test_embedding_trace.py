"""Actual AO embedding/Linear composition and closed source permission checks."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import m2m
from m2m.capture.dynamic_linear_trace import (
    DynamicLinearLineageError,
    LinearInstrumentation,
    validate_selected_source,
)
from m2m.capture.embedding_trace import Int8WeightOnlyEmbeddingPolicy
from m2m.capture.torchao_pipeline import QuantizationConfig

AO = pytest.importorskip("torchao.quantization")
QUANT = QuantizationConfig(scheme="int8_dyn_act_int8_weight")


class Scaled(torch.nn.Embedding):
    def __init__(self, n, k):
        super().__init__(n, k)
        self.output_scale = 7.25

    def forward(self, ids):
        return super().forward(ids) * self.output_scale


class Model(torch.nn.Module):
    def __init__(self, *, scaled=False, tied=False):
        super().__init__()
        self.embedding = Scaled(17, 23) if scaled else torch.nn.Embedding(17, 23)
        self.linear = torch.nn.Linear(23, 17 if tied else 11, bias=False)
        if tied:
            self.linear.weight = self.embedding.weight
        self.switch = False

    def forward(self, ids):
        if self.switch:
            ids = ids.flip(-1)
        value = self.embedding(ids)
        return self.linear(value), value


def setup(**kwargs):
    torch.manual_seed(13)
    model = Model(**kwargs).eval().requires_grad_(False)
    return model, (torch.tensor([[0, 16, 1, 0], [7, 3, 16, 2]], dtype=torch.int64),)


def apply(model):
    AO.quantize_(
        model,
        AO.Int8DynamicActivationInt8WeightConfig(),
        filter_fn=lambda m, f: isinstance(m, torch.nn.Linear),
    )
    AO.quantize_(
        model,
        AO.Int8WeightOnlyConfig(set_inductor_config=False),
        filter_fn=lambda m, f: isinstance(m, torch.nn.Embedding),
    )


@pytest.mark.parametrize("scaled,tied", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
def test_real_AO_lookup_and_unchanged_module_DAG_compose(scaled, tied, indices_dtype):
    model, inputs = setup(scaled=scaled, tied=tied)
    inputs = (inputs[0].to(indices_dtype),)
    with LinearInstrumentation(
        model, embedding_policy=Int8WeightOnlyEmbeddingPolicy()
    ) as owner:
        original = owner.capture_source(model, inputs)
        apply(model)
        exported, proof = owner.export_selected(
            original, model, inputs, quantization=QUANT
        )
        assert (
            proof["embedding_weight_only_sites"] == 1
            and proof["embedding_module_post_arithmetic_unchanged"]
        )
        row = next(
            r
            for r in proof["site_bindings"]
            if r.get("source_operator") == "aten.embedding.default"
        )
        assert row["original_module_post_arithmetic_collapsed"] is False
        assert len(row["selected_call_ids"]) == 1
        converted = m2m.convert(
            exported,
            inputs,
            backend="fx_importer",
            quantization=QUANT,
            quantization_preapplied=True,
            original_frontend_snapshot=original.snapshot,
            capture_trace=True,
        )
        assert converted.ok, converted.diagnostics
        assert converted.capture_trace["status"] == "complete", (
            converted.capture_trace.get("blockers")
        )
        assert validate_selected_source(exported, inputs)
    assert (
        "forward" not in model.embedding.__dict__
        and "forward" not in model.linear.__dict__
    )


def test_default_still_refuses_changed_nonLinear_state():
    model, inputs = setup()
    with LinearInstrumentation(model) as owner:
        original = owner.capture_source(model, inputs)
        apply(model)
        with pytest.raises(
            DynamicLinearLineageError, match="unchanged state payload differs"
        ):
            owner.export_selected(original, model, inputs, quantization=QUANT)


@pytest.mark.parametrize(
    "what",
    [
        "qdata",
        "scale",
        "zero_point",
        "metadata_kind",
        "source_state",
        "source_input",
        "source_graph",
        "indices_caller",
        "indices_dtype",
        "controls",
        "postscale",
        "callback",
    ],
)
def test_changed_source_selected_contract_refuses(what):
    model, inputs = setup(scaled=True)
    with LinearInstrumentation(
        model, embedding_policy=Int8WeightOnlyEmbeddingPolicy()
    ) as owner:
        original = owner.capture_source(model, inputs)
        apply(model)
        if what in ("qdata", "scale", "zero_point"):
            value = getattr(model.embedding.weight, what)
            value.reshape(-1)[0] += 1
        elif what == "metadata_kind":
            model.embedding.weight.reduce_range = 0
        elif what == "source_state":
            next(iter(original.exported.state_dict.values())).reshape(-1)[0] += 1
        elif what == "source_input":
            inputs[0][0, 0] += 1
        elif what == "source_graph":
            op = next(
                n
                for n in original.exported.graph.nodes
                if n.op == "call_function"
                and n.target is torch.ops.aten.embedding.default
            )
            op.kwargs = {"sparse": True}
        elif what == "indices_caller":
            model.switch = True
        elif what == "indices_dtype":
            inputs = (inputs[0].to(torch.int32),)
        elif what == "controls":
            model.embedding.scale_grad_by_freq = 0
        elif what == "postscale":
            model.embedding.output_scale = 8.0
        elif what == "callback":
            model.embedding.weight.dequantize = lambda: torch.zeros(17, 23)
        with pytest.raises(DynamicLinearLineageError):
            owner.export_selected(original, model, inputs, quantization=QUANT)


@pytest.mark.parametrize(
    "what",
    [
        "qdata",
        "input",
        "storage_rebind",
        "metadata_kind",
        "proof",
        "graph",
        "callback",
        "controls",
    ],
)
def test_late_changes_are_revalidated_at_conversion_entry(what):
    model, inputs = setup(scaled=True)
    with LinearInstrumentation(
        model, embedding_policy=Int8WeightOnlyEmbeddingPolicy()
    ) as owner:
        original = owner.capture_source(model, inputs)
        apply(model)
        exported, proof = owner.export_selected(
            original, model, inputs, quantization=QUANT
        )
        if what == "qdata":
            model.embedding.weight.qdata.reshape(-1)[0] += 1
        elif what == "input":
            inputs[0][0, 0] += 1
        elif what == "storage_rebind":
            value = model.embedding.weight.qdata
            value.set_(value.clone())
        elif what == "metadata_kind":
            model.embedding.weight.reduce_range = 0
        elif what == "callback":
            model.embedding.weight.dequantize = lambda: torch.zeros(17, 23)
        elif what == "controls":
            model.embedding.scale_grad_by_freq = 0
        elif what == "proof":
            proof["embedding_weight_only_sites"] = 0
        elif what == "graph":
            op = next(
                n
                for n in exported.graph.nodes
                if n.op == "call_function" and n.target is torch.ops.aten.mul.Tensor
            )
            op.target = torch.ops.aten.add.Tensor
        with pytest.raises(ValueError):
            validate_selected_source(exported, inputs)


def test_source_storage_rebinding_refuses():
    model, inputs = setup()
    with LinearInstrumentation(
        model, embedding_policy=Int8WeightOnlyEmbeddingPolicy()
    ) as owner:
        original = owner.capture_source(model, inputs)
        value = original.exported.state_dict["embedding.weight"]
        value.set_(value.clone())
        apply(model)
        with pytest.raises(DynamicLinearLineageError, match="sealed tensor"):
            owner.export_selected(original, model, inputs, quantization=QUANT)


def test_alias_to_unrelated_source_state_is_not_an_encoding_exemption():
    class WithBuffer(Model):
        def __init__(self):
            super().__init__()
            self.register_buffer("other", torch.zeros(17, 23, dtype=torch.int8))
            self.embedding.weight.data.zero_()

        def forward(self, ids):
            a, b = super().forward(ids)
            return a + self.other.float().sum(), b

    model = WithBuffer().eval().requires_grad_(False)
    inputs = (torch.tensor([[0, 16, 1, 0], [7, 3, 16, 2]]),)
    with LinearInstrumentation(
        model, embedding_policy=Int8WeightOnlyEmbeddingPolicy()
    ) as owner:
        original = owner.capture_source(model, inputs)
        apply(model)
        model.embedding.weight.qdata = model.other
        with pytest.raises(DynamicLinearLineageError, match="alias"):
            owner.export_selected(original, model, inputs, quantization=QUANT)


def test_unrelated_weight_read_refuses_before_quantized_execution():
    class Extra(Model):
        def forward(self, ids):
            a, b = super().forward(ids)
            return a + self.embedding.weight.mean(), b

    model = Extra().eval().requires_grad_(False)
    _, inputs = setup()
    with (
        LinearInstrumentation(
            model, embedding_policy=Int8WeightOnlyEmbeddingPolicy()
        ) as owner,
        pytest.raises(DynamicLinearLineageError, match="unclosed unrelated use"),
    ):
        owner.capture_source(model, inputs)


def test_mutating_embedding_or_aliased_module_admission_is_atomic():
    model, _ = setup()
    model.embedding.max_norm = 1.0
    with pytest.raises(DynamicLinearLineageError):
        LinearInstrumentation(model, embedding_policy=Int8WeightOnlyEmbeddingPolicy())
    assert (
        "forward" not in model.linear.__dict__
        and "forward" not in model.embedding.__dict__
    )
    model.embedding.max_norm = None
    model.alias = model.embedding
    with pytest.raises(DynamicLinearLineageError, match="aliased module"):
        LinearInstrumentation(model, embedding_policy=Int8WeightOnlyEmbeddingPolicy())
    assert (
        "forward" not in model.linear.__dict__
        and "forward" not in model.embedding.__dict__
    )


def test_permission_is_typed_and_explicit():
    model, _ = setup()
    with pytest.raises(DynamicLinearLineageError, match="explicit"):
        LinearInstrumentation(model, embedding_policy=lambda _: True)


def test_storage_token_observes_real_resizable_allocation_capacity():
    from m2m.capture.dynamic_linear_trace import _storage

    value = torch.zeros(8)
    storage = value.untyped_storage()
    before = _storage(value)
    storage.resize_(storage.nbytes() + 64)
    after = _storage(value)
    assert before[0] == after[0] and before[2] + 64 == after[2] and before != after


@pytest.mark.parametrize("when", ["before_export", "after_export"])
@pytest.mark.parametrize("change", ["replace", "remove", "class", "roster"])
def test_capability_reference_and_type_lifetime_is_sealed(when, change):
    model, inputs = setup()
    policy = Int8WeightOnlyEmbeddingPolicy()
    with LinearInstrumentation(model, embedding_policy=policy) as owner:
        original = owner.capture_source(model, inputs)
        apply(model)
        if when == "after_export":
            exported, _ = owner.export_selected(
                original, model, inputs, quantization=QUANT
            )
        site = original.sites["embedding"]
        if change == "replace":
            site["embedding_policy"] = Int8WeightOnlyEmbeddingPolicy()
        elif change == "remove":
            del site["embedding_policy"]
        elif change == "roster":
            object.__setattr__(original, "capabilities", ())
        elif change == "class":

            class Other:
                __slots__ = ()

            object.__setattr__(policy, "__class__", Other)
        with pytest.raises(DynamicLinearLineageError, match="capability"):
            if when == "before_export":
                owner.export_selected(original, model, inputs, quantization=QUANT)
            else:
                validate_selected_source(exported, inputs)


def test_noncontiguous_source_input_allocation_resize_is_observed():
    model, _ = setup()
    inputs = (torch.arange(16, dtype=torch.int64).reshape(2, 8)[:, ::2],)
    with LinearInstrumentation(
        model, embedding_policy=Int8WeightOnlyEmbeddingPolicy()
    ) as owner:
        original = owner.capture_source(model, inputs)
        apply(model)
        storage = inputs[0].untyped_storage()
        storage.resize_(storage.nbytes() + 64)
        with pytest.raises(DynamicLinearLineageError, match="sealed tensor"):
            owner.export_selected(original, model, inputs, quantization=QUANT)


@pytest.mark.parametrize("offset,rebind", [(0, False), (8, False), (8, True)])
def test_distinct_storage_shared_numpy_backing_refuses(offset, rebind):
    model, inputs = setup()
    model.embedding.weight.data.zero_()
    with LinearInstrumentation(
        model, embedding_policy=Int8WeightOnlyEmbeddingPolicy()
    ) as owner:
        original = owner.capture_source(model, inputs)
        source_weight = model.embedding.weight
        apply(model)
        actual = model.embedding.weight.qdata
        backing = source_weight.detach().numpy().view(np.int8).reshape(-1)
        aliased = torch.from_numpy(
            backing[offset : offset + actual.numel()].reshape(actual.shape)
        )
        assert torch.equal(actual, aliased)
        assert (
            aliased.untyped_storage()._cdata != source_weight.untyped_storage()._cdata
        )
        assert aliased.data_ptr() == source_weight.data_ptr() + offset
        if rebind:
            actual.set_(aliased)
        else:
            model.embedding.weight.qdata = aliased
        with pytest.raises(DynamicLinearLineageError, match="alias"):
            owner.export_selected(original, model, inputs, quantization=QUANT)


def test_distinct_storage_shared_input_backing_refuses():
    model, _ = setup()
    model.embedding.weight.data.zero_()
    inputs = (torch.zeros((1, 64), dtype=torch.int64),)
    with LinearInstrumentation(
        model, embedding_policy=Int8WeightOnlyEmbeddingPolicy()
    ) as owner:
        original = owner.capture_source(model, inputs)
        apply(model)
        actual = model.embedding.weight.qdata
        backing = inputs[0].numpy().view(np.int8).reshape(-1)
        model.embedding.weight.qdata = torch.from_numpy(
            backing[8 : 8 + actual.numel()].reshape(actual.shape)
        )
        with pytest.raises(DynamicLinearLineageError, match="alias"):
            owner.export_selected(original, model, inputs, quantization=QUANT)


def test_storage_span_uses_full_capacity_and_device_not_tensor_start():
    from m2m.capture.embedding_trace import _checked_span, _overlap, _storage_span

    base = torch.zeros(100)
    tail = base[99:]
    assert tail.data_ptr() != base.data_ptr()
    assert _storage_span(tail) == _storage_span(base)
    assert _overlap(_storage_span(tail), _storage_span(base))
    cpu = torch.device("cpu")
    other = torch.device("cuda", 0)
    assert not _overlap(_checked_span(cpu, 32, 8), _checked_span(cpu, 40, 1))
    assert not _overlap(_checked_span(cpu, 32, 8), _checked_span(other, 32, 8))
    assert not _overlap(_checked_span(cpu, 0, 0), _checked_span(cpu, 1, 8))


@pytest.mark.parametrize(
    "pointer,capacity",
    [(0, 1), (-1, 1), (1, -1), (2**64, 0), (2**64 - 1, 1), (True, 1), (1, True)],
)
def test_unproved_storage_interval_refuses(pointer, capacity):
    from m2m.capture.embedding_trace import _checked_span

    with pytest.raises(DynamicLinearLineageError, match="interval"):
        _checked_span(torch.device("cpu"), pointer, capacity)
