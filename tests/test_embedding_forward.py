"""Closed executed wrapper grammar, including indirect Python-effect refusals."""

from __future__ import annotations
import __future__

import inspect

import pytest
import torch

from m2m.capture.dynamic_linear_trace import (
    DynamicLinearLineageError,
    LinearInstrumentation,
    validate_selected_source,
)
from m2m.capture.embedding_trace import Int8WeightOnlyEmbeddingPolicy
from m2m.capture.torchao_pipeline import QuantizationConfig

AO = pytest.importorskip("torchao.quantization")


class BufferScaled(torch.nn.Embedding):
    def __init__(self):
        super().__init__(17, 23)
        self.register_buffer("scale", torch.tensor(48.0), persistent=False)

    def forward(self, ids):
        return super().forward(ids) * self.scale.to(self.weight.dtype)


class Counter(torch.nn.Embedding):
    def __init__(self):
        super().__init__(17, 23)
        self.counter = 0

    def forward(self, ids):
        self.counter += 1
        return super().forward(ids)


class Append(torch.nn.Embedding):
    def __init__(self):
        super().__init__(17, 23)
        self.log = []

    def forward(self, ids):
        self.log.append(1)
        return super().forward(ids)


class Helper(torch.nn.Embedding):
    def __init__(self):
        super().__init__(17, 23)
        self.counter = 0

    def helper(self, ids):
        self.counter += 1
        return super().forward(ids)

    def forward(self, ids):
        return self.helper(ids)


class Update(torch.nn.Embedding):
    def __init__(self):
        super().__init__(17, 23)
        self.record = {}

    def forward(self, ids):
        return self.record.update({"called": True}) or super().forward(ids)


class Descriptor(torch.nn.Embedding):
    def __init__(self):
        super().__init__(17, 23)
        self.counter = 0

    @property
    def scale(self):
        self.counter += 1
        return 48.0

    def forward(self, ids):
        return super().forward(ids) * self.scale


class Getter(torch.nn.Embedding):
    def __init__(self):
        super().__init__(17, 23)

    def __getattr__(self, name):
        return super().__getattr__(name)


class Call(torch.nn.Embedding):
    def __init__(self):
        super().__init__(17, 23)
        self.counter = 0

    def __call__(self, *args, **kwargs):
        self.counter += 1
        return super().__call__(*args, **kwargs)


class Model(torch.nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding
        self.linear = torch.nn.Linear(23, 11)

    def forward(self, ids):
        value = self.embedding(ids)
        return self.linear(value), value


def prepare(embedding):
    return Model(embedding).eval().requires_grad_(False), (
        torch.tensor([[0, 16, 3, 3]]),
    )


def quantize(model):
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


@pytest.mark.parametrize("kind", [BufferScaled, torch.nn.Embedding])
def test_pure_public_lookup_and_buffer_postscale_exact(kind):
    model, inputs = prepare(kind() if kind is BufferScaled else kind(17, 23))
    with LinearInstrumentation(
        model, embedding_policy=Int8WeightOnlyEmbeddingPolicy()
    ) as owner:
        original = owner.capture_source(model, inputs)
        quantize(model)
        exported, proof = owner.export_selected(
            original,
            model,
            inputs,
            quantization=QuantizationConfig("int8_dyn_act_int8_weight"),
        )
        assert proof["selected_export_exact_eager_output"]
        assert validate_selected_source(exported, inputs)


def test_real_public_scaled_embedding_grammar():
    mod = pytest.importorskip("transformers.models.gemma2.modeling_gemma2")
    model, inputs = prepare(mod.Gemma2TextScaledWordEmbedding(17, 23, 0, 48.0))
    with LinearInstrumentation(
        model, embedding_policy=Int8WeightOnlyEmbeddingPolicy()
    ) as owner:
        original = owner.capture_source(model, inputs)
        quantize(model)
        exported, proof = owner.export_selected(
            original,
            model,
            inputs,
            quantization=QuantizationConfig("int8_dyn_act_int8_weight"),
        )
        row = next(
            r
            for r in proof["site_bindings"]
            if r.get("source_operator") == "aten.embedding.default"
        )
        assert row["module_output_source_id"] != row["source_output_id"]
        assert proof["selected_export_exact_eager_output"] and validate_selected_source(
            exported, inputs
        )


@pytest.mark.parametrize(
    "kind", [Counter, Append, Helper, Update, Descriptor, Getter, Call]
)
def test_direct_indirect_descriptor_and_call_effects_refuse_before_execution(kind):
    model, _ = prepare(kind())
    with pytest.raises(DynamicLinearLineageError, match="pure embedding wrapper"):
        LinearInstrumentation(model, embedding_policy=Int8WeightOnlyEmbeddingPolicy())
    assert (
        "forward" not in model.embedding.__dict__
        and "forward" not in model.linear.__dict__
    )
    assert model.embedding.__dict__.get("counter", 0) == 0
    assert model.embedding.__dict__.get("log", []) == []
    assert model.embedding.__dict__.get("record", {}) == {}


@pytest.mark.parametrize(
    "change",
    [
        "scale_rebind",
        "scale_value",
        "code",
        "closure",
        "defaults",
        "super",
        "instance_to",
        "hooks",
        "function_metadata",
    ],
)
@pytest.mark.parametrize("when", ["before_export", "after_export"])
def test_forward_scale_code_closure_and_call_binding_lifetime(
    change, when, monkeypatch
):
    model, inputs = prepare(BufferScaled())
    with LinearInstrumentation(
        model, embedding_policy=Int8WeightOnlyEmbeddingPolicy()
    ) as owner:
        original = owner.capture_source(model, inputs)
        quantize(model)
        if when == "after_export":
            exported, _ = owner.export_selected(
                original,
                model,
                inputs,
                quantization=QuantizationConfig("int8_dyn_act_int8_weight"),
            )
        fn = BufferScaled.forward
        if change == "scale_rebind":
            model.embedding.scale = model.embedding.scale.clone()
        elif change == "scale_value":
            model.embedding.scale.add_(1)
        elif change == "code":
            monkeypatch.setattr(
                fn,
                "__code__",
                fn.__code__.replace(
                    co_names=tuple(
                        "other" if n == "scale" else n for n in fn.__code__.co_names
                    )
                ),
            )
        elif change == "closure":
            monkeypatch.setattr(fn.__closure__[0], "cell_contents", torch.nn.Embedding)
        elif change == "defaults":
            monkeypatch.setattr(fn, "__defaults__", (None,))
        elif change == "super":
            monkeypatch.setitem(fn.__globals__, "super", lambda: None)
        elif change == "instance_to":
            model.embedding.scale.to = lambda *args: torch.tensor(48.0)
        elif change == "hooks":
            model.embedding.register_forward_pre_hook(lambda *args: None)
        elif change == "function_metadata":
            monkeypatch.setattr(fn, "__wrapped__", object(), raising=False)
        with pytest.raises(DynamicLinearLineageError):
            if when == "after_export":
                validate_selected_source(exported, inputs)
            else:
                owner.export_selected(
                    original,
                    model,
                    inputs,
                    quantization=QuantizationConfig("int8_dyn_act_int8_weight"),
                )


def test_scale_tensor_subclass_refuses():
    class OtherTensor(torch.Tensor):
        pass

    model, _ = prepare(BufferScaled())
    model.embedding.scale = model.embedding.scale.as_subclass(OtherTensor)
    with pytest.raises(DynamicLinearLineageError, match="scale"):
        LinearInstrumentation(model, embedding_policy=Int8WeightOnlyEmbeddingPolicy())


def test_global_module_hook_refuses(monkeypatch):
    from collections import OrderedDict

    from torch.nn.modules import module

    model, _ = prepare(BufferScaled())
    monkeypatch.setattr(
        module, "_global_forward_pre_hooks", OrderedDict([(1, lambda *args: None)])
    )
    with pytest.raises(DynamicLinearLineageError, match="global module hooks"):
        LinearInstrumentation(model, embedding_policy=Int8WeightOnlyEmbeddingPolicy())


@pytest.mark.parametrize(
    "change",
    [
        "exception_table",
        "stack",
        "locals",
        "optimized",
        "newlocals",
        "nested",
        "future",
    ],
)
def test_executed_code_structure_refuses_before_model_call(change, monkeypatch):
    model, _ = prepare(BufferScaled())
    fn = BufferScaled.forward
    code = fn.__code__
    if change == "exception_table":
        if not hasattr(code, "co_exceptiontable"):
            pytest.skip("exception opcodes already part of co_code on Python3.10")
        changed = code.replace(co_exceptiontable=b"\x80\x01\x00\x00")
    elif change == "stack":
        changed = code.replace(co_stacksize=code.co_stacksize + 1)
    elif change == "locals":
        changed = code.replace(
            co_nlocals=code.co_nlocals + 1, co_varnames=code.co_varnames + ("unproved",)
        )
    else:
        flag = {
            "optimized": inspect.CO_OPTIMIZED,
            "newlocals": inspect.CO_NEWLOCALS,
            "nested": inspect.CO_NESTED,
            "future": __future__.annotations.compiler_flag,
        }[change]
        changed = code.replace(co_flags=code.co_flags ^ flag)
    monkeypatch.setattr(fn, "__code__", changed)
    with pytest.raises(DynamicLinearLineageError, match="source/code mismatch"):
        LinearInstrumentation(model, embedding_policy=Int8WeightOnlyEmbeddingPolicy())
    assert "forward" not in model.embedding.__dict__


def test_real_nested_pure_class_context_is_supported():
    class Nested(torch.nn.Embedding):
        def forward(self, ids):
            return super().forward(ids)

    model, _ = prepare(Nested(17, 23))
    with LinearInstrumentation(model, embedding_policy=Int8WeightOnlyEmbeddingPolicy()):
        pass
