"""Finite-saturating source casts cannot become unbounded standard truncation."""

import pytest
import torch
from xdsl.dialects.arith import FastMathFlagsAttr
from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
    IntegerAttr,
    StringAttr,
    TensorType,
    f32,
)
from xdsl.ir import Block

import m2m
from m2m.ir.decompositions import (
    _e4m3fn_cast_clamp_domain,
    decompose_clamp,
    decompose_to_copy,
)


class Cast(torch.nn.Module):
    def __init__(self, bounds=None):
        super().__init__()
        self.bounds = bounds

    def forward(self, value):
        if self.bounds is not None:
            value = value.clamp(min=self.bounds[0], max=self.bounds[1])
        return value.to(torch.float8_e4m3fn)


@pytest.mark.parametrize(
    "sample",
    [
        torch.zeros(9),
        torch.tensor(
            [
                -float("inf"),
                -465.0,
                -448.0,
                -0.0,
                0.0,
                448.0,
                465.0,
                float("inf"),
                float("nan"),
            ]
        ),
    ],
)
def test_unbounded_actual_export_is_explicit_diagnostic_even_for_zero_examples(sample):
    exported = torch.export.export(Cast(), (sample,))
    result = m2m.convert(exported, (sample,), backend="fx_importer", capture_trace=True)
    assert result.ok, result.diagnostics
    result.module.verify()
    assert not any(op.name == "arith.truncf" for op in result.module.walk())
    assert "aten_to_e4m3fn_unproved_cast_domain" in result.mlir_text
    assert result.capture_trace["status"] == "diagnostic"
    assert result.capture_trace["precision"]["arithmetic_qualified"] is False
    assert result.capture_trace["blockers"]


@pytest.mark.parametrize("bounds", [(-448.0, 448.0), (-240.0, 100.0), (0.0, 448.0)])
def test_actual_existing_clamp_retains_its_complete_source_body(bounds):
    sample = torch.tensor(
        [-float("inf"), -465.0, -0.0, 0.0, 465.0, float("inf"), float("nan")]
    )
    result = m2m.convert(
        Cast(bounds), (sample,), backend="fx_importer", capture_trace=True
    )
    assert result.ok, result.diagnostics
    result.module.verify()
    truncations = [op for op in result.module.walk() if op.name == "arith.truncf"]
    assert len(truncations) == 1
    assert (
        truncations[0]
        .attributes["prov.fp8_cast_domain"]
        .data.startswith("existing_ordered")
    )
    assert sum(op.name == "arith.cmpf" for op in result.module.walk()) == 2
    assert sum(op.name == "arith.select" for op in result.module.walk()) == 2
    assert result.capture_trace["status"] == "complete", result.capture_trace[
        "blockers"
    ]
    # Structural accounting and overflow-domain evidence do not qualify target
    # arithmetic, arbitrary NaN payloads, or the model's numerical accuracy.
    assert result.capture_trace["precision"]["arithmetic_qualified"] is False


def _clamp_source(bounds=(-448.0, 448.0)):
    source = Block(arg_types=[TensorType(f32, [3])]).args[0]
    value = torch.zeros(3)
    produced = decompose_clamp(
        [source],
        {
            "val": value,
            "_fx_args": (value,),
            "_fx_kwargs": {"min": bounds[0], "max": bounds[1]},
        },
        "clamp",
    )
    meta = {
        "val": torch.empty(3, dtype=torch.float8_e4m3fn),
        "_fx_args": (value,),
        "_fx_ssa_args": {0: produced.result},
    }
    return produced.result, meta


@pytest.mark.parametrize(
    "bounds",
    [
        (-449.0, 448.0),
        (-448.0, 449.0),
        (-float("inf"), 448.0),
        (-448.0, float("inf")),
        (float("nan"), 448.0),
        (10.0, -10.0),
    ],
)
def test_out_of_domain_or_unproved_clamps_decline(bounds):
    source, meta = _clamp_source(bounds)
    assert _e4m3fn_cast_clamp_domain(source, meta) is None
    result = decompose_to_copy([source], meta, "cast")
    assert not any(
        op.name == "arith.truncf" for root in result.ops for op in root.walk()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_input",
        "wrong_predicate",
        "wrong_selection",
        "wrong_yield",
        "fastmath",
        "unknown_body_attribute",
        "unknown_generic_attribute",
        "library_call",
        "nonidentity_map",
        "reduction",
        "missing_source_dtype",
        "contradictory_source_dtype",
        "missing_positional",
        "wrong_positional",
        "spoofed_dtype",
        "custom_positional",
    ],
)
def test_domain_requires_the_complete_typed_live_ssa_body(mutation):
    source, meta = _clamp_source()
    producer = source.owner
    block = producer.body.block
    lo, low_cmp, low_select, _hi, high_cmp, _high_select, yielded = list(block.ops)
    if mutation == "wrong_input":
        low_cmp.operands = [block.args[1], lo.result]
    elif mutation == "wrong_predicate":
        low_cmp.properties["predicate"] = IntegerAttr(5, 64)
    elif mutation == "wrong_selection":
        low_select.operands = [low_cmp.result, block.args[0], lo.result]
    elif mutation == "wrong_yield":
        yielded.operands = [block.args[0]]
    elif mutation == "fastmath":
        high_cmp.properties["fastmath"] = FastMathFlagsAttr("fast")
    elif mutation == "unknown_body_attribute":
        low_select.attributes["unknown.effects"] = StringAttr("unproved")
    elif mutation == "unknown_generic_attribute":
        producer.attributes["unknown.effects"] = StringAttr("unproved")
    elif mutation == "library_call":
        producer.properties["library_call"] = StringAttr("unknown")
    elif mutation == "nonidentity_map":
        from xdsl.dialects.builtin import AffineMapAttr
        from xdsl.ir.affine import AffineExpr, AffineMap

        producer.properties["indexing_maps"] = ArrayAttr(
            [
                AffineMapAttr(AffineMap(1, 0, (AffineExpr.constant(0),))),
                producer.indexing_maps.data[1],
            ]
        )
    elif mutation == "reduction":
        from xdsl.dialects.linalg.attrs import IteratorTypeAttr

        producer.properties["iterator_types"] = ArrayAttr(
            [IteratorTypeAttr.reduction()]
        )
    elif mutation == "missing_source_dtype":
        meta["_fx_args"] = ()
    elif mutation == "contradictory_source_dtype":
        meta["_fx_args"] = (torch.zeros(3, dtype=torch.float16),)
    elif mutation == "missing_positional":
        del meta["_fx_ssa_args"]
    elif mutation == "wrong_positional":
        meta["_fx_ssa_args"] = {0: producer.inputs[0]}
    elif mutation == "spoofed_dtype":

        class NotATensor:
            dtype = torch.float32

        meta["_fx_args"] = (NotATensor(),)
    elif mutation == "custom_positional":

        class UnknownDictionary(dict):
            def get(self, *args):
                raise AssertionError("opaque callback must not run")

        meta["_fx_ssa_args"] = UnknownDictionary({0: source})
    assert _e4m3fn_cast_clamp_domain(source, meta) is None


def test_valid_domain_is_derived_from_actual_float_constants():
    source, meta = _clamp_source()
    assert _e4m3fn_cast_clamp_domain(source, meta) == (-448.0, 448.0)
    # Metadata that claims a range cannot rescue a changed actual bound.
    source.owner.body.block.first_op.properties["value"] = FloatAttr(-1000.0, f32)
    meta["range_hint"] = [-448.0, 448.0]
    assert _e4m3fn_cast_clamp_domain(source, meta) is None


@pytest.mark.parametrize("key", ["prov.module", "prov.fqn"])
def test_known_module_provenance_does_not_change_the_clamp_proof(key):
    source, meta = _clamp_source()
    source.owner.attributes[key] = StringAttr("ordinary.nested.module")
    source.owner.body.block.first_op.attributes[key] = StringAttr("another.name")
    assert _e4m3fn_cast_clamp_domain(source, meta) == (-448.0, 448.0)


@pytest.mark.parametrize("key", ["prov.module", "prov.fqn"])
def test_module_provenance_must_have_the_importer_string_type(key):
    source, meta = _clamp_source()
    source.owner.attributes[key] = IntegerAttr(0, 64)
    assert _e4m3fn_cast_clamp_domain(source, meta) is None
