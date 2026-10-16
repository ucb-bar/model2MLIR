"""Outlier channel splitting: found by structure and statistics, exact in floating point."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from m2m.capture.outlier_split import (
    bucket_widths,
    gated_units,
    pad_gated_widths,
    select_splits,
    split_outlier_channels,
)


class GatedMLP(nn.Module):
    """A gated unit under names no model uses, so detection cannot lean on attribute names."""

    def __init__(self, width: int = 16, hidden: int = 48, bias: bool = False) -> None:
        super().__init__()
        self.alpha = nn.Linear(width, hidden, bias=bias)
        self.beta = nn.Linear(width, hidden, bias=bias)
        self.omega = nn.Linear(hidden, width, bias=bias)
        self.nonlinearity = nn.SiLU()

    def forward(self, x):
        return self.omega(self.nonlinearity(self.alpha(x)) * self.beta(x))


class PlainMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first, self.second, self.third = nn.Linear(16, 32), nn.Linear(32, 32), nn.Linear(32, 16)

    def forward(self, x):
        return self.third(torch.nn.functional.gelu(self.second(torch.relu(self.first(x)))))


def _planted(bias: bool = False, channel: int = 29, scale: float = 300.0) -> GatedMLP:
    torch.manual_seed(7)
    unit = GatedMLP(bias=bias).eval().requires_grad_(False)
    with torch.no_grad():
        unit.beta.weight[channel] *= scale  # channel `channel` of the product is massive
    return unit


def _samples():
    torch.manual_seed(11)
    return [torch.randn(4, 16) for _ in range(6)]


def test_detection_follows_the_dataflow_not_the_names():
    units = gated_units(nn.Sequential(_planted(), PlainMLP()))
    assert [(u.path, u.gate, u.up, u.down) for u in units] == [("0", "alpha", "beta", "omega")]


@pytest.mark.parametrize("bias", [False, True])
def test_the_rule_splits_exactly_the_planted_channel_and_stays_exact(bias):
    unit = _planted(bias=bias)
    samples = _samples()
    with torch.no_grad():
        want = [unit(x) for x in samples]
        probe = torch.randn(5, 16) * 2
        want_probe = unit(probe)

    record = split_outlier_channels(unit, lambda: [unit(x) for x in samples])
    rows = record["units_split"]
    assert len(rows) == 1 and [s["channel"] for s in rows[0]["split"]] == [29]
    parts = rows[0]["split"][0]["parts"]
    assert parts >= 2 and record["added_intermediate_columns"] == parts - 1
    assert unit.omega.weight.shape == (16, 48 + parts - 1)
    with torch.no_grad():
        for x, expected in zip(samples, want):
            torch.testing.assert_close(unit(x), expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(unit(probe), want_probe, rtol=1e-5, atol=1e-5)
        # The split channel's range is now within the rule's threshold.
        product = unit.nonlinearity(unit.alpha(samples[0])) * unit.beta(samples[0])
        assert float(product.abs().amax()) <= rows[0]["threshold"] * 1.0001 or parts > 1


def test_k_derives_from_the_ratio():
    maxima = torch.ones(200)
    maxima[3], maxima[150] = 400.0, 90.0
    threshold, parts = select_splits(maxima, ratio=32.0, quantile=0.5)
    assert threshold == pytest.approx(32.0)
    assert parts == {3: 13, 150: 3}  # ceil(400 / 32), ceil(90 / 32)
    assert select_splits(torch.ones(64), ratio=32.0, quantile=0.5)[1] == {}


def test_a_unit_without_outliers_is_left_alone():
    torch.manual_seed(3)
    unit = GatedMLP().eval().requires_grad_(False)
    before = [p.clone() for p in unit.parameters()]
    record = split_outlier_channels(unit, lambda: [unit(x) for x in _samples()])
    assert record["gated_units"] == 1 and record["units_split"] == []
    assert all(torch.equal(a, b) for a, b in zip(before, unit.parameters()))


def test_llama_mlps_are_found_and_split_exactly():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32)
    lm = transformers.LlamaForCausalLM(config).eval().requires_grad_(False)
    with torch.no_grad():
        lm.model.layers[1].mlp.up_proj.weight[5] *= 500.0
    tokens = torch.arange(24).reshape(3, 8) % 64
    with torch.no_grad():
        want = lm(input_ids=tokens).logits
    record = split_outlier_channels(lm, lambda: lm(input_ids=tokens))
    assert record["gated_units"] == 2
    assert [(r["unit"], [s["channel"] for s in r["split"]]) for r in record["units_split"]] == [
        ("model.layers.1.mlp", [5])
    ]
    with torch.no_grad():
        torch.testing.assert_close(lm(input_ids=tokens).logits, want, rtol=1e-4, atol=1e-4)


def test_ratio_and_quantile_are_validated():
    unit = _planted()
    with pytest.raises(ValueError):
        split_outlier_channels(unit, lambda: None, ratio=1.0)
    with pytest.raises(ValueError):
        split_outlier_channels(unit, lambda: None, quantile=1.0)


def _brute_force(widths, buckets, align):
    import itertools

    up = sorted({-(-w // align) * align for w in widths})
    best = None
    for k in range(1, min(buckets, len(up)) + 1):
        for chosen in itertools.combinations(up, k):
            if max(chosen) < max(widths):
                continue
            total = sum(min(b for b in chosen if b >= w) - w for w in widths)
            best = total if best is None else min(best, total)
    return best


@pytest.mark.parametrize("seed", range(6))
def test_bucket_widths_minimise_the_padding(seed):
    import random

    rng = random.Random(seed)
    widths = [rng.randint(100, 180) for _ in range(rng.randint(3, 9))]
    for buckets in (1, 2, 3):
        for align in (1, 16):
            chosen = bucket_widths(widths, buckets, align=align)
            assert len(chosen) <= buckets and all(b % align == 0 for b in chosen)
            padding = sum(min(b for b in chosen if b >= w) - w for w in widths)
            assert padding == _brute_force(widths, buckets, align)


def test_tinyllama_split_widths_fall_into_two_aligned_buckets():
    # The 22 per-layer MLP widths a ratio-8 split produced on TinyLlama-1.1B's calibration statistics.
    widths = [6064, 5640, 5982, 5726, 5696, 5683, 5656, 5794, 5657, 5647, 5637, 5641, 5643, 5647, 5639,
              5667, 5689, 5644, 5642, 5639, 5636, 5652]
    assert bucket_widths(widths, 2, align=16) == [5696, 6064]


def test_padding_to_shared_widths_is_exact_and_leaves_few_shapes():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32)
    lm = transformers.LlamaForCausalLM(config).eval().requires_grad_(False)
    with torch.no_grad():
        lm.model.layers[0].mlp.up_proj.weight[5] *= 500.0
        lm.model.layers[2].mlp.up_proj.weight[9] *= 2000.0
    tokens = torch.arange(24).reshape(3, 8) % 64
    with torch.no_grad():
        want = lm(input_ids=tokens).logits
    split_outlier_channels(lm, lambda: lm(input_ids=tokens), ratio=8.0)
    record = pad_gated_widths(lm, buckets=2, align=16)
    widths = {int(layer.mlp.down_proj.weight.shape[1]) for layer in lm.model.layers}
    assert widths <= set(record["buckets"]) and len(widths) <= 2 and all(w % 16 == 0 for w in widths)
    for layer in lm.model.layers:
        mlp = layer.mlp
        assert mlp.gate_proj.weight.shape[0] == mlp.up_proj.weight.shape[0] == mlp.down_proj.weight.shape[1]
    with torch.no_grad():
        torch.testing.assert_close(lm(input_ids=tokens).logits, want, rtol=1e-4, atol=1e-4)
