"""Causal-session calibration streams and FP-equivalent smoothing (static int8 recipes)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from m2m.capture.causal_session import (
    FixedCacheDecodeStep,
    FixedCachePrefillStage,
    causal_calibration_inputs,
    load_calibration_tokens,
)


def _tiny_llama(seed: int = 0):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(seed)
    config = transformers.LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
        use_cache=True,
    )
    return transformers.LlamaForCausalLM(config).eval().requires_grad_(False)


def _segments(count: int = 3, length: int = 7) -> torch.Tensor:
    return torch.arange(count * length, dtype=torch.long).reshape(count, length) % 64


def test_prefill_stream_is_each_segments_prefix():
    lm = _tiny_llama()
    samples = list(causal_calibration_inputs(
        FixedCachePrefillStage(lm, 7).eval(), _segments(), prefill_tokens=4))
    assert [s[0].tolist() for s in samples] == [row[:4].reshape(1, -1).tolist() for row in _segments()]


@pytest.mark.parametrize("stride", [1, 2])
def test_decode_stream_is_the_teacher_forced_fp_recurrence(stride):
    lm = _tiny_llama()
    decode = FixedCacheDecodeStep(lm, 7).eval()
    segments = _segments()
    samples = list(causal_calibration_inputs(decode, segments, prefill_tokens=4, decode_stride=stride))
    steps = [s for s in range(3) if s % stride == 0]
    assert len(samples) == len(segments) * len(steps)
    prefill = FixedCachePrefillStage(lm, 7).eval()
    for index, row in enumerate(segments):
        with torch.no_grad():
            _, cache, position = prefill(row[:4].reshape(1, -1))
            expected = []
            for step in range(3):
                token = row[4 + step].reshape(1, 1)
                expected.append((token, cache, position))
                _, cache, position = decode(token, cache, position)
        for offset, step in enumerate(steps):
            token, cache, position = samples[index * len(steps) + offset]
            want = expected[step]
            assert torch.equal(token, want[0])
            assert torch.equal(cache, want[1])
            assert torch.equal(position, want[2])
            assert int(position[0]) == 4 + step


def test_decode_stream_refuses_segments_shorter_than_the_cache():
    decode = FixedCacheDecodeStep(_tiny_llama(), 7).eval()
    with pytest.raises(ValueError, match="do not cover"):
        list(causal_calibration_inputs(decode, _segments(length=6), prefill_tokens=4))


def test_calibration_tokens_are_validated_and_hashed(tmp_path):
    path = tmp_path / "calibration.npz"
    np.savez(path, input_ids=np.arange(20, dtype=np.int64).reshape(2, 10))
    tokens, provenance = load_calibration_tokens(str(path), vocab_size=64, min_tokens=7)
    assert tokens.shape == (2, 7) and tokens.dtype == torch.long
    assert provenance["calibration_segments"] == 2 and len(provenance["calibration_token_sha256"]) == 64
    with pytest.raises(ValueError, match="need 12"):
        load_calibration_tokens(str(path), vocab_size=64, min_tokens=12)
    with pytest.raises(ValueError, match="outside"):
        load_calibration_tokens(str(path), vocab_size=8, min_tokens=7)


@pytest.mark.parametrize("groups", [("qkv",), ("gate_up",), ("down",), ("o",), ("qkv", "gate_up", "down", "o")])
def test_smoothing_is_floating_point_equivalent(groups):
    from m2m.capture.smoothing import smooth_decoder

    reference, smoothed = _tiny_llama(1), _tiny_llama(1)
    # Give one channel of every projection input a large range, as in real decoders.
    with torch.no_grad():
        for layer in smoothed.model.layers, reference.model.layers:
            for block in layer:
                block.input_layernorm.weight[3] = 40.0
                block.post_attention_layernorm.weight[5] = 25.0
    tokens = _segments(count=2, length=8)
    record = smooth_decoder(smoothed, tokens, alpha=0.5, groups=groups)
    assert record["groups"] == list(groups) and record["statistics_segments"] == 2
    probe = torch.tensor([[3, 9, 27, 17, 51, 1]])
    with torch.no_grad():
        want = reference(input_ids=probe).logits
        got = smoothed(input_ids=probe).logits
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
    changed = any(not torch.equal(a, b) for a, b in zip(smoothed.parameters(), reference.parameters()))
    assert changed


def test_smoothing_flattens_the_input_range_it_targets():
    from m2m.capture.smoothing import input_absmax, smooth_decoder

    lm = _tiny_llama(2)
    with torch.no_grad():
        for block in lm.model.layers:
            block.input_layernorm.weight[3] = 40.0
    tokens = _segments(count=2, length=8)
    before = input_absmax(lm, tokens)["model.layers.0.self_attn.q_proj"]
    smooth_decoder(lm, tokens, alpha=0.5, groups=("qkv",))
    after = input_absmax(lm, tokens)["model.layers.0.self_attn.q_proj"]
    assert float(after.max() / after.median()) < float(before.max() / before.median())


def test_smoothing_refuses_an_unknown_group_or_alpha():
    from m2m.capture.smoothing import smooth_decoder

    lm = _tiny_llama()
    with pytest.raises(ValueError, match="unknown smoothing groups"):
        smooth_decoder(lm, _segments(), groups=("lm_head",))
    with pytest.raises(ValueError, match="alpha"):
        smooth_decoder(lm, _segments(), alpha=1.5)
