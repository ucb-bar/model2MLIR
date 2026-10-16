"""Fixed-cache causal sessions expose state as an ordinary, functional tensor ABI."""
from __future__ import annotations

import functools
import inspect

import numpy as np
import pytest
import torch

from m2m.capture.causal_session import (
    CausalMultiProgramCapture,
    FixedCacheDecodeStep,
    FixedCachePrefillStage,
    load_token_corpus,
    prefill_state,
)
from m2m.capture.external_runtime import external_runtime_session


def test_fixed_cache_prefill_is_functional_and_exports_fixed_capacity_state():
    lm = _tiny_llama()
    stage = FixedCachePrefillStage(lm, capacity=7).eval()
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    logits, cache, position = stage(tokens)
    assert logits.shape == (1, 64)
    assert cache.shape == (2, 2, 1, 2, 7, 8)
    assert torch.equal(position, torch.full_like(position, 4))
    exported = torch.export.export(stage, (tokens,), strict=False)
    assert len(exported.graph_signature.output_specs) == 3


def test_fixed_cache_prefill_owns_length_with_older_static_layer(monkeypatch):
    from transformers.cache_utils import StaticLayer

    initialize = StaticLayer.lazy_initialization

    @functools.wraps(initialize)  # keep the installed Transformers signature
    def without_length(self, *args):
        initialize(self, *args)
        if hasattr(self, "cumulative_length"):
            del self.cumulative_length

    monkeypatch.setattr(StaticLayer, "lazy_initialization", without_length)
    stage = FixedCachePrefillStage(_tiny_llama(), capacity=7).eval()

    _, first_cache, first_position = stage(torch.tensor([[1, 2, 3, 4]]))
    _, second_cache, second_position = stage(torch.tensor([[5, 6]]))
    assert torch.equal(first_position, torch.full_like(first_position, 4))
    assert torch.equal(second_position, torch.full_like(second_position, 2))
    assert torch.count_nonzero(second_cache[..., 2:, :]) == 0
    assert not torch.equal(first_cache, second_cache)


def test_functional_caches_accept_keys_only_static_layer_initialization(monkeypatch):
    # Transformers 4.x initializes StaticLayer from keys only (5.x passes keys and values).
    from transformers.cache_utils import StaticLayer

    initialize = StaticLayer.lazy_initialization
    takes_values = "value_states" in inspect.signature(initialize).parameters

    def keys_only(self, key_states):
        if takes_values:
            initialize(self, key_states, key_states)
        else:
            initialize(self, key_states)

    monkeypatch.setattr(StaticLayer, "lazy_initialization", keys_only)
    lm = _tiny_llama()
    prefill = FixedCachePrefillStage(lm, capacity=7).eval()
    decode = FixedCacheDecodeStep(lm, capacity=7).eval()
    _, cache, position = prefill(torch.tensor([[1, 2, 3, 4]]))
    _, _, next_position = decode(torch.tensor([[5]]), cache, position)
    assert torch.equal(position, torch.full_like(position, 4))
    assert torch.equal(next_position, torch.full_like(next_position, 5))


def test_fixed_cache_stages_run_eagerly_after_export():
    # torch.export leaves traced tensors bound to the cache objects; the worker's float reference
    # runs the same modules eagerly afterwards.
    lm = _tiny_llama()
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    prefill = FixedCachePrefillStage(lm, capacity=7).eval()
    with torch.no_grad():
        before = prefill(tokens)
    exported = torch.export.export(prefill, (tokens,))
    with torch.no_grad():
        after = prefill(tokens)
    for expected, actual, traced in zip(before, after, exported.module()(tokens)):
        assert torch.equal(expected, actual)
        torch.testing.assert_close(traced, expected)

    decode = FixedCacheDecodeStep(lm, capacity=7).eval()
    inputs = (torch.tensor([[5]]), before[1], before[2])
    with torch.no_grad():
        before = decode(*inputs)
    torch.export.export(decode, inputs)
    with torch.no_grad():
        after = decode(*inputs)
    for expected, actual in zip(before, after):
        assert torch.equal(expected, actual)


def test_actual_causal_bundle_materializes_real_goldens_before_cache_export(tmp_path):
    from m2m.capture.bundle import write_multi_program_bundle
    from m2m.capture.provenance import capture_receipt

    capture = CausalMultiProgramCapture(
        _tiny_llama(), torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long),
        checkpoint="unit/tiny", full_checkpoint=False, paper_ready=False,
        provenance={}, prefill_tokens=4, decode_tokens=2)
    session = external_runtime_session(capture, ())
    write_multi_program_bundle(session.bundle_programs(), dict(session.metadata), tmp_path,
                               capture_trace=True)
    for name in ("prefill", "decode"):
        stage = tmp_path / "stages" / name
        assert np.isfinite(np.load(stage / "golden.npy")).all()
        assert capture_receipt(stage)["materialized_abi"]["complete"]


def _tiny_llama():
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
        use_cache=True,
    )
    return transformers.LlamaForCausalLM(config).eval().requires_grad_(False)


def test_fixed_cache_decode_is_functional_and_exported_as_tensor_state():
    lm = _tiny_llama()
    prefill = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    cache, position = prefill_state(lm, prefill, capacity=7)
    cache_before, position_before = cache.clone(), position.clone()
    step = FixedCacheDecodeStep(lm, capacity=7).eval()

    logits, next_cache, next_position = step(torch.tensor([[5]]), cache, position)
    assert logits.shape == (1, 64)
    assert torch.equal(cache, cache_before)
    assert torch.equal(position, position_before)
    assert torch.equal(next_position, torch.full_like(next_position, 5))
    assert not torch.equal(next_cache, cache)

    exported = torch.export.export(
        step, (torch.tensor([[5]]), cache, position), strict=False)
    outputs = list(exported.graph_signature.output_specs)
    assert len(outputs) == 3
    assert all(spec.kind.name == "USER_OUTPUT" for spec in outputs)


def test_fixed_cache_decode_has_no_opaque_m2m_calls():
    import m2m

    lm = _tiny_llama()
    cache, position = prefill_state(lm, torch.tensor([[1, 2, 3, 4]]), capacity=7)
    result = m2m.convert(
        FixedCacheDecodeStep(lm, capacity=7), (torch.tensor([[5]]), cache, position),
        backend="fx_importer", level="linalg-on-tensors")
    assert result.ok
    assert "func.func private" not in result.mlir_text


def test_paper_ready_token_corpus_fails_closed(tmp_path, monkeypatch):
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(8, dtype=np.int64))
    monkeypatch.setenv("M2M_TEST_TOKEN_IDS", str(path))
    monkeypatch.setenv("M2M_TEST_PAPER_READY", "1")
    with pytest.raises(ValueError, match="TOKEN_SOURCE"):
        load_token_corpus(vocab_size=16, total_tokens=8, env_prefix="M2M_TEST")

    monkeypatch.setenv("M2M_TEST_TOKEN_SOURCE", "unit-test-corpus/v1")
    tokens, provenance, ready = load_token_corpus(
        vocab_size=16, total_tokens=8, env_prefix="M2M_TEST")
    assert ready is True
    assert tokens.shape == (1, 8)
    assert provenance["synthetic_tokens"] is False
    assert len(provenance["token_sha256"]) == 64


def test_deferred_causal_capture_exposes_actual_prefill_decode_runtime_protocol():
    lm = _tiny_llama()
    capture = CausalMultiProgramCapture(
        lm, torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.long),
        checkpoint="unit/tiny", full_checkpoint=False, paper_ready=False,
        provenance={"tokens": "unit"}, prefill_tokens=4, decode_tokens=3)

    exported = external_runtime_session(capture, ())

    assert exported.version == 2
    assert [program.name for program in exported.programs] == ["prefill", "decode"]
    assert isinstance(exported.programs[0].module, FixedCachePrefillStage)
    assert isinstance(exported.programs[1].module, FixedCacheDecodeStep)
    assert exported.observations == 3
    assert [(call.program, call.repeats, call.cadence) for call in exported.execution_schedule] == [
        ("prefill", 1, "once_before_observations"),
        ("decode", 3, "per_observation"),
    ]
    assert exported.streams[0].values.ravel().tolist() == [5, 6, 7]
    assert [(route.name, route.source.program, route.target.program) for route in exported.routes] == [
        ("kv_cache", "decode", "decode"),
        ("position", "decode", "decode"),
        ("kv_cache", "prefill", "decode"),
        ("position", "prefill", "decode"),
    ]
    assert exported.observation_output.program == "decode"
    assert exported.final_output.program == "decode"


def test_deferred_causal_bundle_writer_reuses_protocol_and_preserves_fp32_reference(
        tmp_path, monkeypatch):
    lm = _tiny_llama()
    capture = CausalMultiProgramCapture(
        lm, torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long),
        checkpoint="unit/tiny", full_checkpoint=False, paper_ready=False,
        provenance={}, prefill_tokens=4, decode_tokens=2)
    caught = {}

    def fake_write(programs, root, out, **kwargs):
        caught.update(programs=programs, root=root, out=out, kwargs=kwargs)
        return {"ok": True}

    monkeypatch.setattr("m2m.capture.bundle.write_multi_program_bundle", fake_write)
    assert capture.write_bundle(tmp_path) == {"ok": True}
    assert [program["name"] for program in caught["programs"]] == ["prefill", "decode"]
    reference = caught["programs"][1]["session"]["quality"]["reference_values"]
    assert reference.shape == (2, 1, 64)
    assert caught["root"]["quality_program"] == "decode"
    assert caught["kwargs"]["quantization_preapplied"] is False


def test_exported_decode_tensor_position_matches_complete_causal_prefix():
    lm = _tiny_llama()
    lm.set_attn_implementation('sdpa')
    prefix = torch.tensor([[1, 2, 3]], dtype=torch.long)
    cache, position = prefill_state(lm, prefix, capacity=8)
    step = FixedCacheDecodeStep(lm, capacity=8).eval()
    exported = torch.export.export(step, (torch.tensor([[4]]), cache, position), strict=False).module()
    for token in (4, 5, 6):
        input_ids = torch.tensor([[token]], dtype=torch.long)
        observed, cache, position = exported(input_ids, cache, position)
        prefix = torch.cat((prefix, input_ids), dim=1)
        expected = lm(input_ids=prefix, use_cache=False).logits[:, -1, :]
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)


def test_prefill_eager_after_export_preserves_full_causal_logits():
    lm = _tiny_llama()
    lm.set_attn_implementation('sdpa')
    tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
    stage = FixedCachePrefillStage(lm, capacity=8).eval()
    exported = torch.export.export(stage, (tokens,), strict=False).module()
    expected = lm(input_ids=tokens, use_cache=False).logits[:, -1, :]
    torch.testing.assert_close(exported(tokens)[0], expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(stage(tokens)[0], expected, rtol=1e-5, atol=1e-6)


def test_fixed_cache_export_does_not_retain_fake_tensor_constants():
    from torch._subclasses.fake_tensor import FakeTensor
    lm = _tiny_llama()
    tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
    stage = FixedCachePrefillStage(lm, capacity=8).eval()
    for _ in range(2):
        exported = torch.export.export(stage, (tokens,))
        assert not any(isinstance(value, FakeTensor) for value in exported.constants.values())
        assert all(not isinstance(layer.keys, FakeTensor) for layer in stage._static_cache.layers)
        expected = lm(input_ids=tokens, use_cache=False).logits[:, -1, :]
        torch.testing.assert_close(stage(tokens)[0], expected, rtol=1e-5, atol=1e-6)
