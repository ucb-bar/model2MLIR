"""Fixed-cache causal sessions expose state as an ordinary, functional tensor ABI."""
from __future__ import annotations

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
