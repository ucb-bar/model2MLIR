"""Write a self-contained Merlin capture bundle (mlir + weights + golden + inputs + extra +
input_order) from ONE seeded model instance.

Factored out of ``workloads/capture_consistent.py`` so any capture path (per-model loader, or the
GGUF frontend) produces a byte-compatible bundle for the Merlin RVV runtime. It recovers the two
argument classes m2m elides to the runtime — registered buffers and lifted get_attr constants — by
type, plus quantized subclass inner tensors, exactly as the consistent-capture worker does.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class _LogitsOnly(nn.Module):
    """Wrap a HF causal LM so export sees a clean ``input_ids -> logits`` forward."""

    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        self.lm = lm

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm(input_ids=input_ids, use_cache=False).logits


def capture_hf_bundle(model_dir, out, *, quant_scheme: str | None = None, seq_len: int = 8,
                      dtype: "torch.dtype" = torch.float32) -> dict:
    """Capture a HuggingFace causal-LM (local dir or hub id) into a Merlin bundle via the torch path.

    ``quant_scheme`` is a torchAO scheme name (e.g. ``"int8_weight_only"``) or ``None`` for fp. This
    is the torch/torchAO ingestion arm (the GGUF arm is :func:`m2m.frontends.gguf.capture_gguf_bundle`).
    """
    from transformers import AutoModelForCausalLM

    lm = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype, use_cache=False).eval()
    model = _LogitsOnly(lm).eval()
    vocab = int(lm.config.vocab_size)
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab, (1, seq_len))
    quant = None
    if quant_scheme:
        from m2m.capture.torchao_pipeline import QuantizationConfig
        quant = QuantizationConfig(scheme=quant_scheme)
    summary = write_bundle(model, (input_ids,), out, quant=quant)
    summary["hf"] = str(model_dir)
    summary["quant_scheme"] = quant_scheme
    return summary


def _flatten_subclass(obj: Any, prefix: str, out: dict) -> None:
    """Recursively flatten a tensor-subclass parameter to its leaf tensors (qinner::<attr-path>)."""
    flat = getattr(obj, "__tensor_flatten__", None)
    if callable(flat):
        try:
            names, _ = flat()
        except Exception:  # noqa: BLE001
            names = []
        for nm in names:
            child = getattr(obj, nm, None)
            if child is not None:
                _flatten_subclass(child, f"{prefix}.{nm}", out)
    elif hasattr(obj, "detach"):                       # a leaf torch.Tensor
        t = obj.detach().cpu()
        _dt = str(t.dtype)
        arr = t.float().numpy() if ("float8" in _dt or "bfloat16" in _dt) else t.numpy()
        out[f"qinner::{prefix}"] = arr


def _lifted_constants(mdl, inputs, extra: dict) -> None:
    """Populate c_lifted_tensor_<i> from m2m's own export (graph order matches the importer)."""
    try:
        from m2m.capture.torch_export import capture_frontend_artifact
        from m2m.ir.torchmlir_decomps import torch_mlir_gap_decompositions
        from torch.export.graph_signature import InputKind

        artifact = capture_frontend_artifact(
            mdl, inputs, export_decomposition_table=torch_mlir_gap_decompositions())
        ep = artifact.exported_program or artifact.original_exported_program
        consts = dict(getattr(ep, "constants", {}) or {})
        sd = dict(getattr(ep, "state_dict", {}) or {})
        li = 0
        for spec in ep.graph_signature.input_specs:
            if spec.kind != InputKind.CONSTANT_TENSOR:
                continue
            val = consts.get(str(spec.target), sd.get(str(spec.target)))
            if val is not None and hasattr(val, "detach"):
                extra[f"c_lifted_tensor_{li}"] = val.detach().cpu().numpy()
            li += 1
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"[warn] lifted-constant export failed: {exc}\n{traceback.format_exc()[-800:]}")


def _extract_prov_fqns(mlir_text: str) -> list[str]:
    """The distinct ``prov.fqn`` module paths the export tagged, in first-seen order.

    Structured scan of the emitted MLIR (locate the ``prov.fqn = "..."`` attribute literal and read to
    the closing quote) — deterministic string handling, never a semantic regex.
    """
    marker = 'prov.fqn = "'
    out: list[str] = []
    seen: set[str] = set()
    i = mlir_text.find(marker)
    while i != -1:
        j = mlir_text.find('"', i + len(marker))
        if j == -1:
            break
        fqn = mlir_text[i + len(marker):j]
        if fqn and fqn not in seen:
            seen.add(fqn)
            out.append(fqn)
        i = mlir_text.find(marker, j + 1)
    return out


# npz key convention for region_goldens.npz, SHARED with the Merlin reader
# (merlin.baselines.bundle.CaptureBundle.region_goldens): one entry per (region fqn, slot) where slot
# is ``in<k>`` (the region's k-th boundary input = the upstream region's output) or ``out`` (the
# region's output golden). Flat, self-describing keys so np.load needs no sidecar.
_REGION_KEY = "{fqn}::{slot}"


def _capture_region_goldens(mdl, inputs, fqns) -> dict[str, "np.ndarray"]:
    """Register forward hooks on the nn.Modules named by ``fqns`` and capture, during ONE forward,
    each region's boundary tensors: its positional INPUTS and its OUTPUT. Best-effort — a module that
    is not registered under that exact name, or whose IO is not a plain tensor, is simply skipped."""
    named = dict(mdl.named_modules())
    caught: dict[str, np.ndarray] = {}
    handles = []

    def _mk(fqn: str):
        def hook(_m, args, output):
            for k, a in enumerate(args):
                if isinstance(a, torch.Tensor):
                    caught[_REGION_KEY.format(fqn=fqn, slot=f"in{k}")] = \
                        a.detach().float().cpu().numpy()
            out_t = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(out_t, torch.Tensor):
                caught[_REGION_KEY.format(fqn=fqn, slot="out")] = \
                    out_t.detach().float().cpu().numpy()
        return hook

    for fqn in fqns:
        mod = named.get(fqn)
        if mod is not None:
            handles.append(mod.register_forward_hook(_mk(fqn)))
    try:
        with torch.no_grad():
            g = mdl(*inputs)
    finally:
        for h in handles:
            h.remove()
    return caught, g


def _tensor_outputs(value: Any) -> tuple[torch.Tensor, ...]:
    """Flatten the capture ABI's top-level tensor results, refusing opaque Python state."""
    values = value if isinstance(value, (tuple, list)) else (value,)
    if not values or any(not isinstance(item, torch.Tensor) for item in values):
        raise ValueError("session forwards must return one tensor or a flat tuple/list of tensors")
    return tuple(values)


def capture_session_trajectory(mdl, inputs, session: dict) -> np.ndarray:
    """Execute a loader-authored session and return its selected output at every semantic step.

    This intentionally runs before quantization when called by :func:`write_bundle`. The resulting
    trajectory is the model-quality reference; a second execution after conversion/quantization is
    recorded separately as the compiler-correctness reference.
    """
    inputs = tuple(inputs)
    states = list(session.get("states", ()) or ())
    streams = list(session.get("streams", ()) or ())
    counts: set[int] = set()
    stream_values: list[tuple[int, torch.Tensor]] = []
    for index, stream in enumerate(streams):
        input_index = int(stream["input_index"])
        if input_index < 0 or input_index >= len(inputs):
            raise ValueError(f"session stream {index} references an unknown input index")
        values = stream.get("values")
        values = values if isinstance(values, torch.Tensor) else torch.as_tensor(values)
        if values.ndim < 1 or tuple(values.shape[1:]) != tuple(inputs[input_index].shape):
            raise ValueError(f"session stream {index} shape differs from its model input")
        counts.add(int(values.shape[0]))
        stream_values.append((input_index, values))
    if streams:
        if len(counts) != 1:
            raise ValueError("all semantic streams must have one common step count")
        steps = next(iter(counts))
    else:
        steps = int(session.get("steps", 0))
        if steps < 1 or not states:
            raise ValueError("stream-free trajectory needs positive steps and carried state")
    quality = dict(session.get("quality", {}) or {})
    output_index = int(quality.get("output_index", 0))
    trajectory_inputs = list(inputs)
    outputs_seen: list[np.ndarray] = []
    with torch.no_grad():
        for step in range(steps):
            for input_index, values in stream_values:
                base = inputs[input_index]
                trajectory_inputs[input_index] = values[step].to(
                    dtype=base.dtype, device=base.device)
            outputs = _tensor_outputs(mdl(*trajectory_inputs))
            if output_index < 0 or output_index >= len(outputs):
                raise ValueError(
                    f"quality output index {output_index} is outside {len(outputs)} outputs")
            outputs_seen.append(outputs[output_index].detach().float().cpu().numpy())
            for state in states:
                input_index, state_output = int(state["input_index"]), int(state["output_index"])
                if state_output < 0 or state_output >= len(outputs):
                    raise ValueError(
                        f"state output index {state_output} is outside {len(outputs)} outputs")
                updated = outputs[state_output].detach().clone()
                if tuple(updated.shape) != tuple(inputs[input_index].shape):
                    raise ValueError(
                        f"state {state.get('name', input_index)!r} changes shape from "
                        f"{tuple(inputs[input_index].shape)} to {tuple(updated.shape)}")
                trajectory_inputs[input_index] = updated
    return np.ascontiguousarray(outputs_seen, dtype=np.float32)


def _runtime_args_by_input_index(manifest: dict, input_order: dict[str, int]) -> dict[int, int]:
    """Map loader tuple indices to the exported forward's numeric ABI argument indices.

    The Merlin runtime consumes the complete exported signature (parameters first), so a loader's
    fourth input is not generally ABI argument 3.  Resolve through the safetensors manifest and the
    same ``input_order`` table the runtime uses rather than guessing around the weight count.
    """
    by_input: dict[int, int] = {}
    fallback = 0
    for raw_index in sorted(manifest, key=lambda value: int(value)):
        meta = manifest[raw_index]
        name = str(meta.get("name", "") or "")
        if meta.get("kind") in ("param", "buffer") or "lifted_tensor" in name:
            continue
        input_index = input_order.get(name)
        if input_index is None:
            while fallback in by_input:
                fallback += 1
            input_index = fallback
            fallback += 1
        by_input.setdefault(int(input_index), int(raw_index))
    return by_input


def write_session_artifacts(mdl, inputs, out: str | Path, *, manifest: dict,
                            input_order: dict[str, int], session: dict,
                            quality_reference: np.ndarray | None = None) -> dict:
    """Write a semantic trajectory contract, distinct observations, and eager trajectory goldens.

    ``session`` uses loader tuple indices because those are stable at authoring time:

    - ``states``: ``{name, input_index, output_index}``
    - ``streams``: ``{name, input_index, key, values}``, where values have a leading step dimension
    - ``quality``: ``{output_index, key}``

    This function resolves tuple indices to the exported numeric ABI, executes the quantized model over
    the whole trajectory while carrying state, and emits the files consumed by Merlin's generic runtime.
    It never infers state from names or tuple positions.
    """
    import yaml

    out = Path(out)
    inputs = tuple(inputs)
    states = list(session.get("states", ()) or ())
    streams = list(session.get("streams", ()) or ())
    abi_args = _runtime_args_by_input_index(manifest, input_order)
    state_inputs = {int(item["input_index"]) for item in states}
    stream_inputs = {int(item["input_index"]) for item in streams}
    if state_inputs & stream_inputs:
        raise ValueError("a session input cannot be both carried state and an observation stream")

    arrays: dict[str, np.ndarray] = {}
    contract_streams = []
    step_counts = set()
    for index, stream in enumerate(streams):
        input_index = int(stream["input_index"])
        if input_index < 0 or input_index >= len(inputs) or input_index not in abi_args:
            raise ValueError(f"session stream {index} references an unknown input index {input_index}")
        key = str(stream.get("key") or stream.get("name") or f"stream{index}")
        value = stream.get("values")
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.ascontiguousarray(value)
        if array.ndim < 1 or list(array.shape[1:]) != list(inputs[input_index].shape):
            raise ValueError(
                f"session stream {key!r} must have shape [steps, {list(inputs[input_index].shape)}], "
                f"got {list(array.shape)}")
        arrays[key] = array
        step_counts.add(int(array.shape[0]))
        contract_streams.append({"name": str(stream.get("name", key)),
                                 "input_arg": abi_args[input_index], "key": key})
    if streams:
        if len(step_counts) != 1 or next(iter(step_counts)) < 1:
            raise ValueError("all semantic input streams must have one common positive step count")
        steps = next(iter(step_counts))
    else:
        steps = int(session.get("steps", 0))
        if steps < 1 or not states:
            raise ValueError(
                "a stream-free semantic session needs positive steps and explicit carried state")

    contract_states = []
    for index, state in enumerate(states):
        input_index, output_index = int(state["input_index"]), int(state["output_index"])
        if input_index < 0 or input_index >= len(inputs) or input_index not in abi_args:
            raise ValueError(f"session state {index} references an unknown input index {input_index}")
        contract_states.append({"name": str(state.get("name", f"state{index}")),
                                "input_arg": abi_args[input_index], "output_index": output_index})

    quality = dict(session.get("quality", {}) or {})
    quality_output = int(quality.get("output_index", 0))
    quality_key = str(quality.get("key", "output0"))
    trajectory_inputs = list(inputs)
    goldens = []
    with torch.no_grad():
        for step in range(steps):
            for stream, contract_stream in zip(streams, contract_streams, strict=True):
                input_index = int(stream["input_index"])
                base = inputs[input_index]
                trajectory_inputs[input_index] = torch.as_tensor(
                    arrays[contract_stream["key"]][step], dtype=base.dtype, device=base.device)
            outputs = _tensor_outputs(mdl(*trajectory_inputs))
            if quality_output < 0 or quality_output >= len(outputs):
                raise ValueError(f"quality output index {quality_output} is outside {len(outputs)} outputs")
            goldens.append(outputs[quality_output].detach().float().cpu().numpy())
            for state in states:
                input_index, output_index = int(state["input_index"]), int(state["output_index"])
                if output_index < 0 or output_index >= len(outputs):
                    raise ValueError(f"state output index {output_index} is outside {len(outputs)} outputs")
                updated = outputs[output_index].detach().clone()
                if tuple(updated.shape) != tuple(inputs[input_index].shape):
                    raise ValueError(
                        f"state {state.get('name', input_index)!r} changes shape from "
                        f"{tuple(inputs[input_index].shape)} to {tuple(updated.shape)}")
                trajectory_inputs[input_index] = updated

    correctness_values = np.ascontiguousarray(goldens, dtype=np.float32)
    if quality_reference is None:
        quality_reference = correctness_values
    quality_reference = np.ascontiguousarray(quality_reference, dtype=np.float32)
    if quality_reference.shape != correctness_values.shape:
        raise ValueError(
            f"FP32 quality trajectory shape {quality_reference.shape} differs from compiled-precision "
            f"correctness trajectory {correctness_values.shape}")
    np.savez(out / "session_inputs.npz", **arrays)
    np.savez(out / "session_goldens.npz", **{quality_key: correctness_values})
    np.savez(out / "session_quality_fp32.npz", **{quality_key: quality_reference})
    import hashlib
    correctness_sha256 = hashlib.sha256(correctness_values.tobytes()).hexdigest()
    quality_sha256 = hashlib.sha256(quality_reference.tobytes()).hexdigest()
    contract = {
        "version": 1,
        "kind": str(session["kind"]),
        "paper_ready": bool(session.get("paper_ready", False)),
        "stages": [str(value) for value in session.get("stages", ())],
        "steps": steps,
        "inputs": "session_inputs.npz",
        "states": contract_states,
        "streams": contract_streams,
        "correctness": {"scope": "trajectory", "golden": "session_goldens.npz",
                        "key": quality_key, "output_index": quality_output,
                        "reference": "eager_same_precision",
                        "reference_sha256": correctness_sha256},
        "quality": {"scope": "trajectory", "golden": "session_quality_fp32.npz",
                    "key": quality_key, "output_index": quality_output,
                    "reference": str(quality.get("reference", "eager_fp32")),
                    "reference_sha256": quality_sha256},
    }
    if session.get("provenance"):
        contract["provenance"] = session["provenance"]
    if session.get("stage_schedule"):
        contract["stage_schedule"] = session["stage_schedule"]
    if session.get("parameters"):
        contract["parameters"] = session["parameters"]
    (out / "session_contract.yaml").write_text(yaml.safe_dump(contract, sort_keys=False))
    return {"session_kind": contract["kind"], "session_steps": steps,
            "session_states": len(contract_states), "paper_ready": contract["paper_ready"]}


#: Torch dtypes numpy has no equivalent for, so a bundle must store them widened. Named rather than
#: caught-by-exception: a TypeError from `.numpy()` also covers a genuinely broken tensor, and silently
#: widening that would hide it.
def _numpy_unrepresentable() -> frozenset:
    import torch
    out = [torch.bfloat16]
    for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
        dt = getattr(torch, name, None)
        if dt is not None:
            out.append(dt)
    return frozenset(out)


def _numpy_safe(x):
    """A tensor as numpy, converting ONLY the dtypes numpy cannot represent.

    `numpy` has no bfloat16 (nor any float8), so `.numpy()` on such a tensor raises
    `TypeError: Got unsupported ScalarType BFloat16`. Measured on smolvla, whose vision tower takes
    bf16 inputs: the export succeeded, the goldens were written, and the bundle died on `inputs.npz` --
    the one place here that lacked the conversion `golden.npy` and the buffer writes already do.

    ⚠️ NOT a blanket `.float()`. An LLM's `input_ids` are int64, and casting those to float32 corrupts
    the token ids silently -- the input would still load, still have the right shape, and index a
    different embedding row. So the conversion is keyed on the dtype actually being unrepresentable,
    and every other dtype passes through untouched.

    bf16 -> f32 is lossless (bf16 is a truncated f32), so no precision is traded for the storage.
    """
    import torch

    t = x.detach().cpu()
    if t.dtype in _numpy_unrepresentable():
        t = t.float()
    return t.numpy()


def write_bundle(mdl, inputs, out: str | Path, *, quant=None, capture_regions: bool = True,
                 session: dict | None = None, quantization_preapplied: bool = False) -> dict:
    """Convert ``mdl`` and write the full bundle to ``out``. Returns a summary dict.

    ``quant`` is an m2m ``QuantizationConfig`` (or ``None`` for an unquantized/fp bundle). The golden
    is the forward of the SAME instance on the SAME inputs, so the bundle is self-consistent.

    ``capture_regions`` (default on) additionally records per-region boundary tensors to
    ``region_goldens.npz`` (keyed by the ``prov.fqn`` modules the export tagged) — the shared substrate
    for per-region equivalence + standalone-section profiling. Captured in the SAME golden forward.
    """
    import m2m

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mdl.eval()
    inputs = tuple(inputs)

    quality_reference = None
    if session is not None:
        quality = dict(session.get("quality", {}) or {})
        supplied = quality.get("reference_values")
        if supplied is not None:
            if isinstance(supplied, torch.Tensor):
                supplied = supplied.detach().float().cpu().numpy()
            quality_reference = np.ascontiguousarray(supplied, dtype=np.float32)
        elif not quantization_preapplied:
            quality_reference = capture_session_trajectory(mdl, inputs, session)
        elif session.get("paper_ready") is True:
            raise ValueError(
                "paper-ready prequantized capture must supply an independently generated "
                "eager_fp32 quality.reference_values trajectory")

    weights_path = str(out / "weights.safetensors")
    r = m2m.convert(mdl, inputs, backend="fx_importer", quantization=quant,
                    level="linalg-on-tensors", weights_path=weights_path,
                    quantization_preapplied=quantization_preapplied)
    assert r.ok, "m2m.convert failed"
    (out / "model.mlir").write_text(r.mlir_text)

    n_regions = 0
    if capture_regions:
        fqns = _extract_prov_fqns(r.mlir_text)
        region_goldens, g = _capture_region_goldens(mdl, inputs, fqns)
        if region_goldens:
            np.savez(out / "region_goldens.npz", **region_goldens)
        n_regions = len({k.split("::", 1)[0] for k in region_goldens})
    else:
        with torch.no_grad():
            g = mdl(*inputs)
    golden = g[0] if isinstance(g, (tuple, list)) else g
    np.save(out / "golden.npy", golden.detach().float().cpu().numpy())

    np.savez(out / "inputs.npz",
             **{f"in{i}": _numpy_safe(x) for i, x in enumerate(inputs)})

    extra: dict = {}
    for name, t in mdl.named_buffers():
        extra["buf::" + name] = t.detach().float().cpu().numpy()
    for pname, p in mdl.named_parameters():
        if type(p).__name__ not in ("Parameter", "Tensor") or hasattr(p, "__tensor_flatten__"):
            _flatten_subclass(p, pname, extra)
    _lifted_constants(mdl, inputs, extra)
    np.savez(out / "extra.npz", **extra)

    man = json.loads(Path(weights_path + ".manifest.json").read_text())
    order: dict = {}
    k = 0
    for i in range(len(man)):
        meta = man[str(i)]
        nm = meta.get("name", "") or ""
        if meta["kind"] in ("param", "buffer") or "lifted_tensor" in nm:
            continue
        order[nm] = k
        k += 1
    (out / "input_order.json").write_text(json.dumps(order, indent=2))

    session_summary = {}
    if session is not None:
        session_summary = write_session_artifacts(
            mdl, inputs, out, manifest=man, input_order=order, session=session,
            quality_reference=quality_reference)

    return {
        "out": str(out), "n_inputs": len(inputs),
        "n_buffers": sum(1 for kk in extra if kk.startswith("buf::")),
        "n_lifted": sum(1 for kk in extra if kk.startswith("c_lifted")),
        "n_qinner": sum(1 for kk in extra if kk.startswith("qinner::")),
        "golden_shape": list(golden.shape), "linalg": r.mlir_text.count("linalg."),
        "input_order": order, "n_regions": n_regions, **session_summary,
    }


def write_multi_program_bundle(programs: list[dict], root_session: dict, out: str | Path, *,
                               quant=None, quantization_preapplied: bool = False) -> dict:
    """Write several fixed-shape compiled programs plus an ABI-resolved root session contract.

    Loader-authored bindings use stable tuple ``input_index`` values.  Each stage conversion can
    prepend hundreds of weight arguments, so this function resolves those indices through that
    stage's emitted manifest and records the numeric ``input_arg`` consumed by Merlin.  Cross-stage
    output ordinals are already stable exported ABI values.
    """
    import yaml

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = [str(program["name"]) for program in programs]
    if not names or len(set(names)) != len(names):
        raise ValueError("multi-program capture needs unique program names")
    if names != [str(value) for value in root_session.get("stages", ())]:
        raise ValueError("program order must exactly equal the root session stages")
    stage_records: dict[str, dict] = {}
    summaries = []
    for program in programs:
        name = str(program["name"])
        stage_out = out / "stages" / name
        summary = write_bundle(
            program["model"], tuple(program["inputs"]), stage_out, quant=quant,
            capture_regions=bool(program.get("capture_regions", False)),
            session=program.get("session"), quantization_preapplied=quantization_preapplied)
        manifest = json.loads((stage_out / "weights.safetensors.manifest.json").read_text())
        input_order = json.loads((stage_out / "input_order.json").read_text())
        stage_records[name] = {
            "summary": summary,
            "abi_args": _runtime_args_by_input_index(manifest, input_order),
            "steps": int(program["steps"]),
        }
        summaries.append({"name": name, **summary})

    bindings = []
    for index, binding in enumerate(root_session.get("bindings", ()) or ()):
        source = dict(binding["from"])
        target = dict(binding["to"])
        target_program = str(target["program"])
        input_index = int(target.pop("input_index"))
        abi_args = stage_records[target_program]["abi_args"]
        if input_index not in abi_args:
            raise ValueError(
                f"root binding {index} target input index {input_index} is not in "
                f"program {target_program}'s exported ABI")
        target["input_arg"] = abi_args[input_index]
        bindings.append({"name": str(binding["name"]), "from": source, "to": target})

    schedule = list(root_session.get("stage_schedule", ()) or ())
    expected_schedule = [(name, stage_records[name]["steps"]) for name in names]
    actual_schedule = [(str(row.get("name")), int(row.get("steps", 0))) for row in schedule]
    if actual_schedule != expected_schedule:
        raise ValueError("stage schedule names/steps differ from the captured programs")
    contract = {
        "version": 2,
        "kind": str(root_session["kind"]),
        "paper_ready": bool(root_session.get("paper_ready", False)),
        "stages": names,
        "stage_schedule": schedule,
        "parameters": dict(root_session.get("parameters", {}) or {}),
        "programs": [{"name": name, "bundle": f"stages/{name}",
                      "steps": stage_records[name]["steps"]} for name in names],
        "bindings": bindings,
        "states": [{"name": str(value)} for value in root_session.get("states", ()) or ()],
        "streams": [],
        "quality": {"scope": "trajectory", "program": str(root_session["quality_program"])},
        "provenance": dict(root_session.get("provenance", {}) or {}),
    }
    (out / "session_contract.yaml").write_text(
        yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    return {"out": str(out), "session_kind": contract["kind"], "paper_ready": contract["paper_ready"],
            "n_programs": len(programs), "programs": summaries,
            "n_bindings": len(bindings)}
