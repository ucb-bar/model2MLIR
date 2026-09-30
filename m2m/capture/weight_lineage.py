"""Source-bound weight lineage for graph-level PT2E quantization.

The numerical format is read from the converted graph, never inferred from a
target name.  In particular, a per-channel PT2E capture is not a claim that a
particular accelerator implements per-channel requantization.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "m2m.quant_weight_lineage.v1"
FORMULA = "torch.quantized_decomposed.quantize.v1"


def _tensor(value: Any, *, label: str) -> Any:
    import torch

    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{label} is not a tensor")
    if value.device.type != "cpu" or value.is_sparse:
        raise ValueError(f"{label} must be a dense CPU tensor")
    return value.detach().contiguous().clone()


def _bytes(value: Any) -> bytes:
    return value.contiguous().view(__import__("torch").uint8).numpy().tobytes()


def _sha(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attr(module: Any, name: str) -> Any:
    value = module
    for part in name.split("."):
        value = getattr(value, part)
    return value


def _contractions(module: Any) -> dict[str, Any]:
    import torch

    targets = {torch.ops.aten.conv2d.default, torch.ops.aten.linear.default}
    return {
        node.name: node
        for node in module.graph.nodes
        if node.op == "call_function" and node.target in targets
    }


def _prepared_weight_attr(node: Any) -> str | None:
    """Select observed weights; leave unannotated host contractions untouched."""
    weight = node.args[1]
    if weight.op != "call_module" or not str(weight.target).startswith("activation_post_process_"):
        annotation = node.meta.get("quantization_annotation")
        if (annotation is not None and getattr(annotation, "_annotated", False)
                and weight in getattr(annotation, "input_qspec_map", {})):
            raise ValueError(f"{node.name}: annotated weight has no PT2E observer output")
        return None
    if len(weight.args) != 1 or weight.args[0].op != "get_attr":
        raise ValueError(f"{node.name}: observer does not read one named weight")
    return str(weight.args[0].target)


def snapshot_source_weights(model: Any, exported: Any) -> dict[str, Any]:
    """Clone named source weight candidates before the recipe selects them.

    This runs before PT2E preparation/folding, where the quantizer has not yet
    annotated its selected nodes.  Dynamic or non-state host operands are not
    candidates; if PT2E later observes one, effective binding refuses it.
    """
    source = model.state_dict()
    selected: dict[str, Any] = {}
    for node in _contractions(exported).values():
        if len(node.args) < 2 or getattr(node.args[1], "op", None) != "get_attr":
            continue
        key = str(node.args[1].target)
        if key not in source:
            raise ValueError(f"{node.name}: source state has no weight {key!r}")
        selected[key] = _tensor(source[key], label=f"original {key}")
    return selected


def snapshot_effective_weights(original_state: dict[str, Any], prepared: Any) -> dict[str, dict[str, Any]]:
    """Retain original and post-prepare float bytes before conversion drops them.

    ``prepare_pt2e`` folds inference Conv+BatchNorm.  The *effective* weight
    therefore comes from the prepared graph, not the original module's state.
    Requiring the same state key on both sides refuses an untraceable rewrite.
    """
    import torch

    result: dict[str, dict[str, Any]] = {}
    for node_name, node in _contractions(prepared).items():
        state_key = _prepared_weight_attr(node)
        if state_key is None:
            continue
        if state_key not in original_state:
            raise ValueError(f"{node_name}: original state has no weight {state_key!r}")
        original = _tensor(original_state[state_key], label=f"original {state_key}")
        effective = _tensor(_attr(prepared, state_key), label=f"effective {state_key}")
        if original.dtype != torch.float32 or effective.dtype != torch.float32:
            raise ValueError(f"{node_name}: expected float32 pre-quantization weights")
        if original.shape != effective.shape:
            raise ValueError(f"{node_name}: effective weight shape changed without a relation")
        result[node_name] = {
            "operator": str(node.target),
            "state_key": state_key,
            "original": original,
            "effective": effective,
        }
    if not result:
        raise ValueError("PT2E capture has no observed Conv2d/Linear weight")
    return result


def bind_frozen_weights(prepared_weights: dict[str, dict[str, Any]], converted: Any) -> list[dict[str, Any]]:
    """Bind each converted operand to its frozen buffer and exact qparams.

    A changed FX shape is an error, not a positional guess.  The frozen bytes
    must equal the versioned Torch formula on the effective bytes.
    """
    import torch

    nodes = _contractions(converted)
    if not set(prepared_weights) <= set(nodes):
        raise ValueError("PT2E conversion lost selected contraction node identities")
    qdq_targets = {
        torch.ops.quantized_decomposed.dequantize_per_channel.default,
        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    }
    for node_name, node in nodes.items():
        if node_name in prepared_weights:
            continue
        weight = node.args[1]
        if getattr(weight, "op", None) == "call_function" and weight.target in qdq_targets:
            raise ValueError(f"{node_name}: converted weight dequantization has no source lineage")
    rows: list[dict[str, Any]] = []
    for node_name in sorted(prepared_weights):
        node = nodes[node_name]
        prior = prepared_weights[node_name]
        if str(node.target) != prior["operator"]:
            raise ValueError(f"{node_name}: converted contraction target changed")
        dq = node.args[1]
        if dq.op != "call_function":
            raise ValueError(f"{node_name}: weight is not a PT2E dequantization")
        per_channel = dq.target == torch.ops.quantized_decomposed.dequantize_per_channel.default
        per_tensor = dq.target == torch.ops.quantized_decomposed.dequantize_per_tensor.default
        if not (per_channel or per_tensor):
            raise ValueError(f"{node_name}: unsupported weight dequantization {dq.target}")
        if len(dq.args) < (7 if per_channel else 6):
            raise ValueError(f"{node_name}: incomplete weight dequantization parameters")
        qnode = dq.args[0]
        if qnode.op != "get_attr":
            raise ValueError(f"{node_name}: frozen weight is not a named graph buffer")
        frozen_name = str(qnode.target)
        frozen = _tensor(_attr(converted, frozen_name), label=f"frozen {frozen_name}")
        if per_channel:
            snode, znode = dq.args[1:3]
            if any(n.op != "get_attr" for n in (snode, znode)):
                raise ValueError(f"{node_name}: per-channel qparams are not named graph buffers")
            scale_name, zero_name = str(snode.target), str(znode.target)
            scale = _tensor(_attr(converted, scale_name), label=f"scale {scale_name}")
            zero = _tensor(_attr(converted, zero_name), label=f"zero-point {zero_name}")
            axis, qmin, qmax, dtype = dq.args[3:7]
            granularity = "per_channel"
        else:
            scale_name = zero_name = None
            scale, zero, qmin, qmax, dtype = dq.args[1:6]
            if type(scale) is not float or type(zero) is not int:
                raise ValueError(f"{node_name}: per-tensor qparams are not literal float/int scalars")
            axis = None
            granularity = "per_tensor"
        if dtype != torch.int8 or frozen.dtype != torch.int8:
            raise ValueError(f"{node_name}: frozen weight is not int8")
        qmin, qmax = int(qmin), int(qmax)
        effective = prior["effective"]
        if frozen.shape != effective.shape or not torch.isfinite(effective).all():
            raise ValueError(f"{node_name}: frozen weight shape or finite status differs")
        if per_channel:
            axis = int(axis)
            if not 0 <= axis < effective.ndim:
                raise ValueError(f"{node_name}: invalid channel axis")
            if scale.dtype != torch.float32 or scale.shape != (effective.shape[axis],):
                raise ValueError(f"{node_name}: scale dtype/shape does not match channel axis")
            if zero.shape != scale.shape or zero.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{node_name}: zero-point dtype/shape does not match scale")
            if not torch.isfinite(scale).all() or not torch.all(scale > 0):
                raise ValueError(f"{node_name}: non-finite or nonpositive scale")
            expected = torch.ops.quantized_decomposed.quantize_per_channel.default(
                effective, scale, zero, axis, qmin, qmax, dtype
            )
        else:
            import math

            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"{node_name}: non-finite or nonpositive scalar scale")
            expected = torch.ops.quantized_decomposed.quantize_per_tensor.default(
                effective, scale, zero, qmin, qmax, dtype
            )
        if not torch.equal(expected, frozen):
            raise ValueError(f"{node_name}: frozen bytes disagree with declared quantization formula")
        identity = {
            "original_state_key": prior["state_key"],
            "original_sha256": _sha(prior["original"]),
            "fx_node": node_name,
            "operand_index": 1,
        }
        weight_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        rows.append({
            "weight_id": weight_id,
            "fx_node": node_name,
            "operator": prior["operator"],
            "operand_index": 1,
            "original_state_key": prior["state_key"],
            "original": prior["original"],
            "effective": effective,
            "frozen_buffer": frozen_name,
            "frozen": frozen,
            "scale_buffer": scale_name,
            "scale": scale,
            "zero_point_buffer": zero_name,
            "zero_point": zero,
            "granularity": granularity,
            "axis": axis,
            "quant_min": qmin,
            "quant_max": qmax,
        })
    bound = {row["frozen_buffer"] for row in rows}
    unbound = {
        name for name, _ in converted.named_buffers(recurse=True)
        if name.rsplit(".", 1)[-1].startswith("_frozen_param") and name not in bound
    }
    if unbound:
        raise ValueError(f"converted graph has frozen buffers without source lineage: {sorted(unbound)}")
    return rows


def _tensor_record(name: str, tensor: Any) -> dict[str, Any]:
    return {
        "key": name,
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "shape": list(tensor.shape),
        "sha256": _sha(tensor),
    }


def write_weight_lineage(rows: list[dict[str, Any]], weights_path: str | Path) -> dict[str, str]:
    """Write deterministic JSON + safetensors beside the exported weight file.

    Verify that each graph-bound frozen buffer is actually in the exported
    sidecar before publishing lineage.  A failed externalization is fatal here.
    """
    import torch
    import torchao
    from safetensors.torch import load_file, save_file

    if not rows:
        raise ValueError("PT2E weight lineage requires at least one bound weight")
    weights_path = Path(weights_path)
    manifest_path = Path(str(weights_path) + ".manifest.json")
    if not weights_path.is_file() or not manifest_path.is_file():
        raise ValueError("PT2E weight lineage requires successful weight externalization")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exported = load_file(str(weights_path), device="cpu")
    records: list[dict[str, Any]] = []
    prequant: dict[str, Any] = {}

    def check_exported(node_name: str, name: str, tensor: Any) -> None:
        matches = [entry for entry in manifest.values() if entry.get("weight") == name]
        if len(matches) != 1 or matches[0].get("stub") or name not in exported:
            raise ValueError(f"{node_name}: state tensor not uniquely exported: {name}")
        if matches[0].get("dtype") != str(tensor.dtype).replace("torch.", "") or matches[0].get("shape") != list(tensor.shape):
            raise ValueError(f"{node_name}: externalized state descriptor changed: {name}")
        if _bytes(exported[name]) != _bytes(tensor):
            raise ValueError(f"{node_name}: exported state bytes changed: {name}")

    for row in sorted(rows, key=lambda item: item["weight_id"]):
        frozen_name = row["frozen_buffer"]
        checks = [(frozen_name, row["frozen"])]
        if row["granularity"] == "per_channel":
            checks.extend(((row["scale_buffer"], row["scale"]), (row["zero_point_buffer"], row["zero_point"])))
        for name, tensor in checks:
            check_exported(row["fx_node"], name, tensor)
        weight_id = row["weight_id"]
        original_key = f"{weight_id}.original"
        effective_key = f"{weight_id}.effective"
        prequant[original_key] = row["original"]
        prequant[effective_key] = row["effective"]
        qparams = {
            "granularity": row["granularity"],
            "axis": row["axis"],
            "quant_min": row["quant_min"],
            "quant_max": row["quant_max"],
            "dtype": "int8",
        }
        if row["granularity"] == "per_channel":
            qparams["scale"] = _tensor_record(row["scale_buffer"], row["scale"])
            qparams["zero_point"] = _tensor_record(row["zero_point_buffer"], row["zero_point"])
        else:
            qparams["scale"] = {"literal_float64_hex": row["scale"].hex()}
            qparams["zero_point"] = {"literal_int": row["zero_point"]}
        records.append({
            "weight_id": weight_id,
            "fx_operand": {"node": row["fx_node"], "index": row["operand_index"], "operator": row["operator"]},
            "original": _tensor_record(original_key, row["original"]) | {"state_key": row["original_state_key"]},
            "effective": _tensor_record(effective_key, row["effective"]),
            "frozen": _tensor_record(frozen_name, row["frozen"]),
            "qparams": qparams,
        })
    prequant_path = Path(str(weights_path) + ".prequant.safetensors")
    receipt_path = Path(str(weights_path) + ".quantization.json")
    tmp_prequant = prequant_path.with_name(prequant_path.name + ".tmp")
    tmp_receipt = receipt_path.with_name(receipt_path.name + ".tmp")
    try:
        save_file(prequant, str(tmp_prequant))
        document = {
            "schema": SCHEMA,
            "formula": {
                "id": FORMULA,
                "expression": "int8(clamp(round_to_nearest_even(effective_f32 * reciprocal(scale)) + zero_point, quant_min, quant_max))",
                "evaluation_order": ["reciprocal", "f32 multiplication", "torch.round ties-to-even", "add zero_point", "clamp inclusive", "cast int8"],
                "verified_against_frozen_bytes": True,
            },
            "producer": {"torch": torch.__version__, "torchao": torchao.__version__},
            "exported_weights": {"file": weights_path.name, "sha256": _file_sha(weights_path)},
            "prequant_weights": {"file": prequant_path.name, "sha256": _file_sha(tmp_prequant)},
            "weights": records,
        }
        tmp_receipt.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_prequant, prequant_path)
        os.replace(tmp_receipt, receipt_path)
    finally:
        tmp_prequant.unlink(missing_ok=True)
        tmp_receipt.unlink(missing_ok=True)
    return {"prequant_weights": str(prequant_path), "receipt": str(receipt_path)}
