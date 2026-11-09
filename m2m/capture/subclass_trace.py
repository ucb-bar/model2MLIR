"""Observe actual tensor-subclass field lifting across export decomposition.

Bindings use live source/selected tensor storage, exact dtype/shape/strides,
owned bytes, the public graph state roster and all source write effects. Names
and similar tensor contents never establish a field binding.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch

from m2m.capture.trace import _digest, snapshot_exported_program
from m2m.capture.view_trace import SourceStorageAliases


def _storage(tensor):
    if (
        type(tensor) not in (torch.Tensor, torch.nn.Parameter)
        or tensor.layout != torch.strided
    ):
        return None
    try:
        return tensor.untyped_storage()._cdata
    except RuntimeError:
        return None


def _description(tensor):
    if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
        return None
    shape, strides, offset = (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
    )
    if any(type(value) is not int for value in (*shape, *strides, offset)):
        return None
    return str(tensor.dtype), shape, strides, offset, str(tensor.device)


def _hash(tensor):
    value = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(memoryview(value)).hexdigest()


@dataclass(frozen=True)
class Field:
    source_id: str
    source_target: str
    field: str
    owner: Any
    value: Any
    storage: int
    description: tuple
    sha256: str


@dataclass(frozen=True)
class Observation:
    source: Any
    source_graph_sha256: str
    fields: tuple[Field, ...]


def _states(exported):
    return {**exported.state_dict, **exported.constants}


def _canonical_quantized_ids(exported):
    """A public trace stage must not be silently renumbered after a rewrite."""
    return all(
        (node.meta.get("custom") or {}).get("m2m_node_id")
        == f"g:quantized:root:n{index}"
        for index, node in enumerate(exported.graph.nodes)
    )


def observe(exported, *, quantized_frontend_snapshot=None) -> Observation | None:
    """Collect only immutable, actual state-field accesses with complete effects."""
    packet = getattr(torch.ops.export, "access_subclass_inner_tensor", None)
    accessor = getattr(packet, "default", None)
    nodes = list(exported.graph.nodes)
    if accessor is None or not any(
        node.op == "call_function" and node.target is accessor for node in nodes
    ):
        return None
    if (
        exported.graph_signature.buffers_to_mutate
        or exported.graph_signature.user_inputs_to_mutate
    ):
        return None
    # api.convert may already have inlined higher-order regions. Old public
    # qIDs still describe the archived raw graph, not new root ordinals. Refuse
    # this proof before any snapshot can overwrite them.
    if not _canonical_quantized_ids(exported):
        return None
    if quantized_frontend_snapshot is not None:
        supplied = dict(quantized_frontend_snapshot)
        claimed = supplied.pop("sha256", None)
        if (
            supplied.get("stage") != "quantized"
            or supplied.get("status") != "complete"
            or claimed != _digest(supplied)
        ):
            return None
    snapshot = snapshot_exported_program(exported, stage="quantized")
    if (
        quantized_frontend_snapshot is not None
        and snapshot != quantized_frontend_snapshot
    ):
        return None
    aliases = SourceStorageAliases(snapshot)
    states = _states(exported)
    rows = snapshot["nodes"]
    if len(nodes) != len(rows):
        return None  # Nested scopes require their own explicit call/state proof.
    by_node = dict(zip(nodes, rows, strict=True))
    specs = {spec.arg.name: spec for spec in exported.graph_signature.input_specs}
    fields = []
    guarded_roots = set()
    for node in nodes:
        if node.op != "call_function" or node.target is not accessor:
            continue
        if (
            len(node.args) != 2
            or node.kwargs
            or not isinstance(node.args[0], torch.fx.Node)
            or not isinstance(node.args[1], str)
        ):
            return None
        owner, name = node.args
        spec = specs.get(owner.name)
        if owner.op != "placeholder" or spec is None or spec.target not in states:
            return None
        wrapper = states[spec.target]
        flatten = getattr(wrapper, "__tensor_flatten__", None)
        if not callable(flatten):
            return None
        names, _context = flatten()
        if (
            not isinstance(names, (list, tuple))
            or name not in names
            or any(not isinstance(n, str) for n in names)
        ):
            return None
        value = torch.ops.export.access_subclass_inner_tensor.default(wrapper, name)
        storage = _storage(value)
        description = _description(value)
        if (
            storage is None
            or description is None
            or _description(node.meta.get("val")) != description
        ):
            return None
        row = by_node[node]
        root = aliases.root(row)
        parent_root = aliases.root(by_node[owner])
        if root is None or parent_root is None:
            return None
        guarded_roots.update((root, parent_root))
        fields.append(
            Field(
                row["id"],
                str(spec.target),
                name,
                wrapper,
                value,
                storage,
                description,
                _hash(value),
            )
        )
    if not fields:
        return None
    # No external input/state may be mutated during this lifting proof. This
    # also covers another wrapper or a user input aliasing a field's storage;
    # runtime aliasing cannot turn that write into a fresh-intermediate write.
    for node, row in by_node.items():
        if node.op == "placeholder":
            root = aliases.root(row)
            if root is None:
                return None
            guarded_roots.add(root)
    for row in rows:
        if row["op"] in {"placeholder", "get_attr", "output"}:
            continue
        if row["op"] != "call_function":
            return None
        schema = aliases.schema(row)
        if schema is None:
            if aliases.tuple_projection_root(row) is None:
                return None
            continue
        for index, arg in enumerate(schema.arguments):
            if arg.alias_info is not None and arg.alias_info.is_write:
                owner = aliases.tensor_ref(aliases.argument(row, index, arg.name))
                root = aliases.root(owner) if owner is not None else None
                if root is None or root in guarded_roots:
                    return None
    return Observation(exported, snapshot["sha256"], tuple(fields))


def bind(observation: Observation | None, prepared) -> list[dict]:
    """Stamp only uniquely observed, unchanged state-field parameters."""
    if observation is None or not _canonical_quantized_ids(observation.source):
        return []
    fresh = snapshot_exported_program(observation.source, stage="quantized")
    if fresh["sha256"] != observation.source_graph_sha256:
        return []
    fields = observation.fields
    source_states = _states(observation.source)
    if (
        prepared.graph_signature.buffers_to_mutate
        or prepared.graph_signature.user_inputs_to_mutate
    ):
        return []
    states = _states(prepared)
    specs = {spec.arg.name: spec for spec in prepared.graph_signature.input_specs}
    nodes = [node for node in prepared.graph.nodes if node.op == "placeholder"]
    plans = []
    for field in fields:
        current_owner = source_states.get(field.source_target)
        if current_owner is not field.owner:
            return []
        try:
            current = torch.ops.export.access_subclass_inner_tensor.default(
                current_owner, field.field
            )
        except (AttributeError, RuntimeError):
            return []
        if (
            _storage(current) != field.storage
            or _description(current) != field.description
            or _hash(current) != field.sha256
        ):
            return []
        if (
            source_states.get(field.source_target) is not field.owner
            or _storage(field.value) != field.storage
            or _description(field.value) != field.description
            or _hash(field.value) != field.sha256
        ):
            return []
        candidates = []
        for node in nodes:
            spec = specs.get(node.name)
            if spec is None or spec.target not in states:
                continue
            value = states[spec.target]
            if (
                _storage(value) == field.storage
                and _description(value) == field.description
                and _description(node.meta.get("val")) == field.description
            ):
                candidates.append((node, spec, value))
        if len(candidates) != 1:
            return []
        node, spec, value = candidates[0]
        if _hash(value) != field.sha256:
            return []
        plans.append((field, node, spec))
    records = []
    for field, node, spec in plans:
        custom = dict(node.meta.get("custom") or {})
        lineage = list(custom.get("m2m_lineage") or ())
        if field.source_id not in lineage:
            lineage.append(field.source_id)
        custom.update(m2m_lineage=lineage, m2m_transform="subclass_field_parameter")
        node.meta["custom"] = custom
        records.append(
            {
                "source_id": field.source_id,
                "source_state_target": field.source_target,
                "source_graph_sha256": observation.source_graph_sha256,
                "field": field.field,
                "prepared_state_target": str(spec.target),
                "same_live_storage": True,
                "source_state_unwritten": True,
                "dtype_shape_stride_offset": str(field.description),
                "sha256": field.sha256,
            }
        )
    return records
