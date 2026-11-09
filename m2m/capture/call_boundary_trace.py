"""Preserve actual callee values when decomposition erases no-op type_as."""

from __future__ import annotations

from dataclasses import replace

import torch

from m2m.capture.trace import _forward_identity_value, snapshot_exported_program
from m2m.capture.trace_copy import copy_trace_program
from m2m.capture.view_trace import SourceStorageAliases, unchanged_through_uses


def normalize_call_boundary_aliases(exported):
    """Return an owned EP with exactly proved tensor-value signature aliases.

    The arithmetic graph is unchanged. Raw caller signatures remain untouched.
    A module-call argument is redirected only through an exact identity type_as
    and its unchanged alias lifetime, never through names/types/shapes alone.
    This handles framework signature maintenance, not model accuracy or target
    execution. Unsupported identity/effect cases remain a precise refusal.
    """
    from torch._subclasses.fake_tensor import FakeTensor
    from torch.export.graph_signature import TensorArgument

    nodes = {node.name: node for node in exported.graph.nodes}
    candidates = []
    for entry_index, entry in enumerate(exported.module_call_graph):
        if entry.signature is None:
            continue
        for direction in ("inputs", "outputs"):
            for index, arg in enumerate(getattr(entry.signature, direction)):
                if not isinstance(arg, TensorArgument):
                    continue
                node = nodes.get(arg.name)
                if node is not None and node.target is torch.ops.aten.type_as.default:
                    candidates.append((entry_index, entry.fqn, direction, index, node))
    if not candidates:
        return exported, []
    # Analyze a separate graph so temporary IDs cannot overwrite raw qIDs or
    # their source origins in the actual conversion program.
    analysis = copy_trace_program(exported)
    snapshot = snapshot_exported_program(analysis, stage="boundary_analysis")
    root_rows = [
        row
        for row in snapshot["nodes"]
        if row["graph_id"] == "g:boundary_analysis:root"
    ]
    rows = {
        node.name: row
        for node, row in zip(analysis.graph.nodes, root_rows, strict=True)
    }
    aliases = SourceStorageAliases(snapshot)
    uses = {}
    for edge in snapshot["edges"]:
        uses.setdefault(edge["producer_node_id"], []).append(edge)
    plans = []
    for entry_index, fqn, direction, index, node in candidates:
        cursor = node
        traversed = []
        while cursor.target is torch.ops.aten.type_as.default:
            if (
                len(cursor.args) != 2
                or cursor.kwargs
                or any(not isinstance(arg, torch.fx.Node) for arg in cursor.args)
                or any(
                    type(value) not in (torch.Tensor, torch.nn.Parameter, FakeTensor)
                    for value in (
                        cursor.meta.get("val"),
                        *(arg.meta.get("val") for arg in cursor.args),
                    )
                )
            ):
                raise ValueError(
                    "callee type_as alias lacks ordinary dense tensor metadata"
                )
            row = rows[cursor.name]
            checked = _forward_identity_value(snapshot, row)
            if checked is None:
                # A real dtype/device conversion remains an actual operation;
                # framework decomposition handles its ordinary provenance.
                traversed = []
                break
            state = aliases.last_writer(row)
            if (
                state is None
                or (
                    state[1] is not None and state[1]["ordinal"] > checked[0]["ordinal"]
                )
                or not unchanged_through_uses(aliases, row, uses.get(row["id"], []))
            ):
                raise ValueError(
                    "callee type_as alias/effect/consumer lifetime is unproved"
                )
            traversed.append(cursor.name)
            cursor = cursor.args[0]
        if not traversed:
            continue
        plans.append(
            (entry_index, fqn, direction, index, node.name, cursor.name, traversed)
        )
    if not plans:
        return exported, []
    owned = copy_trace_program(exported)
    records = []
    for entry_index, fqn, direction, index, old_name, new_name, traversed in plans:
        entry = owned.module_call_graph[entry_index]
        if entry.fqn != fqn or entry.signature is None:
            raise ValueError("callee entry occurrence changed during alias preparation")
        slots = getattr(entry.signature, direction)
        before = slots[index]
        if before.name != old_name:
            raise ValueError(
                "callee signature occurrence changed during alias preparation"
            )
        slots[index] = replace(before, name=new_name)
        records.append(
            {
                "module_path": fqn,
                "module_call_entry": entry_index,
                "direction": direction,
                "slot": index,
                "original_value": old_name,
                "same_tensor_value": new_name,
                "identity_chain": traversed,
                "source_id": (nodes[old_name].meta.get("custom") or {}).get(
                    "m2m_node_id"
                ),
                "same_value_source_id": (nodes[new_name].meta.get("custom") or {}).get(
                    "m2m_node_id"
                ),
                "analysis_graph_sha256": snapshot["sha256"],
                "graph_arithmetic_changed": False,
                "typed_identity_and_consumer_lifetime_verified": True,
            }
        )
    owned.validate()
    return owned, records
