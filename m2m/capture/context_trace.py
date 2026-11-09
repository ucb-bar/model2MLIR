"""Typed read-only grad-context calls for source correspondence.

The context operation remains in both exports. No context is erased, no output
is declared fresh, and schema-unknown or writing bodies are refused.
"""

from __future__ import annotations

import json
import operator
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch

from m2m.capture.trace import _graph_modules, _structural_rows
from m2m.capture.view_trace import SourceStorageAliases


@dataclass(frozen=True)
class ContextCall:
    identity: str
    function_id: str
    scope: str
    control: bool
    arguments: list
    body: tuple[dict, ...]
    canonical_body: str


def _specs(row):
    return [
        {key: value for key, value in result.items() if key != "id"}
        for result in row.get("results") or []
    ]


def live_nodes(exported, snapshot) -> dict[str, Any]:
    actual = [
        (scope, node)
        for scope, module in _graph_modules(exported.graph_module)
        for node in module.graph.nodes
    ]
    if len(actual) != len(snapshot["nodes"]):
        raise ValueError("source graph/module roster differs from snapshot")
    result = {}
    for (scope, node), row in zip(actual, snapshot["nodes"], strict=True):
        if (
            row["graph_id"] != f"g:{snapshot['stage']}:{scope}"
            or row["op"] != node.op
            or row["target"] != str(node.target)
            or node.meta.get("_m2m_node_id") != row["id"]
        ):
            raise ValueError("live source node does not match its captured occurrence")
        result[row["id"]] = node
    return result


def observe_contexts(exported, snapshot) -> dict[str, ContextCall]:
    """Observe exact, flat, read-only set-grad bodies and their typed callers."""
    live = live_nodes(exported, snapshot)
    by_id = {row["id"]: row for row in snapshot["nodes"]}
    by_scope = {}
    for row in snapshot["nodes"]:
        by_scope.setdefault(row["graph_id"], []).append(row)
    scopes = list(_graph_modules(exported.graph_module))
    aliases = SourceStorageAliases(snapshot)
    expected = getattr(torch.ops.higher_order, "wrap_with_set_grad_enabled", None)
    contexts = {}
    for row in snapshot["nodes"]:
        node = live[row["id"]]
        if (
            row["graph_id"] != f"g:{snapshot['stage']}:root"
            or node.target is not expected
        ):
            continue
        args = node.args
        if (
            node.op != "call_function"
            or len(args) < 2
            or type(args[0]) is not bool
            or not isinstance(args[1], torch.fx.Node)
            or args[1].op != "get_attr"
            or node.kwargs
        ):
            continue
        function = args[1]
        try:
            body_module = exported.graph_module.get_submodule(str(function.target))
        except AttributeError:
            continue
        matches = [scope for scope, module in scopes if module is body_module]
        if len(matches) != 1 or matches[0] == "root":
            continue
        scope = f"g:{snapshot['stage']}:{matches[0]}"
        body = by_scope.get(scope, [])
        if not body or any(
            part["op"] not in {"placeholder", "call_function", "output"}
            for part in body
        ):
            continue
        safe = True
        for part in body:
            if part["op"] != "call_function":
                continue
            schema = aliases.schema(part)
            if schema is None or any(
                arg.alias_info is not None and arg.alias_info.is_write
                for arg in schema.arguments
            ):
                safe = False
                break
        formal = [part for part in body if part["op"] == "placeholder"]
        data = row["args"][2:]
        if not safe or len(formal) != len(data):
            continue
        for incoming, parameter in zip(data, formal, strict=True):
            if not isinstance(incoming, dict) or set(incoming) != {
                "node_id",
                "value_id",
            }:
                safe = False
                break
            owner = by_id.get(incoming["node_id"])
            values = owner.get("results") or [] if owner else []
            if (
                len(values) != 1
                or incoming["value_id"] != values[0]["id"]
                or owner["graph_id"] != row["graph_id"]
                or values[0].get("kind") != "tensor"
                or _specs(owner) != _specs(parameter)
            ):
                safe = False
                break
        if not safe:
            continue
        outputs = [part for part in body if part["op"] == "output"]
        if len(outputs) != 1 or len(outputs[0]["args"]) != 1:
            continue
        returned = outputs[0]["args"][0]
        returned = returned if isinstance(returned, list) else [returned]
        if len(returned) != len(row["results"]) or not returned:
            continue
        for ref, result in zip(returned, row["results"], strict=True):
            owner = by_id.get(ref.get("node_id")) if isinstance(ref, dict) else None
            if (
                owner is None
                or owner["graph_id"] != scope
                or len(owner["results"]) != 1
                or ref.get("value_id") != owner["results"][0]["id"]
                or result.get("kind") != "tensor"
                or _specs(owner)
                != [{key: value for key, value in result.items() if key != "id"}]
            ):
                safe = False
                break
        if not safe:
            continue
        canonical = _structural_rows({"nodes": body})
        for index, part in enumerate(canonical):
            if part["op"] == "placeholder":
                part["target"] = "formal_" + str(index)
        key = json.dumps(
            {
                "body": canonical,
                "schemas": {
                    part["target"]: snapshot["operator_schemas"].get(part["target"])
                    for part in body
                    if part["op"] == "call_function"
                },
            },
            sort_keys=True,
        )
        function_id = function.meta.get("_m2m_node_id")
        if function_id not in by_id or by_id[function_id]["op"] != "get_attr":
            continue
        contexts[row["id"]] = ContextCall(
            row["id"], function_id, scope, args[0], data, tuple(body), key
        )
    counts = Counter(context.scope for context in contexts.values())
    return {
        identity: context
        for identity, context in contexts.items()
        if counts[context.scope] == 1
    }


def context_projection(row, snapshot, contexts, live) -> bool:
    """A tuple selection is read-only; no storage-freshness fact is inferred."""
    if (
        live[row["id"]].target is not operator.getitem
        or row["op"] != "call_function"
        or row["kwargs"]
        or len(row["args"]) != 2
        or type(row["args"][1]) is not int
    ):
        return False
    ref, slot = row["args"]
    if (
        not isinstance(ref, dict)
        or set(ref) != {"node_id", "value_id"}
        or ref["node_id"] not in contexts
    ):
        return False
    parent = next(part for part in snapshot["nodes"] if part["id"] == ref["node_id"])
    values = parent["results"]
    if not -len(values) <= slot < len(values):
        return False
    selected = values[slot % len(values)]
    return (
        ref["value_id"] == selected["id"]
        and len(row["results"]) == 1
        and _specs(row)
        == [{key: value for key, value in selected.items() if key != "id"}]
    )
