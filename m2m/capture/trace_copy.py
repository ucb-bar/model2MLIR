"""Own FX graph/metadata without copying checkpoint tensor storage."""

from __future__ import annotations

import copy
from collections import defaultdict

import torch


def _metadata(value):
    if isinstance(value, torch.Tensor):
        # Tensor metadata and state storage are observed, never mutated here.
        return value
    if isinstance(value, dict):
        items = {key: _metadata(item) for key, item in value.items()}
        if isinstance(value, defaultdict):
            return type(value)(value.default_factory, items)
        return type(value)(items)
    if isinstance(value, list):
        return type(value)(_metadata(item) for item in value)
    if isinstance(value, tuple):
        items = tuple(_metadata(item) for item in value)
        return type(value)(*items) if hasattr(value, "_fields") else items
    return copy.deepcopy(value)


def copy_trace_program(exported):
    """Clone all live graph bodies/custom metadata, retaining actual state owners.

    Transform only the returned verified EP. The raw source graph remains live
    and unchanged for its archived-snapshot/field binding checks. This is not a
    tensor clone, source relabelling, alias permission or numerical proof.
    """
    memo = {}

    def clone(module):
        if id(module) in memo:
            return memo[id(module)]
        graph = torch.fx.Graph()
        nodes = {}
        for node in module.graph.nodes:
            copied = graph.node_copy(node, lambda incoming: nodes[incoming])
            copied.meta = _metadata(node.meta)
            nodes[node] = copied
        owned = torch.fx.GraphModule(module, graph)
        memo[id(module)] = owned
        owned.meta = _metadata(module.meta)
        for path, child in module.named_modules(remove_duplicate=False):
            if not path or not isinstance(child, torch.fx.GraphModule):
                continue
            parent_path, _, name = path.rpartition(".")
            try:
                parent = owned.get_submodule(parent_path) if parent_path else owned
            except AttributeError:
                raise ValueError("graph-copy body has no owned module path") from None
            if parent_path and parent is module.get_submodule(parent_path):
                raise ValueError("nested graph body parent is not independently owned")
            setattr(parent, name, clone(child))
        for name, value in module.__dict__.items():
            if name.startswith("_m2m_trace_"):
                setattr(owned, name, _metadata(value))
        return owned

    result = exported._update(
        clone(exported.graph_module),
        copy.deepcopy(exported.graph_signature),
        state_dict=dict(exported.state_dict),
        constants=dict(exported.constants),
    )
    for name, value in exported.__dict__.items():
        if name.startswith("_m2m_trace_"):
            setattr(result, name, _metadata(value))
    result.validate()
    return result
