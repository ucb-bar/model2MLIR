"""Own FX graph/metadata without copying checkpoint tensor storage."""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import fields
from types import BuiltinFunctionType, ModuleType

import torch


def _metadata(value):
    if isinstance(value, torch.Tensor):
        # Tensor metadata and state storage are observed, never mutated here.
        return value
    if isinstance(value, dict):
        items = {_metadata(key): _metadata(item) for key, item in value.items()}
        if isinstance(value, defaultdict):
            return type(value)(value.default_factory, items)
        return type(value)(items)
    if isinstance(value, list):
        return type(value)(_metadata(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return type(value)(_metadata(item) for item in value)
    if isinstance(value, tuple):
        items = tuple(_metadata(item) for item in value)
        return type(value)(*items) if hasattr(value, "_fields") else items
    if type(value) in {type(None), bool, int, float, complex, str, bytes} or isinstance(
        value,
        (
            type,
            torch.dtype,
            torch.device,
            torch.layout,
            torch.memory_format,
            torch._ops.OpOverload,
            torch._ops.OpOverloadPacket,
        ),
    ):
        return value
    if type(value) is BuiltinFunctionType and (
        value.__self__ is None or isinstance(value.__self__, (ModuleType, type))
    ):
        return value  # Global callable identity, never a bound tensor receiver.
    from torch.fx import traceback

    source_type = getattr(traceback, "NodeSource", None)
    if source_type is not None and type(value) is source_type:
        allowed = {
            "pass_name",
            "action",
            "from_node",
            "node_info",
            "_dict",
            "_action_string",
        }
        action_type = getattr(traceback, "NodeSourceAction", None)
        info = value.node_info
        if (
            set(vars(value)) != allowed
            or type(value.pass_name) is not str
            or type(value.action) is not list
            or any(type(action) is not action_type for action in value.action)
            or type(value.from_node) is not list
            or any(type(parent) is not source_type for parent in value.from_node)
            or (
                info is not None
                and (
                    type(info) is not source_type.NodeInfo
                    or set(vars(info)) != {"name", "target", "graph_id"}
                    or type(info.name) is not str
                    or type(info.target) is not str
                    or type(info.graph_id) is not int
                )
            )
        ):
            raise ValueError("unsupported trace NodeSource metadata")
        owned = source_type(None, value.pass_name, list(value.action))
        if info is not None:
            owned.node_info = source_type.NodeInfo(
                info.name, info.target, info.graph_id
            )
        owned.from_node = [_metadata(parent) for parent in value.from_node]
        owned._dict = _metadata(value._dict)
        owned._action_string = _metadata(value._action_string)
        return owned
    from torch import _guards
    from torch.utils._pytree import LeafSpec, TreeSpec

    CompileId = getattr(_guards, "CompileId", None)
    if CompileId is not None and type(value) is CompileId:
        names = tuple(field.name for field in fields(CompileId) if field.init)
        if any(
            getattr(value, name) is not None and type(getattr(value, name)) is not int
            for name in names
        ):
            raise ValueError("unsupported trace CompileId metadata")
        return CompileId(**{name: getattr(value, name) for name in names})
    if type(value) in (TreeSpec, LeafSpec):
        if value.is_leaf():
            return LeafSpec()
        return TreeSpec(
            value.type,
            _metadata(value.context),
            [_metadata(child) for child in value.children_specs],
        )
    raise ValueError(
        "unsupported opaque trace metadata: "
        + type(value).__module__
        + "."
        + type(value).__qualname__
    )


def copy_trace_program(exported):
    """Clone all live graph bodies/custom metadata, retaining actual state owners.

    Transform only the returned verified EP. The raw source graph remains live
    and unchanged for its archived-snapshot/field binding checks. This is not a
    tensor clone, source relabelling, alias permission or numerical proof.
    """
    exported.validate()  # Existing verified source names/signature are authoritative.
    memo = {}

    def clone(module):
        if id(module) in memo:
            return memo[id(module)]
        graph = torch.fx.Graph()
        nodes = {}
        for node in module.graph.nodes:
            copied = graph.node_copy(node, lambda incoming: nodes[incoming])
            # FX's fresh namespace may rename valid exported names such as
            # `input` to avoid shadowing a builtin. Keep the verified source
            # occurrence name and namespace binding, never guess a signature
            # remap or allow an implicit input/input_1 collision.
            copied.name = node.name
            graph._graph_namespace._rename_object(copied, node.name)
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
