"""Immutable typed public/callee contracts for one owned source export."""

from __future__ import annotations

import dataclasses
import struct


def _value(value):
    if any(type(value) is kind for kind in (type(None), bool, int, str)):
        return (type(value), value)
    if type(value) is float:
        return (float, struct.pack("!d", value))
    if type(value) is tuple or type(value) is list:
        return (type(value), tuple(_value(item) for item in value))
    if type(value) is dict:
        return (dict, tuple((_value(k), _value(v)) for k, v in value.items()))
    raise ValueError("unsupported call-contract context value")


def _tree(spec):
    from torch.utils._pytree import LeafSpec, TreeSpec

    if type(spec) is not TreeSpec and type(spec) is not LeafSpec:
        raise ValueError("unsupported call-contract TreeSpec")
    if hasattr(spec, "__dict__") and set(vars(spec)) != {
        "type",
        "context",
        "children_specs",
        "num_nodes",
        "num_leaves",
        "num_children",
    }:
        raise ValueError("unsupported extra call-contract tree fields")
    children = spec.children_specs
    if type(children) is not list:
        raise ValueError("unsupported call-contract tree child container")
    if any(
        type(child) is not TreeSpec and type(child) is not LeafSpec
        for child in children
    ):
        raise ValueError("unsupported call-contract tree child kind")
    counts = (spec.num_nodes, spec.num_leaves, spec.num_children)
    if any(type(count) is not int for count in counts):
        raise ValueError("unsupported call-contract tree counts")
    if spec.type is None:
        if spec.context is not None or children or counts != (1, 1, 0):
            raise ValueError("inconsistent call-contract leaf fields")
        return ("leaf", type(spec), counts)
    if not any(spec.type is kind for kind in (tuple, list, dict)):
        raise ValueError("unsupported call-contract tree container")
    # Recurse through only exact known child records before reading any child
    # counts. A mutated counter is not an integer and cannot reach sum/add.
    child_records = tuple(_tree(child) for child in children)
    expected = (
        1 + sum(child[-1][0] for child in child_records),
        sum(child[-1][1] for child in child_records),
        len(children),
    )
    if counts != expected:
        raise ValueError("inconsistent call-contract tree counts")
    if ((spec.type is tuple or spec.type is list) and spec.context is not None) or (
        spec.type is dict
        and (type(spec.context) is not list or len(spec.context) != len(children))
    ):
        raise ValueError("inconsistent call-contract tree context")
    return (
        "tree",
        type(spec),
        spec.type,
        _value(spec.context),
        child_records,
        counts,
    )


def _fields(value):
    declared = tuple(field.name for field in dataclasses.fields(type(value)))
    if set(vars(value)) != set(declared):
        raise ValueError("unsupported extra call-contract fields")
    return declared


def _argument(value):
    from torch.export import graph_signature

    known = tuple(
        cls
        for name in (
            "TensorArgument",
            "SymIntArgument",
            "SymFloatArgument",
            "SymBoolArgument",
            "ConstantArgument",
            "TokenArgument",
        )
        if (cls := getattr(graph_signature, name, None)) is not None
    )
    if not any(type(value) is kind for kind in known):
        raise ValueError("unsupported call-contract argument kind")
    return (
        type(value),
        tuple((key, _value(getattr(value, key))) for key in _fields(value)),
    )


def _ranges(exported):
    import sympy
    from torch.utils._sympy.value_ranges import ValueRanges

    def bound(value):
        if type(value) is int:
            return (int, value)
        if isinstance(value, sympy.Basic):
            return (type(value), sympy.srepr(value))
        raise ValueError("unsupported call-contract range expression")

    if type(exported.range_constraints) is not dict:
        raise ValueError("unsupported call-contract range mapping")
    result = []
    for key, value in exported.range_constraints.items():
        if type(value) is not ValueRanges or set(_fields(value)) != {
            "lower",
            "upper",
            "is_bool",
            "is_int",
            "is_float",
        }:
            raise ValueError("unsupported call-contract range bounds")
        flags = tuple(
            getattr(value, name) for name in ("is_bool", "is_int", "is_float")
        )
        if any(type(flag) is not bool for flag in flags):
            raise ValueError("unsupported call-contract range flags")
        result.append((bound(key), bound(value.lower), bound(value.upper), flags))
    return tuple(result)


def freeze_call_contract(exported):
    """Observe typed values without invoking callbacks or copying tensor state.

    This opt-in Linear proof accepts static ordinary tensor arguments and
    builtin tuple/list/dict output trees. Unknown trees, arguments, contexts or
    extra record fields refuse. Range expressions are recorded without solving
    or evaluating guards. Neither this observation nor equality rewrites an EP.
    """
    from torch.export.exported_program import ModuleCallEntry, ModuleCallSignature

    if type(exported.module_call_graph) is not list:
        raise ValueError("unsupported call-contract module entry list")
    entries = []
    for entry in exported.module_call_graph:
        if type(entry) is not ModuleCallEntry or set(_fields(entry)) != {
            "fqn",
            "signature",
        }:
            raise ValueError("unsupported call-contract module entry")
        if type(entry.fqn) is not str:
            raise ValueError("unsupported call-contract module path")
        signature = entry.signature
        if signature is None:
            entries.append((entry.fqn, None))
            continue
        if type(signature) is not ModuleCallSignature:
            raise ValueError("unsupported call-contract signature")
        fields = set(_fields(signature))
        required = {"inputs", "outputs", "in_spec", "out_spec"}
        if not required <= fields <= required | {"forward_arg_names"}:
            raise ValueError("unsupported call-contract signature fields")
        if type(signature.inputs) is not list or type(signature.outputs) is not list:
            raise ValueError("unsupported call-contract argument lists")
        names = getattr(signature, "forward_arg_names", None)
        if names is not None and (
            type(names) is not list or any(type(name) is not str for name in names)
        ):
            raise ValueError("unsupported call-contract forward names")
        entries.append(
            (
                entry.fqn,
                tuple(_argument(arg) for arg in signature.inputs),
                tuple(_argument(arg) for arg in signature.outputs),
                _tree(signature.in_spec),
                _tree(signature.out_spec),
                _value(names),
            )
        )
    return tuple(entries), _ranges(exported)
