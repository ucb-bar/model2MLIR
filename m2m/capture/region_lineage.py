"""Exact, call-bound receipts for re-exported grad-disabled regions.

This module proves correspondence only. It grants no execution admission and
never infers an identity from an operator, attribute name or tensor shape alone.
"""
from __future__ import annotations

KEY = "m2m_nested_lineage"
REQUIRED = "m2m_nested_lineage_required"


def wire_value(value):
    """Only finite, JSON-shaped certificate fields belong to this version."""
    import math

    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(wire_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and wire_value(item) for key, item in value.items())
    return False


def string_ids(value):
    return isinstance(value, list) and all(isinstance(item, str) and item for item in value)


def valid_specs(value, *, with_ids=False):
    if not isinstance(value, list):
        return False
    for row in value:
        if not isinstance(row, dict) or not isinstance(row.get("kind"), str):
            return False
        if with_ids and not isinstance(row.get("id"), str):
            return False
        for key in ("dtype", "storage_dtype", "compute_dtype", "device", "layout"):
            if key not in row or row[key] is not None and not isinstance(row[key], str):
                return False
        for key in ("shape", "stride"):
            if key not in row or row[key] is not None and not (
                    isinstance(row[key], list) and all(type(item) in {int, str} for item in row[key])):
                return False
    return True


def valid_records(rows, *, scoped=False):
    if not isinstance(rows, list) or not rows:
        return False
    for row in rows:
        if (not isinstance(row, dict)
                or any(not isinstance(row.get(key), str) for key in ("id", "op", "target"))
                or scoped and not isinstance(row.get("graph_id"), str)
                or not isinstance(row.get("args"), list) or not isinstance(row.get("kwargs"), dict)
                or not valid_specs(row.get("results"), with_ids=True)):
            return False
    return len({row["id"] for row in rows}) == len(rows)


def valid_region(region):
    if (not isinstance(region, dict) or region.get("schema") != "m2m.autocast_lineage.v1"
            or region.get("status") != "validated" or not isinstance(region.get("source_node_id"), str)
            or not string_ids(region.get("body_source_ids"))):
        return False
    for key, fields in (("operand_bindings", ("placeholder", "operand")),
                        ("output_bindings", ("selector", "body_result"))):
        if not isinstance(region.get(key), list):
            return False
        for binding in region[key]:
            if not isinstance(binding, dict):
                return False
            if key == "output_bindings" and type(binding.get("result_index")) is not int:
                return False
            for field in fields:
                value = binding.get(field)
                if (not isinstance(value, dict) or not isinstance(value.get("node_id"), str)
                        or not valid_specs(value.get("results"))):
                    return False
    moves = region.get("ownership_moves")
    if not isinstance(moves, list) or any(not isinstance(move, dict) or any(
            not isinstance(move.get(key), str) for key in
            ("source_node_id", "from_body_node_id", "to_wrapper_scope_node_id")) for move in moves):
        return False
    eliminations = region.get("eliminations")
    return isinstance(eliminations, list) and bool(eliminations) and all(
        isinstance(row, dict) and string_ids(row.get("source_ids"))
        and row.get("kind") == "eliminated" and isinstance(row.get("proof"), dict)
        for row in eliminations)


def valid_receipt(receipt):
    """Check the versioned wire structure before any proof reader indexes it."""
    return (isinstance(receipt, dict) and wire_value(receipt) and receipt.get("schema") == "m2m.nested_lineage.v1"
            and isinstance(receipt.get("torch_version"), str)
            and valid_records([receipt.get("source_wrapper")], scoped=True)
            and valid_records(receipt.get("source_body"), scoped=True)
            and valid_records(receipt.get("normalized_body"))
            and isinstance(receipt.get("lineages"), list)
            and len(receipt["lineages"]) == len(receipt["normalized_body"])
            and all(string_ids(ids) for ids in receipt["lineages"])
            and isinstance(receipt.get("bindings"), list)
            and all(isinstance(row, dict) and string_ids(row.get("source_ids"))
                    and bool(row["source_ids"]) and valid_specs(row.get("spec"))
                    for row in receipt["bindings"])
            and isinstance(receipt.get("regions"), list) and bool(receipt["regions"])
            and all(valid_region(region) for region in receipt["regions"])
            and len({region["source_node_id"] for region in receipt["regions"]}) == len(receipt["regions"]))


def mark_required(nodes, wrapper_id):
    for node in nodes:
        node.meta = dict(node.meta)
        custom = dict(node.meta.get("custom") or {})
        custom[REQUIRED] = wrapper_id
        node.meta["custom"] = custom


def precision_modules(graph_module):
    """Visit call-reachable bodies, retaining descendants of opaque owners.

    A's ownership check may keep the old autocast body's registration because
    its grad-disabled ancestor remains registered. Default FX execution of a
    uniquely bound grad call reaches only the ancestor's actual graph operands;
    the retired registration must not be interpreted as a second source site.
    """
    from m2m.capture.trace import _graph_modules

    pending, visited = [("root", graph_module)], set()
    while pending:
        scope, module = pending.pop(0)
        if id(module) in visited:
            continue
        visited.add(id(module))
        yield scope, module
        for node in module.graph.nodes:
            if node.op not in {"get_attr", "call_module"}:
                continue
            value = module
            for part in str(node.target).split("."):
                value = getattr(value, part)
            if not hasattr(value, "graph"):
                continue
            child_scope = str(node.target) if scope == "root" else f"{scope}.{node.target}"
            exact_call = (node.op == "get_attr" and len(node.users) == 1
                          and grad_body(module, next(iter(node.users))) is value)
            if exact_call:
                pending.append((child_scope, value))
            else:
                pending.extend((child_scope if name == "root" else f"{child_scope}.{name}", child)
                               for name, child in _graph_modules(value))


def specs(node):
    return [{key: value for key, value in result.items() if key != "id"}
            for result in node.get("results") or []]


def core(node):
    return {key: node.get(key) for key in
            ("id", "graph_id", "op", "target", "args", "kwargs", "results")}


def guard_input(node, nodes):
    if node["target"] != "aten._assert_tensor_metadata.default":
        return None
    if any(result.get("kind") != "unknown" or result.get("dtype") is not None
           for result in node.get("results") or []):
        return None
    fields = ("a", "size", "stride", "dtype", "device", "layout")
    args, kwargs = node["args"], node["kwargs"]
    if (not args or len(args) > len(fields) or set(kwargs) - set(fields)
            or any(field in kwargs for field in fields[:len(args)])):
        return None
    values = {field: args[i] if i < len(args) else kwargs.get(field)
              for i, field in enumerate(fields)}
    ref = values.pop("a")
    producer = nodes.get(ref.get("node_id")) if isinstance(ref, dict) else None
    if (producer is None or len(producer["results"]) != 1
            or ref.get("value_id") != producer["results"][0]["id"]):
        return None
    value = producer["results"][0]
    expected = {"size": value["shape"], "stride": value["stride"],
                "dtype": {"kind": "dtype", "value": f"torch.{value['dtype']}"},
                "device": {"kind": "device", "value": value["device"]},
                "layout": {"kind": "layout", "value": value["layout"]}}
    asserted = {field: item for field, item in values.items() if item is not None}
    if not asserted or any(expected[field] != item for field, item in asserted.items()):
        return None
    return producer


def reduced_body(nodes):
    """Erase only typed identity casts and already-satisfied metadata guards.

    A write or unknown call anywhere in the body prevents these reductions;
    a metadata observation alone cannot establish storage stability.
    """
    import torch
    from m2m.capture.trace import _forward_identity_value

    stable = True
    for node in nodes:
        if node["op"] != "call_function" or node["target"] == "<built-in function getitem>":
            continue
        parts = node["target"].split(".")
        if len(parts) != 3 or parts[0] != "aten":
            stable = False
            break
        overload = getattr(getattr(torch.ops.aten, parts[1], None), parts[2], None)
        schema = getattr(overload, "_schema", None)
        if (schema is None or schema.is_mutable or any(
                field.alias_info is not None and field.alias_info.is_write
                for field in [*schema.arguments, *schema.returns])):
            stable = False
            break
    retained, aliases, guards = [], {}, []
    lookup = {node["id"]: node for node in nodes}
    for node in nodes:
        identity = (_forward_identity_value({"nodes": nodes}, node)
                    if stable and node["target"] in {"aten.to.dtype", "aten.to.dtype_layout", "aten.to.device"}
                    else None)
        if identity is not None:
            parent = identity[0]["id"]
            aliases[node["id"]] = aliases.get(parent, parent)
        elif stable and guard_input(node, lookup) is not None:
            guards.append(node["id"])
        else:
            retained.append(node)
    return retained, aliases, guards


def canonical_body(nodes):
    """Malformed proof data is an unavailable body, never an exception."""
    if not valid_records(nodes):
        return None
    try:
        return _canonical_body(nodes)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        return None


def _canonical_body(nodes):
    """Ordered typed graph, including output order and every producer slot."""
    if len({node["id"] for node in nodes}) != len(nodes):
        return None
    all_nodes = {node["id"]: node for node in nodes}
    nodes, aliases, guards = reduced_body(nodes)
    positions = {node["id"]: index for index, node in enumerate(nodes)}

    def encode(value):
        if isinstance(value, list):
            return [encode(item) for item in value]
        if isinstance(value, dict):
            if "node_id" in value:
                if set(value) != {"node_id", "value_id"}:
                    raise ValueError("unknown region edge fields")
                producer = all_nodes[value["node_id"]]
                owner = aliases.get(producer["id"], producer["id"])
                indices = [i for i, result in enumerate(producer.get("results") or [])
                           if result["id"] == value["value_id"]]
                if len(indices) != 1:
                    raise ValueError("unbound region result")
                return {"producer": positions[owner], "result": indices[0]}
            return {key: encode(item) for key, item in value.items()}
        return value

    result = []
    try:
        for node in nodes:
            args, kwargs = node["args"], node["kwargs"]
            # Export changes only schema argument placement for these guards.
            if node["target"] == "aten._assert_tensor_metadata.default":
                fields = ("a", "size", "stride", "dtype", "device", "layout")
                if (len(args) > len(fields) or set(kwargs) - set(fields)
                        or any(field in kwargs for field in fields[:len(args)])):
                    return None
                args = [args[i] if i < len(args) else kwargs.get(field)
                        for i, field in enumerate(fields)]
                kwargs = {}
            result.append({"op": node["op"],
                           "target": None if node["op"] == "placeholder" else node["target"],
                           "args": encode(args), "kwargs": encode(kwargs),
                           "results": specs(node)})
    except (KeyError, TypeError, ValueError):
        return None
    return result


def module_records(module):
    """Observe an owned FX body without stamping or changing its metadata."""
    from torch.fx import Node
    from m2m.capture.trace import _literal, _types

    ids = {node: f"body:n{i}" for i, node in enumerate(module.graph.nodes)}
    records = []
    for node in module.graph.nodes:
        def encode(value):
            if isinstance(value, Node):
                values = _types(value.meta.get("val"), ids[value])
                index = node.args[1] if (str(node.target) == "<built-in function getitem>"
                                        and len(node.args) == 2 and type(node.args[1]) is int) else 0
                selected = values[index] if -len(values) <= index < len(values) else {}
                return {"node_id": ids[value], "value_id": selected.get("id")}
            if isinstance(value, (tuple, list)):
                return [encode(item) for item in value]
            if isinstance(value, dict):
                return {str(key): encode(item) for key, item in value.items()}
            return _literal(value)
        records.append({"id": ids[node], "op": node.op, "target": str(node.target),
                        "args": encode(node.args), "kwargs": encode(node.kwargs),
                        "results": _types(node.meta.get("val"), ids[node])})
    return records


def grad_body(module, wrapper):
    """Resolve exactly one disabled call; shared and unrecognized ABIs refuse."""
    import torch
    from torch.fx import Node

    if (wrapper.op != "call_function"
            or wrapper.target is not torch.ops.higher_order.wrap_with_set_grad_enabled
            or len(wrapper.args) < 2 or wrapper.args[0] is not False or wrapper.kwargs):
        return None
    attr = wrapper.args[1]
    if not isinstance(attr, Node) or attr.op != "get_attr" or set(attr.users) != {wrapper}:
        return None
    body = module.get_submodule(str(attr.target))
    def resolve(target):
        owner = module
        for part in str(target).split("."):
            owner = getattr(owner, part)
        return owner

    references = [node for node in module.graph.nodes if node.op == "get_attr"
                  and resolve(node.target) is body]
    if len(references) != 1:
        return None
    return body


def publish(graph_module, original, regions):
    """Carry an exact normalized body plus its original ownership evidence."""
    import torch
    from m2m.capture.trace import _graph_modules, _types

    source = {node["id"]: node for node in original["nodes"]}
    for _, module in _graph_modules(graph_module):
        for wrapper in module.graph.nodes:
            body = grad_body(module, wrapper)
            if body is None:
                continue
            custom = dict(wrapper.meta.get("custom") or {})
            source_wrapper = source.get(custom.get("m2m_node_id"))
            if source_wrapper is None:
                continue
            ref = source.get(source_wrapper["args"][1].get("node_id"))
            if ref is None:
                continue
            parent_scope = source_wrapper["graph_id"].split(":", 2)[-1]
            scope = str(ref["target"]) if parent_scope == "root" else f"{parent_scope}.{ref['target']}"
            prefix = f"g:{original['stage']}:{scope}"
            source_body = [core(node) for node in original["nodes"]
                           if node["graph_id"] == prefix or node["graph_id"].startswith(prefix + ".")]
            owned_regions = [row for row in regions if row.get("source_node_id") in
                             {node["id"] for node in source_body}]
            if not owned_regions:
                continue
            # A failed publication must not leave provisional splice lineage
            # looking like ordinary ancestry in a later snapshot.
            mark_required([wrapper, *wrapper.users, *body.graph.nodes], source_wrapper["id"])
            bindings = []
            for operand in wrapper.args[2:]:
                if not hasattr(operand, "meta"):
                    break
                identities = (operand.meta.get("custom") or {}).get("m2m_lineage") or []
                values = _types(operand.meta.get("val"), "operand")
                if len(values) != 1 or not identities:
                    break
                bindings.append({"source_ids": list(identities), "spec": specs({"results": values})})
            if len(bindings) != len(wrapper.args) - 2:
                continue
            nodes = list(body.graph.nodes)
            normalized = module_records(body)
            if canonical_body(normalized) is None:
                continue
            custom[KEY] = {
                "schema": "m2m.nested_lineage.v1", "source_wrapper": core(source_wrapper),
                "torch_version": torch.__version__,
                "source_body": source_body, "normalized_body": normalized,
                "bindings": bindings,
                "lineages": [list((node.meta.get("custom") or {}).get("m2m_lineage") or [])
                             for node in nodes],
                "regions": owned_regions,
            }
            wrapper.meta["custom"] = custom


def aligned_lineages(receipt, observed):
    """Return aliases and eliminations from an exact ordered body proof."""
    if not valid_receipt(receipt) or not valid_records(observed):
        return None
    expected = receipt["normalized_body"]
    expected_form = canonical_body(expected)
    if expected_form is None or canonical_body(observed) != expected_form:
        return None
    left, aliases, guards = reduced_body(expected)
    right, right_aliases, right_guards = reduced_body(observed)
    if len(left) != len(right):
        return None
    if len(expected) != len(receipt["lineages"]):
        return None
    lineages = {node["id"]: origins for node, origins in
                zip(expected, receipt["lineages"], strict=True)}
    for identity, owner in aliases.items():
        lineages[owner] = list(dict.fromkeys([*lineages[owner], *lineages[identity]]))
    restored = {node["id"]: lineages[source["id"]]
                for source, node in zip(left, right, strict=True)}
    for identity, owner in right_aliases.items():
        restored[identity] = restored[owner]
    eliminated = [identity for guard in guards for identity in lineages[guard]]
    return restored, eliminated, right_guards


def call_receipt(wrapper, body):
    """Torch may put HOP custom metadata on its selectors and inner calls."""
    from m2m.capture.trace import _digest

    present = [custom[KEY] for node in [wrapper, *wrapper.users, *body.graph.nodes]
               if KEY in (custom := node.meta.get("custom") or {})]
    if not present or any(not valid_receipt(row) for row in present):
        return None
    unique = {_digest(row): row for row in present}
    return next(iter(unique.values())) if len(unique) == 1 else None


def restore(graph_module):
    """Restore inner identities only after rechecking the full body and caller."""
    import torch
    from m2m.capture.trace import _graph_modules, _types

    for _, module in _graph_modules(graph_module):
        for wrapper in module.graph.nodes:
            body = grad_body(module, wrapper)
            if body is None:
                continue
            receipt = call_receipt(wrapper, body)
            if (not valid_receipt(receipt) or receipt.get("torch_version") != torch.__version__
                    or any(region.get("status") != "validated" for region in receipt["regions"])):
                continue
            observed = module_records(body)
            alignment = aligned_lineages(receipt, observed)
            if alignment is None:
                continue
            bindings = receipt["bindings"]
            if len(bindings) != len(wrapper.args) - 2:
                continue
            valid = True
            for operand, binding in zip(wrapper.args[2:], bindings, strict=True):
                if not hasattr(operand, "meta"):
                    valid = False
                    break
                lineage = set((operand.meta.get("custom") or {}).get("m2m_lineage") or [])
                if (not lineage.intersection(binding["source_ids"])
                        or specs({"results": _types(operand.meta.get("val"), "operand")}) != binding["spec"]):
                    valid = False
                    break
            if not valid:
                continue
            restored, _, _ = alignment
            custom = dict(wrapper.meta.get("custom") or {})
            custom[KEY] = receipt
            wrapper.meta["custom"] = custom
            mark_required([wrapper, *wrapper.users, *body.graph.nodes], receipt["source_wrapper"]["id"])
            for node, row in zip(body.graph.nodes, observed, strict=True):
                lineage = restored.get(row["id"], [])
                node.meta = dict(node.meta)
                custom = dict(node.meta.get("custom") or {})
                # Only the proven per-node identities survive. Export may have
                # copied any earlier stage's wrapper tags onto this call.
                custom["m2m_lineage"] = list(lineage)
                # A wrapper's receipt copied wholesale into every inner node is
                # not evidence that the node is another region invocation.
                custom.pop(KEY, None)
                node.meta["custom"] = custom


def snapshot_receipts(graph_module):
    from m2m.capture.trace import _graph_modules

    entries = []
    for _, module in _graph_modules(graph_module):
        for node in module.graph.nodes:
            custom = node.meta.get("custom") or {}
            if (node.op != "call_function" or str(node.target) != "wrap_with_set_grad_enabled"
                    or KEY not in custom):
                continue
            body = grad_body(module, node)
            receipt = call_receipt(node, body) if body is not None else custom[KEY]
            entries.append({"wrapper_node_id": node.meta["_m2m_node_id"], "receipt": receipt})
    return entries


def snapshot_caller(snapshot, scope):
    """Unique structural ownership for either supported disabled region ABI."""
    parent, _, local = scope.rpartition(".")
    parent_id = f"g:{snapshot['stage']}:{parent or 'root'}"
    attrs = [node for node in snapshot["nodes"] if node["graph_id"] == parent_id
             and node["op"] == "get_attr" and node["target"] == (local or scope)]
    if len(attrs) != 1:
        return None
    attr = attrs[0]
    users = [node for node in snapshot["nodes"] if node["graph_id"] == parent_id
             and any(isinstance(arg, dict) and arg.get("node_id") == attr["id"]
                     for arg in node.get("args") or [])]
    if len(users) != 1:
        return None
    wrapper = users[0]
    args = wrapper.get("args") or []
    if wrapper.get("kwargs") or wrapper["op"] != "call_function":
        return None
    if wrapper["target"] == "wrap_with_set_grad_enabled":
        if len(args) < 2 or args[0] is not False:
            return None
        offset = 2
    elif wrapper["target"] == "wrap_with_autocast":
        if (len(args) < 5 or not isinstance(args[0], str) or args[2] is not False
                or not (args[3] is None or type(args[3]) is bool)
                or not (args[1] is None or isinstance(args[1], dict)
                        and set(args[1]) == {"kind", "value"} and args[1]["kind"] == "dtype")):
            return None
        offset = 5
    else:
        return None
    if (len(attr.get("results") or []) != 1 or not isinstance(args[offset - 1], dict)
            or args[offset - 1] != {"node_id": attr["id"], "value_id": attr["results"][0]["id"]}):
        return None
    return wrapper, offset


def relations(source, dest, established):
    """Validate the receipt against both observed graphs, never just its tags."""
    src = {node["id"]: node for node in source["nodes"]}
    dst = {node["id"]: node for node in dest["nodes"]}
    mapped = {}
    for row in established:
        for identity in row["source_ids"]:
            mapped.setdefault(identity, set()).update(row["destination_ids"])
    result = []
    if source.get("runtime_versions") != dest.get("runtime_versions"):
        return result
    for entry in dest.get("nested_lineage", []):
        if not isinstance(entry, dict) or not valid_receipt(entry.get("receipt")):
            continue
        receipt = entry["receipt"]
        if (receipt.get("torch_version") != (source.get("runtime_versions") or {}).get("torch")
                or any(region.get("status") != "validated" for region in receipt["regions"])):
            continue
        sw = receipt["source_wrapper"]
        dw = dst.get(entry["wrapper_node_id"])
        if src.get(sw["id"]) is None or core(src[sw["id"]]) != sw or dw is None:
            continue
        sb = receipt["source_body"]
        if any(identity["id"] not in src or core(src[identity["id"]]) != identity for identity in sb):
            continue
        source_scope = sb[0]["graph_id"].split(":", 2)[-1] if sb else None
        source_call = snapshot_caller(source, source_scope) if source_scope else None
        if source_call is None or source_call[0]["id"] != sw["id"]:
            continue
        prefix = f"g:{source['stage']}:{source_scope}"
        actual_source_body = [core(node) for node in source["nodes"]
                              if node["graph_id"] == prefix or node["graph_id"].startswith(prefix + ".")]
        if actual_source_body != sb:
            continue
        # Derive destination ownership from the actual wrapper operand.
        args = dw.get("args") or []
        if (dw["target"] != "wrap_with_set_grad_enabled" or len(args) < 2
                or args[0] is not False or dw.get("kwargs") or not isinstance(args[1], dict)):
            continue
        attr = dst.get(args[1].get("node_id"))
        if attr is None or attr["op"] != "get_attr":
            continue
        parent = dw["graph_id"].split(":", 2)[-1]
        scope = attr["target"] if parent == "root" else f"{parent}.{attr['target']}"
        dest_call = snapshot_caller(dest, scope)
        if dest_call is None or dest_call[0]["id"] != dw["id"]:
            continue
        db = [node for node in dest["nodes"] if node["graph_id"] == f"g:{dest['stage']}:{scope}"]
        users = [node for node in dest["nodes"] if node["op"] == "call_function"
                 and any(isinstance(arg, dict) and arg.get("node_id") == attr["id"]
                         for arg in node.get("args") or [])]
        if len(users) != 1 or users[0]["id"] != dw["id"]:
            continue
        alignment = aligned_lineages(receipt, db)
        if alignment is None:
            continue
        if len(sw["args"]) != len(args) or specs(sw) != specs(dw):
            continue
        inputs = [node for node in db if node["op"] == "placeholder"]
        if len(inputs) != len(args) - 2:
            continue
        valid = True
        for left, right, placeholder in zip(sw["args"][2:], args[2:], inputs, strict=True):
            if not isinstance(left, dict) or not isinstance(right, dict):
                valid = False
                break
            sp, dp = src.get(left.get("node_id")), dst.get(right.get("node_id"))
            if (sp is None or dp is None or mapped.get(sp["id"]) != {dp["id"]}
                    or len(specs(sp)) != 1 or specs(sp) != specs(dp) or specs(dp) != specs(placeholder)
                    or left.get("value_id") != sp["results"][0]["id"]
                    or right.get("value_id") != dp["results"][0]["id"]):
                valid = False
                break
        if not valid:
            continue
        proof = {"exact_typed_body": True, "exact_caller_bindings": True,
                 "normalization_receipt": True, "ordered_outputs": True,
                 "identity_cast_aliases": [
                     {"node_id": identity, "input_node_id": owner,
                      "no_copy_same_dtype": True, "exact_tensor_metadata": True}
                     for identity, owner in reduced_body(receipt["normalized_body"])[1].items()],
                 "satisfied_metadata_guard_ids": reduced_body(receipt["normalized_body"])[2]}
        restored, eliminated, guards = alignment
        for node in db:
            lineage = restored.get(node["id"], [])
            origins = sorted(set(lineage).intersection(src))
            if origins:
                result.append({"source_ids": origins, "destination_ids": [node["id"]],
                               "kind": "nested_structural_identity", "proof": proof})
        result.append({"source_ids": [sw["id"]], "destination_ids": [dw["id"]],
                       "kind": "nested_call_identity", "proof": proof})
        if eliminated:
            result.append({"source_ids": sorted(set(eliminated).intersection(src)),
                           "destination_ids": [], "kind": "eliminated",
                           "proof": {**proof, "satisfied_typed_metadata_guards": True}})
        for region in receipt["regions"]:
            for elimination in region["eliminations"]:
                origins = sorted(set(elimination["source_ids"]).intersection(src))
                if origins:
                    result.append({"source_ids": origins, "destination_ids": [],
                                   "kind": "eliminated", "proof": {**proof, **elimination["proof"]}})
    return result


def observed_certificate(snapshot, entry):
    """Validate an invocation in a snapshot independently of inherited tags."""
    nodes = {node['id']: node for node in snapshot['nodes']}
    wrapper = nodes.get(entry['wrapper_node_id'])
    receipt = entry['receipt']
    if wrapper is None:
        return None
    args = wrapper.get('args') or []
    if len(args) < 2 or not isinstance(args[1], dict):
        return None
    attr = nodes.get(args[1].get('node_id'))
    if attr is None or attr['op'] != 'get_attr':
        return None
    parent = wrapper['graph_id'].split(':', 2)[-1]
    scope = attr['target'] if parent == 'root' else f"{parent}.{attr['target']}"
    caller = snapshot_caller(snapshot, scope)
    if caller is None or caller[0]['id'] != wrapper['id'] or caller[1] != 2:
        return None
    body = [node for node in snapshot['nodes'] if node['graph_id'] == f"g:{snapshot['stage']}:{scope}"]
    if aligned_lineages(receipt, body) is None or len(receipt['bindings']) != len(args) - 2:
        return None
    for arg, binding in zip(args[2:], receipt['bindings'], strict=True):
        operand = nodes.get(arg.get('node_id')) if isinstance(arg, dict) else None
        if (operand is None or specs(operand) != binding['spec'] or len(operand['results']) != 1
                or arg.get('value_id') != operand['results'][0]['id']
                or not set(binding['source_ids']).intersection(
                    [operand['id'], *operand.get('origin_node_ids', [])])):
            return None
    return {wrapper['id'], *(node['id'] for node in body)}


def certificate_entries(snapshot):
    """Reject missing, malformed or conflicting rows before using ownership."""
    from m2m.capture.trace import _digest

    entries = snapshot.get('nested_lineage', [])
    if not isinstance(entries, list):
        return None
    unique, callers = {}, {}
    for entry in entries:
        if (not isinstance(entry, dict) or not isinstance(entry.get('wrapper_node_id'), str)
                or not valid_receipt(entry.get('receipt'))
                or entry['receipt']['torch_version'] != (snapshot.get('runtime_versions') or {}).get('torch')):
            return None
        identity = entry['receipt']['source_wrapper']['id']
        if (identity in unique and _digest(unique[identity]) != _digest(entry)
                or entry['wrapper_node_id'] in callers and callers[entry['wrapper_node_id']] != identity):
            return None
        unique[identity] = entry
        callers[entry['wrapper_node_id']] = identity
    for node in snapshot['nodes']:
        # Older snapshots predate the explicit requirement marker. Moved
        # descendant identities inside a surviving scope are still provisional.
        scope = str(node.get('graph_id') or f"g:{snapshot['stage']}:root").split(':', 2)[-1]
        moved_origins = [origin for origin in node.get('origin_node_ids', [])
                         if isinstance(origin, str) and len(origin.split(':', 3)) == 4
                         and origin.split(':', 3)[2].startswith(scope + '.')]
        if scope != 'root' and moved_origins:
            caller = snapshot_caller(snapshot, scope)
            if caller is None or caller[0]['id'] not in callers:
                return None
        required = node.get('nested_lineage_required')
        if required is not None and (not isinstance(required, str) or required not in unique):
            return None
    return unique


def certificate_policy(source, dest):
    """Return only the IDs owned by certificates; unrelated scopes stay direct.

    A lost receipt is a diagnostic even if all restored IDs remain on the
    nodes. No other structural inference may replace this required evidence.
    """
    from m2m.capture.trace import _digest

    if 'flattened_nested_lineage' in dest:
        return flattened_certificate_policy(source, dest)
    left, right = certificate_entries(source), certificate_entries(dest)
    if left is None or right is None:
        return set(), set(), 'missing, malformed or conflicting nested lineage certificate'
    if any(identity not in right or _digest(entry['receipt']) != _digest(right[identity]['receipt'])
           for identity, entry in left.items()):
        return set(), set(), 'nested lineage certificate lost or changed between snapshots'
    protected = []
    for snapshot, entries in ((source, left), (dest, right)):
        identities = {node['id'] for node in snapshot['nodes'] if node.get('nested_lineage_required')}
        for entry in entries.values():
            owned = observed_certificate(snapshot, entry)
            if owned is None:
                return set(), set(), 'nested lineage certificate does not match the observed invocation'
            identities.update(owned)
        # The original graph predates publication, but its source body and
        # caller still require the certificate carried by the later graph.
        all_ids = {node['id'] for node in snapshot['nodes']}
        for entry in right.values():
            receipt = entry['receipt']
            identities.update(all_ids.intersection(
                [receipt['source_wrapper']['id'], *(node['id'] for node in receipt['source_body'])]))
        calls = set(identities)
        identities.update(node['id'] for node in snapshot['nodes']
                          if node['target'] == '<built-in function getitem>' and node['args']
                          and isinstance(node['args'][0], dict)
                          and node['args'][0].get('node_id') in calls)
        protected.append(identities)
    return *protected, None


def certify_flattened_regions(source, exported):
    """Seal the owned inline/decomposition transition before its final snapshot.

    Flattening deliberately removes the certified caller. Its replacement is
    an explicit transition receipt binding both complete observed graphs, not
    permission to fall back to stale inner identities when metadata is lost.
    """
    from copy import deepcopy
    from m2m.capture.trace import snapshot_exported_program

    if not source.get('nested_lineage'):
        return
    _, _, error = certificate_policy(source, source)
    if error:
        return
    observed = snapshot_exported_program(exported, stage='prepared')
    exported._m2m_flattened_lineage = deepcopy({
        'schema': 'm2m.flattened_nested_lineage.v1',
        'runtime_versions': source['runtime_versions'],
        'source_nodes': [core(node) for node in source['nodes']],
        'destination_nodes': [core(node) for node in observed['nodes']],
        'certificates': source['nested_lineage'],
    })


def flattened_certificate_policy(source, dest):
    from m2m.capture.trace import _digest

    receipt = dest.get('flattened_nested_lineage')
    if (not isinstance(receipt, dict) or receipt.get('schema') != 'm2m.flattened_nested_lineage.v1'
            or receipt.get('runtime_versions') != source.get('runtime_versions')
            or source.get('runtime_versions') != dest.get('runtime_versions')
            or not valid_records(receipt.get('source_nodes'), scoped=True)
            or not valid_records(receipt.get('destination_nodes'), scoped=True)
            or receipt['source_nodes'] != [core(node) for node in source['nodes']]
            or receipt['destination_nodes'] != [core(node) for node in dest['nodes']]
            or _digest(receipt.get('certificates')) != _digest(source.get('nested_lineage'))):
        return set(), set(), 'missing or mismatched flattened nested lineage certificate'
    entries = certificate_entries(source)
    if not entries or any(observed_certificate(source, entry) is None for entry in entries.values()):
        return set(), set(), 'invalid source certificate for flattened nested lineage'
    if any(node.get('nested_lineage_required') not in {None, *entries} for node in dest['nodes']):
        return set(), set(), 'unbound flattened nested lineage requirement'
    return set(), set(), None
