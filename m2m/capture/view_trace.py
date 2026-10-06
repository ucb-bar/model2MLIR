"""Exact view ownership after functionalization regenerates an updated alias.

This is source correspondence, not a proof of arbitrary functionalization.
Selected operator schemas supply storage alias/write ownership; the matched destination must
be the identical static view of the storage version attributed to the last write.
A write through a slice may return only the slice while functionalization
reconstructs the full base. This does not certify the writer's transformation.
"""

from __future__ import annotations


class SourceStorageAliases:
    """Conservative storage ownership from exact selected operator schemas.

    New captures record the actual registered schemas, so offline inspection
    does not need to register framework/custom libraries a second time. A live
    schema, when available, must agree; old captures use only the exact runtime.
    The enclosing capture receipt, not this parser, attests those schema bytes.
    """

    def __init__(self, source: dict):
        import torch

        self.source = source
        self.valid = (source.get("runtime_versions") or {}).get("torch") == torch.__version__
        self.nodes = {node["id"]: node for node in source["nodes"]}
        self.inputs: dict[str, list[str]] = {}
        for edge in source.get("edges") or []:
            self.inputs.setdefault(edge["consumer_node_id"], []).append(edge["producer_node_id"])
        self.schemas = {}
        self.roots: dict[str, str | None] = {}

    def schema(self, node):
        import torch

        if not self.valid:
            return None
        target = str(node.get("target"))
        if target in self.schemas:
            return self.schemas[target]
        parts = target.split(".")
        if len(parts) != 3:
            return None
        captured = (self.source.get("operator_schemas") or {}).get(target)
        selected = None
        if captured is not None:
            if not isinstance(captured, str):
                return None
            try:
                selected = torch._C.parse_schema(captured)
            except RuntimeError:
                return None
            qualified = selected.name.replace("::", ".")
            if qualified + "." + (selected.overload_name or "default") != target:
                return None
        try:
            operation = getattr(getattr(getattr(torch.ops, parts[0]), parts[1]), parts[2])
            live = operation._schema if str(operation) == target else None
        except AttributeError:
            live = None
        if captured is not None and live is not None and str(selected) != str(live):
            return None
        self.schemas[target] = selected if captured is not None else live
        return self.schemas[target]

    @staticmethod
    def argument(node, index, name):
        if name in (node.get("kwargs") or {}):
            return node["kwargs"][name]
        args = node.get("args") or []
        return args[index] if index < len(args) else None

    def tensor_ref(self, value):
        if not isinstance(value, dict) or set(value) != {"node_id", "value_id"}:
            return None
        producer = self.nodes.get(value["node_id"])
        results = producer.get("results") or [] if producer else []
        if len(results) != 1 or results[0].get("id") != value["value_id"]:
            return None
        return producer

    def fresh_dtype_conversion(self, node):
        """Recognize an exact ``to.dtype`` that must allocate new storage.

        The selected schema permits an alias for the no-op case. A changed
        dtype cannot share tensor storage; keep all other ``to`` forms unknown
        unless their existing identity proof applies.
        """
        if node.get("op") != "call_function" or node.get("target") != "aten.to.dtype":
            return False
        args = node.get("args") or []
        kwargs = node.get("kwargs") or {}
        if not isinstance(args, list) or not isinstance(kwargs, dict) or not 2 <= len(args) <= 5:
            return False
        if set(kwargs) - {"non_blocking", "copy", "memory_format"}:
            return False
        if any(name in kwargs for name in ("non_blocking", "copy", "memory_format")[: len(args) - 2]):
            return False
        producer = self.tensor_ref(args[0])
        if producer is None or producer.get("graph_id") != node.get("graph_id"):
            return False
        source = _tensor_value(producer)
        result = _tensor_value(node)

        def complete_tensor(value):
            if value is None or not isinstance(value.get("id"), str) or not value["id"]:
                return False
            if any(
                not isinstance(value.get(field), str) or not value[field]
                for field in ("dtype", "storage_dtype", "device", "layout")
            ):
                return False
            if "compute_dtype" not in value or (
                value["compute_dtype"] is not None and not isinstance(value["compute_dtype"], str)
            ):
                return False
            if value["storage_dtype"] != value["dtype"] or value["layout"] != "torch.strided":
                return False
            shape, stride = value.get("shape"), value.get("stride")
            if not isinstance(shape, list) or not isinstance(stride, list) or len(shape) != len(stride):
                return False
            return all(
                type(dim) in {int, str} and (dim >= 0 if type(dim) is int else bool(dim)) for dim in [*shape, *stride]
            )

        if not complete_tensor(source) or not complete_tensor(result) or source["dtype"] == result["dtype"]:
            return False
        requested = args[1]
        if requested != {"kind": "dtype", "value": f"torch.{result.get('dtype')}"}:
            return False
        if (args[2] if len(args) > 2 else kwargs.get("non_blocking", False)) is not False:
            return False
        if (args[3] if len(args) > 3 else kwargs.get("copy", False)) is not False:
            return False
        if (args[4] if len(args) > 4 else kwargs.get("memory_format")) is not None:
            return False
        if not all(source[field] == result[field] for field in ("kind", "shape", "device", "layout", "stride")):
            return False
        incoming = [edge for edge in self.source.get("edges") or [] if edge.get("consumer_node_id") == node["id"]]
        return (
            len(incoming) == 1
            and incoming[0].get("producer_node_id") == producer["id"]
            and incoming[0].get("producer_value_id") == source["id"]
            and incoming[0].get("argument_path") == "args/0"
            and incoming[0].get("value_kind") == "tensor"
            and incoming[0].get("dtype") == source["dtype"]
            and incoming[0].get("shape") == source["shape"]
        )

    def root(self, node):
        from m2m.capture.trace import _forward_identity_value

        identity = node["id"]
        if not self.valid:
            return None
        if identity in self.roots:
            return self.roots[identity]
        self.roots[identity] = None
        if node["op"] in {"placeholder", "get_attr"}:
            self.roots[identity] = identity
            return identity
        spec = self.schema(node)
        if spec is None or len(spec.returns) != 1:
            return None
        if node["target"] in {"aten.to.dtype", "aten.to.dtype_layout"}:
            exact = _forward_identity_value(self.source, node)
            if exact is not None:
                self.roots[identity] = self.root(exact[0])
            elif self.fresh_dtype_conversion(node):
                self.roots[identity] = identity
            return self.roots[identity]
        alias = spec.returns[0].alias_info
        if alias is None:
            self.roots[identity] = identity
            return identity
        owners = []
        for index, arg in enumerate(spec.arguments):
            info = arg.alias_info
            if info and alias.before_set.intersection(info.before_set):
                owner = self.tensor_ref(self.argument(node, index, arg.name))
                if owner is None:
                    return None
                owners.append(self.root(owner))
        if len(owners) == 1:
            self.roots[identity] = owners[0]
        return self.roots[identity]

    def last_writer(self, node):
        """Return (storage root, last writer), or None for unknown ownership."""
        root = self.root(node)
        if root is None:
            return None
        writers = []
        for prior in self.source["nodes"]:
            if prior["graph_id"] != node["graph_id"]:
                continue
            if prior["ordinal"] >= node["ordinal"]:
                break
            spec = self.schema(prior)
            if spec is None:
                for owner_id in self.inputs.get(prior["id"], ()):
                    owner = self.nodes.get(owner_id)
                    if owner and self.root(owner) == root:
                        return None
                continue
            for index, arg in enumerate(spec.arguments):
                info = arg.alias_info
                if info is not None and info.is_write:
                    owner = self.tensor_ref(self.argument(prior, index, arg.name))
                    if owner is None:
                        return None
                    if self.root(owner) == root:
                        writers.append(prior)
        writer = writers[-1] if writers else None
        if writer is not None and (self.root(writer) != root or _tensor_value(writer) is None):
            return None
        return root, writer


def _tensor_value(node):
    values = node.get("results") or []
    return values[0] if len(values) == 1 and values[0].get("kind") == "tensor" else None


def replayed_view_relations(source: dict, dest: dict, relations: list[dict], consumed: set[str]) -> list[dict]:
    aliases = SourceStorageAliases(source)
    if not aliases.valid or source.get("runtime_versions") != dest.get("runtime_versions"):
        return []
    mapped: dict[str, set[str]] = {}
    for relation in relations:
        for identity in relation["source_ids"]:
            mapped.setdefault(identity, set()).update(relation["destination_ids"])

    def typed(value):
        return {key: field for key, field in value.items() if key != "id"}

    def view_key(node, input_value):
        shape = input_value.get("shape") or []
        if not shape or any(type(dim) is not int or dim < 0 for dim in shape):
            return None
        args = node.get("args") or []
        if node.get("kwargs"):
            return None
        rank = len(shape)
        if node["target"] == "aten.unsqueeze.default" and len(args) == 2:
            axis = args[1]
            if type(axis) is int and -rank - 1 <= axis <= rank:
                return ("insert_axis", axis % (rank + 1))
        if node["target"] == "aten.transpose.int" and len(args) == 3:
            if all(type(axis) is int and -rank <= axis < rank for axis in args[1:]):
                order = list(range(rank))
                first, second = (axis % rank for axis in args[1:])
                order[first], order[second] = order[second], order[first]
                return ("permute", tuple(order))
        if node["target"] == "aten.permute.default" and len(args) == 2:
            axes = args[1]
            if (
                isinstance(axes, list)
                and len(axes) == rank
                and all(type(axis) is int and -rank <= axis < rank for axis in axes)
                and set(axis % rank for axis in axes) == set(range(rank))
            ):
                return ("permute", tuple(axis % rank for axis in axes))
        return None

    destination_nodes = {node["id"]: node for node in dest["nodes"]}
    source_uses = {}
    destination_uses = {}
    for edge in source.get("edges") or []:
        source_uses.setdefault(edge["producer_node_id"], []).append(edge)
    for edge in dest.get("edges") or []:
        destination_uses.setdefault(edge["producer_node_id"], []).append(edge)
    inferred = []
    for node in source["nodes"]:
        if node["id"] in consumed or node["target"] not in {
            "aten.unsqueeze.default",
            "aten.transpose.int",
            "aten.permute.default",
        }:
            continue
        args = node.get("args") or []
        producer = aliases.tensor_ref(args[0]) if args else None
        if producer is None or _tensor_value(producer) is None or _tensor_value(node) is None:
            continue
        state = aliases.last_writer(node)
        expected = view_key(node, _tensor_value(producer))
        if state is None or state[1] is None or expected is None:
            continue
        root, writer = state
        candidates = []
        for candidate in dest["nodes"]:
            cargs = candidate.get("args") or []
            ref = cargs[0] if cargs else None
            if (
                not isinstance(ref, dict)
                or set(ref) != {"node_id", "value_id"}
                or ref["node_id"] not in mapped.get(writer["id"], ())
            ):
                continue
            owner = destination_nodes[ref["node_id"]]
            observed_input, observed_output = _tensor_value(owner), _tensor_value(candidate)
            if (
                observed_input is None
                or observed_output is None
                or ref["value_id"] != observed_input["id"]
                or candidate["graph_id"].split(":", 2)[-1] != node["graph_id"].split(":", 2)[-1]
                or typed(observed_input) != typed(_tensor_value(producer))
                or typed(observed_output) != typed(_tensor_value(node))
                or view_key(candidate, observed_input) != expected
            ):
                continue
            uses = source_uses.get(node["id"], ())
            matched_uses = []
            for use in uses:
                source_output = _tensor_value(node)
                if (
                    use.get("producer_value_id") != source_output["id"]
                    or use.get("dtype") != source_output.get("dtype")
                    or use.get("shape") != source_output.get("shape")
                ):
                    break
                matches = [
                    edge
                    for edge in destination_uses.get(candidate["id"], ())
                    if edge["consumer_node_id"] in mapped.get(use["consumer_node_id"], ())
                    and edge.get("argument_path") == use.get("argument_path")
                    and edge.get("producer_value_id") == observed_output["id"]
                    and edge.get("dtype") == use.get("dtype")
                    and edge.get("shape") == use.get("shape")
                ]
                if len(matches) != 1:
                    break
                matched_uses.append(matches[0])
            if uses and len(matched_uses) == len(uses):
                candidates.append((candidate, matched_uses))
        if len(candidates) == 1:
            candidate, uses = candidates[0]
            inferred.append(
                {
                    "source_ids": [node["id"]],
                    "destination_ids": [candidate["id"]],
                    "kind": "replayed_view",
                    "proof": {
                        "source_storage_root": root,
                        "last_writer_node_id": writer["id"],
                        "exact_static_view": list(expected),
                        "exact_tensor_metadata": True,
                        "typed_consumer_edges": uses,
                        "scope": "view source ownership after functionalization; not writer equivalence",
                    },
                }
            )
    return inferred
