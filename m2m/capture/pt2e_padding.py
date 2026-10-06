"""Canonicalize static Conv2d string padding before PT2E observes tensor edges."""

from __future__ import annotations

from typing import Any


def normalize_conv2d_padding(graph_module: Any) -> list[dict[str, Any]]:
    """Use numeric Conv2d, explicitly padding asymmetric ``same`` inputs.

    TorchAO's portable quantizer and Conv/BN folder consume the numeric-padding
    overload. Leaving ``aten.conv2d.padding`` untouched silently excludes it
    from W8A8 even though its static semantics are identical. Mutating the
    existing Conv node preserves its source identity. A generated zero-pad
    carries ancestry, not a fabricated one-to-one source-node identity.
    """
    import torch
    from torch._subclasses.fake_tensor import FakeTensor
    from torch.fx import Node

    overload = torch.ops.aten.conv2d.padding
    graph = graph_module.graph
    normalized = []
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target != overload:
            continue
        arguments = list(overload._schema.arguments)
        if (
            len(node.args) > len(arguments)
            or set(node.kwargs) - {argument.name for argument in arguments}
            or any(argument.name in node.kwargs for argument in arguments[: len(node.args)])
        ):
            raise ValueError("Conv2d padding has extra or duplicate schema arguments")
        values = {}
        for index, argument in enumerate(arguments):
            if index < len(node.args):
                values[argument.name] = node.args[index]
            elif argument.name in node.kwargs:
                values[argument.name] = node.kwargs[argument.name]
            elif argument.has_default_value():
                values[argument.name] = argument.default_value
            else:
                raise ValueError(f"Conv2d padding lacks {argument.name}")
        padding = values["padding"]
        if padding not in ("valid", "same"):
            raise ValueError(f"unsupported Conv2d string padding: {padding!r}")
        pad_input = []
        numeric_padding = [0, 0]
        if padding == "same":
            weight = values["weight"]
            shape = getattr(weight.meta.get("val"), "shape", None) if isinstance(weight, Node) else None
            dilation = values["dilation"]
            stride = values["stride"]
            if (
                shape is None
                or len(shape) != 4
                or any(type(size) is not int or size <= 0 for size in shape)
                or len(dilation) != 2
                or any(type(d) is not int or d <= 0 for d in dilation)
                or list(stride) != [1, 1]
            ):
                raise ValueError("same-padding W8A8 requires static 2D weights, positive dilation and unit stride")
            totals = [dilation[axis] * (shape[axis + 2] - 1) for axis in range(2)]
            if all(total % 2 == 0 for total in totals):
                numeric_padding = [total // 2 for total in totals]
            else:
                # PyTorch SAME_UPPER: extra padding is on the right/bottom.
                pad_input = [value for total in reversed(totals) for value in (total // 2, total - total // 2)]
                input_node = values["input"]
                input_value = input_node.meta.get("val") if isinstance(input_node, Node) else None
                if (
                    not isinstance(input_value, FakeTensor)
                    or len(input_value.shape) != 4
                    or any(type(size) is not int or size <= 0 for size in input_value.shape)
                ):
                    raise ValueError("asymmetric Conv2d padding requires a static input FakeTensor")
                # Execute the actual ATen padding operation in the exported
                # input's FakeTensorMode. The generated edge must have the
                # exact dtype/device/shape metadata PT2E uses for eligibility;
                # a guessed shape or a real tensor would not be source proof.
                with input_value.fake_mode:
                    padded_value = torch.ops.aten.constant_pad_nd.default(input_value, pad_input, 0.0)
                if (
                    not isinstance(padded_value, FakeTensor)
                    or padded_value.fake_mode is not input_value.fake_mode
                    or padded_value.dtype != input_value.dtype
                    or padded_value.device != input_value.device
                    or len(padded_value.shape) != 4
                    or any(type(size) is not int or size <= 0 for size in padded_value.shape)
                ):
                    raise ValueError("asymmetric Conv2d padding has invalid generated FakeTensor metadata")
                with graph.inserting_before(node):
                    padded = graph.call_function(
                        torch.ops.aten.constant_pad_nd.default, (values["input"], pad_input, 0.0)
                    )
                padded.meta["val"] = padded_value
                custom = node.meta.get("custom") or {}
                padded.meta["custom"] = {
                    "m2m_lineage": list(custom.get("m2m_lineage") or ()),
                    "m2m_transform": "conv2d_string_padding",
                }
                values["input"] = padded
        node.target = torch.ops.aten.conv2d.default
        node.args = (
            values["input"],
            values["weight"],
            values["bias"],
            values["stride"],
            numeric_padding,
            values["dilation"],
            values["groups"],
        )
        node.kwargs = {}
        normalized.append(
            {
                "node": node.name,
                "source_overload": str(overload),
                "padding": padding,
                "numeric_padding": numeric_padding,
                "explicit_input_padding_lrtb": pad_input,
                "source_ids": list((node.meta.get("custom") or {}).get("m2m_lineage") or ()),
            }
        )
    if normalized:
        graph.lint()
        graph_module.recompile()
    return normalized
