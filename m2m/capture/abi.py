"""Identify captured constants consistently across bundle data and receipts."""

from collections.abc import Mapping


def is_lifted_tensor_argument(entry: Mapping) -> bool:
    name = str(entry.get("name", ""))
    return entry.get("kind") == "constant" or (
        entry.get("kind") not in {"param", "buffer"}
        and name.startswith("c_")
        and "lifted_tensor" in name
    )
