"""Unknown tree storage cannot invoke iterator, property or arithmetic hooks."""

import pytest
from torch.utils._pytree import tree_structure

from m2m.capture.call_contract import _tree


class Calls:
    count = 0


class OpaqueChildren:
    def __iter__(self):
        Calls.count += 1
        raise AssertionError("opaque iterator executed")


class ListSubclass(list):
    def __iter__(self):
        Calls.count += 1
        raise AssertionError("subclass iterator executed")


class OpaqueChild:
    @property
    def num_nodes(self):
        Calls.count += 1
        raise AssertionError("opaque child property executed")


class OpaqueCount:
    def __radd__(self, other):
        Calls.count += 1
        raise AssertionError("opaque count arithmetic executed")


class EqualityMeta(type):
    def __eq__(self, other):
        Calls.count += 1
        raise AssertionError("opaque metaclass comparison executed")


class EqualityChild(metaclass=EqualityMeta):
    pass


def set_children(spec, value):
    # The same known public TreeSpec record has dataclass storage on older
    # Torch and direct slots on current Torch; never evaluate its children.
    if hasattr(spec, "__dict__"):
        object.__setattr__(spec, "children_specs", value)
    else:
        object.__setattr__(spec, "_children", value)


@pytest.mark.parametrize("children", [OpaqueChildren(), ListSubclass([]), (), object()])
def test_unknown_child_container_refuses_before_iteration(children):
    spec = tree_structure((1,))
    set_children(spec, children)
    Calls.count = 0
    with pytest.raises(ValueError, match="child container"):
        _tree(spec)
    assert Calls.count == 0


@pytest.mark.parametrize("child", [OpaqueChild(), EqualityChild(), object()])
def test_unknown_child_refuses_before_property_or_type_comparison(child):
    spec = tree_structure((1,))
    set_children(spec, [child])
    Calls.count = 0
    with pytest.raises(ValueError, match="child kind"):
        _tree(spec)
    assert Calls.count == 0


def test_known_child_with_opaque_count_refuses_before_parent_sum():
    spec = tree_structure((1,))
    child = spec.children_specs[0]
    previous = child.num_nodes
    try:
        object.__setattr__(child, "num_nodes", OpaqueCount())
        Calls.count = 0
        with pytest.raises(ValueError, match="tree counts"):
            _tree(spec)
        assert Calls.count == 0
    finally:
        # Torch may intern leaf specs; never poison later independent captures.
        object.__setattr__(child, "num_nodes", previous)


def test_unknown_root_refuses_without_metaclass_comparison():
    Calls.count = 0
    with pytest.raises(ValueError, match="TreeSpec"):
        _tree(EqualityChild())
    assert Calls.count == 0


def test_nested_ordinary_tree_counts_are_retained():
    spec = tree_structure(({"x": [1, 2]},))
    frozen = _tree(spec)
    assert frozen[-1] == (5, 2, 1)
