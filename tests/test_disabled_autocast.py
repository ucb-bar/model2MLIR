"""Disabled autocast is precision-neutral only with the exact captured ABI."""
import operator

import pytest
import torch

from m2m.capture.trace import (
    _inline_disabled_autocast_regions, _retarget_captured_float_dtypes,
    materialize_frontend_precision,
)


class Region(torch.nn.Module):
    def __init__(self, enabled=False):
        super().__init__()
        self.enabled = enabled

    def forward(self, x):
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=self.enabled):
            return (x @ x.T).cos()


def test_disabled_region_receipt_and_exact_outputs():
    inputs = (torch.randn(4, 4),)
    model = Region()
    gm, args, original, receipt = materialize_frontend_precision(
        model, inputs, dtype=torch.float32, retarget_float_dtype_arguments=True)
    assert torch.equal(model(*inputs), gm(*args))
    row, = receipt['inlined_autocast_regions']
    assert row['source_node_id'] in {n['id'] for n in original['nodes']}
    assert row['autocast_dtype'] == 'torch.bfloat16'
    assert row['enabled'] is False
    assert receipt['staged_precision_audit']['non_target_floating_values'] == 0


def test_enabled_region_is_refused():
    with pytest.raises(ValueError, match='enabled autocast'):
        materialize_frontend_precision(Region(True), (torch.randn(4, 4),),
            dtype=torch.float32, retarget_float_dtype_arguments=True)


@pytest.mark.parametrize('change', ['dtype', 'enabled', 'cache', 'device', 'kwargs', 'arity'])
def test_unknown_signature_is_refused(change):
    gm = torch.export.export(Region(), (torch.randn(4, 4),)).module()
    node = next(n for n in gm.graph.nodes
                if n.target is torch.ops.higher_order.wrap_with_autocast)
    args = list(node.args)
    index = {'device': 0, 'dtype': 1, 'enabled': 2, 'cache': 3}
    if change in index:
        args[index[change]] = object()
    elif change == 'kwargs':
        node.kwargs = {'surprise': False}
    else:
        args = args[:4]
    node.args = tuple(args)
    with pytest.raises(ValueError, match='unknown signature'):
        _inline_disabled_autocast_regions(gm)


def test_body_with_child_module_is_refused():
    child = torch.nn.Module()
    child.register_buffer('bias', torch.ones(2))
    child.register_parameter('weight', torch.nn.Parameter(torch.ones(2)))
    child.add_module('relu', torch.nn.ReLU())
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    bias = graph.get_attr('child.bias')
    weight = graph.get_attr('child.weight')
    add = graph.call_function(torch.add, (x, bias))
    mul = graph.call_function(torch.mul, (add, weight))
    relu = graph.call_module('child.relu', (mul,))
    relu.meta['custom'] = {'m2m_node_id': 'body:relu', 'm2m_lineage': ['source:relu']}
    graph.output((relu,))
    root = torch.nn.Module()
    root.add_module('child', child)
    body = torch.fx.GraphModule(root, graph)
    outer = torch.fx.Graph()
    arg = outer.placeholder('x')
    attr = outer.get_attr('body')
    wrap = outer.call_function(torch.ops.higher_order.wrap_with_autocast,
        ('cpu', torch.bfloat16, False, False, attr, arg))
    result = outer.call_function(operator.getitem, (wrap, 0))
    outer.output(result)
    gm = torch.fx.GraphModule({'body': body}, outer)
    before = (gm.code, body.code)
    with pytest.raises(ValueError, match='unsupported execution'):
        _inline_disabled_autocast_regions(gm)
    assert (gm.code, body.code) == before


def test_unowned_dtype_literal_still_refused():
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    value = graph.call_function(tuple, ((x, torch.bfloat16),))
    graph.output(value)
    gm = torch.fx.GraphModule({}, graph)
    with pytest.raises(ValueError, match='dtype'):
        _retarget_captured_float_dtypes(gm, torch.float32, {'nodes': []})


def _autocast_body(gm):
    wrapper = next(n for n in gm.graph.nodes
                   if n.target is torch.ops.higher_order.wrap_with_autocast)
    return gm.get_submodule(wrapper.args[4].target)


def test_tensor_body_output_is_refused_before_splicing():
    gm = torch.export.export(Region(), (torch.randn(2, 2),)).module()
    body = _autocast_body(gm)
    output = next(n for n in body.graph.nodes if n.op == 'output')
    output.args = (output.args[0][0],)
    body.recompile()
    # getitem(wrapper, 0) now means the first tensor row, not the tensor.
    assert gm(torch.ones(2, 2)).shape == (2,)
    before = str(gm.graph)
    with pytest.raises(ValueError, match='tuple-output ABI'):
        _inline_disabled_autocast_regions(gm)
    assert str(gm.graph) == before


@pytest.mark.parametrize('location', ['body_output', 'unused_operand'])
def test_public_materialization_refuses_discarded_dtype_literals(location):
    inputs = (torch.randn(2, 2),)
    exported = torch.export.export(Region(), inputs)
    body = _autocast_body(exported.graph_module)
    if location == 'body_output':
        output = next(n for n in body.graph.nodes if n.op == 'output')
        output.args = ((*output.args[0], torch.bfloat16),)
    else:
        wrapper = next(n for n in exported.graph.nodes
                       if n.target is torch.ops.higher_order.wrap_with_autocast)
        wrapper.args = (*wrapper.args, torch.bfloat16)
        last_input = [n for n in body.graph.nodes if n.op == 'placeholder'][-1]
        with body.graph.inserting_after(last_input):
            body.graph.placeholder('unused_dtype')
    body.recompile()
    exported.graph_module.recompile()
    with pytest.raises(ValueError, match='dtype literal'):
        materialize_frontend_precision(exported, inputs, dtype=torch.float32,
                                      retarget_float_dtype_arguments=True)


class NoGradRegion(Region):
    @torch.no_grad()
    def forward(self, x):
        return super().forward(x)


def test_public_materialization_preserves_source_nested_graphs():
    inputs = (torch.randn(2, 2),)
    exported = torch.export.export(NoGradRegion(), inputs)
    source = exported.module()
    expected = source(*inputs).clone()
    modules = {name: module for name, module in exported.graph_module.named_modules()}
    graphs = {name: module.code for name, module in modules.items()
              if isinstance(module, torch.fx.GraphModule)}
    metadata = {name: repr([n.meta for n in module.graph.nodes])
                for name, module in modules.items() if hasattr(module, 'graph')}
    gm, args, _, receipt = materialize_frontend_precision(
        exported, inputs, dtype=torch.float32, retarget_float_dtype_arguments=True)
    assert receipt['inlined_autocast_regions']
    assert dict(exported.graph_module.named_modules()) == modules
    for name, code in graphs.items():
        assert modules[name].code == code
        assert repr([n.meta for n in modules[name].graph.nodes]) == metadata[name]
    assert torch.equal(source(*inputs), expected)
    assert torch.equal(exported.module()(*inputs), expected)
    assert torch.equal(gm(*args), expected)


@pytest.mark.parametrize('entry', ['call', 'forward', 'forward_kwargs'])
@pytest.mark.parametrize('model_type', [Region, NoGradRegion])
def test_external_enabled_autocast_is_refused_and_precondition_recorded(model_type, entry):
    model = model_type()
    inputs = (torch.randn(2, 2),)
    gm, args, _, receipt = materialize_frontend_precision(
        model, inputs, dtype=torch.float32, retarget_float_dtype_arguments=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        assert model(*inputs).dtype == torch.float32
        with pytest.raises(ValueError, match='requires ambient autocast disabled for cpu'):
            if entry == 'call':
                gm(*args)
            elif entry == 'forward':
                gm.forward(*args)
            else:
                gm.forward(x=args[0])
    assert receipt['runtime_preconditions'] == {
        'autocast_disabled_for_devices': sorted(torch._C._autocast_supported_devices()),
        'global_execution_hooks_absent': True,
        'torch_dispatch_modes_absent': True,
        'torch_function_modes_absent': True,
        'tensor_input_devices': [{'leaf_ordinal': 0, 'device': 'cpu'}]}
    assert receipt['inlined_autocast_regions'][0]['action'] == 'inlined_precision_neutral_region'
    assert torch.equal(model(*inputs), gm(*args))


def test_invalid_runtime_device_is_refused():
    gm = torch.export.export(Region(), (torch.randn(2, 2),)).module()
    wrapper = next(n for n in gm.graph.nodes
                   if n.target is torch.ops.higher_order.wrap_with_autocast)
    wrapper.args = ('not_a_device', *wrapper.args[1:])
    before = str(gm.graph)
    with pytest.raises(ValueError, match='invalid runtime device contract'):
        _inline_disabled_autocast_regions(gm)
    assert str(gm.graph) == before


def _wrap_body(body, *, repeat=False):
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    results = []
    for _ in range(2 if repeat else 1):
        attr = graph.get_attr('body')
        wrap = graph.call_function(torch.ops.higher_order.wrap_with_autocast,
            ('cpu', torch.bfloat16, False, False, attr, x))
        results.append(graph.call_function(operator.getitem, (wrap, 0)))
    graph.output(tuple(results) if repeat else results[0])
    return torch.fx.GraphModule({'body': body}, graph)


def test_body_with_rebinding_child_is_refused():
    class RebindBias(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('bias', torch.ones(2))

        def forward(self, x):
            self.bias = self.bias + 1
            return x

    child = RebindBias()
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    # Register both uses before normalization; the call rebinds the buffer
    # before the graph reads its value through the same owning child.
    graph.get_attr('child.bias')
    graph.call_module('child', (x,))
    bias = graph.get_attr('child.bias')
    graph.output((bias,))
    root = torch.nn.Module()
    root.add_module('child', child)
    body = torch.fx.GraphModule(root, graph)
    gm = _wrap_body(body)
    before = (gm.code, body.code)
    with pytest.raises(ValueError, match='unsupported execution'):
        _inline_disabled_autocast_regions(gm)
    assert (gm.code, body.code) == before


def test_distinct_body_attributes_remain_live_until_last_wrapper_is_removed():
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    value = graph.call_function(torch.ops.aten.sin.default, (x,))
    graph.output((value,))
    gm = _wrap_body(torch.fx.GraphModule({}, graph), repeat=True)
    assert sum(n.op == 'get_attr' and n.target == 'body' for n in gm.graph.nodes) == 2
    rows = _inline_disabled_autocast_regions(gm)
    assert len(rows) == 2
    gm.graph.lint()
    assert not hasattr(gm, 'body')
    for output in gm(torch.ones(2)):
        assert torch.equal(output, torch.sin(torch.ones(2)))


@pytest.mark.parametrize('retarget', [False, True])
def test_public_materialization_refuses_body_forward_hook_before_copy(retarget):
    class MatmulRegion(torch.nn.Module):
        def forward(self, x):
            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=False):
                return x @ x.T

    inputs = (torch.ones(2, 2),)
    exported = torch.export.export(MatmulRegion(), inputs)
    body = _autocast_body(exported.graph_module)
    handle = body.register_forward_hook(
        lambda module, args, result: (result[0] + 10,))
    source = exported.module()
    assert torch.equal(source(*inputs), torch.full_like(inputs[0], 12))
    before = body.code
    with pytest.raises(ValueError, match='module hooks'):
        materialize_frontend_precision(
            exported, inputs, dtype=torch.float32,
            retarget_float_dtype_arguments=retarget)
    assert body.code == before
    assert handle.id in body._forward_hooks
    assert torch.equal(source(*inputs), torch.full_like(inputs[0], 12))


@pytest.mark.parametrize('container', [
    'dict_key', 'nested_dict_key', 'set', 'frozenset',
    'slice_start', 'slice_stop', 'slice_step',
])
def test_public_materialization_refuses_dtype_in_discarded_container(container):
    inputs = (torch.ones(2, 2),)
    exported = torch.export.export(Region(), inputs)
    body = _autocast_body(exported.graph_module)
    output = next(n for n in body.graph.nodes if n.op == 'output')
    result = output.args[0][0]
    containers = {
        'dict_key': {torch.bfloat16: result},
        'nested_dict_key': ([({torch.bfloat16: result},)],),
        'set': {torch.bfloat16},
        'frozenset': frozenset({torch.bfloat16}),
        'slice_start': slice(torch.bfloat16, None, None),
        'slice_stop': slice(None, torch.bfloat16, None),
        'slice_step': slice(None, None, torch.bfloat16),
    }
    output.args = ((result, containers[container]),)
    body.recompile()
    before = body.code
    with pytest.raises(ValueError, match='dtype literal'):
        materialize_frontend_precision(exported, inputs, dtype=torch.float32,
                                      retarget_float_dtype_arguments=True)
    assert body.code == before


@pytest.mark.parametrize('reference', ['call_module', 'get_attr', 'alias', 'outer_scope'])
def test_body_survives_live_ancestor_reference(reference):
    class Holder(torch.nn.Module):
        def __init__(self, body):
            super().__init__()
            self.body = body

        def forward(self, x):
            return self.body(x)[0]

    body_graph = torch.fx.Graph()
    x = body_graph.placeholder('x')
    value = body_graph.call_function(torch.ops.aten.sin.default, (x,))
    body_graph.output((value,))
    body = torch.fx.GraphModule({}, body_graph)
    holder = Holder(body)
    root = torch.nn.Module()
    root.holder = holder
    root.alias = holder
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    attr = graph.get_attr('holder.body')
    wrapper = graph.call_function(torch.ops.higher_order.wrap_with_autocast,
        ('cpu', torch.bfloat16, False, False, attr, x))
    selected = graph.call_function(operator.getitem, (wrapper, 0))
    if reference == 'get_attr':
        ancestor = graph.get_attr('holder')
        other = graph.call_method('__call__', (ancestor, x))
    elif reference != 'outer_scope':
        other = graph.call_module('alias' if reference == 'alias' else 'holder', (x,))
    graph.output(selected if reference == 'outer_scope' else (selected, other))
    gm = torch.fx.GraphModule(root, graph)
    # GraphModule can rebuild a namespace for holder.body independently of
    # alias. Keep the actual shared ancestor to exercise alias reachability.
    gm.holder = holder
    if reference == 'outer_scope':
        outer = torch.fx.Graph()
        x = outer.placeholder('x')
        inlined = outer.call_module('region', (x,))
        other = outer.call_module('holder', (x,))
        outer.output((inlined, other))
        gm = torch.fx.GraphModule({'region': gm, 'holder': holder}, outer)
    inputs = (torch.ones(2),)
    expected = gm(*inputs)
    rows = _inline_disabled_autocast_regions(gm)
    assert len(rows) == 1
    gm.graph.lint()
    actual = gm(*inputs)
    assert gm.holder.body is body
    for left, right in zip(actual, expected, strict=True):
        assert torch.equal(left, right)


@pytest.mark.parametrize('registration', [
    'register_forward_hook', 'register_forward_pre_hook',
    'register_full_backward_hook', 'register_full_backward_pre_hook',
    'register_state_dict_pre_hook', 'register_state_dict_post_hook',
    'register_load_state_dict_pre_hook', 'register_load_state_dict_post_hook',
])
@pytest.mark.parametrize('location', ['root', 'body', 'child', 'splice_body'])
def test_module_hooks_are_refused_before_copy_or_splice(registration, location, monkeypatch):
    import copy

    inputs = (torch.ones(2, 2),)
    exported = torch.export.export(Region(), inputs)
    gm = exported.module() if location == 'splice_body' else exported.graph_module
    body = _autocast_body(gm)
    child = torch.nn.Identity()
    body.add_module('unused_child', child)
    owner = gm if location == 'root' else child if location == 'child' else body

    def hook(*args):
        return None

    getattr(owner, registration)(hook)
    before = (gm.code, body.code)

    def unexpected_copy(*args, **kwargs):
        pytest.fail('hook preflight must run before deepcopy')

    monkeypatch.setattr(copy, 'deepcopy', unexpected_copy)
    with pytest.raises(ValueError, match='module hooks'):
        if location == 'splice_body':
            _inline_disabled_autocast_regions(gm)
        else:
            materialize_frontend_precision(exported, inputs, dtype=torch.float32,
                                          retarget_float_dtype_arguments=True)
    assert (gm.code, body.code) == before


class MatmulRegion(torch.nn.Module):
    def forward(self, x):
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=False):
            return x @ x.T


@pytest.mark.parametrize('public', [False, True])
def test_pytree_codegen_cannot_change_tuple_callable_abi(public):
    from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
    from torch.utils._pytree import tree_flatten

    inputs = (torch.ones(2, 2),)
    exported = torch.export.export(MatmulRegion(), inputs)
    body = _autocast_body(exported.graph_module)
    x = next(n for n in body.graph.nodes if n.op == 'placeholder')
    body.graph.set_codegen(_PyTreeCodeGen(_PyTreeInfo(
        ['x'], tree_flatten([x])[1], tree_flatten(x)[1])))
    body.recompile()
    assert isinstance(next(n for n in body.graph.nodes if n.op == 'output').args[0], tuple)
    assert exported.module()(*inputs).shape == (2,)
    before = body.code
    with pytest.raises(ValueError, match='default FX codegen'):
        if public:
            materialize_frontend_precision(exported, inputs, dtype=torch.float32,
                                          retarget_float_dtype_arguments=True)
        else:
            _inline_disabled_autocast_regions(exported.module())
    assert body.code == before


def test_custom_codegen_is_refused():
    from torch.fx.graph import CodeGen

    class CustomCodeGen(CodeGen):
        pass

    gm = torch.export.export(MatmulRegion(), (torch.ones(2, 2),)).module()
    _autocast_body(gm).graph.set_codegen(CustomCodeGen())
    with pytest.raises(ValueError, match='default FX codegen'):
        _inline_disabled_autocast_regions(gm)


@pytest.mark.parametrize('public', [False, True])
def test_global_forward_pre_hook_cannot_be_lost(public):
    from torch.nn.modules import module

    inputs = (torch.full((2, 2), 2.),)
    exported = torch.export.export(MatmulRegion(), inputs)
    source = exported.module()
    body = _autocast_body(source)

    def hook(mod, args):
        if mod is body:
            return (args[0] + 1,)

    handle = module.register_module_forward_pre_hook(hook)
    try:
        assert torch.equal(source(*inputs), torch.full((2, 2), 18.))
        before = body.code
        with pytest.raises(ValueError, match='global execution hooks'):
            if public:
                materialize_frontend_precision(exported, inputs, dtype=torch.float32,
                                              retarget_float_dtype_arguments=True)
            else:
                _inline_disabled_autocast_regions(source)
        assert body.code == before
    finally:
        handle.remove()


@pytest.mark.parametrize('registration', [
    'register_module_forward_hook', 'register_module_forward_pre_hook',
    'register_module_backward_hook', 'register_module_full_backward_hook',
    'register_module_full_backward_pre_hook',
])
def test_all_global_execution_hook_kinds_are_refused(registration):
    from torch.nn.modules import module

    gm = torch.export.export(MatmulRegion(), (torch.ones(2, 2),)).module()
    # Torch remembers whether backward hooks are full even after handle removal.
    previous = module._global_is_full_backward_hook
    handle = getattr(module, registration)(lambda *args: None)
    try:
        with pytest.raises(ValueError, match='global execution hooks'):
            _inline_disabled_autocast_regions(gm)
    finally:
        handle.remove()
        module._global_is_full_backward_hook = previous


@pytest.mark.parametrize('registry', [
    '_global_forward_pre_hooks', '_global_forward_hooks',
    '_global_forward_hooks_always_called', '_global_forward_hooks_with_kwargs',
    '_global_backward_pre_hooks', '_global_backward_hooks',
])
@pytest.mark.parametrize('condition', ['missing', 'wrong_type', 'nonempty'])
def test_global_registry_inspection_fails_closed(registry, condition, monkeypatch):
    from collections import OrderedDict
    from torch.nn.modules import module

    gm = torch.export.export(MatmulRegion(), (torch.ones(2, 2),)).module()
    if condition == 'missing':
        monkeypatch.delattr(module, registry)
    else:
        monkeypatch.setattr(module, registry, {} if condition == 'wrong_type'
                            else OrderedDict([(0, lambda *args: None)]))
    with pytest.raises(ValueError, match='global execution hooks'):
        _inline_disabled_autocast_regions(gm)


@pytest.mark.parametrize('entry', ['call', 'forward'])
def test_returned_module_refuses_later_global_hooks(entry):
    from torch.nn.modules import module

    gm, inputs, _, receipt = materialize_frontend_precision(
        MatmulRegion(), (torch.ones(2, 2),), dtype=torch.float32,
        retarget_float_dtype_arguments=True)
    assert receipt['runtime_preconditions']['global_execution_hooks_absent'] is True
    handle = module.register_module_forward_hook(lambda *args: None)
    try:
        with pytest.raises(ValueError, match='global execution hooks'):
            (gm if entry == 'call' else gm.forward)(*inputs)
    finally:
        handle.remove()
    assert torch.equal(gm(*inputs), torch.full((2, 2), 2.))


def test_body_autocast_cache_observation_is_refused():
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    matmul = graph.call_function(torch.ops.aten.matmul.default, (x, x))
    cache = graph.call_function(torch.is_autocast_cache_enabled, ())
    value = graph.call_function(torch.ops.aten.add.Scalar, (matmul, cache))
    graph.output((value,))
    gm = _wrap_body(torch.fx.GraphModule({}, graph))
    wrapper = next(n for n in gm.graph.nodes
                   if n.target is torch.ops.higher_order.wrap_with_autocast)
    wrapper.args = (*wrapper.args[:3], True, *wrapper.args[4:])
    gm.recompile()
    with torch.autocast('cpu', enabled=False, cache_enabled=False):
        assert torch.equal(gm(torch.ones(2, 2)), torch.full((2, 2), 3.))
        with pytest.raises(ValueError, match='unsupported execution'):
            _inline_disabled_autocast_regions(gm)


def test_child_autocast_dtype_observation_is_refused():
    class DtypeReader(torch.nn.Module):
        def forward(self, x):
            return x + (10 if torch.get_autocast_dtype('cpu') == torch.bfloat16 else 0)

    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    matmul = graph.call_function(torch.ops.aten.matmul.default, (x, x))
    value = graph.call_module('child', (matmul,))
    graph.output((value,))
    gm = _wrap_body(torch.fx.GraphModule({'child': DtypeReader()}, graph))
    with torch.autocast('cpu', dtype=torch.float16, enabled=False):
        assert torch.equal(gm(torch.ones(2, 2)), torch.full((2, 2), 12.))
        with pytest.raises(ValueError, match='unsupported execution'):
            _inline_disabled_autocast_regions(gm)


def _python_identity(x):
    return x


@pytest.mark.parametrize('kind', [
    'torch_function', 'python_function', 'python_callable', 'python_operator', 'aten_packet',
    'custom_overload', 'autocast_dtype', 'autocast_cache', 'nested_autocast', 'nested_grad',
])
def test_non_aten_call_function_kinds_are_refused(kind):
    graph = torch.fx.Graph()
    x = graph.placeholder('x')

    class IdentityCallable:
        __name__ = 'identity_callable'

        def __call__(self, x):
            return x

    targets = {
        'python_callable': (IdentityCallable(), (x,)),
        'torch_function': (torch.sin, (x,)),
        'python_function': (_python_identity, (x,)),
        'python_operator': (operator.add, (x, x)),
        'aten_packet': (torch.ops.aten.sin, (x,)),
        'autocast_dtype': (torch.get_autocast_dtype, ('cpu',)),
        'autocast_cache': (torch.is_autocast_cache_enabled, ()),
    }
    library = torch.library.Library('autocast_admission_test', 'FRAGMENT')
    if kind == 'custom_overload':
        library.define('identity(Tensor x) -> Tensor')
        target, args = torch.ops.autocast_admission_test.identity.default, (x,)
    elif kind.startswith('nested_'):
        child_graph = torch.fx.Graph()
        child_x = child_graph.placeholder('x')
        child_graph.output((child_x,))
        child = torch.fx.GraphModule({}, child_graph)
        attr = graph.get_attr('child')
        if kind == 'nested_autocast':
            target = torch.ops.higher_order.wrap_with_autocast
            args = ('cpu', None, False, False, attr, x)
        else:
            target = torch.ops.higher_order.wrap_with_set_grad_enabled
            args = (False, attr, x)
    else:
        target, args = targets[kind]
    value = graph.call_function(target, args)
    if kind.startswith('nested_'):
        value = graph.call_function(operator.getitem, (value, 0))
    graph.output((value,))
    body = torch.fx.GraphModule({'child': child} if kind.startswith('nested_') else {}, graph)
    gm = _wrap_body(body)
    before = (gm.code, body.code)
    with pytest.raises(ValueError, match='unsupported execution'):
        _inline_disabled_autocast_regions(gm)
    assert (gm.code, body.code) == before


@pytest.mark.parametrize('producer_kind', ['tuple', 'tensor', 'list', 'placeholder'])
def test_getitem_requires_schema_proven_body_tuple(producer_kind):
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    if producer_kind == 'tuple':
        producer = graph.call_function(torch.ops.aten.max.dim, (x, 0))
    elif producer_kind == 'tensor':
        producer = graph.call_function(torch.ops.aten.sin.default, (x,))
        # Metadata alone cannot establish the callable's result ABI.
        producer.meta['val'] = (torch.ones(2), torch.ones(2))
    elif producer_kind == 'list':
        producer = graph.call_function(torch.ops.aten.split.Tensor, (x, 1))
    else:
        producer = x
    selected = graph.call_function(operator.getitem, (producer, 0))
    graph.output((selected,))
    gm = _wrap_body(torch.fx.GraphModule({}, graph))
    if producer_kind == 'tuple':
        expected = gm(torch.ones(2, 2))
        assert len(_inline_disabled_autocast_regions(gm)) == 1
        assert torch.equal(gm(torch.ones(2, 2)), expected)
    else:
        with pytest.raises(ValueError, match='unsupported execution'):
            _inline_disabled_autocast_regions(gm)


def test_call_method_body_is_refused():
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    value = graph.call_method('sin', (x,))
    graph.output((value,))
    with pytest.raises(ValueError, match='unsupported execution'):
        _inline_disabled_autocast_regions(_wrap_body(torch.fx.GraphModule({}, graph)))


@pytest.mark.parametrize('dotted', [False, True])
@pytest.mark.parametrize('parameter', [False, True])
def test_exclusively_owned_tensor_state_and_metadata_stay_admitted(dotted, parameter):
    state_owner = torch.nn.Module()
    value = torch.ones(2)
    if parameter:
        state_owner.register_parameter('bias', torch.nn.Parameter(value))
    else:
        state_owner.register_buffer('bias', value, persistent=False)
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    bias = graph.get_attr('child.bias' if dotted else 'bias')
    result = graph.call_function(torch.ops.aten.add.Tensor, (x, bias))
    result.meta['custom'] = {'m2m_node_id': 'body:add', 'm2m_lineage': ['source:add']}
    graph.output((result,))
    root = torch.nn.Module()
    root.add_module('child', state_owner)
    body = torch.fx.GraphModule(root if dotted else state_owner, graph)
    gm = _wrap_body(body)
    _inline_disabled_autocast_regions(gm)
    assert not hasattr(gm, 'body')
    assert torch.equal(gm(torch.ones(2)), torch.full((2,), 2.))
    assert next(n for n in gm.graph.nodes if n.op == 'call_function').meta == result.meta
    owned, = list(gm.parameters() if parameter else gm.buffers())
    assert owned is state_owner.bias


@pytest.mark.parametrize('reference', ['call_module', 'get_attr', 'alias', 'outer_scope'])
@pytest.mark.parametrize('parameter', [False, True])
def test_direct_state_refuses_live_owner_rebinding(reference, parameter):
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    bias = graph.get_attr('bias')
    value = graph.call_function(torch.ops.aten.add.Tensor, (x, bias))
    graph.output((value,))
    state = torch.nn.Module()
    if parameter:
        state.register_parameter('bias', torch.nn.Parameter(torch.ones(2)))
    else:
        state.register_buffer('bias', torch.ones(2))
    body = torch.fx.GraphModule(state, graph)

    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = body

        def forward(self, x):
            next_bias = self.body.bias + 1
            self.body.bias = torch.nn.Parameter(next_bias) if parameter else next_bias
            return x

    holder = Holder()
    gm = _wrap_body(body)
    gm.add_module('holder', holder)
    gm.add_module('alias', holder)
    arg = next(n for n in gm.graph.nodes if n.op == 'placeholder')
    if reference != 'outer_scope':
        wrapper_attr = next(n for n in gm.graph.nodes if n.op == 'get_attr')
        with gm.graph.inserting_before(wrapper_attr):
            if reference == 'get_attr':
                owner = gm.graph.get_attr('holder')
                gm.graph.call_method('__call__', (owner, arg))
            else:
                gm.graph.call_module('alias' if reference == 'alias' else 'holder', (arg,))
        gm.recompile()
    else:
        outer = torch.fx.Graph()
        x = outer.placeholder('x')
        outer.call_module('holder', (x,))
        value = outer.call_module('region', (x,))
        outer.output(value)
        gm = torch.fx.GraphModule({'holder': holder, 'region': gm}, outer)
    assert torch.equal(gm(torch.ones(2)), torch.full((2,), 3.))
    before = (gm.code, body.code)
    with pytest.raises(ValueError, match='live retained owner'):
        _inline_disabled_autocast_regions(gm)
    assert (gm.code, body.code) == before
    assert torch.equal(gm(torch.ones(2)), torch.full((2,), 4.))


def test_reported_rotary_body_with_unchanged_assertions_admits():
    class RotaryRegion(torch.nn.Module):
        @torch.no_grad()
        def forward(self, frequencies, positions):
            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=False):
                inv = frequencies.float().unsqueeze(0).unsqueeze(-1).expand(positions.shape[0], -1, 1)
                pos = positions.float().unsqueeze(1)
                angles = (inv @ pos).transpose(1, 2)
                embedding = torch.cat((angles, angles), dim=-1)
                return (embedding.cos() * 0.5).to(frequencies.dtype), (embedding.sin() * 0.5).to(frequencies.dtype)

    inputs = (torch.randn(2), torch.arange(3).reshape(1, -1))
    model = RotaryRegion()
    exported = torch.export.export(model, inputs)
    bodies = [module.get_submodule(node.args[4].target)
              for module in exported.graph_module.modules() if hasattr(module, 'graph')
              for node in module.graph.nodes
              if node.target is torch.ops.higher_order.wrap_with_autocast]
    assert len(bodies) == 1
    assert all(node.op in {'placeholder', 'output'} or (
        node.op == 'call_function' and isinstance(node.target, torch._ops.OpOverload)
        and node.target.namespace == 'aten') for node in bodies[0].graph.nodes)
    assertions = [node for node in bodies[0].graph.nodes
                  if node.target is torch.ops.aten._assert_tensor_metadata.default]
    assert assertions
    source_code = bodies[0].code
    gm = exported.module()
    owner = next(module for module in gm.modules() if hasattr(module, 'graph')
                 if any(n.target is torch.ops.higher_order.wrap_with_autocast
                        for n in module.graph.nodes))
    assert len(_inline_disabled_autocast_regions(gm)) == 1
    assert sum(n.target is torch.ops.aten._assert_tensor_metadata.default
               for n in owner.graph.nodes) == len(assertions)
    for actual, expected in zip(gm(*inputs), model(*inputs), strict=True):
        assert torch.equal(actual, expected)
    assert bodies[0].code == source_code


@pytest.mark.parametrize('attribute', ['plain_tensor', 'module', 'scalar'])
def test_unregistered_body_state_is_refused(attribute):
    graph = torch.fx.Graph()
    graph.placeholder('x')
    value = graph.get_attr('state')
    graph.output((value,))
    values = {'plain_tensor': torch.ones(2), 'module': torch.nn.Identity(), 'scalar': 1}
    body = torch.fx.GraphModule({'state': values[attribute]}, graph)
    # FX automatically registers tensor attributes as buffers; explicitly make
    # this case an ordinary attribute to verify the registration requirement.
    if attribute == 'plain_tensor':
        value = body.state
        del body.state
        body.state = value
    with pytest.raises(ValueError, match='registered tensor state'):
        _inline_disabled_autocast_regions(_wrap_body(body))


@pytest.mark.parametrize('public', [False, True])
@pytest.mark.parametrize('reader,args', [
    (torch.ops.aten.get_autocast_dtype.default, ('cpu',)),
    (torch.ops.aten.is_autocast_enabled.default, ()),
    (torch.ops.aten.is_grad_enabled.default, ()),
])
def test_aten_context_readers_are_refused(reader, args, public):
    inputs = (torch.ones(2, 2),)
    exported = torch.export.export(MatmulRegion(), inputs)
    body = _autocast_body(exported.graph_module)
    output = next(n for n in body.graph.nodes if n.op == 'output')
    with body.graph.inserting_before(output):
        observed = body.graph.call_function(reader, args)
        # ScalarType is returned as an integer. Feeding every reader into
        # arithmetic also bypasses the unrelated dynamic-dtype operand guard.
        value = body.graph.call_function(torch.ops.aten.add.Scalar, (output.args[0][0], observed))
    output.args = ((value,),)
    body.recompile()
    before = body.code
    with pytest.raises(ValueError, match='unsupported execution'):
        if public:
            materialize_frontend_precision(exported, inputs, dtype=torch.float32,
                                          retarget_float_dtype_arguments=True)
        else:
            _inline_disabled_autocast_regions(exported.module())
    assert body.code == before


@pytest.mark.parametrize('target,args,kwargs', [
    (torch.ops.aten.arange.default, (2,), {}),
    (torch.ops.aten.zeros.default, ([2],), {}),
    (torch.ops.aten.empty.memory_format, ([2],), {}),
    (torch.ops.aten.rand.default, ([2],), {}),
    (torch.ops.aten.scalar_tensor.default, (1.,), {}),
])
def test_tensorless_aten_factories_are_refused(target, args, kwargs):
    graph = torch.fx.Graph()
    graph.placeholder('x')
    value = graph.call_function(target, args, kwargs)
    graph.output((value,))
    gm = _wrap_body(torch.fx.GraphModule({}, graph))
    before = gm.code
    with pytest.raises(ValueError, match='unsupported execution'):
        _inline_disabled_autocast_regions(gm)
    assert gm.code == before


@pytest.mark.parametrize('target', [torch.ops.aten.add_.Tensor, torch.ops.aten.copy_.default,
                                   torch.ops.aten.add.out])
def test_mutable_aten_schemas_are_refused(target):
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    kwargs = {'out': x} if target is torch.ops.aten.add.out else {}
    value = graph.call_function(target, (x, x), kwargs)
    graph.output((value,))
    with pytest.raises(ValueError, match='unsupported execution'):
        _inline_disabled_autocast_regions(_wrap_body(torch.fx.GraphModule({}, graph)))


@pytest.mark.parametrize('target,args', [
    (torch.ops.aten.rand_like.default, ()),
    (torch.ops.aten.randn_like.default, ()),
    (torch.ops.aten.normal.Tensor_float, (1.,)),
    (torch.ops.aten.native_dropout.default, (0.5, True)),
])
def test_rng_tagged_tensor_ops_are_refused(target, args):
    assert torch.Tag.nondeterministic_seeded in target.tags
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    value = graph.call_function(target, (x, *args))
    graph.output((value,))
    with pytest.raises(ValueError, match='unsupported execution'):
        _inline_disabled_autocast_regions(_wrap_body(torch.fx.GraphModule({}, graph)))


@pytest.mark.parametrize('tags', [None, [torch.Tag.nondeterministic_bitwise], ['unknown']])
def test_uninspectable_or_nondeterministic_tags_refuse(monkeypatch, tags):
    target = torch.ops.aten.sin.default
    monkeypatch.setattr(target, '_tags', tags)
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    value = graph.call_function(target, (x,))
    graph.output((value,))
    with pytest.raises(ValueError, match='unsupported execution'):
        _inline_disabled_autocast_regions(_wrap_body(torch.fx.GraphModule({}, graph)))


def _cache_sensitive_cuda_scope():
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    weight = graph.placeholder('weight')
    first = graph.call_function(torch.ops.aten.matmul.default, (x, weight))
    graph.call_function(torch.ops.aten.add_.Scalar, (weight, 1.))
    second = graph.call_function(torch.ops.aten.matmul.default, (x, weight))
    result = graph.call_function(torch.ops.aten.add.Tensor, (first, second))
    graph.output((result,))
    body = torch.fx.GraphModule({}, graph)
    outer = torch.fx.Graph()
    x = outer.placeholder('x')
    weight = outer.placeholder('weight')
    attr = outer.get_attr('body')
    wrapped = outer.call_function(torch.ops.higher_order.wrap_with_autocast,
        ('cuda', torch.float16, False, False, attr, x, weight))
    result = outer.call_function(operator.getitem, (wrapped, 0))
    outer.output(result)
    return torch.fx.GraphModule({'body': body}, outer), body


def test_cuda_scope_cpu_cache_repro_is_refused():
    gm, body = _cache_sensitive_cuda_scope()
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=True):
        x = torch.ones(2, 2)
        assert torch.equal(gm(x, torch.ones(2, 2, requires_grad=True)), torch.full((2, 2), 6.))
        # Calling the same body without the disabled CUDA wrapper uses the CPU
        # cast cache and reuses the pre-update weight for the second matmul.
        assert torch.equal(body(x, torch.ones(2, 2, requires_grad=True))[0], torch.full((2, 2), 4.))
    before = (gm.code, body.code)
    with pytest.raises(ValueError, match='unsupported execution'):
        _inline_disabled_autocast_regions(gm)
    assert (gm.code, body.code) == before


@pytest.mark.parametrize('wrapper_cache', [False, True, None])
def test_cuda_scope_returned_module_guards_cpu_and_cache_is_irrelevant_when_disabled(wrapper_cache):
    inputs = (torch.ones(2, 2),)
    exported = torch.export.export(MatmulRegion(), inputs)
    wrapper = next(n for n in exported.graph.nodes
                   if n.target is torch.ops.higher_order.wrap_with_autocast)
    wrapper.args = ('cuda', torch.float16, False, wrapper_cache, *wrapper.args[4:])
    exported.graph_module.recompile()
    gm, args, _, receipt = materialize_frontend_precision(
        exported, inputs, dtype=torch.float32, retarget_float_dtype_arguments=True)
    devices = sorted(torch._C._autocast_supported_devices())
    assert receipt['runtime_preconditions']['autocast_disabled_for_devices'] == devices
    for cache in [False, True]:
        with torch.autocast('cpu', enabled=False, cache_enabled=cache):
            assert torch.equal(exported.module()(*inputs), gm(*args))
        with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=cache):
            with pytest.raises(ValueError, match='requires ambient autocast disabled for cpu'):
                gm(*args)


@pytest.mark.parametrize('condition', ['missing', 'empty', 'wrong_type'])
def test_autocast_device_discovery_fails_closed(monkeypatch, condition):
    exported = torch.export.export(MatmulRegion(), (torch.ones(2, 2),))
    if condition == 'missing':
        monkeypatch.delattr(torch._C, '_autocast_supported_devices')
    else:
        monkeypatch.setattr(torch._C, '_autocast_supported_devices',
                            lambda: [] if condition == 'empty' else ['invalid_device'])
    with pytest.raises(ValueError, match='autocast devices cannot be inspected'):
        materialize_frontend_precision(exported, (torch.ones(2, 2),), dtype=torch.float32,
                                      retarget_float_dtype_arguments=True)


@pytest.mark.parametrize('value,known', [(torch.device('cpu'), False),
                                       (torch.device('meta'), True), ('meta', True)])
def test_explicit_unproven_device_is_refused(value, known):
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    if known:
        x.meta['val'] = torch.ones(2)
    result = graph.call_function(torch.ops.aten._to_copy.default, (x,), {'device': value})
    graph.output((result,))
    with pytest.raises(ValueError, match='unsupported execution'):
        _inline_disabled_autocast_regions(_wrap_body(torch.fx.GraphModule({}, graph)))


@pytest.mark.parametrize('operand', ['device', 'layout', 'memory_format'])
def test_non_dtype_control_cannot_hide_dtype_literals(operand):
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    result = graph.call_function(torch.ops.aten._to_copy.default, (x,), {operand: torch.bfloat16})
    graph.output((result,))
    with pytest.raises(ValueError, match='schema-owned dtype'):
        _inline_disabled_autocast_regions(_wrap_body(torch.fx.GraphModule({}, graph)))


@pytest.mark.parametrize('include_assertion', [False, True])
def test_tensor_metadata_assertion_and_literal_controls_admit(include_assertion):
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    if include_assertion:
        graph.call_function(torch.ops.aten._assert_tensor_metadata.default,
                            (x,), {'dtype': torch.float32, 'layout': torch.strided})
    result = graph.call_function(torch.ops.aten.to.dtype, (x, torch.float32),
                                 {'memory_format': torch.contiguous_format})
    graph.output((result,))
    gm = _wrap_body(torch.fx.GraphModule({}, graph))
    assert len(_inline_disabled_autocast_regions(gm)) == 1
    assert torch.equal(gm(torch.ones(2)), torch.ones(2))


@pytest.mark.parametrize('entry', ['call', 'forward'])
@pytest.mark.parametrize('device', torch._C._autocast_supported_devices())
def test_returned_module_refuses_every_exposed_autocast_device(device, entry):
    gm, inputs, _, receipt = materialize_frontend_precision(
        MatmulRegion(), (torch.ones(2, 2),), dtype=torch.float32,
        retarget_float_dtype_arguments=True)
    previous = torch.is_autocast_enabled(device)
    try:
        # Set the dispatch flag directly, including for unavailable hardware.
        torch.set_autocast_enabled(device, True)
        with pytest.raises(ValueError, match='requires ambient autocast disabled'):
            (gm if entry == 'call' else gm.forward)(*inputs)
    finally:
        torch.set_autocast_enabled(device, previous)
    assert device in receipt['runtime_preconditions']['autocast_disabled_for_devices']


@pytest.mark.parametrize('entry', ['call', 'forward'])
def test_returned_module_binds_tensor_input_devices(entry):
    gm, inputs, _, receipt = materialize_frontend_precision(
        MatmulRegion(), (torch.ones(2, 2),), dtype=torch.float32,
        retarget_float_dtype_arguments=True)
    assert receipt['runtime_preconditions']['tensor_input_devices'] == [
        {'leaf_ordinal': 0, 'device': 'cpu'}]
    with pytest.raises(ValueError, match='requires captured tensor input devices'):
        (gm if entry == 'call' else gm.forward)(inputs[0].to('meta'))


@pytest.mark.parametrize('source', ['metadata', 'registered_state'])
def test_proven_same_device_controls_stay_admitted(source):
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    root = torch.nn.Module()
    if source == 'registered_state':
        root.register_buffer('state', torch.ones(2))
        x = graph.get_attr('state')
    else:
        x.meta['val'] = torch.ones(2)
    value = graph.call_function(torch.ops.aten._to_copy.default, (x,),
                                {'device': torch.device('cpu'), 'dtype': torch.float32,
                                 'layout': torch.strided, 'memory_format': torch.contiguous_format})
    graph.output((value,))
    gm = _wrap_body(torch.fx.GraphModule(root, graph))
    assert len(_inline_disabled_autocast_regions(gm)) == 1
    assert torch.equal(gm(torch.ones(2)), torch.ones(2))


def test_missing_operator_tags_refuse(monkeypatch):
    monkeypatch.delattr(torch.ops.aten.sin.default, '_tags')
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    result = graph.call_function(torch.ops.aten.sin.default, (x,))
    graph.output((result,))
    with pytest.raises(ValueError, match='unsupported execution'):
        _inline_disabled_autocast_regions(_wrap_body(torch.fx.GraphModule({}, graph)))


@pytest.mark.parametrize('type_name,expected', [
    ('Tensor', True), ('Tensor?', True), ('Tensor[]', True), ('Tensor?[]', True),
    ('int', False), ('int[]', False), ('Device?', False),
])
def test_schema_tensor_argument_types(type_name, expected):
    from m2m.capture.trace import _autocast_tensor_argument

    schema = torch._C.parse_schema(f'probe({type_name} value) -> ()')
    assert _autocast_tensor_argument(schema.arguments[0].type) is expected


class AddRegion(torch.nn.Module):
    def forward(self, x):
        with torch.autocast('cpu', enabled=False, cache_enabled=False):
            return x + x


@pytest.mark.parametrize('public', [False, True])
@pytest.mark.parametrize('location', ['body', 'enclosing'])
@pytest.mark.parametrize('active', [False, True])
def test_generated_arithmetic_cannot_be_discarded(public, location, active):
    inputs = (torch.ones(2),)
    exported = torch.export.export(AddRegion(), inputs)
    gm = exported.graph_module
    owner = _autocast_body(gm) if location == 'body' else gm
    arg = next(n for n in owner.graph.nodes if n.op == 'placeholder')
    # Restoring the callback does not restore already compiled executable code.
    context = owner.graph.on_generate_code(
        lambda previous: lambda lines: [f'{arg.target} = {arg.target} + 1\n'] + lines)
    owner.recompile()
    if not active:
        with context:
            pass
    assert torch.equal(gm(*inputs)[0], torch.full((2,), 4.))
    before = (gm.code, owner.code)
    with pytest.raises(ValueError, match='codegen|executable code'):
        if public:
            materialize_frontend_precision(exported, inputs, dtype=torch.float32,
                                          retarget_float_dtype_arguments=True)
        else:
            _inline_disabled_autocast_regions(gm)
    assert (gm.code, owner.code) == before


@pytest.mark.parametrize('location', ['body', 'enclosing'])
@pytest.mark.parametrize('state', ['missing', 'malformed'])
def test_codegen_transformer_inspection_fails_closed(location, state):
    gm = torch.export.export(AddRegion(), (torch.ones(2),)).graph_module
    owner = _autocast_body(gm) if location == 'body' else gm
    if state == 'missing':
        del owner.graph._codegen._body_transformer
    else:
        owner.graph._codegen._body_transformer = 1
    with pytest.raises(ValueError, match='codegen'):
        _inline_disabled_autocast_regions(gm)


@pytest.mark.parametrize('mode_kind', ['dispatch', 'function'])
def test_returned_module_refuses_cache_observing_modes(mode_kind):
    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import TorchDispatchMode

    class CacheDispatch(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if func is torch.ops.aten.add.Tensor:
                return result + int(torch.is_autocast_cache_enabled())
            return result

    class CacheFunction(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if func is torch.ops.aten.add.Tensor:
                return result + int(torch.is_autocast_cache_enabled())
            return result

    inputs = (torch.ones(2),)
    exported = torch.export.export(AddRegion(), inputs)
    source = exported.module()
    gm, args, _, receipt = materialize_frontend_precision(
        exported, inputs, dtype=torch.float32, retarget_float_dtype_arguments=True)
    mode = CacheDispatch if mode_kind == 'dispatch' else CacheFunction
    with torch.autocast('cpu', enabled=False, cache_enabled=True):
        with mode():
            original = source(*inputs)
            assert torch.equal(original, torch.full((2,), 2.))
            for invoke in (gm, gm.forward):
                with pytest.raises(ValueError, match='modes'):
                    invoke(*args)
    assert receipt['runtime_preconditions']['torch_dispatch_modes_absent'] is True
    assert receipt['runtime_preconditions']['torch_function_modes_absent'] is True


@pytest.mark.parametrize('entry', ['call', 'forward'])
@pytest.mark.parametrize('mode_kind', ['dispatch', 'function'])
@pytest.mark.parametrize('fault', ['missing', 'malformed', 'raises', 'length'])
def test_runtime_mode_inspection_fails_closed(mode_kind, fault, entry, monkeypatch):
    from torch.utils import _python_dispatch

    gm, args, _, _ = materialize_frontend_precision(
        AddRegion(), (torch.ones(2),), dtype=torch.float32,
        retarget_float_dtype_arguments=True)
    namespace = _python_dispatch if mode_kind == 'dispatch' else torch.overrides
    stack_name = f'_get_current_{mode_kind}_mode_stack'
    if fault == 'missing':
        monkeypatch.delattr(namespace, stack_name)
    elif fault == 'malformed':
        monkeypatch.setattr(namespace, stack_name, lambda: ())
    elif fault == 'length':
        monkeypatch.setattr(namespace, f'_len_torch_{mode_kind}_stack', lambda: True)
    else:
        def unavailable():
            raise RuntimeError('unavailable')
        monkeypatch.setattr(namespace, stack_name, unavailable)
    with pytest.raises(ValueError, match='modes cannot be inspected'):
        (gm if entry == 'call' else gm.forward)(*args)


@pytest.mark.parametrize('override', ['__getattribute__', '__getattr__'])
@pytest.mark.parametrize('location', ['leaf', 'ancestor'])
def test_computed_registered_state_lookup_is_refused(override, location):
    def lookup(self, name):
        if override == '__getattr__':
            value = torch.nn.Module.__getattr__(self, name)
        else:
            try:
                value = torch.nn.Module.__getattribute__(self, name)
            except AttributeError:
                value = torch.nn.Module.__getattr__(self, name)
        if name == 'bias':
            return value + int(torch.is_autocast_cache_enabled())
        return value

    state_type = type('State', (torch.nn.Module,), {override: lookup})
    state = state_type()
    state.register_buffer('bias', torch.ones(2))
    root = torch.nn.Module()
    if location == 'ancestor':
        ancestor = state_type()
        ancestor.add_module('child', torch.nn.Module())
        ancestor.child.register_buffer('bias', torch.ones(2))
        root.add_module('state', ancestor)
        target = 'state.child.bias'
    else:
        root.add_module('state', state)
        target = 'state.bias'
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    bias = graph.get_attr(target)
    result = graph.call_function(torch.ops.aten.add.Tensor, (x, bias))
    graph.output((result,))
    body = torch.fx.GraphModule(root, graph)
    # GraphModule copies a dotted target into fresh ordinary intermediate
    # modules. Restore the actual custom owner present in this capture.
    body.state = root.state
    gm = _wrap_body(body)
    before = gm.code
    with torch.autocast('cpu', enabled=False, cache_enabled=True):
        assert torch.equal(gm(torch.ones(2)), torch.full((2,), 2.))
        with pytest.raises(ValueError, match='ordinary attribute lookup'):
            _inline_disabled_autocast_regions(gm)
    assert gm.code == before


def test_registered_state_descriptor_must_return_registered_identity():
    class State(torch.nn.Module):
        @property
        def bias(self):
            return self._buffers['bias'] + int(torch.is_autocast_cache_enabled())

    state = State()
    state._buffers['bias'] = torch.ones(2)
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    bias = graph.get_attr('state.bias')
    result = graph.call_function(torch.ops.aten.add.Tensor, (x, bias))
    graph.output((result,))
    root = torch.nn.Module()
    root.add_module('state', state)
    body = torch.fx.GraphModule(root, graph)
    body.state = state
    gm = _wrap_body(body)
    with pytest.raises(ValueError, match='registered tensor identity'):
        _inline_disabled_autocast_regions(gm)


@pytest.mark.parametrize('effect', ['renorm', 'backward'])
def test_unannotated_native_effects_are_refused(effect):
    root = torch.nn.Module()
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    if effect == 'renorm':
        target = torch.ops.aten._no_grad_embedding_renorm_.default
        root.register_buffer('indices', torch.tensor([0]))
        indices = graph.get_attr('indices')
        graph.call_function(target, (x, indices, 1., 2.))
        inputs = torch.full((2, 2), 2.)
    elif effect == 'backward':
        target = torch.ops.aten._backward.default
        graph.call_function(target, (x, []))
        inputs = torch.tensor(2., requires_grad=True)
    assert target._schema.is_mutable is False
    graph.output((x,))
    gm = _wrap_body(torch.fx.GraphModule(root, graph))
    if effect == 'renorm':
        gm(inputs)
        assert torch.linalg.vector_norm(inputs[0]) < 1.01
    elif effect == 'backward':
        observed = []
        handle = inputs.register_hook(lambda grad: observed.append(grad.clone()))
        gm(inputs)
        handle.remove()
        assert len(observed) == 1
    with pytest.raises(ValueError, match='unsupported execution'):
        _inline_disabled_autocast_regions(gm)


@pytest.mark.parametrize('entry', ['call', 'forward', 'forward_kwargs'])
def test_matmul_direct_forward_obeys_runtime_receipt(entry):
    inputs = (torch.tensor([[0.2, 0.9]]),)
    exported = torch.export.export(MatmulRegion(), inputs)
    gm, args, _, receipt = materialize_frontend_precision(
        exported, inputs, dtype=torch.float32, retarget_float_dtype_arguments=True)
    invoke = gm if entry == 'call' else gm.forward
    assert torch.equal(invoke(*args), exported.module()(*inputs))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        original = exported.module().forward(*inputs)
        assert original.dtype == torch.float32
        assert original.item() == pytest.approx(0.8500000238)
        with pytest.raises(ValueError, match='requires ambient autocast disabled for cpu'):
            if entry == 'forward_kwargs':
                invoke(x=args[0])
            else:
                invoke(*args)
    assert receipt['runtime_preconditions'] == {
        'autocast_disabled_for_devices': sorted(torch._C._autocast_supported_devices()),
        'global_execution_hooks_absent': True,
        'torch_dispatch_modes_absent': True,
        'torch_function_modes_absent': True,
        'tensor_input_devices': [{'leaf_ordinal': 0, 'device': 'cpu'}]}


@pytest.mark.parametrize('retarget', [False, True])
@pytest.mark.parametrize('target_dtype', [torch.float32, torch.float64])
def test_nonleaf_buffer_materialization_owns_state_and_preserves_aliases(retarget, target_dtype):
    from m2m.capture.trace import _owned_precision_capture

    class Derived(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2., 3.]))
            self.weight_alias = self.weight
            self.register_buffer('derived', self.weight * 2)
            self.register_buffer('alias', self.derived)
            self.register_buffer('view', self.derived.view(-1))

        def forward(self, x):
            return x + self.derived, self.alias, self.view, self.weight, self.weight_alias

    model = Derived()
    inputs = (torch.ones(2),)
    exported = torch.export.export(model, inputs)
    before = {name: value.detach().clone() for name, value in exported.state_dict.items()}
    source_code = exported.graph_module.code
    owned = _owned_precision_capture(exported, normalize_autocast=retarget)
    assert owned.state_dict['derived'] is owned.state_dict['alias']
    assert owned.state_dict['weight'] is owned.state_dict['weight_alias']
    assert isinstance(owned.state_dict['weight'], torch.nn.Parameter)
    assert owned.state_dict['derived'].requires_grad
    assert owned.state_dict['derived'].untyped_storage().data_ptr() == owned.state_dict['view'].untyped_storage().data_ptr()
    with torch.no_grad():
        owned.state_dict['derived'].add_(1)
    assert torch.equal(owned.state_dict['view'], torch.tensor([5., 7.]))
    gm, args, _, _ = materialize_frontend_precision(
        exported, inputs, dtype=target_dtype, retarget_float_dtype_arguments=retarget)
    output = gm(*args)
    assert torch.equal(output[0], torch.tensor([5., 7.], dtype=target_dtype))
    assert gm.derived is gm.alias
    assert gm.weight is gm.weight_alias
    assert gm.weight.requires_grad and gm.derived.requires_grad
    assert gm.derived.dtype == target_dtype
    assert gm.derived is not model.derived and gm.weight is not model.weight
    with torch.no_grad():
        gm.derived.add_(10)
        gm.weight.add_(10)
    assert exported.graph_module.code == source_code
    for name, value in exported.state_dict.items():
        assert torch.equal(value, before[name])
    assert model.derived.grad_fn is not None


@pytest.mark.parametrize('target,args,kwargs', [
    (torch.ops.aten._assert_tensor_metadata.default, (), {'size': [2], 'dtype': torch.float32}),
    (torch.ops.aten._assert_scalar.default, (True, 'shape assertion'), {}),
    (torch.ops.aten.sym_constrain_range.default, (2,), {'min': 1, 'max': 4}),
    (torch.ops.aten.sym_constrain_range_for_size.default, (2,), {'min': 1, 'max': 4}),
])
def test_exact_zero_return_checks_remain_in_body(target, args, kwargs):
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    graph.call_function(target, (x,) if not args else args, kwargs)
    graph.output((x,))
    gm = _wrap_body(torch.fx.GraphModule({}, graph))
    expected = gm(torch.ones(2))
    assert len(_inline_disabled_autocast_regions(gm)) == 1
    assert any(n.target is target for n in gm.graph.nodes)
    assert torch.equal(gm(torch.ones(2)), expected)


def test_metadata_assertion_still_raises_after_inlining():
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    graph.call_function(torch.ops.aten._assert_tensor_metadata.default,
                        (x,), {'dtype': torch.float64})
    graph.output((x,))
    gm = _wrap_body(torch.fx.GraphModule({}, graph))
    with pytest.raises(RuntimeError, match='dtype'):
        gm(torch.ones(2))
    _inline_disabled_autocast_regions(gm)
    with pytest.raises(RuntimeError, match='dtype'):
        gm(torch.ones(2))


@pytest.mark.parametrize('retarget', [False, True])
def test_nonleaf_buffer_plain_model_entry(retarget):
    class Derived(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2., 3.]))
            self.register_buffer('derived', self.weight * 2)

        def forward(self, x):
            return x + self.derived

    model = Derived()
    gm, args, _, _ = materialize_frontend_precision(
        model, (torch.ones(2),), dtype=torch.float32,
        retarget_float_dtype_arguments=retarget)
    assert torch.equal(gm(*args), torch.tensor([5., 7.]))
    assert model.derived.grad_fn is not None
    assert torch.equal(model.derived, torch.tensor([4., 6.]))
