"""Fake propagation keeps data-dependent sizes without dynamic contractions."""
import copy

import pytest
import torch
import torch.ao.quantization.fx._decomposed  # Register Q/DQ operators.
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from m2m.capture.pt2e_integerize import _geometry_fake_mode, integerize_pt2e
from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference


class BooleanIndex(torch.nn.Module):
    def forward(self, x, mask):
        positions = torch.zeros_like(x)
        positions[mask] = x[mask]
        return positions @ x.T


def test_data_dependent_intermediate_reuses_capture_environment():
    inputs = (torch.randn(4, 4), torch.eye(4, dtype=torch.bool))
    gm = torch.export.export(BooleanIndex(), inputs).module()
    index = next(n for n in gm.graph.nodes if n.target == torch.ops.aten.index.Tensor)
    symbol = index.meta['val'].shape[0]
    mode = _geometry_fake_mode(gm, inputs)
    assert mode.shape_env is symbol.node.shape_env
    assert mode.static_shapes is True
    integerize_pt2e(gm, inputs)
    assert isinstance(index.meta['tensor_meta'].shape[0], torch.SymInt)
    assert str(index.meta['tensor_meta'].shape[0]) == str(symbol)
    mm = next(n for n in gm.graph.nodes if n.target == torch.ops.aten.matmul.default)
    assert mm.meta['tensor_meta'].shape == (4, 4)


def test_incompatible_symbol_owners_refused():
    gm = torch.fx.symbolic_trace(torch.nn.Identity())
    nodes = list(gm.graph.nodes)
    for node in nodes:
        node.meta['val'] = ShapeEnv().create_unbacked_symint()
    with pytest.raises(ValueError, match='incompatible ShapeEnv'):
        _geometry_fake_mode(gm, (torch.ones(2),))


@pytest.mark.parametrize('shared_owner', [False, True])
def test_nested_symbol_owners_are_collected(shared_owner):
    root = torch.fx.symbolic_trace(torch.nn.Identity())
    environment = ShapeEnv()
    output = next(n for n in root.graph.nodes if n.op == 'output')
    value = output.args[0]
    for name in ('first', 'second'):
        child = torch.fx.symbolic_trace(torch.nn.Identity())
        owner = environment if shared_owner else ShapeEnv()
        next(iter(child.graph.nodes)).meta['val'] = owner.create_unbacked_symint()
        root.add_submodule(name, child)
        with root.graph.inserting_before(output):
            value = root.graph.call_module(name, (value,))
    output.args = (value,)
    root.recompile()
    # The root itself has no symbolic metadata. Only its nested bodies own it.
    if shared_owner:
        assert _geometry_fake_mode(root, (torch.ones(2),)).shape_env is environment
    else:
        with pytest.raises(ValueError, match='incompatible ShapeEnv'):
            _geometry_fake_mode(root, (torch.ones(2),))


class QuantizedMatmul(torch.nn.Module):
    def __init__(self, dynamic=False):
        super().__init__()
        self.dynamic = dynamic
        self.register_buffer('weight', torch.randint(-5, 6, (8, 8), dtype=torch.int8))

    def forward(self, x, mask):
        qd = torch.ops.quantized_decomposed
        if self.dynamic:
            x = x[mask]
        q = qd.quantize_per_tensor.default(x, .125, 0, -128, 127, torch.int8)
        x = qd.dequantize_per_tensor.default(q, .125, 0, -128, 127, torch.int8)
        w = qd.dequantize_per_tensor.default(self.weight, .25, 0, -128, 127, torch.int8)
        return x @ w


@pytest.mark.parametrize('dynamic', [False, True])
def test_static_contraction_exact_and_dynamic_contraction_refused(dynamic):
    inputs = (torch.randn(4, 8), torch.tensor([True, False, True, False]))
    gm = torch.export.export(QuantizedMatmul(dynamic), inputs).module()
    reference = None if dynamic else run_pt2e_integer_reference(copy.deepcopy(gm), inputs).output
    gm, receipt = integerize_pt2e(gm, inputs)
    if dynamic:
        assert receipt['quantized_contractions_integerized'] == 0
        assert 'static positive tensor shape is unavailable' in receipt['refusals'][0]['reason']
    else:
        assert receipt['quantized_contractions_integerized'] == 1
        assert not receipt['refusals']
        assert torch.equal(reference, gm(*inputs))


def test_no_captured_symbols_creates_shape_environment():
    gm = torch.fx.symbolic_trace(torch.nn.Identity())
    assert isinstance(_geometry_fake_mode(gm, (torch.ones(2),)).shape_env, ShapeEnv)
