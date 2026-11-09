"""Static diff lowers to exact tensor operations, including dtype edge cases."""
import numpy as np
import pytest
import torch
from xdsl.dialects.func import FuncOp
from m2m.ir.import_fx import FXImporter


class Diff(torch.nn.Module):
    def __init__(self, n, dim, ends):
        super().__init__()
        self.n, self.dim, self.ends = n, dim, ends

    def forward(self, x, prepend, append):
        return torch.diff(x, n=self.n, dim=self.dim,
                          prepend=prepend if 'p' in self.ends else None,
                          append=append if 'a' in self.ends else None)


def evaluate(module, inputs):
    function = next(op for op in module.walk() if isinstance(op, FuncOp))
    env = {arg: value.numpy() for arg, value in zip(function.body.block.args, inputs)}
    for op in function.body.block.ops:
        if op.name == 'func.return':
            return env[op.operands[0]]
        if op.name == 'tensor.concat':
            value = np.concatenate([env[o] for o in op.operands], axis=op.dim.value.data)
        elif op.name == 'tensor.extract_slice':
            offsets = op.static_offsets.get_values()
            sizes = op.static_sizes.get_values()
            strides = op.static_strides.get_values()
            value = env[op.source][tuple(slice(start, start + size * step, step)
                                         for start, size, step in zip(offsets, sizes, strides))]
        elif op.name == 'tensor.empty':
            value = np.empty(op.results[0].type.get_shape(), dtype=inputs[0].numpy().dtype)
        elif op.name == 'linalg.generic':
            body = list(op.body.block.ops)
            assert len(body) == 2 and body[-1].name == 'linalg.yield'
            left, right = (env[o] for o in op.inputs)
            if body[0].name in ('arith.subi', 'arith.subf'):
                value = np.subtract(left, right)
            elif body[0].name == 'arith.xori':
                value = np.logical_xor(left, right)
            else:
                raise AssertionError(f'unsupported arithmetic {body[0].name}')
        else:
            raise AssertionError(f'unsupported operation {op.name}')
        env[op.results[0]] = value
    raise AssertionError('missing return')


@pytest.mark.parametrize('dtype', [torch.int8, torch.uint8, torch.int64, torch.float32, torch.bool])
@pytest.mark.parametrize('n,dim,ends', [(0,-1,'pa'), (1,-1,''), (1,-1,'p'), (2,0,'a'), (3,1,'pa'), (9,-1,'')])
def test_diff_exact_dtype_arithmetic(dtype, n, dim, ends):
    data = np.array([[127, -128, 0, 3], [-3, 1, 127, -128], [1, 0, -1, 9]])
    if dtype == torch.int64:
        data[0, 0], data[0, 1] = 2**63 - 1, -(2**63)
    x = torch.tensor(data, dtype=torch.int64).to(dtype)
    axis = dim % x.ndim
    end_shape = list(x.shape)
    end_shape[axis] = 1
    inputs = (x, torch.full(end_shape, 1, dtype=dtype), torch.full(end_shape, 0, dtype=dtype))
    model = Diff(n,dim,ends)
    expected = model(*inputs).numpy()
    module = FXImporter(allow_opaque_fallback=False).import_graph(torch.export.export(model, inputs))
    module.verify()
    assert not any(op.name == 'func.call' for op in module.walk())
    np.testing.assert_array_equal(evaluate(module, inputs), expected)
    if n and dtype in (torch.int8,torch.uint8,torch.int64):
        assert not any(op.name in ('arith.sitofp','arith.uitofp','arith.subf') for op in module.walk())
