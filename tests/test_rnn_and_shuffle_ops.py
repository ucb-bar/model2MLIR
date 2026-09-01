"""LSTM, pixel-shuffle and string-padded conv: the ops that blocked lstmnetvit entirely.

Two gates per op, because they fail differently and one cannot substitute for the other:

* **structural** -- the op lowers with no opaque ``func.call``. An opaque call is an undefined
  symbol at link time, not a slow path, and one opaque op makes a whole capture unusable.
* **numerical** -- the ARITHMETIC the decomposition emits reproduces torch. A shape check cannot
  see a wrong gate order or a wrong permutation: both produce a correctly-shaped, wrong tensor.
  ``m2m.coverage.validate_op`` checks structure and shape only (its module docstring mentions a
  differential oracle, but none is wired), so the arithmetic is re-derived here and compared.

Executing the emitted MLIR is a consumer-side concern -- this repo hosts no runtime -- so these
tests pin the two halves that CAN be checked here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from m2m.coverage import validate_op


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


_LSTM_CASES = [
    ("1layer_batch_first", 1, True, True),
    ("3layer_batch_first", 3, True, True),
    ("2layer_seq_first", 2, False, True),
    ("2layer_no_bias", 2, True, False),
]


class _LSTMOut(nn.Module):
    def __init__(self, layers: int, batch_first: bool, bias: bool):
        super().__init__()
        self.l = nn.LSTM(5, 4, num_layers=layers, batch_first=batch_first, bias=bias)

    def forward(self, x, h, c):
        y, _ = self.l(x, (h, c))
        return y


@pytest.mark.parametrize("name,layers,batch_first,bias", _LSTM_CASES)
def test_lstm_lowers_with_no_opaque_call(name, layers, batch_first, bias):
    m = _LSTMOut(layers, batch_first, bias).eval()
    x = torch.randn(1, 3, 5) if batch_first else torch.randn(3, 1, 5)
    v = validate_op(m, (x, torch.zeros(layers, 1, 4), torch.zeros(layers, 1, 4)), name=f"lstm_{name}")
    assert v.lowered, f"opaque calls remain: {v.opaque_calls} ({v.error})"
    assert v.shape_ok


@pytest.mark.parametrize("name,layers,batch_first,bias", _LSTM_CASES)
def test_lstm_recurrence_matches_torch(name, layers, batch_first, bias):
    """The gate order is the trap. torch packs i,f,g,o along the 4H axis; any other reading still
    type-checks, still has the right shape, and is silently a different network."""
    torch.manual_seed(0)
    m = nn.LSTM(5, 4, num_layers=layers, batch_first=batch_first, bias=bias).eval()
    x = torch.randn(2, 3, 5) if batch_first else torch.randn(3, 2, 5)
    h0, c0 = torch.randn(layers, 2, 4), torch.randn(layers, 2, 4)
    with torch.no_grad():
        y, (hn, cn) = m(x, (h0, c0))

    per = 4 if bias else 2
    ps = [p.detach().numpy() for p in m._flat_weights]
    X = np.transpose(x.numpy(), (1, 0, 2)) if batch_first else x.numpy()
    xs = [X[t] for t in range(X.shape[0])]
    h_fin, c_fin = [], []
    for layer in range(layers):
        w_ih, w_hh = ps[layer * per + 0], ps[layer * per + 1]
        b = (ps[layer * per + 2] + ps[layer * per + 3]) if per == 4 else 0.0
        h, c = h0.numpy()[layer], c0.numpy()[layer]
        outs = []
        for xt in xs:
            z = xt @ w_ih.T + h @ w_hh.T + b
            H = w_hh.shape[1]
            zi, zf, zg, zo = (z[:, k * H:(k + 1) * H] for k in range(4))
            c = _sigmoid(zf) * c + _sigmoid(zi) * np.tanh(zg)
            h = _sigmoid(zo) * np.tanh(c)
            outs.append(h)
        xs = outs
        h_fin.append(h)
        c_fin.append(c)
    out = np.stack(xs, 0)
    if batch_first:
        out = np.transpose(out, (1, 0, 2))

    assert np.allclose(out, y.numpy(), atol=1e-5), "output sequence diverges from torch"
    assert np.allclose(np.stack(h_fin, 0), hn.numpy(), atol=1e-5), "h_n diverges from torch"
    assert np.allclose(np.stack(c_fin, 0), cn.numpy(), atol=1e-5), "c_n diverges from torch"


def test_lstm_declares_all_three_results():
    """h_n and c_n must be real results, not dropped.

    While the op stayed opaque its stub carried only the first of three tensors, so every
    ``getitem(node, 1|2)`` consumer had no producer and went opaque too -- one opaque LSTM was
    three opaque ops in lstmnetvit's capture.
    """
    from m2m.ir.decompositions import DECOMPOSITION_TABLE

    assert "aten.lstm.input" in DECOMPOSITION_TABLE

    class Full(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.LSTM(5, 4, num_layers=2, batch_first=True)

        def forward(self, x, h, c):
            y, (hn, cn) = self.l(x, (h, c))
            return y.sum() + hn.sum() + cn.sum()

    v = validate_op(Full().eval(), (torch.randn(1, 3, 5), torch.zeros(2, 1, 4),
                                    torch.zeros(2, 1, 4)), name="lstm_all_outputs")
    assert v.lowered, f"consuming h_n/c_n left opaque calls: {v.opaque_calls}"


@pytest.mark.parametrize("n,c,r,h,w", [(1, 16, 2, 8, 12), (2, 3, 3, 4, 5), (1, 4, 4, 2, 2)])
def test_pixel_shuffle_lowers_and_permutes_correctly(n, c, r, h, w):
    torch.manual_seed(0)
    x = torch.randn(n, c * r * r, h, w)
    shuffle = nn.PixelShuffle(r)
    v = validate_op(lambda t: shuffle(t), (x,), name=f"pixel_shuffle_r{r}")
    assert v.lowered, f"opaque calls remain: {v.opaque_calls}"

    # The interleave is the content of the op: leaving the two upscale axes next to C produces a
    # tensor of the RIGHT SHAPE holding the wrong pixels, which no shape check would catch.
    a = np.transpose(x.numpy().reshape(n, c, r, r, h, w), (0, 1, 4, 2, 5, 3))
    assert np.array_equal(a.reshape(n, c, h * r, w * r), shuffle(x).numpy())


@pytest.mark.parametrize("name,mod,shape", [
    ("same_depthwise", nn.Conv2d(8, 8, 3, padding="same", groups=8), (1, 8, 6, 6)),
    ("same_grouped", nn.Conv2d(16, 16, 3, padding="same", groups=4), (1, 16, 7, 9)),
    ("same_plain", nn.Conv2d(4, 8, 3, padding="same"), (1, 4, 6, 6)),
    ("valid_string", nn.Conv2d(4, 8, 3, padding="valid"), (1, 4, 6, 6)),
    ("same_dilated", nn.Conv2d(4, 8, 3, padding="same", dilation=2), (1, 4, 8, 8)),
])
def test_string_padded_conv_lowers(name, mod, shape):
    """``aten.conv2d.padding`` is the overload torch emits for a STRING padding. It used to try
    only the direct-conv path, which refuses non-zero padding AND groups>1, and went opaque
    otherwise -- so an ordinary padded depthwise conv never lowered."""
    v = validate_op(lambda t: mod.eval()(t), (torch.randn(*shape),), name=f"conv2d_{name}")
    assert v.lowered, f"opaque calls remain: {v.opaque_calls} ({v.error})"
    assert v.shape_ok


def test_an_even_kernel_same_pad_is_refused_not_guessed():
    """torch pads an even kernel ASYMMETRICALLY under padding="same". The shared conv path pads
    both sides equally, so answering here would shift the output by a pixel while still looking
    like a clean lowering. Refusing keeps it visible."""
    from m2m.ir.decompositions import _same_padding

    assert _same_padding([8, 8, 3, 3], [1, 1], spatial=2) == [1, 1]
    assert _same_padding([8, 8, 5, 5], [2, 2], spatial=2) == [4, 4]
    assert _same_padding([8, 8, 4, 4], [1, 1], spatial=2) is None
