"""Regression tests: ops that must lower to real standard-dialect MLIR (no opaque).

Gated by the torch-mlir differential oracle where available; falls back to a structural
check (lowered + no opaque func.call) otherwise.
"""

from __future__ import annotations

import pytest
import torch

from m2m.coverage import validate_op

X = (torch.randn(4, 8),)
XY = (torch.randn(4, 8), torch.randn(4, 8))
XPOS = (torch.randn(4, 8).abs() + 1,)


def _conv_cases():
    """The conv/pad variants the vision, audio and control workloads need.

    Before these lowered, only a rank-4 groups=1 zero-padded conv reached real linalg;
    Conv1d, any non-zero padding, groups>1 and ConvTranspose each left an opaque
    ``func.call`` -- an undefined symbol at link time, not a slow path. Numerical agreement
    with torch is gated on the consumer side (the emitted MLIR has to be executed to check
    an index map, which needs a runtime this repo does not host); here we gate that they
    lower with no opaque calls and with the result type eager produced.
    """
    from torch import nn

    specs = [
        ("2d_patchembed", nn.Conv2d(3, 16, 4, stride=4), (1, 3, 16, 16)),
        ("2d_pad1", nn.Conv2d(4, 8, 3, padding=1), (1, 4, 8, 8)),
        ("2d_pad1_stride2", nn.Conv2d(4, 8, 3, stride=2, padding=1), (1, 4, 8, 8)),
        ("2d_k7_s4_p3", nn.Conv2d(1, 8, 7, stride=4, padding=3), (1, 1, 12, 16)),
        ("2d_depthwise", nn.Conv2d(8, 8, 3, padding=1, groups=8), (1, 8, 6, 6)),
        ("2d_groups2", nn.Conv2d(4, 8, 3, padding=1, groups=2), (1, 4, 6, 6)),
        ("2d_dilation2", nn.Conv2d(4, 8, 3, padding=2, dilation=2), (1, 4, 8, 8)),
        ("2d_nobias", nn.Conv2d(4, 8, 3, padding=1, bias=False), (1, 4, 6, 6)),
        ("1d_k3_p1", nn.Conv1d(8, 12, 3, padding=1), (1, 8, 20)),
        ("1d_k3_s2_p1", nn.Conv1d(8, 12, 3, stride=2, padding=1), (1, 8, 20)),
        ("transpose2d_s2_p1", nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1), (1, 8, 6, 6)),
        ("transpose2d_outpad", nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1,
                                                 output_padding=1), (1, 8, 6, 6)),
        ("transpose2d_s1_p0", nn.ConvTranspose2d(8, 4, 3), (1, 8, 6, 6)),
        ("transpose2d_s3_p2", nn.ConvTranspose2d(4, 6, 5, stride=3, padding=2), (1, 4, 5, 5)),
        ("pad2d_sym", nn.ZeroPad2d(2), (1, 3, 6, 6)),
        ("pad2d_asym", nn.ZeroPad2d((1, 2, 3, 0)), (1, 3, 6, 6)),
        ("pad1d", nn.ConstantPad1d((2, 3), 0.0), (1, 4, 9)),
    ]
    torch.manual_seed(0)
    cases = []
    for nm, mod, shape in specs:
        mod = mod.eval()
        cases.append((f"conv_{nm}", (lambda x, _m=mod: _m(x)), (torch.randn(*shape),)))
    return cases


CASES = [
    ("view", lambda a: a.view(2, 16), X),
    ("reshape", lambda a: a.reshape(8, 4), X),
    ("unsqueeze", lambda a: a.unsqueeze(1), X),
    ("flatten", lambda a: a.flatten(), X),
    ("mul", lambda a, b: a * b, XY),
    ("add", lambda a, b: a + b, XY),
    ("sub", lambda a, b: a - b, XY),
    ("div", lambda a, b: a / b, XY),
    ("neg", lambda a: -a, X),
    ("rsqrt", torch.rsqrt, XPOS),
    ("sigmoid", torch.sigmoid, X),
    ("silu", torch.nn.functional.silu, X),
    ("cos", torch.cos, X),
    ("sin", torch.sin, X),
    ("to_f16", lambda a: a.to(torch.float16), X),
    ("exp", torch.exp, X),
    ("sqrt", lambda a: torch.sqrt(a.abs() + 1), X),
    ("tanh", torch.tanh, X),
    ("abs", torch.abs, X),
    ("floor", torch.floor, X),
    ("log", lambda a: torch.log(a.abs() + 1), X),
    ("gelu", torch.nn.functional.gelu, X),
    ("permute3d", lambda a: a.permute(0, 2, 1), (torch.randn(2, 3, 4),)),
    ("clone", torch.clone, X),
    ("mul_bcast", lambda a, b: a * b, (torch.randn(4, 8), torch.randn(8))),
    ("mm", torch.mm, (torch.randn(4, 8), torch.randn(8, 16))),
    ("addmm", lambda b, x, w: torch.addmm(b, x, w), (torch.randn(16), torch.randn(4, 8), torch.randn(8, 16))),
    ("relu", torch.relu, X),
    ("relu_inplace", lambda a: torch.relu_(a.clone()), X),
    ("batch_norm_eval", lambda a, w, b, mean, var:
     torch.nn.functional.batch_norm(a, mean, var, w, b, training=False),
     (torch.randn(1, 4, 8, 8), torch.randn(4), torch.randn(4),
      torch.randn(4), torch.rand(4) + 0.5)),
    ("max_pool2d", lambda a: torch.nn.functional.max_pool2d(a, 3, stride=2, padding=1),
     (torch.randn(1, 4, 8, 8),)),
    ("adaptive_avg_pool2d_global",
     lambda a: torch.nn.functional.adaptive_avg_pool2d(a, (1, 1)),
     (torch.randn(1, 4, 7, 7),)),
    ("clamp", lambda a: a.clamp(-1, 1), X),
    ("maximum", torch.maximum, XY),
    ("minimum", torch.minimum, XY),
    ("where", lambda c, a, b: torch.where(c, a, b), (torch.randn(4, 8) > 0, torch.randn(4, 8), torch.randn(4, 8))),
    ("lt", lambda a, b: a < b, XY),
    ("eq_scalar", lambda a: a == 0, X),
    ("logical_not", lambda a: torch.logical_not(a > 0), X),
    ("expand", lambda a: a.expand(4, 8), (torch.randn(1, 8),)),
    ("slice", lambda a: a[:, 2:6], X),
    ("pow2", lambda a: a ** 2, X),
    ("mean", lambda a: a.mean(-1), X),
    ("softmax", lambda a: torch.softmax(a, -1), X),
    ("layer_norm", lambda a: torch.nn.functional.layer_norm(a, (8,), torch.ones(8), torch.zeros(8)), X),
    ("bmm", torch.bmm, (torch.randn(2, 4, 8), torch.randn(2, 8, 16))),
    # scan / arg-reduce / dtype-edge families
    ("cumsum", lambda a: torch.cumsum(a, -1), X),
    ("cumsum_bool", lambda a: torch.cumsum(a > 0, -1), X),
    ("min_dim_vals", lambda a: torch.min(a, dim=1, keepdim=True)[0], X),
    ("min_dim_idx", lambda a: torch.min(a, dim=1)[1], X),
    ("argmax", lambda a: torch.argmax(a, dim=-1), X),
    ("sum_bool_keepdim", lambda a: (a > 0).sum(0, keepdim=True), (torch.randn(8),)),
    ("reciprocal_int", lambda a: torch.reciprocal(a), (torch.tensor([2, 3, 4]),)),
    ("to_copy_bool2int", lambda a: (a > 0).to(torch.int64), X),
    ("rsub_scalar", lambda a: 1.0 - a, X),            # reverse sub: scalar - tensor
    ("rdiv_scalar", lambda a: 2.0 / (a.abs() + 1), X),
    ("bucketize", lambda a: torch.bucketize(a, torch.linspace(-1, 1, 7)), X),
    ("index_gather", lambda s, i0, i1: s[i0, i1],
     (torch.randn(4, 8), torch.zeros(1, 1, dtype=torch.int64), torch.arange(8).view(1, 8))),
    # generalized advanced indexing: partial / None-leading / mid-block (free dims preserved)
    ("index_partial_lead", lambda s, i: s[i], (torch.randn(4, 8), torch.tensor([0, 2, 1]))),
    ("index_none_lead", lambda s, i: s[:, i], (torch.randn(4, 8), torch.tensor([0, 2, 1]))),
    ("index_mid", lambda s, i: s[:, i, :], (torch.randn(2, 5, 7), torch.tensor([0, 1, 2]))),
    ("bitwise_not_bool", lambda a: ~(a > 0), X),
    ("bitwise_not_int", lambda a: ~a, (torch.randint(0, 100, (4, 8), dtype=torch.int32),)),
    ("repeat_tile", lambda a: a.repeat(2, 3), X),
    ("repeat_prepend", lambda a: a.repeat(2, 1, 1), (torch.randn(3, 8),)),
    ("mean_full", lambda a: a.mean(), X),                 # mean.default -> scalar
    ("mean_absmean", lambda a: a.abs().mean(), X),        # BitNet W1.58 absmean
    ("sdpa_4d", lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(q, k, v),
     (torch.randn(1, 4, 16, 8), torch.randn(1, 4, 16, 8), torch.randn(1, 4, 16, 8))),
    ("sdpa_mask", lambda q, k, v, m: torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=m),
     (torch.randn(1, 4, 8, 16), torch.randn(1, 4, 12, 16), torch.randn(1, 4, 12, 16), torch.randn(1, 1, 8, 12))),
    ("chunk", lambda a: a.chunk(3, -1)[1], (torch.randn(1, 1, 12),)),
    ("unbind", lambda a: torch.unbind(a, 2)[1], (torch.randn(1, 4, 3, 8),)),
    ("repeat_interleave", lambda a: a.repeat_interleave(2, dim=1), (torch.randn(1, 4, 8),)),
    ("matmul_3d_2d", lambda a, b: a @ b, (torch.randn(1, 30, 32), torch.randn(32, 64))),
    *_conv_cases(),
]

# Data-dependent ops: output has a dynamic (?) dim, so shape can't match eager exactly --
# assert only that they lower to standard dialects (scf + tensor) with no opaque calls.
DYNAMIC_CASES = [
    ("bool_mask_gather", lambda x, m: x[m], (torch.arange(8), torch.tensor([True, False] * 4))),
    ("masked_scatter", _masked_scatter := (lambda s, src, m: s.clone().index_put_((m,), src[m])),
     (torch.zeros(8, dtype=torch.int64), torch.arange(8), torch.tensor([True, False] * 4))),
]


@pytest.mark.parametrize("name,fn,inputs", CASES, ids=[c[0] for c in CASES])
def test_op_lowers_to_standard_dialects(name, fn, inputs):
    v = validate_op(fn, inputs, name=name)
    assert v.error is None, v.error
    assert v.lowered, f"{name} left opaque calls: {v.opaque_calls}"
    assert v.shape_ok, f"{name} result type {v.mlir_result_type} != eager {v.eager_shape}"


@pytest.mark.parametrize("name,fn,inputs", DYNAMIC_CASES, ids=[c[0] for c in DYNAMIC_CASES])
def test_dynamic_op_lowers_to_standard_dialects(name, fn, inputs):
    """Data-dependent ops lower to scf+tensor with a dynamic (?) dim -- no opaque calls."""
    v = validate_op(fn, inputs, name=name)
    assert v.error is None, v.error
    assert v.lowered, f"{name} left opaque calls: {v.opaque_calls}"


# The conv variants a consumer's vector schedule can claim. A schedule matches contractions
# BY OP NAME, so a conv that lowers to a fused conv-shaped linalg.generic is correct but gets
# neither vectorization nor a parallel-loop split -- it runs scalar. These assert conv lands on
# a real linalg.matmul, which is the property that makes conv fast on a matmul-only target.
_CONV_TO_MATMUL = [
    ("patchembed", torch.nn.Conv2d(3, 16, 4, stride=4), (1, 3, 16, 16)),
    ("padded", torch.nn.Conv2d(4, 8, 3, padding=1), (1, 4, 8, 8)),
    ("strided_padded", torch.nn.Conv2d(4, 8, 3, stride=2, padding=1), (1, 4, 8, 8)),
    ("dilated", torch.nn.Conv2d(4, 8, 3, padding=2, dilation=2), (1, 4, 8, 8)),
    ("conv1d", torch.nn.Conv1d(8, 12, 3, padding=1), (1, 8, 20)),
    ("transposed", torch.nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1), (1, 8, 6, 6)),
]


@pytest.mark.parametrize("name,mod,shape", _CONV_TO_MATMUL, ids=[c[0] for c in _CONV_TO_MATMUL])
def test_conv_lowers_to_a_named_matmul(name, mod, shape):
    import m2m

    torch.manual_seed(0)
    mod = mod.eval()

    class _W(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)

    r = m2m.convert(_W(mod).eval(), (torch.randn(*shape),), backend="fx_importer",
                    level="linalg-on-tensors")
    assert r.ok, f"{name} failed to convert"
    assert "linalg.matmul" in r.mlir_text, (
        f"{name} did not lower to a named linalg.matmul; a matmul-matching vector schedule "
        f"cannot claim it and it would run scalar")
    assert 'prov.conv_path = "im2col_matmul"' in r.mlir_text, (
        f"{name} took an unexpected conv path (prov.conv_path should record it)")


def test_grouped_conv_records_its_non_matmul_path():
    """A grouped conv is batched over the group axis, and xDSL registers no
    ``linalg.batch_matmul``, so it lands on a batched ``linalg.generic``. That is correct but
    NOT claimable by a name-matching contraction schedule -- assert the provenance says so
    rather than letting a scalar fallback be discovered on a board."""
    import m2m

    torch.manual_seed(0)
    dw = torch.nn.Conv2d(8, 8, 3, padding=1, groups=8).eval()

    class _W(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)

    r = m2m.convert(_W(dw).eval(), (torch.randn(1, 8, 6, 6),), backend="fx_importer",
                    level="linalg-on-tensors")
    assert r.ok
    assert "aten_convolution" not in r.mlir_text, "grouped conv fell back to an opaque call"
    assert 'prov.conv_path = "im2col_matmul"' in r.mlir_text
    assert "linalg.generic" in r.mlir_text


# ---------------------------------------------------------------------------
# torch.fft: complex tensors are carried as a real trailing-(re, im) pair and every transform
# is emitted as linalg.matmul against constant twiddles, so a spectral model inherits the
# consumer's contraction schedule with no FFT-specific lowering.
# ---------------------------------------------------------------------------


class _SpectralGating(torch.nn.Module):
    """SpectFormer's SpectralGatingNetwork: rfft2 -> complex gate -> irfft2, norm='ortho'."""

    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = torch.nn.Parameter(torch.randn(h, w, dim, 2) * 0.02)

    def forward(self, x):
        import math

        b_, n, c = x.shape
        a = b = int(math.sqrt(n))
        y = x.view(b_, a, b, c).to(torch.float32)
        y = torch.fft.rfft2(y, dim=(1, 2), norm="ortho")
        y = y * torch.view_as_complex(self.complex_weight)
        y = torch.fft.irfft2(y, s=(a, b), dim=(1, 2), norm="ortho")
        return y.reshape(b_, n, c)


def test_spectral_gating_lowers_to_contractions():
    """The whole rfft2 -> gate -> irfft2 chain must reach real matmuls with nothing opaque."""
    import m2m
    from m2m.coverage import opaque_report

    torch.manual_seed(0)
    r = m2m.convert(_SpectralGating(8).eval(), (torch.randn(1, 196, 8),),
                    backend="fx_importer", level="linalg-on-tensors")
    assert r.ok
    assert sum(opaque_report(r.mlir_text).values()) == 0, opaque_report(r.mlir_text)
    # 2 for the real-input forward axis, 4 for the complex forward axis, 4 + 2 inverse.
    assert r.mlir_text.count("linalg.matmul") == 12, r.mlir_text.count("linalg.matmul")
    # No complex ELEMENT TYPE may survive (a `prov.orig_dtype = "complex64"` annotation is
    # expected and wanted -- it records what the value logically was).
    assert "complex<" not in r.mlir_text, "a complex element type leaked into the IR"


@pytest.mark.parametrize("norm", ["ortho", "backward", "forward"])
def test_rfft_roundtrip_lowers(norm):
    """rfft2 -> irfft2 lowers for each normalization mode (the aten norm enum is read as data)."""
    import m2m
    from m2m.coverage import opaque_report

    class _RT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(6, 6, bias=False)

        def forward(self, x):
            y = self.lin(x)
            y = torch.fft.rfft2(y, dim=(1, 2), norm=norm)
            return torch.fft.irfft2(y, s=(8, 8), dim=(1, 2), norm=norm)

    torch.manual_seed(0)
    r = m2m.convert(_RT().eval(), (torch.randn(1, 8, 8, 6),), backend="fx_importer",
                    level="linalg-on-tensors")
    assert r.ok
    assert sum(opaque_report(r.mlir_text).values()) == 0, opaque_report(r.mlir_text)


def test_rfft1d_lowers():
    """A 1-D rfft goes through the same generic path -- no per-rank special case."""
    import m2m
    from m2m.coverage import opaque_report

    class _R1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(16, 16, bias=False)

        def forward(self, x):
            y = self.lin(x)
            y = torch.fft.rfft(y, dim=-1, norm="ortho")
            return torch.view_as_real(y)

    torch.manual_seed(0)
    r = m2m.convert(_R1().eval(), (torch.randn(1, 4, 16),), backend="fx_importer",
                    level="linalg-on-tensors")
    assert r.ok
    assert sum(opaque_report(r.mlir_text).values()) == 0, opaque_report(r.mlir_text)


def test_complex_unaware_op_fails_closed():
    """An op that is NOT written against the (re, im) pair layout must refuse a complex operand.

    Its xDSL value has one more axis than torch's logical shape, so slicing or padding by
    logical dim would hit the wrong axis and produce plausible wrong numbers. The guard turns
    that into a visible opaque call instead.
    """
    import m2m
    from m2m.coverage import opaque_report

    class _SliceComplex(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4, bias=False)

        def forward(self, x):
            y = torch.fft.rfft2(self.lin(x), dim=(1, 2))
            return torch.view_as_real(y[:, :, 1:])

    torch.manual_seed(0)
    r = m2m.convert(_SliceComplex().eval(), (torch.randn(1, 6, 6, 4),), backend="fx_importer",
                    level="linalg-on-tensors")
    assert r.ok, "must still produce a module"
    opq = opaque_report(r.mlir_text)
    assert sum(opq.values()) > 0, "a complex slice must NOT be silently lowered as a real slice"
