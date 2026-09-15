""" Gradient correctness of the custom autograd Functions.

These run on small float64 inputs so they are fast enough to run every time,
unlike the ones in test_dwt_grad.py / test_dtcwt_grad.py which are skipped.

These Functions used to compute their backward pass as a synthesis filter bank
applied to the incoming gradient, which is the exact adjoint of the forward
convolution only when the padding is zero or periodic. The padded modes now go
through a real adjoint instead: transposed convolution followed by a
scatter-add that folds every padded copy's gradient back onto the sample it was
copied from.
"""
import pytest
import torch
from torch.autograd import gradcheck

from pytorch_wavelets import (DWTForward, DWTInverse, DWT1DForward,
                              DWT1DInverse)
from pytorch_wavelets.dwt.lowlevel import (AFB1D, AFB2D, SFB1D, SFB2D,
                                           AFB2D_nonsep, SFB2D_nonsep,
                                           afb2d, mode_to_int)

MODES = ['zero', 'symmetric', 'reflect', 'periodization', 'periodic']


@pytest.fixture
def double():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.float64, requires_grad=True)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('wave', ['db1', 'db2', 'db3'])
def test_afb2d_gradient(double, wave, mode):
    f = DWTForward(J=1, wave=wave, mode=mode)
    assert gradcheck(AFB2D.apply,
                     (_rand(1, 2, 12, 12), f.h0_col, f.h1_col, f.h0_row,
                      f.h1_row, mode_to_int(mode)), eps=1e-6, atol=1e-6)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('wave', ['db1', 'db2', 'db3'])
def test_afb2d_nonsep_gradient(double, wave, mode):
    f = DWTForward(J=1, wave=wave, mode=mode, separable=False)
    assert gradcheck(AFB2D_nonsep.apply,
                     (_rand(1, 2, 12, 12), f.h, mode_to_int(mode)),
                     eps=1e-6, atol=1e-6)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('wave', ['db1', 'db2', 'db3'])
def test_sfb2d_gradient(double, wave, mode):
    f = DWTInverse(wave=wave, mode=mode)
    assert gradcheck(SFB2D.apply,
                     (_rand(1, 2, 8, 8), _rand(1, 2, 3, 8, 8), f.g0_col,
                      f.g1_col, f.g0_row, f.g1_row, mode_to_int(mode)),
                     eps=1e-6, atol=1e-6)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('wave', ['db1', 'db2', 'db3'])
def test_sfb2d_nonsep_gradient(double, wave, mode):
    f = DWTInverse(wave=wave, mode=mode, separable=False)
    assert gradcheck(SFB2D_nonsep.apply,
                     (_rand(1, 2, 8, 8), _rand(1, 2, 3, 8, 8), f.g,
                      mode_to_int(mode)), eps=1e-6, atol=1e-6)


@pytest.mark.parametrize('mode', MODES)
def test_afb1d_gradient(double, mode):
    f = DWT1DForward(J=1, wave='db3', mode=mode)
    assert gradcheck(AFB1D.apply,
                     (_rand(1, 2, 16), f.h0, f.h1, mode_to_int(mode)),
                     eps=1e-6, atol=1e-6)


@pytest.mark.parametrize('mode', MODES)
def test_sfb1d_gradient(double, mode):
    f = DWT1DInverse(wave='db3', mode=mode)
    assert gradcheck(SFB1D.apply,
                     (_rand(1, 2, 8), _rand(1, 2, 8), f.g0, f.g1,
                      mode_to_int(mode)), eps=1e-6, atol=1e-6)


@pytest.mark.parametrize('size', [11, 12, 16])
@pytest.mark.parametrize('mode', MODES)
def test_gradient_matches_autograd(double, mode, size):
    """ afb2d() is built purely from differentiable primitives, so autograd
    through it is the true gradient of the very same forward computation that
    AFB2D performs. They must agree.

    This is the regression test for two defects: the backward pass was not the
    adjoint at all for symmetric/reflect padding, and in periodization mode it
    cropped away the gradient of the sample afb1d appends to odd-length inputs
    instead of folding it back.
    """
    f = DWTForward(J=1, wave='db3', mode=mode)
    filts = (f.h0_col, f.h1_col, f.h0_row, f.h1_row)

    x1 = _rand(1, 1, size, size)
    x2 = x1.detach().clone().requires_grad_(True)

    y_ref = afb2d(x1, filts, mode=mode)
    g = torch.randn_like(y_ref)
    y_ref.backward(g)

    low, highs = AFB2D.apply(x2, *filts, mode_to_int(mode))
    torch.cat((low[:, :, None], highs), dim=2).reshape(y_ref.shape).backward(g)

    torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize('size', [11, 16])
@pytest.mark.parametrize('mode', MODES)
def test_end_to_end_gradient(double, mode, size):
    """ The user-facing modules must pass gradients too, both backends. """
    for separable in (True, False):
        x = _rand(1, 2, size, size)
        xfm = DWTForward(J=2, wave='db2', mode=mode, separable=separable)
        ifm = DWTInverse(wave='db2', mode=mode, separable=separable)
        ifm(xfm(x)).pow(2).sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
