""" Gradient correctness of the DTCWT and ScatterNet autograd Functions.

The gradchecks in test_dtcwt_grad.py and test_scatnet_bwd.py run on full-size
inputs and take minutes, so they are marked slow and never run on a pull
request. These cover the same backward passes - every branch of them - on
inputs small enough to check in a second or two, so that a change breaking a
gradient fails CI where it is proposed rather than on the next weekly run.
"""
import pytest
import torch
from torch.autograd import gradcheck

from pytorch_wavelets import DTCWTForward, DTCWTInverse, ScatLayer, ScatLayerj2
from pytorch_wavelets.dtcwt import transform_funcs as tf
from pytorch_wavelets.dwt.lowlevel import mode_to_int

SYMMETRIC = mode_to_int('symmetric')


@pytest.fixture
def double():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _rand(*shape, grad=True):
    return torch.randn(*shape, dtype=torch.float64, requires_grad=grad)


def _check(fn, inputs):
    # fast_mode compares the analytical and numerical Jacobians along random
    # directions rather than building both in full. A wrong backward still
    # fails it, and the ScatterNet's many outputs make the full check cost a
    # minute here rather than seconds.
    assert gradcheck(fn, inputs, eps=1e-6, atol=1e-6, fast_mode=True)


# The DTCWT Functions, one level at a time. (o_dim, ri_dim) pairs cover the
# default layout and one with the orientations before the channels.
LAYOUTS = [(2, -1), (1, 2)]


@pytest.mark.parametrize('o_dim, ri_dim', LAYOUTS)
@pytest.mark.parametrize('skip_hps', [False, True])
def test_fwd_j1(double, skip_hps, o_dim, ri_dim):
    f = DTCWTForward(J=1)
    _check(lambda x: tf.FWD_J1.apply(x, f.h0o, f.h1o, skip_hps, o_dim, ri_dim,
                                     SYMMETRIC),
           (_rand(1, 2, 8, 8),))


@pytest.mark.parametrize('o_dim, ri_dim', LAYOUTS)
@pytest.mark.parametrize('skip_hps', [False, True])
def test_fwd_j2plus(double, skip_hps, o_dim, ri_dim):
    f = DTCWTForward(J=2)
    _check(lambda x: tf.FWD_J2PLUS.apply(x, f.h0a, f.h1a, f.h0b, f.h1b,
                                         skip_hps, o_dim, ri_dim, SYMMETRIC),
           (_rand(1, 2, 8, 8),))


# INV_J1 and INV_J2PLUS have a separate branch for each combination of
# inputs needing a gradient.
GRADS = [(True, True), (True, False), (False, True)]


@pytest.mark.parametrize('low_grad, high_grad', GRADS)
def test_inv_j1(double, low_grad, high_grad):
    g = DTCWTInverse()
    low, high = _rand(1, 2, 8, 8, grad=low_grad), \
        _rand(1, 2, 6, 4, 4, 2, grad=high_grad)
    _check(lambda lo, hi: tf.INV_J1.apply(lo, hi, g.g0o, g.g1o, 2, -1,
                                          SYMMETRIC),
           (low, high))


@pytest.mark.parametrize('low_grad, high_grad', GRADS)
def test_inv_j2plus(double, low_grad, high_grad):
    g = DTCWTInverse()
    low, high = _rand(1, 2, 8, 8, grad=low_grad), \
        _rand(1, 2, 6, 4, 4, 2, grad=high_grad)
    _check(lambda lo, hi: tf.INV_J2PLUS.apply(lo, hi, g.g0a, g.g1a, g.g0b,
                                              g.g1b, 2, -1, SYMMETRIC),
           (low, high))


def test_dtcwt_end_to_end(double):
    """ Through the modules, three levels, including the odd-size and
    divisible-by-4 padding they do between levels. """
    xfm, ifm = DTCWTForward(J=3), DTCWTInverse()
    _check(lambda x: ifm(xfm(x)), (_rand(1, 1, 13, 18),))


# The ScatterNet layers. near_sym_b_bp takes the *_rot code paths, which
# nothing else exercises; combine_colour has its own backward branches.
@pytest.mark.parametrize('combine_colour', [False, True])
@pytest.mark.parametrize('biort', ['near_sym_a', 'near_sym_b_bp'])
def test_scatlayer(double, biort, combine_colour):
    scat = ScatLayer(biort=biort, combine_colour=combine_colour)
    _check(scat, (_rand(1, 3, 8, 8),))


@pytest.mark.parametrize('combine_colour', [False, True])
@pytest.mark.parametrize('biort, qshift', [('near_sym_a', 'qshift_a'),
                                           ('near_sym_b_bp', 'qshift_b_bp')])
def test_scatlayerj2(double, biort, qshift, combine_colour):
    scat = ScatLayerj2(biort=biort, qshift=qshift,
                       combine_colour=combine_colour)
    _check(scat, (_rand(1, 3, 8, 8),))


@pytest.mark.parametrize('size', [7, 10])
def test_scatlayerj2_pads_odd_sizes(double, size):
    _check(ScatLayerj2(), (_rand(1, 1, size, size),))
