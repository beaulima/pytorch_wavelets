""" Executes the examples printed in the documentation.

The docs are the contract this library advertises, and every claim below is
copied from a .rst file. If one of these fails, either the code regressed or
the doc is lying - both are bugs.
"""
import numpy as np
import pytest
import torch

from pytorch_wavelets import (DWTForward, DWTInverse, DWT1DForward,
                              DWT1DInverse, DTCWTForward, DTCWTInverse,
                              ScatLayer, ScatLayerj2, SWTForward,
                              SWTInverse)


def test_dwt_example():
    """ docs/dwt.rst, "Example" """
    xfm = DWTForward(J=3, mode='zero', wave='db3')
    ifm = DWTInverse(mode='zero', wave='db3')
    X = torch.randn(10, 5, 64, 64)
    Yl, Yh = xfm(X)
    assert tuple(Yl.shape) == (10, 5, 12, 12)
    assert tuple(Yh[0].shape) == (10, 5, 3, 34, 34)
    assert tuple(Yh[1].shape) == (10, 5, 3, 19, 19)
    assert tuple(Yh[2].shape) == (10, 5, 3, 12, 12)
    Y = ifm((Yl, Yh))
    np.testing.assert_array_almost_equal(Y.numpy(), X.numpy())


@pytest.mark.parametrize('mode', ['symmetric', 'reflect', 'zero',
                                  'periodization'])
def test_dwt_padding_schemes(mode):
    """ docs/dwt.rst lists these four schemes as supported. """
    X = torch.randn(1, 1, 32, 32)
    xfm = DWTForward(J=2, mode=mode, wave='db2')
    ifm = DWTInverse(mode=mode, wave='db2')
    np.testing.assert_allclose(ifm(xfm(X)).numpy(), X.numpy(), atol=1e-5)


def test_dtcwt_example():
    """ docs/dtcwt.rst, "Example" """
    xfm = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')
    X = torch.randn(10, 5, 64, 64)
    Yl, Yh = xfm(X)
    assert tuple(Yl.shape) == (10, 5, 16, 16)
    assert tuple(Yh[0].shape) == (10, 5, 6, 32, 32, 2)
    assert tuple(Yh[1].shape) == (10, 5, 6, 16, 16, 2)
    assert tuple(Yh[2].shape) == (10, 5, 6, 8, 8, 2)
    ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
    np.testing.assert_allclose(ifm((Yl, Yh)).numpy(), X.numpy(), atol=1e-5)


def test_dtcwt_include_scale():
    """ docs/dtcwt.rst, "Including all the lowpasses" """
    x = torch.randn(1, 1, 64, 64)
    yl, _ = DTCWTForward(J=3, include_scale=True)(x)
    assert [tuple(s.shape) for s in yl] == [(1, 1, 64, 64), (1, 1, 32, 32),
                                            (1, 1, 16, 16)]
    yl, _ = DTCWTForward(J=3, include_scale=[False, True, True])(x)
    # A skipped scale is a 0-dim placeholder, i.e. torch.Size([]).
    assert tuple(yl[0].shape) == ()
    assert tuple(yl[1].shape) == (1, 1, 32, 32)


def test_dtcwt_inverse_takes_no_J():
    """ README "New in version 1.1.0": removed the need to specify the number
    of scales for DTCWTInverse. docs/readme.rst used to still pass J=3. """
    with pytest.raises(TypeError):
        DTCWTInverse(J=3, biort='near_sym_b', qshift='qshift_b')


def test_scatternet_example():
    """ README, "New in version 1.2.0" """
    X = torch.randn(10, 5, 64, 64)
    assert tuple(ScatLayer()(X).shape) == (10, 35, 32, 32)
    stacked = torch.nn.Sequential(ScatLayer(), ScatLayer())
    assert tuple(stacked(X).shape) == (10, 245, 16, 16)
    assert tuple(ScatLayerj2()(X).shape) == (10, 245, 16, 16)


def test_scatlayerj2_records_its_qshift():
    """ ScatLayerj2 used to store `biort` in self.qshift, so introspection and
    the repr reported the wrong second-level filter. """
    layer = ScatLayerj2(biort='near_sym_a', qshift='qshift_b')
    assert layer.qshift == 'qshift_b'
    assert "qshift='qshift_b'" in repr(layer)


def test_dwt1d_roundtrip():
    """ README "New in version 1.3.0": added 1D DWT support. """
    x = torch.randn(10, 5, 64)
    xfm, ifm = DWT1DForward(J=3, wave='db3'), DWT1DInverse(wave='db3')
    np.testing.assert_allclose(ifm(xfm(x)).numpy(), x.numpy(), atol=1e-5)


def test_float64_via_default_dtype():
    """ docs/readme.rst, "Floating Point Type": set the default dtype before
    constructing the transforms to get 64-bit filters. """
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        x = torch.randn(1, 2, 32, 32, dtype=torch.float64)
        xfm = DWTForward(J=2, wave='db3', mode='periodization')
        ifm = DWTInverse(wave='db3', mode='periodization')
        assert xfm.h0_col.dtype == torch.float64
        np.testing.assert_allclose(ifm(xfm(x)).numpy(), x.numpy(), atol=1e-12)
    finally:
        torch.set_default_dtype(old)


def test_dwt_separable_flag():
    """ docs/dwt.rst: "You can test the two out to see which is better for you
    by changing the `separable` flag in the DWT/IDWT constructor." """
    x = torch.randn(1, 2, 32, 32)
    for separable in (True, False):
        xfm = DWTForward(J=2, wave='db2', separable=separable)
        ifm = DWTInverse(wave='db2', separable=separable)
        np.testing.assert_allclose(
            ifm(xfm(x)).numpy(), x.numpy(), atol=1e-5)


def test_swt_example():
    """ docs/dwt.rst, "Stationary (undecimated) WT" """
    sfm = SWTForward(J=3, wave='db3')
    ifm = SWTInverse(wave='db3')
    X = torch.randn(10, 5, 64, 64)
    coeffs = sfm(X)
    assert len(coeffs) == 3
    assert tuple(coeffs[0].shape) == (10, 5, 4, 64, 64)
    np.testing.assert_allclose(ifm(coeffs).numpy(), X.numpy(), atol=1e-4)


def test_scatternet_filters_are_buffers_not_parameters():
    """ The filters are fixed, so they belong in the buffer registry. As
    nn.Parameter(requires_grad=False) they showed up in .parameters() and an
    optimiser would apply weight decay to them.

    The state_dict keys are unchanged either way, so existing checkpoints keep
    loading.
    """
    for layer in (ScatLayer(), ScatLayerj2()):
        assert list(layer.parameters()) == []
        assert len(list(layer.buffers())) > 0
    assert sorted(ScatLayer().state_dict()) == ['h0o', 'h1o']
    assert sorted(ScatLayerj2().state_dict()) == [
        'h0a', 'h0b', 'h0o', 'h1a', 'h1b', 'h1o']


def test_mode_helpers_are_shared():
    """ mode_to_int / int_to_mode were duplicated verbatim between the DWT and
    scatternet modules; they are now one definition. """
    from pytorch_wavelets.dwt.lowlevel import mode_to_int as dwt_m2i
    from pytorch_wavelets.scatternet.lowlevel import mode_to_int as scat_m2i
    assert dwt_m2i is scat_m2i
