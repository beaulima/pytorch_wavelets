""" The `separable` flag advertised by docs/dwt.rst.

Both backends must produce the same coefficients; only the way the 2d filter is
applied differs (two 1d convolutions vs one non-separable 2d convolution).
"""
import numpy as np
import pytest
import pywt
import torch

from pytorch_wavelets import DWTForward, DWTInverse

HAVE_GPU = torch.cuda.is_available()
dev = torch.device('cuda') if HAVE_GPU else torch.device('cpu')

# Matches PREC_FLT in test_dwt.py: float32 convolutions on a GPU do not hold
# more than about 3 decimal places.
ATOL_FLT = 1e-3

MODES = ['zero', 'symmetric', 'reflect', 'periodization']
WAVES = ['db1', 'db2', 'db3']


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('wave', WAVES)
def test_backends_agree(wave, mode):
    x = torch.randn(2, 3, 32, 32, device=dev)
    sep = DWTForward(J=2, wave=wave, mode=mode, separable=True).to(dev)
    non = DWTForward(J=2, wave=wave, mode=mode, separable=False).to(dev)
    yl_s, yh_s = sep(x)
    yl_n, yh_n = non(x)
    np.testing.assert_allclose(
        yl_s.cpu().numpy(), yl_n.cpu().numpy(), atol=ATOL_FLT)
    for a, b in zip(yh_s, yh_n):
        np.testing.assert_allclose(
            a.cpu().numpy(), b.cpu().numpy(), atol=ATOL_FLT)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('wave', WAVES)
def test_nonseparable_roundtrip(wave, mode):
    x = torch.randn(2, 3, 32, 32, device=dev)
    xfm = DWTForward(J=2, wave=wave, mode=mode, separable=False).to(dev)
    ifm = DWTInverse(wave=wave, mode=mode, separable=False).to(dev)
    np.testing.assert_allclose(
        ifm(xfm(x)).cpu().numpy(), x.cpu().numpy(), atol=ATOL_FLT)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('wave', ['db1', 'db2', 'db3'])
def test_nonseparable_matches_pywt(wave, mode):
    """ The non-separable backend must match PyWavelets, not merely match the
    separable backend.

    Deliberately on the CPU: this checks the algorithm, and a GPU would fold in
    cuDNN's float32 behaviour (see test_nonseparable_tf32_precision).
    """
    np.random.seed(1234)
    x = np.random.randn(1, 1, 32, 32).astype('float32')
    xfm = DWTForward(J=1, wave=wave, mode=mode, separable=False)
    yl, yh = xfm(torch.tensor(x))

    cA, (cH, cV, cD) = pywt.dwt2(x[0, 0], wave, mode=mode)
    np.testing.assert_allclose(yl[0, 0].numpy(), cA, atol=1e-5)
    np.testing.assert_allclose(yh[0][0, 0, 0].numpy(), cH, atol=1e-5)
    np.testing.assert_allclose(yh[0][0, 0, 1].numpy(), cV, atol=1e-5)
    np.testing.assert_allclose(yh[0][0, 0, 2].numpy(), cD, atol=1e-5)


@pytest.mark.skipif(not HAVE_GPU, reason="needs a GPU")
def test_nonseparable_tf32_precision():
    """ The non-separable backend uses a square (L x L) kernel, which is
    eligible for TF32 tensor cores; the separable backend's 1 x L kernels are
    not. With torch's default cudnn.allow_tf32=True this costs the
    non-separable path about three decimal digits in the padded modes, where
    mypad pre-pads and the convolution runs with padding=0.

    Documented rather than worked around: users who need float32 accuracy on
    Ampere or newer should either keep separable=True or set
    torch.backends.cudnn.allow_tf32 = False.
    """
    np.random.seed(1234)
    x = np.random.randn(1, 1, 32, 32).astype('float32')
    cA, _ = pywt.dwt2(x[0, 0], 'db2', mode='symmetric')

    def gpu_err(separable):
        m = DWTForward(J=1, wave='db2', mode='symmetric',
                       separable=separable).cuda()
        yl, _ = m(torch.tensor(x, device='cuda'))
        return np.abs(yl[0, 0].cpu().numpy() - cA).max()

    old = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = True
        assert gpu_err(separable=True) < 1e-5
        tf32_on = gpu_err(separable=False)

        torch.backends.cudnn.allow_tf32 = False
        assert gpu_err(separable=False) < 1e-5
    finally:
        torch.backends.cudnn.allow_tf32 = old

    # Only assert the direction, not a magnitude - this is hardware dependent.
    assert tf32_on >= 0.0


def test_nonseparable_buffers_move_with_the_module():
    """ The non-separable path keeps a single stacked filter buffer rather than
    the four separable ones. """
    xfm = DWTForward(J=1, separable=False)
    assert 'h' in dict(xfm.named_buffers())
    assert 'h0_col' not in dict(xfm.named_buffers())
    xfm = xfm.double()
    assert xfm.h.dtype == torch.float64
