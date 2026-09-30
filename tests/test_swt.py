import numpy as np
import pytest
import pywt
import torch

from pytorch_wavelets import SWTForward, SWTInverse

HAVE_GPU = torch.cuda.is_available()
if HAVE_GPU:
    dev = torch.device('cuda')
else:
    dev = torch.device('cpu')


def setup_module():
    global x, x_t
    np.random.seed(0)
    x = np.random.randn(3, 32, 32).astype('float32')
    x_t = torch.tensor(x, device=dev)[None]


@pytest.mark.parametrize('J', [1, 2, 3])
def test_shape(J):
    """ Each scale keeps the input size and splits the 4 subbands onto their
    own axis - see the docstring of SWTForward. """
    sfm = SWTForward(J=J, wave='db2').to(dev)
    coeffs = sfm(x_t)
    assert len(coeffs) == J
    for c in coeffs:
        assert tuple(c.shape) == (1, 3, 4, 32, 32)


@pytest.mark.parametrize('wave', ['db1', 'db2', 'db3'])
@pytest.mark.parametrize('J', [1, 2, 3])
def test_equal_pywt(wave, J):
    sfm = SWTForward(J=J, wave=wave, mode='periodization').to(dev)
    coeffs = sfm(x_t)

    for ch in range(x.shape[0]):
        # pywt returns coarsest scale first, we return finest first.
        ref = pywt.swt2(x[ch], wave, level=J)[::-1]
        for j in range(J):
            cA, (cH, cV, cD) = ref[j]
            got = coeffs[j][0, ch].cpu().numpy()
            np.testing.assert_allclose(got[0], cA, atol=1e-5)
            np.testing.assert_allclose(got[1], cH, atol=1e-5)
            np.testing.assert_allclose(got[2], cV, atol=1e-5)
            np.testing.assert_allclose(got[3], cD, atol=1e-5)


def test_multilevel_lowpass_feeds_next_scale():
    """ Regression: the lowpass used to be sliced off the wrong axis, which
    silently produced a 3d tensor and broke every J > 1 transform. """
    sfm = SWTForward(J=2, wave='db2').to(dev)
    c1, c2 = sfm(x_t)
    sfm1 = SWTForward(J=1, wave='db2').to(dev)
    np.testing.assert_allclose(
        c1.cpu().numpy(), sfm1(x_t)[0].cpu().numpy(), atol=1e-6)


# ---------------------------------------------------------------- inverse ---

@pytest.mark.parametrize('wave', ['db1', 'db2', 'db3', 'sym4', 'coif2',
                                  'bior2.4'])
@pytest.mark.parametrize('J', [1, 2, 3])
def test_inverse_roundtrip(wave, J):
    """ Perfect reconstruction, at float64 precision. """
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        x = torch.randn(2, 3, 32, 32, dtype=torch.float64, device=dev)
        sfm = SWTForward(J=J, wave=wave).to(dev)
        ifm = SWTInverse(wave=wave).to(dev)
        np.testing.assert_allclose(
            ifm(sfm(x)).cpu().numpy(), x.cpu().numpy(), atol=1e-10)
    finally:
        torch.set_default_dtype(old)


@pytest.mark.parametrize('wave', ['db1', 'db2', 'db3'])
@pytest.mark.parametrize('J', [1, 2, 3])
def test_inverse_equals_pywt_iswt2(wave, J):
    """ Fed PyWavelets' own coefficients, we must return what pywt.iswt2
    returns. """
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        np.random.seed(0)
        im = np.random.randn(32, 32)
        ref = pywt.swt2(im, wave, level=J)

        # pywt is coarsest first, we are finest first.
        coeffs = [torch.tensor(np.stack([cA, cH, cV, cD])[None, None],
                               dtype=torch.float64, device=dev)
                  for cA, (cH, cV, cD) in ref[::-1]]
        ours = SWTInverse(wave=wave).to(dev)(coeffs).cpu().numpy()[0, 0]

        np.testing.assert_allclose(ours, pywt.iswt2(ref, wave), atol=1e-10)
    finally:
        torch.set_default_dtype(old)


@pytest.mark.parametrize('mode', ['zero', 'symmetric', 'reflect'])
def test_inverse_non_periodic_recovers_the_interior(mode):
    """ The undecimated reconstruction identity assumes circular convolution,
    so only periodic extension gives perfect reconstruction. The interior is
    still exact - see the note on SWTInverse. """
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        x = torch.randn(1, 1, 32, 32, dtype=torch.float64, device=dev)
        rec = SWTInverse(wave='db2', mode=mode).to(dev)(
            SWTForward(J=2, wave='db2', mode=mode).to(dev)(x))
        d = (rec - x).abs().cpu().numpy()
        np.testing.assert_allclose(d[..., 8:-8, 8:-8], 0, atol=1e-10)
        assert d.max() > 1e-6, "expected a boundary mismatch"
    finally:
        torch.set_default_dtype(old)


def test_swt_is_exported():
    import pytorch_wavelets as pw
    assert pw.SWTForward is SWTForward and pw.SWTInverse is SWTInverse
    assert pw.SWT is pw.SWTForward and pw.ISWT is pw.SWTInverse


@pytest.mark.parametrize('mode', ['periodization', 'symmetric', 'zero'])
def test_gradients_flow(mode):
    """ The SWT is built from differentiable primitives with no custom
    backward, so autograd differentiates it exactly. """
    x = torch.randn(1, 2, 32, 32, device=dev, requires_grad=True)
    sfm = SWTForward(J=2, wave='db2', mode=mode).to(dev)
    ifm = SWTInverse(wave='db2', mode=mode).to(dev)
    ifm(sfm(x)).pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
