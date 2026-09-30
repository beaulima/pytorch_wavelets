"""
Getting started with the DWT
============================

The 2D discrete wavelet transform, its inverse, and the shape of what comes
back. Everything here runs on a batch of images at once, on CPU or GPU, and
gradients flow through it - which is the reason for doing wavelets in pytorch
rather than in numpy.
"""

# %%
# Objective
# ---------
#
# Establish what this package's DWT returns and how to read it: the shape and
# ordering of the coefficients, which padding schemes reconstruct exactly, and
# that the transform is an ordinary batched, differentiable pytorch module.
# Later examples build on these conventions.

# %%
# Environment
# -----------
#
# Imports, versions and the random seed in one cell, so that running it is
# enough to set the notebook up. Every number below can be checked
# against a rerun. Following the reproducibility conventions in Rule et al.
# (2019), every figure and every quantity in this notebook is produced by the
# code above it - nothing is quoted from a previous run.

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch
from skimage import data

import pytorch_wavelets
from pytorch_wavelets import DWTForward, DWTInverse


def to_tensor(im):
    """Greyscale float image in [0, 1] as a (1, 1, H, W) tensor."""
    if im.ndim == 3:
        im = im.mean(axis=2)
    im = im.astype('float32')
    im = (im - im.min()) / (np.ptp(im) + 1e-9)
    return torch.tensor(im)[None, None]


SEED = 0
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

for name, mod in [('pytorch_wavelets', pytorch_wavelets), ('torch', torch),
                  ('numpy', np), ('scikit-image', skimage),
                  ('matplotlib', matplotlib)]:
    print('%-16s %s' % (name, mod.__version__))
print('%-16s %d' % ('seed', SEED))

# %%
# Data
# ----
#
# ``camera`` and ``astronaut`` from ``skimage.data``, which ships them with the
# library under documented provenance. Both are converted to float in [0, 1].
# No image files are committed to this repository.


# %%
# The transform expects the usual pytorch image layout, ``(N, C, H, W)``.

x = to_tensor(data.camera())
print('input:', tuple(x.shape))

# %%
# A three level decomposition. ``yl`` is the final lowpass; ``yh`` is a list
# with one entry per scale, **finest first** - the opposite of PyWavelets, but
# it matches the dtcwt convention. Each entry stacks the three subbands
# (LH, HL, HH) along a new third axis, so a 4D input gives 5D bandpasses.

xfm = DWTForward(J=3, wave='db3', mode='symmetric')
yl, yh = xfm(x)

print('lowpass: ', tuple(yl.shape))
for j, band in enumerate(yh):
    print('scale %d:  %s' % (j + 1, tuple(band.shape)))

# %%
# Reconstruction is exact to floating point precision.

ifm = DWTInverse(wave='db3', mode='symmetric')
print('max reconstruction error: %.2e' % (ifm((yl, yh)) - x).abs().max())

# %%
# What the subbands look like. The lowpass is a blurred, decimated copy; each
# bandpass holds the detail lost at that step, split three ways.

fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
axes[0].imshow(yl[0, 0], cmap='gray')
axes[0].set_title('lowpass, scale 3')
for i, name in enumerate(['LH (horizontal)', 'HL (vertical)', 'HH (diagonal)']):
    band = yh[0][0, 0, i]
    lim = band.abs().max() * 0.4      # symmetric limits put zero at mid grey
    axes[i + 1].imshow(band, cmap='gray', vmin=-lim, vmax=lim)
    axes[i + 1].set_title(name)
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle('One level of the DWT: three orientations')
plt.tight_layout()

# %%
# Padding schemes
# ---------------
#
# The signal has to be extended at the borders, and the choice shows up in the
# coefficient count. ``periodization`` is the only one that stays critically
# sampled - as many coefficients out as samples in - at the cost of wrapping
# the image onto itself. All of them reconstruct exactly.

for mode in ['zero', 'symmetric', 'reflect', 'periodization']:
    xf = DWTForward(J=1, wave='db3', mode=mode)
    inv = DWTInverse(wave='db3', mode=mode)
    coeffs = xf(x)
    print('%-14s lowpass %s  reconstruction %.1e'
          % (mode, tuple(coeffs[0].shape[-2:]), (inv(coeffs) - x).abs().max()))

# %%
# Batches, channels and the GPU
# -----------------------------
#
# Nothing above is special-cased for one greyscale image. Colour is three
# channels, a batch is a batch, and the filters are applied per channel.

colour = torch.tensor(
    data.astronaut().astype('float32').transpose(2, 0, 1) / 255.)[None]
batch = colour.repeat(4, 1, 1, 1)
yl_b, yh_b = xfm(batch)
print('batch in: ', tuple(batch.shape))
print('lowpass:  ', tuple(yl_b.shape))
print('finest:   ', tuple(yh_b[0].shape))

if torch.cuda.is_available():
    yl_g, _ = DWTForward(J=3, wave='db3').cuda()(batch.cuda())
    print('on gpu:   ', tuple(yl_g.shape), yl_g.device)
else:
    print('no gpu here; .cuda() on the module is all it takes')

# %%
# Set the default dtype *before* building the transform to work in float64.
# The filter coefficients are built at construction time, so converting the
# module afterwards would leave them rounded to float32.

old = torch.get_default_dtype()
torch.set_default_dtype(torch.float64)
try:
    xfm64 = DWTForward(J=3, wave='db3', mode='symmetric')
    rec = DWTInverse(wave='db3', mode='symmetric')(xfm64(x.double()))
    print('float64 reconstruction error: %.2e' % (rec - x.double()).abs().max())
finally:
    torch.set_default_dtype(old)

# %%
# References
# ----------
#
# - S. Mallat, "A theory for multiresolution signal decomposition: the wavelet
#   representation", *IEEE Transactions on Pattern Analysis and Machine
#   Intelligence*, 11(7):674-693, 1989.
# - A. Rule et al., "Ten simple rules for writing and sharing computational
#   analyses in Jupyter Notebooks", *PLOS Computational Biology*,
#   15(7):e1007007, 2019.
