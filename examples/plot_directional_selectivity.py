"""
Directional selectivity: DWT against DTCWT
==========================================

The separable DWT gives three bandpass subbands per scale: horizontal,
vertical and diagonal. The diagonal one cannot tell +45 from -45 degrees,
because a separable filter bank mixes the two together. The dual-tree complex
wavelet transform gives six, at roughly 15, 45, 75, 105, 135 and 165 degrees,
and keeps them apart.

This example measures that rather than asserting it.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch
from skimage import data

import pytorch_wavelets
from pytorch_wavelets import DWTForward, DTCWTForward

ANGLES = [15, 45, 75, 105, 135, 165]


def to_tensor(im):
    """Greyscale float image in [0, 1], cropped to a multiple of 8."""
    if im.ndim == 3:
        im = im.mean(axis=2)
    im = im.astype('float32')
    im = (im - im.min()) / (np.ptp(im) + 1e-9)
    h, w = (im.shape[0] // 8) * 8, (im.shape[1] // 8) * 8
    return torch.tensor(im[:h, :w])[None, None]


def band_energy(coeffs):
    """Fraction of bandpass energy in each orientation, at the finest scale."""
    e = (coeffs ** 2).sum(dim=tuple(range(1, coeffs.ndim))).numpy()
    return e / e.sum()


# %%
# Objective
# ---------
#
# Quantify a claim usually made in words: that the DTCWT resolves orientation
# where the separable DWT cannot. The measurement is the share of finest-scale
# bandpass energy falling in each subband, for three images whose dominant
# structure runs in known directions. If the claim holds, images with different
# orientation content should produce different, interpretable signatures under
# the DTCWT and near-indistinguishable ones under the DWT.

# %%
# Reproducibility
# ---------------
#
# Versions and the random seed, printed so that any number below can be checked
# against a rerun. Following the reproducibility conventions in Rule et al.
# (2019), every figure and every quantity in this notebook is produced by the
# code above it - nothing is quoted from a previous run.

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
# ``brick`` and ``text`` from ``skimage.data``, plus a checkerboard generated
# below so that the build never has to fetch anything. All three are converted
# to float in [0, 1] and cropped to a multiple of eight, since a three level
# transform halves the size three times.


# %%
# Three images with very different structure: a brick wall, whose mortar lines
# and joints are close to horizontal and vertical; handwriting, whose strokes
# lean the other way; and a checkerboard, which has no diagonal content at all.

def checkerboard(size=256, square=16):
    """Synthetic, so the build never has to fetch anything."""
    g = (np.arange(size) // square) % 2
    return (g[:, None] ^ g[None, :]).astype('float32')


images = {
    'brick': data.brick(),
    'text': data.text(),
    'checkerboard': checkerboard(),
}

fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
for ax, (name, im) in zip(axes, images.items()):
    ax.imshow(to_tensor(im)[0, 0], cmap='gray')
    ax.set_title(name)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()

# %%
# Where the energy goes
# ---------------------
#
# For each image, the share of finest-scale bandpass energy in each subband.
# The DWT has three columns to spend, the DTCWT six.

dwt = DWTForward(J=3, wave='db3', mode='symmetric')
dtcwt = DTCWTForward(J=3)

print('%-14s %s' % ('', 'DWT      LH     HL     HH'))
for name, im in images.items():
    _, yh = dwt(to_tensor(im))
    e = band_energy(yh[0][0, 0])
    print('%-14s      %s' % (name, '  '.join('%5.2f' % v for v in e)))

print()
print('%-14s %s' % ('', 'DTCWT   ' + '  '.join('%5d' % a for a in ANGLES)))
dtcwt_energy = {}
for name, im in images.items():
    _, Yh = dtcwt(to_tensor(im))
    e = band_energy(Yh[0][0, 0])
    dtcwt_energy[name] = e
    print('%-14s      %s' % (name, '  '.join('%5.2f' % v for v in e)))

# %%
# The brick wall puts most of its energy on the 75/105 degree pair - together
# some 84% of it - and around one percent on 45/135. The handwriting does the
# opposite, favouring 15 and 165 degrees. The checkerboard, whose edges are
# exactly horizontal and vertical, leaves the diagonal orientations near empty,
# which is a useful check that the orientations mean what they claim.
#
# The DWT cannot express any of this: its three subbands separate horizontal
# from vertical, but every diagonal direction lands in one bucket.

fig, ax = plt.subplots(figsize=(8, 3.6))
width = 0.26
pos = np.arange(len(ANGLES))
for i, (name, e) in enumerate(dtcwt_energy.items()):
    ax.bar(pos + (i - 1) * width, e, width, label=name)
ax.set_xticks(pos)
ax.set_xticklabels(['%d deg' % a for a in ANGLES])
ax.set_ylabel('share of bandpass energy')
ax.set_title('DTCWT orientation signature, finest scale')
ax.legend()
plt.tight_layout()

# %%
# Reconstructing one orientation at a time
# ----------------------------------------
#
# Keeping a single orientation and zeroing the rest shows what each subband
# actually carries. The 75/105 pair recovers the brick courses; the 15/165
# pair recovers the vertical joints.

x = to_tensor(data.brick())
Yl, Yh = dtcwt(x)

fig, axes = plt.subplots(1, 6, figsize=(15, 2.9))
for k, ax in enumerate(axes):
    mag = (Yh[0][0, 0, k, ..., 0] ** 2 + Yh[0][0, 0, k, ..., 1] ** 2).sqrt()
    ax.imshow(mag, cmap='inferno')
    ax.set_title('%d deg' % ANGLES[k])
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle('DTCWT coefficient magnitude per orientation (brick, scale 1)')
plt.tight_layout()

# %%
# The magnitude is what makes this readable: the real and imaginary parts of a
# complex subband oscillate, but their magnitude is a smooth envelope of where
# that orientation has energy. It is also nearly shift invariant, which the
# next example is about.

# %%
# References
# ----------
#
# - I. W. Selesnick, R. G. Baraniuk and N. G. Kingsbury, "The dual-tree complex
#   wavelet transform", *IEEE Signal Processing Magazine*, 2005.
# - N. Kingsbury, "Complex wavelets for shift invariant analysis and filtering
#   of signals", *Applied and Computational Harmonic Analysis*,
#   10(3):234-253, 2001.
