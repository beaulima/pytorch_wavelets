"""
Denoising by thresholding
=========================

The oldest wavelet application there is: transform, shrink the small
coefficients towards zero, transform back. Noise spreads itself thinly over
every coefficient while image structure concentrates in a few large ones, so a
threshold removes much more noise than signal.

It also shows why the redundancy of the stationary transform is worth paying
for, which is the counterpart to :doc:`plot_shift_invariance`.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import data

from pytorch_wavelets import (DWTForward, DWTInverse, SWTForward, SWTInverse)

SIGMA = 0.1
WAVE = 'db4'
J = 3


def psnr(a, b):
    return 10 * np.log10(1.0 / ((a - b) ** 2).mean().item())


def soft(t, thresh):
    """Soft threshold: shrink towards zero rather than clipping."""
    return torch.sign(t) * torch.clamp(t.abs() - thresh, min=0)


rng = np.random.RandomState(0)
clean = torch.tensor(data.camera().astype('float32')[:256, :256] / 255.)
clean = clean[None, None]
noisy = clean + torch.tensor(rng.randn(*clean.shape).astype('float32')) * SIGMA
print('noisy: %.2f dB' % psnr(noisy, clean))

# %%
# Decimated: threshold the bandpasses, leave the lowpass alone
# ------------------------------------------------------------
#
# The threshold is the usual three sigma. The lowpass carries the image's
# coarse structure and almost no noise relative to its amplitude, so it is left
# untouched.

thresh = 3 * SIGMA

xfm = DWTForward(J=J, wave=WAVE, mode='periodization')
ifm = DWTInverse(wave=WAVE, mode='periodization')
yl, yh = xfm(noisy)
dwt_rec = ifm((yl, [soft(band, thresh) for band in yh]))
print('DWT:   %.2f dB' % psnr(dwt_rec, clean))

# %%
# Undecimated: the same recipe, more coefficients
# -----------------------------------------------
#
# The stationary transform keeps every scale at full resolution, so each pixel
# is described many times over. Averaging those descriptions back together
# suppresses the blocky artefacts that decimated thresholding leaves behind -
# this is the same idea as cycle spinning, but without having to average over
# shifts by hand.

swt = SWTForward(J=J, wave=WAVE)
iswt = SWTInverse(wave=WAVE)
coeffs = swt(noisy)
shrunk = []
for scale in coeffs:
    c = scale.clone()
    c[:, :, 1:] = soft(scale[:, :, 1:], thresh)   # index 0 is the lowpass
    shrunk.append(c)
swt_rec = iswt(shrunk)
print('SWT:   %.2f dB' % psnr(swt_rec, clean))

# %%
# Side by side. The difference is easiest to see in the flat regions of the
# sky, where decimated thresholding leaves a faint checkerboard.

panels = [('clean', clean), ('noisy', noisy),
          ('DWT', dwt_rec), ('SWT', swt_rec)]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
for ax, (name, img) in zip(axes, panels):
    ax.imshow(img[0, 0].detach(), cmap='gray', vmin=0, vmax=1)
    title = name if name == 'clean' else '%s  %.2f dB' % (name, psnr(img, clean))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()

# %%
# Choosing the threshold
# ----------------------
#
# Too low and noise survives; too high and edges go with it. Sweeping it shows
# the trade-off, and that the stationary transform is ahead across the range
# rather than at one lucky setting.

factors = np.linspace(0.5, 5.0, 10)
curves = {'DWT': [], 'SWT': []}
for f in factors:
    t = f * SIGMA
    curves['DWT'].append(psnr(ifm((yl, [soft(b, t) for b in yh])), clean))
    sh = []
    for scale in coeffs:
        c = scale.clone()
        c[:, :, 1:] = soft(scale[:, :, 1:], t)
        sh.append(c)
    curves['SWT'].append(psnr(iswt(sh), clean))

fig, ax = plt.subplots(figsize=(7, 3.8))
for name, y in curves.items():
    ax.plot(factors, y, marker='o', label=name)
ax.axhline(psnr(noisy, clean), color='grey', ls='--', label='noisy')
ax.set_xlabel('threshold / sigma')
ax.set_ylabel('PSNR (dB)')
ax.set_title('Soft thresholding, %s, %d scales' % (WAVE, J))
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()

print('best DWT: %.2f dB   best SWT: %.2f dB'
      % (max(curves['DWT']), max(curves['SWT'])))
