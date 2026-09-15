"""
Shift invariance
================

Move an image by one pixel and a decimated transform rearranges its
coefficients: the decimation keeps every other sample, so which samples
survive depends on where the image started. This example measures what each
transform in the package does about that, using circular shifts and
periodization padding so the content really is identical at every offset -
sliding a window across a larger image would change the content as well as
its position, and that change swamps the effect.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import data

from pytorch_wavelets import DWTForward, DTCWTForward, SWTForward

SHIFTS = 16


def to_tensor(im, size=256):
    if im.ndim == 3:
        im = im.mean(axis=2)
    im = im.astype('float32')
    im = (im - im.min()) / (np.ptp(im) + 1e-9)
    return torch.tensor(im[:size, :size])[None, None]


def spread(values):
    """Peak-to-peak variation as a percentage of the mean."""
    return 100 * (values.max() - values.min()) / values.mean()


x = to_tensor(data.camera())
batch = torch.cat([torch.roll(x, s, dims=-1) for s in range(SHIFTS)], dim=0)

# %%
# The stationary transform is exactly invariant
# ---------------------------------------------
#
# Nothing is decimated, so a shifted input gives the same coefficients,
# shifted. Any measure taken over a whole subband is therefore constant to
# floating point precision.

swt = SWTForward(J=2, wave='db3')
coeffs = swt(batch)
e_swt = coeffs[0][:, :, 1:].abs().sum(dim=(1, 2, 3, 4)).numpy()
print('SWT bandpass L1 varies by %.2f%% over 16 one-pixel shifts'
      % spread(e_swt))

# %%
# Better than an aggregate: undo the shift on the coefficient map itself and
# compare. For the SWT this recovers the same map every time.

c0 = coeffs[0][0:1]
err = max(
    (torch.roll(coeffs[0][s:s + 1], -s, dims=-1) - c0).abs().max().item()
    for s in range(SHIFTS))
print('SWT coefficient map, shift undone: max difference %.2e' % err)

# %%
# What the price is
# -----------------
#
# That invariance is bought with redundancy, and the stationary transform is
# the expensive option: every scale stays at full resolution.

n_in = batch[0].numel()
dwt = DWTForward(J=2, wave='db3', mode='periodization')
dtcwt = DTCWTForward(J=2, mode='periodization')
yl, yh = dwt(batch)
Yl, Yh = dtcwt(batch)
print('coefficients per input sample, 2 scales:')
print('  DWT   %.2f' % ((yl[0].numel() + sum(b[0].numel() for b in yh)) / n_in))
print('  DTCWT %.2f' % ((Yl[0].numel() + sum(b[0].numel() for b in Yh)) / n_in))
print('  SWT   %.2f' % (sum(c[0].numel() for c in coeffs) / n_in))

# %%
# The DTCWT: magnitude is the steady part
# ---------------------------------------
#
# The dual-tree transform stays decimated, so its coefficients do move. What
# holds still is their *magnitude*: the real and imaginary parts are in
# quadrature, so as the image slides the pair rotates while its length barely
# changes - the same way :math:`\\sin^2 + \\cos^2` is constant while either
# term alone oscillates.


def mag(y):
    return (y[..., 0] ** 2 + y[..., 1] ** 2).sqrt()


ref_real, ref_mag = Yh[0][0:1, ..., 0], mag(Yh[0][0:1])
rel_real = [(Yh[0][s:s + 1, ..., 0] - ref_real).abs().mean().item()
            / ref_real.abs().mean().item() for s in range(1, SHIFTS)]
rel_mag = [(mag(Yh[0][s:s + 1]) - ref_mag).abs().mean().item()
           / ref_mag.abs().mean().item() for s in range(1, SHIFTS)]

print('mean change against the unshifted image:')
print('  real part  %5.1f%%' % (100 * np.mean(rel_real)))
print('  magnitude  %5.1f%%' % (100 * np.mean(rel_mag)))

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(range(1, SHIFTS), 100 * np.array(rel_real), marker='o',
        label='real part')
ax.plot(range(1, SHIFTS), 100 * np.array(rel_mag), marker='s',
        label='magnitude')
ax.set_xlabel('shift (pixels)')
ax.set_ylabel('mean change vs unshifted (%)')
ax.set_title('DTCWT finest scale under circular shift')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()

# %%
# A caveat on measuring this
# --------------------------
#
# Aggregate numbers over a whole subband - total energy, total absolute value -
# are a poor way to compare the DWT and the DTCWT here. A shift rearranges
# coefficients without changing their sum by much, so those measures look
# stable for both and separate nothing.
#
# Comparing the two properly means measuring what the representation does for
# a downstream task over many images, which is what
# ``tests/Measure of Stability.ipynb`` in this repository does: it draws 1000
# samples, applies shifts, noise and deformations, and reports the distance
# between scattering outputs. The numbers in :doc:`../scatternet` come from it.
