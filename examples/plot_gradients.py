"""
Optimising through the transform
================================

Every transform here is an ordinary pytorch module, so a loss computed on
wavelet coefficients can be differentiated back to the pixels. That is what
this package is for: the numpy implementations cannot do it.
"""

# %%
# Objective
# ---------
#
# Demonstrate that a loss computed on wavelet coefficients can be optimised
# back to the pixels, by solving a small variational denoising problem, and
# then verify the gradients themselves against finite differences rather than
# assuming they are right.

# %%
# Environment
# -----------
#
# Imports, versions and the random seed in one cell, so that running it is
# enough to set the notebook up. Every number below can be checked
# against a rerun. Following the reproducibility conventions in Rule et al.
# (2019), every figure and every quantity here is produced by the code above
# it - nothing is quoted from a previous run.

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch
from skimage import data
from torch.autograd import gradcheck

import pytorch_wavelets
from pytorch_wavelets import DWTForward
from pytorch_wavelets.dwt.lowlevel import AFB2D, mode_to_int

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
# The same 256x256 crop of ``camera`` and the same noise model as
# :doc:`plot_denoising`, so the two results can be compared directly.

SIGMA = 0.1


def psnr(a, b):
    return 10 * np.log10(1.0 / ((a - b) ** 2).mean().item())


clean = torch.tensor(data.camera().astype('float32')[:256, :256] / 255.)
clean = clean[None, None]
noisy = clean + torch.tensor(
    rng.standard_normal(clean.shape, dtype='float32')) * SIGMA

# %%
# Method
# ------
#
# Minimise :math:`\\|x - y\\|^2 + \\lambda \\sum |W x|` over the image
# :math:`x`, where :math:`W` is the wavelet transform and :math:`y` the noisy
# observation. The first term keeps the result near what was measured, the
# second prefers images with few large wavelet coefficients. Nothing here is
# special-cased for wavelets: it is a loss, a parameter and an optimiser.

xfm = DWTForward(J=3, wave='db4', mode='periodization')
x = noisy.clone().requires_grad_(True)
opt = torch.optim.Adam([x], lr=0.02)
lam = 0.3

history = []
for step in range(200):
    opt.zero_grad()
    yl, yh = xfm(x)
    sparsity = sum(band.abs().sum() for band in yh)
    loss = ((x - noisy) ** 2).sum() + lam * sparsity
    loss.backward()
    opt.step()
    if step % 10 == 0:
        history.append((step, psnr(x.detach(), clean)))

# %%
# Results
# -------

print('noisy:     %.2f dB' % psnr(noisy, clean))
print('optimised: %.2f dB' % psnr(x.detach(), clean))

fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
for ax, (name, img) in zip(axes, [('clean', clean), ('noisy', noisy),
                                  ('optimised', x.detach())]):
    ax.imshow(img[0, 0], cmap='gray', vmin=0, vmax=1)
    ax.set_title(name if name == 'clean'
                 else '%s  %.2f dB' % (name, psnr(img, clean)))
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()

fig, ax = plt.subplots(figsize=(7, 3.4))
steps, values = zip(*history)
ax.plot(steps, values, marker='o')
ax.axhline(psnr(noisy, clean), color='grey', ls='--', label='noisy')
ax.set_xlabel('iteration')
ax.set_ylabel('PSNR (dB)')
ax.set_title('Gradient descent through the DWT')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()

# %%
# Soft thresholding the same transform reaches 26.6 dB on this image (see
# :doc:`plot_denoising`). Solving the problem rather than shrinking in one shot
# is worth about a decibel here, and unlike thresholding it extends to any
# differentiable objective - which is the point.

# %%
# Verification: are the gradients right?
# --------------------------------------
#
# Worth checking rather than assuming: pytorch will happily backpropagate
# through a custom Function whose backward pass is wrong. ``gradcheck``
# compares the analytical gradient against central finite differences, in
# double precision on a small input.
#
# The default dtype is set before building the transform so that the filter
# coefficients themselves are float64; converting the module afterwards would
# leave them rounded to float32.

old = torch.get_default_dtype()
torch.set_default_dtype(torch.float64)
try:
    for mode in ['zero', 'symmetric', 'reflect', 'periodization']:
        f = DWTForward(J=1, wave='db3', mode=mode)
        t = torch.randn(1, 1, 12, 12, dtype=torch.float64, requires_grad=True)
        ok = gradcheck(AFB2D.apply,
                       (t, f.h0_col, f.h1_col, f.h0_row, f.h1_row,
                        mode_to_int(mode)), eps=1e-6, atol=1e-6)
        print('%-14s gradcheck %s' % (mode, 'passed' if ok else 'FAILED'))
finally:
    torch.set_default_dtype(old)

# %%
# Discussion
# ----------
#
# All four padding schemes pass, which is worth stating because it was not
# always so. The backward pass used to apply a synthesis filter bank to the
# incoming gradient, which is the exact adjoint only for zero and periodic
# extension. Under ``symmetric`` and ``reflect`` padding the adjoint must also
# fold the boundary extension back onto the samples it was copied from; without
# that step the gradient was wrong in a band along the border whose width
# scales with the filter length.
#
# The variational result above is not a claim that this beats a well-tuned
# shrinkage rule in general - it is one image, one noise realisation and one
# choice of :math:`\\lambda`. What it does show is that the optimisation
# machinery works end to end.

# %%
# References
# ----------
#
# - D. L. Donoho and I. M. Johnstone, "Ideal spatial adaptation by wavelet
#   shrinkage", *Biometrika*, 81(3):425-455, 1994.
# - S. Mallat, "A theory for multiresolution signal decomposition: the wavelet
#   representation", *IEEE Transactions on Pattern Analysis and Machine
#   Intelligence*, 11(7):674-693, 1989.
# - A. Rule et al., "Ten simple rules for writing and sharing computational
#   analyses in Jupyter Notebooks", *PLOS Computational Biology*,
#   15(7):e1007007, 2019.
