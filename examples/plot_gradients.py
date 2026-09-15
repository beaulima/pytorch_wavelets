"""
Optimising through the transform
================================

Every transform here is an ordinary pytorch module, so a loss computed on
wavelet coefficients can be differentiated back to the pixels. That is what
this package is for: the numpy implementations cannot do it.

The example solves a small variational denoising problem - find the image that
stays close to the noisy one while having sparse wavelet coefficients - by
gradient descent rather than by thresholding, and then checks that the
gradients are the real thing.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import data

from pytorch_wavelets import DWTForward

SIGMA = 0.1


def psnr(a, b):
    return 10 * np.log10(1.0 / ((a - b) ** 2).mean().item())


rng = np.random.RandomState(0)
clean = torch.tensor(data.camera().astype('float32')[:256, :256] / 255.)
clean = clean[None, None]
noisy = clean + torch.tensor(rng.randn(*clean.shape).astype('float32')) * SIGMA

# %%
# Minimise :math:`\\|x - y\\|^2 + \\lambda \\sum |W x|` over the image
# :math:`x`, where :math:`W` is the wavelet transform. Nothing here is
# special-cased for wavelets - it is a loss, a parameter and an optimiser.

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

print('noisy:     %.2f dB' % psnr(noisy, clean))
print('optimised: %.2f dB' % psnr(x.detach(), clean))

# %%
# For comparison, soft thresholding the same transform reaches 26.6 dB on this
# image (see :doc:`plot_denoising`). Solving the problem properly rather than
# in one shot is worth about a decibel here - and unlike thresholding it
# extends to any differentiable objective, which is the point.

# %%
# The result, and how it got there.

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
# Are the gradients right?
# ------------------------
#
# Worth checking rather than assuming: pytorch will happily backpropagate
# through a custom Function whose backward pass is wrong. ``gradcheck``
# compares the analytical gradient against finite differences, in double
# precision on a small input.
#
# Set the default dtype before building the transform, so the filter
# coefficients themselves are float64.

from torch.autograd import gradcheck  # noqa: E402
from pytorch_wavelets.dwt.lowlevel import AFB2D, mode_to_int  # noqa: E402

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
# All four padding schemes pass. That is worth stating because it was not
# always true: the backward pass used to apply a synthesis filter bank to the
# incoming gradient, which is the exact adjoint only for zero and periodic
# extension. Under ``symmetric`` and ``reflect`` padding the adjoint also has
# to fold the boundary extension back onto the samples it was copied from, and
# without that the gradient was wrong well away from the edges.
