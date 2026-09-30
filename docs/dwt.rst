DWT in Pytorch Wavelets
=======================

While pytorch_wavelets was initially built as a repo to do the dual tree wavelet
transform efficiently in pytorch, I have also built a thin wrapper over
PyWavelets, allowing the calculation of the 2D-DWT in pytorch on a GPU on
a batch of images. 

Older versions did the DWT non separably. As of v1.0.0 we now
have code to do it separably, and this is the default. The non-separable code
is still there, reachable with the ``separable`` flag in the DWT/IDWT
constructor::

    xfm = DWTForward(J=2, wave='db2', separable=False)
    ifm = DWTInverse(wave='db2', separable=False)

Both backends return identical coefficients; they differ only in how the 2d
filter is applied. The separable one does two 1d convolutions (:math:`2L`
multiply-accumulates per output sample), the non-separable one a single
convolution with the outer-product kernel (:math:`L^2` per sample). So the
non-separable backend wins for short filters and loses for long ones, with the
crossover around ``db3``/``db4``. Measured on a 2-level transform, with the
non-separable time as a multiple of the separable one (below 1.0 means the
non-separable backend is faster):

+----------+---------------------+----------------------+
| Wavelet  | 16x3x128x128        | 4x16x512x512         |
+==========+==========+==========+===========+==========+
|          | CPU      | GPU      | CPU       | GPU      |
+----------+----------+----------+-----------+----------+
| ``db1``  | 0.57x    | 0.74x    | 0.60x     | 0.59x    |
+----------+----------+----------+-----------+----------+
| ``db2``  | 0.99x    | 0.85x    | 0.71x     | 0.75x    |
+----------+----------+----------+-----------+----------+
| ``db4``  | 1.35x    | 1.61x    | 1.02x     | 1.27x    |
+----------+----------+----------+-----------+----------+
| ``db8``  | 3.63x    | 3.19x    | 2.20x     | 2.80x    |
+----------+----------+----------+-----------+----------+

(14 Intel Xeon cores and an RTX A2000; treat the numbers as indicative and
measure on your own hardware and input sizes.)

.. warning::

    On Ampere and newer GPUs, the non-separable backend's square
    :math:`L \times L` kernel is eligible for TF32 tensor cores, which
    PyTorch enables by default for convolutions. In the ``symmetric`` and
    ``reflect`` padding modes this costs it roughly three decimal digits of
    float32 accuracy. The separable backend's :math:`1 \times L` kernels are
    not affected. If you need full float32 accuracy there, either keep
    ``separable=True`` or set
    :code:`torch.backends.cudnn.allow_tf32 = False`.

The DWT/IDWT now supports most of the padding schemes that PyWavelets uses. In
particular:

- symmetric padding
- reflection padding
- zero padding
- periodization 

You can see the source `here <_modules/pytorch_wavelets/dwt/transform2d.html#DWTForward>`_. 
It is pretty minimal and should be clear what is going on.

In particular, the DWT and IWT classes initialize the filter banks as pytorch
tensors (taking care to flip them as pytorch uses cross-correlation not
convolution). It then uses strided convolution to calculate the LL, LH, HL and
HH subbands - by default as two 1d convolutions, or as a single 2d convolution
when ``separable=False``. It also takes care of padding to match the
PyWavelets implementation.

Differences to PyWavelets
-------------------------

Inputs
~~~~~~
The pytorch_wavelets DWT expects the standard pytorch image format of NCHW
- i.e., a batch of N images, with C channels, height H and width W. For a single
RGB image, you would need to make it a torch tensor of size :code:`(1, 3, H,
W)`, or for a batch of 100 grayscale images, you would need to make it a tensor
of size :code:`(100, 1, H, W)`.

Returned Coefficients
~~~~~~~~~~~~~~~~~~~~~
We deviate slightly from PyWavelets with the format of the returned
coefficients.  In particular, we return a tuple of :code:`(yl, yh)` where yl is
the LL band, and `yh` is a list. The first list entry `yh[0]` are the scale
1 bandpass coefficients (finest resolution), and the last list entry `yh[-1]`
are the coarsest bandpass coefficients. Note that this is the reverse of the
PyWavelets format (but fits with the dtcwt standard output). Each of the bands
is a single stacked tensor of the LH (horiz), HL (vertic), and HH (diag)
coefficients for each scale (as opposed to PyWavelets style of returning as
a tuple) with the stack along the third dimension. As the input had
4 dimensions, this output has 5 dimensions, with shape :code:`(N, C, 3, H, W)`. 
This is easily transformed into the PyWavelets style by unstacking the
list elements in `yh`.

Example
-------

.. code:: python

    import torch
    from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)
    xfm = DWTForward(J=3, mode='zero', wave='db3')  # Accepts all wave types available to PyWavelets
    ifm = DWTInverse(mode='zero', wave='db3')
    X = torch.randn(10,5,64,64)
    Yl, Yh = xfm(X) 
    print(Yl.shape)
    >>> torch.Size([10, 5, 12, 12])
    print(Yh[0].shape) 
    >>> torch.Size([10, 5, 3, 34, 34])
    print(Yh[1].shape)
    >>> torch.Size([10, 5, 3, 19, 19])
    print(Yh[2].shape)
    >>> torch.Size([10, 5, 3, 12, 12])
    Y = ifm((Yl, Yh))
    import numpy as np
    np.testing.assert_array_almost_equal(Y.cpu().numpy(), X.cpu().numpy())

Stationary (undecimated) WT
---------------------------

:class:`pytorch_wavelets.SWTForward` and :class:`pytorch_wavelets.SWTInverse`
compute the 2d stationary wavelet transform, also called the undecimated or a
trous wavelet transform. Nothing is downsampled, so every scale keeps the input
resolution and the filters are dilated by :math:`2^j` instead. That costs
:math:`4^J` redundancy but makes the transform shift invariant, which the
decimated DWT is not.

.. code:: python

    import torch
    from pytorch_wavelets import SWTForward, SWTInverse
    sfm = SWTForward(J=3, wave='db3')
    ifm = SWTInverse(wave='db3')
    X = torch.randn(10, 5, 64, 64)
    coeffs = sfm(X)
    print(len(coeffs))
    >>> 3
    print(coeffs[0].shape)
    >>> torch.Size([10, 5, 4, 64, 64])
    Y = ifm(coeffs)

Unlike the DWT, which returns ``(yl, yh)``, the SWT returns a plain list with
one entry per scale, finest first. Each entry stacks all **four** subbands
along a new third dimension, in the order (ll, lh, hl, hh) - the lowpass is
included at every scale because it is what feeds the next one. This matches
the layout of ``pywt.swt2``, other than pywt returning the coarsest scale
first.

The inverse only needs the coarsest lowpass; the finer ones are redundant and
are recomputed as it goes.

.. warning::

    Perfect reconstruction only holds for periodic extension
    (``mode='periodization'``, the default), because the undecimated
    reconstruction identity
    :math:`\tfrac{1}{2}\left[G_0(z)H_0(z) + G_1(z)H_1(z)\right] = 1`
    assumes circular convolution. With the other padding schemes the interior
    is recovered exactly but samples within roughly a filter length of the
    border are not. PyWavelets has the same restriction: its ``swt2`` is
    periodic only.

Other Notes
-----------
GPU Calculations
~~~~~~~~~~~~~~~~
As you would expect, you can move the transforms to the GPU by calling
:code:`xfm.cuda()` or :code:`ifm.cuda()`, where `xfm`, `ifm` are instances of
:class:`pytorch_wavelets.DWTForward` and :class:`pytorch_wavelets.DWTInverse`.

