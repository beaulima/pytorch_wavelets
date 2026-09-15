2D Wavelet Transforms in Pytorch
================================

|build-status| |docs| |doi|

.. |build-status| image:: https://github.com/fbcotter/pytorch_wavelets/actions/workflows/tests.yml/badge.svg?branch=master
    :alt: build status
    :target: https://github.com/fbcotter/pytorch_wavelets/actions/workflows/tests.yml

.. |docs| image:: https://readthedocs.org/projects/pytorch-wavelets/badge/?version=latest
    :target: https://pytorch-wavelets.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |doi| image:: https://zenodo.org/badge/146817005.svg
   :target: https://zenodo.org/badge/latestdoi/146817005
   
The full documentation is also available `here`__, including a gallery of
worked examples covering the DWT basics, the DTCWT's directional selectivity,
shift invariance, denoising and optimising through the transform. Every
example is executed when the documentation is built, and can be downloaded as
a script or a notebook. The same demos are committed as notebooks under
``notebooks/``, generated from the gallery scripts and checked in CI to be in
step with them.

__ http://pytorch-wavelets.readthedocs.io/

This package provides support for computing the 2D discrete wavelet and 
the 2d dual-tree complex wavelet transforms, their inverses, and passing 
gradients through both using pytorch.

The implementation is designed to be used with batches of multichannel images.
We use the standard pytorch implementation of having 'NCHW' data format.

We also have added layers to do the 2-D DTCWT based scatternet. This is similar
to the Morlet based scatternet in `KymatIO`__, but is roughly 10 times faster.

If you use this repo, please cite my PhD thesis, chapter 3: https://doi.org/10.17863/CAM.53748.

__ https://github.com/kymatio/kymatio

New in version 1.4.0
~~~~~~~~~~~~~~~~~~~~

**This release requires torch 2.6 or newer.** The previous floor of 1.0.0 had
not reflected anything the library was tested against for years.

Two further changes alter results for existing code; both are corrections
rather than API changes, but they are worth reading before upgrading.

- **The DWT gradients were wrong for the** ``symmetric`` **and** ``reflect``
  **padding schemes.** Every analysis and synthesis Function computed its
  backward pass as a synthesis filter bank applied to the incoming gradient,
  which is the exact adjoint only for zero or periodic extension. The error
  reached 96% in relative terms and was not confined to the boundary. The
  ``periodization`` gradient was also wrong for odd-length inputs, where the
  sample the forward transform appends was cropped instead of folded back.
  Anything trained with those padding schemes will now see different - correct
  - gradients. The default ``zero`` mode was always exact.
- **The ScatterNet filters are registered as buffers** rather than
  ``nn.Parameter(requires_grad=False)``, so they no longer appear in
  ``.parameters()`` and an optimiser will not apply weight decay to them. The
  ``state_dict`` keys are unchanged, so existing checkpoints keep loading, but
  ``optim.SGD(ScatLayer().parameters())`` now raises "empty parameter list".

New features:

- The 2D stationary (undecimated) wavelet transform is now part of the public
  API. ``SWTForward`` was reachable but broken for more than one level, and
  ``SWTInverse`` could not even be imported and performed a decimated
  synthesis. Both are fixed, exported and documented; the inverse is a real
  undecimated reconstruction, validated against ``pywt.iswt2``.
- ``DWTForward``/``DWTInverse`` accept the ``separable`` flag the docs have
  advertised since v1.0.0. See the measured trade-off in the DWT docs.

Other fixes:

- ``mode='reflect'`` works when the wavelet is longer than the signal, where it
  used to raise RuntimeError.
- ``mypad`` raised IndexError whenever both axes were padded.
- ``AFB2D``/``SFB2D`` applied the column filters across the rows when given
  four distinct filters.
- ``SmoothMagFn`` raised UnboundLocalError when only its second input needed a
  gradient, and ``ScatLayerj2`` recorded ``biort`` as its ``qshift``.
- Coefficients load through ``importlib.resources`` instead of the deprecated
  ``pkg_resources``.

Packaging and infrastructure:

- ``pyproject.toml`` replaces ``setup.py``, and ``PyWavelets`` is declared as
  the runtime dependency it has always been - a clean ``pip install`` used to
  fail on import.
- A ``Makefile`` builds a conda/mamba environment; ``make help`` lists the
  targets.
- GitHub Actions replaces the dead Travis config, covering Python 3.9 to 3.12.
  The long gradchecks are marked ``slow`` and run in their own job rather than
  being skipped outright.
- The documentation builds again, and a test suite executes every example it
  prints.

New in version 1.3.0
~~~~~~~~~~~~~~~~~~~~

- Added 1D DWT support

.. code:: python

    import torch
    from pytorch_wavelets import DWT1DForward, DWT1DInverse  # or simply DWT1D, IDWT1D
    dwt = DWT1DForward(wave='db6', J=3)
    X = torch.randn(10, 5, 100)
    yl, yh = dwt(X)
    print(yl.shape)
    >>> torch.Size([10, 5, 22])
    print(yh[0].shape)
    >>> torch.Size([10, 5, 55])
    print(yh[1].shape)
    >>> torch.Size([10, 5, 33])
    print(yh[2].shape)
    >>> torch.Size([10, 5, 22])
    idwt = DWT1DInverse(wave='db6')
    x = idwt((yl, yh))

New in version 1.2.0
~~~~~~~~~~~~~~~~~~~~

- Added a DTCWT based ScatterNet

.. code:: python

    import torch
    from pytorch_wavelets import ScatLayer
    scat = ScatLayer()
    X = torch.randn(10,5,64,64)
    # A first order scatternet with 6 orientations and one lowpass channels
    # gives 7 times the input channel dimension
    Z = scat(X)
    print(Z.shape)
    >>> torch.Size([10, 35, 32, 32])
    # A second order scatternet with 6 orientations and one lowpass channels
    # gives 7^2 times the input channel dimension
    scat2 = torch.nn.Sequential(ScatLayer(), ScatLayer())
    Z = scat2(X)
    print(Z.shape)
    >>> torch.Size([10, 245, 16, 16])
    # We also have a slightly more specialized, but slower, second order scatternet
    from pytorch_wavelets import ScatLayerj2
    scat2a = ScatLayerj2()
    Z = scat2a(X)
    print(Z.shape)
    >>> torch.Size([10, 245, 16, 16])
    # These all of course work with cuda
    scat2a.cuda()
    Z = scat2a(X.cuda())

New in version 1.1.0
~~~~~~~~~~~~~~~~~~~~

- Fixed memory problem with dwt 
- Fixed the backend code for the dtcwt calculation - much cleaner now but similar performance
- Both dtcwt and dwt should be more memory efficient/aware now. 
- Removed need to specify number of scales for DTCWTInverse

New in version 1.0.0
~~~~~~~~~~~~~~~~~~~~
Version 1.0.0 has now added support for separable DWT calculation, and more
padding schemes, such as symmetric, zero and periodization.

Also, no longer need to specify the number of channels when creating the wavelet
transform classes.

Speed Tests
~~~~~~~~~~~
We compare doing the dtcwt with the python package and doing the dwt with
PyWavelets to doing both in pytorch_wavelets, using a GTX1080. The numpy methods
were run on a 14 core Xeon Phi machine using intel's parallel python. For the
dtwcwt we use the `near_sym_a` filters for the first scale and the `qshift_a`
filters for subsequent scales. For the dwt we use the `db4` filters.

For a fixed input size, but varying the number of scales (from 1 to 4) we have
the following speeds (averaged over 5 runs):

.. image:: https://raw.githubusercontent.com/fbcotter/pytorch_wavelets/master/docs/scale.png
    :width: 700px

For an input size with height and width 512 by 512, we also vary the batch size
for a 3 scale transform. The resulting speeds were:

.. image:: https://raw.githubusercontent.com/fbcotter/pytorch_wavelets/master/docs/batchsize.png
    :width: 700px

Installation
````````````
The recommended way is to use the provided ``Makefile``, which builds a
self-contained conda/mamba environment (a `miniforge
<https://github.com/conda-forge/miniforge>`_ install gives you both)::

    $ git clone https://github.com/fbcotter/pytorch_wavelets
    $ cd pytorch_wavelets
    $ make dev

``make dev`` creates the environment described in ``environment.yml``, installs
``pytorch_wavelets`` into it in editable mode, and checks that it imports. The
environment name and python version are configurable::

    $ make dev ENV_NAME=pw-cuda PYTHON_VERSION=3.12

Run ``make help`` for the full list of targets (``test``, ``test-cov``,
``lint``, ``docs``, ``build``, ``clean``, ``clean-env``). A test suite is
provided so that you may verify the code works on your system::

    $ make test

If you would rather manage the environment yourself, a plain pip install works
too - the dependencies are declared in ``pyproject.toml``::

    $ pip install .
    $ pip install ".[test]" && pytest

Example Use
```````````
For the DWT - note that the highpass output has an extra dimension, in which we
stack the (lh, hl, hh) coefficients.  Also note that the Yh output has the
finest detail coefficients first, and the coarsest last (the opposite to
PyWavelets).

.. code:: python

    import torch
    from pytorch_wavelets import DWTForward, DWTInverse
    xfm = DWTForward(J=3, wave='db3', mode='zero')
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
    ifm = DWTInverse(wave='db3', mode='zero')
    Y = ifm((Yl, Yh))

For the DTCWT:

.. code:: python

    import torch
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
    xfm = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')
    X = torch.randn(10,5,64,64)
    Yl, Yh = xfm(X) 
    print(Yl.shape)
    >>> torch.Size([10, 5, 16, 16])
    print(Yh[0].shape) 
    >>> torch.Size([10, 5, 6, 32, 32, 2])
    print(Yh[1].shape)
    >>> torch.Size([10, 5, 6, 16, 16, 2])
    print(Yh[2].shape)
    >>> torch.Size([10, 5, 6, 8, 8, 2])
    ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
    Y = ifm((Yl, Yh))

Some initial notes:

- Yh returned is a tuple. There are 2 extra dimensions - the first comes between
  the channel dimension of the input and the row dimension. This is the
  6 orientations of the DTCWT. The second is the final dimension, which is the
  real an imaginary parts (complex numbers are not native to pytorch)

Running on the GPU
~~~~~~~~~~~~~~~~~~
This should come as no surprise to pytorch users. The DWT and DTCWT transforms support
cuda calling:

.. code:: python

    import torch
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
    xfm = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b').cuda()
    X = torch.randn(10,5,64,64).cuda()
    Yl, Yh = xfm(X) 
    ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b').cuda()
    Y = ifm((Yl, Yh))

The automated tests cannot test the gpu functionality, but do check cpu running.
To test whether the repo is working on your gpu, you can download the repo,
ensure you have pytorch with cuda enabled (the tests will check to see if
:code:`torch.cuda.is_available()` returns true), and run:

.. code:: 

    pip install -r tests/requirements.txt
    pytest tests/

From the base of the repo.

Backpropagation
~~~~~~~~~~~~~~~
It is possible to pass gradients through the forward and backward transforms.
All you need to do is ensure that the input to each has the required_grad
attribute set to true.



Provenance
~~~~~~~~~~
Based on the Dual-Tree Complex Wavelet Transform Pack for MATLAB by Nick
Kingsbury, Cambridge University. The conditions of use of the original MATLAB
toolbox are summarised in the ``LICENSE`` file at the root of this repo.

Further information on the DT CWT can be obtained from papers
downloadable from my website (given below). The best tutorial is in
the 1999 Royal Society Paper. In particular this explains the conversion
between 'real' quad-number subimages and pairs of complex subimages. 
The Q-shift filters are explained in the ICIP 2000 paper and in more detail
in the May 2001 paper for the Journal on Applied and Computational 
Harmonic Analysis.

This code is copyright and is supplied free of charge for research
purposes only. In return for supplying the code, all I ask is that, if
you use the algorithms, you give due reference to this work in any
papers that you write and that you let me know if you find any good
applications for the DT CWT. If the applications are good, I would be
very interested in collaboration. I accept no liability arising from use
of these algorithms.

Nick Kingsbury, 
Cambridge University, June 2003.

Dr N G Kingsbury,
Dept. of Engineering, University of Cambridge,
Trumpington St., Cambridge CB2 1PZ, UK., or
Trinity College, Cambridge CB2 1TQ, UK.
Phone: (0 or +44) 1223 338514 / 332647;  Home: 1954 211152;
Fax: 1223 338564 / 332662;  E-mail: ngk@eng.cam.ac.uk
Web home page: http://www.eng.cam.ac.uk/~ngk/

.. vim:sw=4:sts=4:et
