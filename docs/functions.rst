API Guide
=========

.. currentmodule:: pytorch_wavelets

Decimated WT
------------

.. autoclass:: DWTForward
    :members:
    :show-inheritance:

.. autoclass:: DWTInverse
    :members:
    :show-inheritance:

Stationary (undecimated) WT
---------------------------

.. autoclass:: SWTForward
    :members:
    :show-inheritance:

.. autoclass:: SWTInverse
    :members:
    :show-inheritance:

1D Decimated WT
---------------

.. autoclass:: DWT1DForward
    :members:
    :show-inheritance:

.. autoclass:: DWT1DInverse
    :members:
    :show-inheritance:

Dual Tree Complex WT
--------------------

.. autoclass:: DTCWTForward
    :members:
    :show-inheritance:

.. autoclass:: DTCWTInverse
    :members:
    :show-inheritance:

DTCWT ScatterNet
----------------

.. autoclass:: ScatLayer
    :members:
    :show-inheritance:

.. autoclass:: ScatLayerj2
    :members:
    :show-inheritance:

Aliases
-------

For convenience the following aliases are also exported: ``DWT``/``IDWT`` and
``DWT2D``/``IDWT2D`` for :class:`DWTForward`/:class:`DWTInverse`,
``DWT1D``/``IDWT1D`` for :class:`DWT1DForward`/:class:`DWT1DInverse`,
``SWT``/``ISWT`` for :class:`SWTForward`/:class:`SWTInverse`, and
``DTCWT``/``IDTCWT`` for :class:`DTCWTForward`/:class:`DTCWTInverse`.
