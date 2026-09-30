""" The numpy helpers in pytorch_wavelets.utils.

asfarray and appropriate_complex_type_for used np.asfarray and
np.issubsctype, both removed in NumPy 2.0, so they raised AttributeError on
any current numpy. These pin down the behaviour they had, which the
replacements must keep.
"""
import numpy as np
import pytest

from pytorch_wavelets.utils import appropriate_complex_type_for, asfarray


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64,
                                   np.complex64, np.complex128])
def test_asfarray_keeps_inexact_dtypes_without_copying(dtype):
    x = np.arange(6).astype(dtype)
    y = asfarray(x)
    assert y.dtype == dtype
    assert np.shares_memory(x, y)


@pytest.mark.parametrize('x', [[1, 2, 3], np.arange(3), np.arange(3, dtype=np.uint8),
                               np.array([True, False])])
def test_asfarray_turns_everything_else_into_float64(x):
    y = asfarray(x)
    assert y.dtype == np.float64
    np.testing.assert_array_equal(y, np.asarray(x, dtype=np.float64))


def test_asfarray_returns_a_plain_ndarray():
    # np.asfarray went through np.asarray, which drops subclasses.
    x = np.ma.masked_array([1.0, 2.0])
    assert type(asfarray(x)) is np.ndarray


@pytest.mark.parametrize('x, expected', [
    (np.zeros(2, np.float32), np.complex64),
    (np.zeros(2, np.float64), np.complex128),
    (np.zeros(2, np.complex64), np.complex64),
    (np.zeros(2, np.complex128), np.complex128),
    ([1, 2], np.complex128),
    (np.zeros(2, np.float16), np.complex128),
])
def test_appropriate_complex_type_for(x, expected):
    assert appropriate_complex_type_for(x) == expected
