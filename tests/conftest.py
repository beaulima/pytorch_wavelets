""" Test configuration.

A handful of test files compare against the numpy reference implementation in
the ``dtcwt`` package. That package declares numpy<2, so an environment with
NumPy 2 - what a plain ``pip install`` of this package gives today - cannot
have it. Leave those files out of collection there rather than failing on the
import, and say so in the header so a short run is never mistaken for a full
one. Everything else, the library itself included, must pass under NumPy 2.
"""
import importlib.util

NEEDS_DTCWT = [
    'test_coldfilt.py',
    'test_colfilter.py',
    'test_dtcwt.py',
    'test_rowdfilt.py',
    'test_rowfilter.py',
    'test_scatnet_fwd.py',
]

HAVE_DTCWT = importlib.util.find_spec('dtcwt') is not None

collect_ignore = [] if HAVE_DTCWT else NEEDS_DTCWT


def pytest_report_header(config):
    if not HAVE_DTCWT:
        return ('dtcwt reference package not installed: skipping %d files '
                'that compare against it (%s)'
                % (len(NEEDS_DTCWT), ', '.join(NEEDS_DTCWT)))
