"""Tests for nnls 0.7 <= scipy.__version__ < 1.12"""

import numpy as np
import pytest
from numba import njit
from numpy.testing import assert_

try:
    from scipy import __version__
    from scipy.optimize import nnls

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from numba_nnls import nnls_old

REASON = "test only for 0.7 <= scipy.__version__ < 1.12 if scipy is installed"


class TestNNLS:
    def _test_nnls(self, func):
        a = np.arange(25.0).reshape(-1, 5)
        x = np.arange(5.0)
        y = np.dot(a, x)
        x, res = func(a, y)
        assert_(res < 1e-7)
        assert_(np.linalg.norm(np.dot(a, x) - y) < 1e-7)

    def test_nnls_old(self):
        self._test_nnls(nnls_old)

    @pytest.mark.skipif(
        not HAS_SCIPY or __version__ < "0.7" or __version__ >= "1.12", reason=REASON
    )
    def test_nnls_njit(self):
        self._test_nnls(njit(lambda a, b: nnls(a, b)))

    def _test_maxiter(self, func):
        # test that maxiter argument does stop iterations
        # NB: did not manage to find a test case where the default value
        # of maxiter is not sufficient, so use a too-small value
        rndm = np.random.RandomState(1234)
        a = rndm.uniform(size=(100, 100))
        b = rndm.uniform(size=100)
        with pytest.raises(RuntimeError):
            func(a, b, maxiter=1)

    def test_maxiter_old(self):
        self._test_maxiter(nnls_old)

    @pytest.mark.skipif(
        not HAS_SCIPY or __version__ < "0.7" or __version__ >= "1.12", reason=REASON
    )
    def test_maxiter_njit(self):
        self._test_maxiter(njit(lambda a, b, maxiter: nnls(a, b, maxiter)))
