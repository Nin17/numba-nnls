"""_summary_"""

from numba import extending

try:
    from scipy import __version__, optimize

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ._nnls import _nnls_old, nnls_new

if HAS_SCIPY:
    if __version__ >= "1.12":  # TODO check this
        extending.overload(optimize.nnls)(
            lambda A, b, maxiter=None, atol=None: nnls_new
        )
    elif __version__ >= "0.7":
        extending.overload(optimize.nnls)(lambda A, b, maxiter=-1: _nnls_old)
