"""_summary_"""

from numba import extending

try:
    from scipy import __version__, optimize

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ._nnls import _nnls_007_111, nnls_112_114, nnls_115_

if HAS_SCIPY:
    # if __version__ >= "1.15":
    #     extending.overload(optimize.nnls)(
    #         lambda A, b, maxiter=None, atol=None: nnls_115_
    #     )
    if __version__ >= "1.12":
        extending.overload(optimize.nnls)(
            lambda A, b, maxiter=None, atol=None: nnls_112_114
        )
    elif __version__ >= "0.7":
        extending.overload(optimize.nnls)(lambda A, b, maxiter=-1: _nnls_007_111)
    else:
        raise NotImplementedError(f"Scipy version {__version__} not supported")
