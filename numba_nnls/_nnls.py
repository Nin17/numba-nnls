"""_summary_"""

from __future__ import annotations

from ctypes.util import find_library

import numpy as np
from llvmlite import binding
from numba import extending, njit, types
from numpy.typing import NDArray

from ._dsysv import numba_dsysv
from .utils import get_extension_path, ptr_from_val, val_from_ptr

__all__ = ["nnls_007_111", "nnls_007_111_", "nnls_112_114", "nnls_115_"]


_path = find_library("nnlsnumba") or get_extension_path("libnnlsnumba")

if _path is not None:
    binding.load_library_permanently(_path)
else:
    raise RuntimeError("Could not find the library libnnlsnumba")


# ------------------------ "0.7" <= scipy.__version__ < "1.12" ----------------------- #
_nnls_fortran_c = types.ExternalFunction(
    "nnls_c",
    types.void(
        types.CPointer(types.float64),  # A(MDA, N)
        types.int32,  # mda
        types.int32,  # m
        types.int32,  # n
        types.CPointer(types.float64),  # b(M)
        types.CPointer(types.float64),  # x(N)
        types.CPointer(types.float64),  # rnorm
        types.CPointer(types.float64),  # w(N)
        types.CPointer(types.float64),  # zz(M)
        types.CPointer(types.int32),  # index(N)
        types.CPointer(types.int32),  # mode
        types.int32,  # maxiter
    ),
)


@njit
def nnls_007_111_(
    A: NDArray[np.float64],
    m: np.int32,
    n: np.int32,
    b: NDArray[np.float64],
    x: NDArray[np.float64],
    rnorm: NDArray[np.float64],
    w: NDArray[np.float64],
    zz: NDArray[np.float64],
    index: NDArray[np.int32],
    mode: NDArray[np.int32],
    maxiter: np.int32,
) -> tuple[NDArray[np.float64], float, int]:
    """
    Wrapper of the nnls subroutine used in scipy.optimize.nnls from version 0.7 to 1.11
    without allocating the work arrays.

    Parameters
    ----------
    A : NDArray[np.float64]
        m by n matrix
    m : np.int32
        Size of the first dimension of A
    n : np.int32
        Size of the second dimension of A
    b : NDArray[np.float64]
        m vector
    x : NDArray[np.float64]
        Contains the solution vector on output (needs not be initialized)
    rnorm : NDArray[np.float64]
        Containst the euclidian norm of the residual vector on output
    w : NDArray[np.float64]
        An n work array, will contain the dual solution vector on output
    zz : NDArray[np.float64]
        An m work array
    index : NDArray[np.int32]
        An integer work array of length >= n
    mode : NDArray[np.int32]
        Success flag: 1 success, 2 bad dimensions, 3 maxiter exceeded
    maxiter : np.int32
        Maximum number of iterations, if negative, 3 * n

    Returns
    -------
    tuple[NDArray[np.float64], float, int]
        The solution vector, the residual and the success flag
        All modified in place
    """
    _nnls_fortran_c(
        np.asfortranarray(A).view(np.float64).ctypes,
        m,
        m,
        n,
        b.view(np.float64).ctypes,
        x.view(np.float64).ctypes,
        rnorm.view(np.float64).ctypes,
        w.view(np.float64).ctypes,
        zz.view(np.float64).ctypes,
        index.view(np.int32).ctypes,
        mode.view(np.int32).ctypes,
        maxiter,
    )
    return x, rnorm.item(), mode.item()


_nnls_fortran_c_wrapper = types.ExternalFunction(
    "nnls_c_wrapper",
    types.void(
        types.CPointer(types.float64),
        types.int32,
        types.int32,
        types.CPointer(types.float64),
        types.CPointer(types.float64),
        types.CPointer(types.float64),
        types.CPointer(types.int32),
        types.int32,
    ),
)


@extending.register_jitable
def _nnls_007_111(A, b, maxiter=-1) -> tuple[NDArray[np.float64], float]:
    A = np.asarray_chkfinite(A).copy()
    b = np.asarray_chkfinite(b).copy()
    if A.ndim != 2:
        raise ValueError(
            "Expected a two-dimensional array (matrix)"
            + f", but the shape of A is {A.shape}"
        )
    if b.ndim != 1:
        raise ValueError(
            "Expected a one-dimensional array (vector)"
            + f", but the shape of b is {b.shape}"
        )
    m, n = A.shape

    if m != b.size:
        raise ValueError(
            "Incompatible dimensions. The first dimension of "
            + f"A is {m}, while the shape of b is {(b.shape[0], )}"
        )

    rnorm_ptr = ptr_from_val(1.0)
    mode_ptr = ptr_from_val(np.int32(0))

    x = np.empty(n, dtype=np.float64)

    _nnls_fortran_c_wrapper(
        np.asfortranarray(A).view(np.float64).ctypes,
        m,
        n,
        b.view(np.float64).ctypes,
        x.ctypes,
        rnorm_ptr,
        mode_ptr,
        maxiter,
    )
    if val_from_ptr(mode_ptr) != 1:
        raise RuntimeError("Maximum number of iterations reached.")

    return x, val_from_ptr(rnorm_ptr)


@njit
def nnls_007_111(A, b, maxiter=-1):
    """
    Numba wrapper of the nnls subroutine used in scipy.optimize.nnls, from version
    0.7 to 1.11

    Solve ``argmin_x || Ax - b ||_2`` for ``x>=0``. This is a wrapper
    for a FORTRAN non-negative least squares solver.

    Parameters
    ----------
    A : ArrayLike
        Matrix ``A`` as shown above.
    b : ArrayLike
        Right-hand side vector.
    maxiter : int, optional
        Maximum number of iterations, by default -1
        -1  -> 3 * A.shape[1]

    Returns
    -------
    tuple[NDArray[np.float64], float]
        Solution vector, Residual ``|| Ax-b ||_2``

    Raises
    ------
    ValueError
        If A is not 2D
    ValueError
        If b is not 1D
    ValueError
        If the first dimension of A is not equal to the size of b
    RuntimeError
        If the maximum number of iterations is reached
    """
    return _nnls_007_111(A, b, maxiter)

# ----------------------- "1.12" <= scipy.__version__ < "1.15" ----------------------- #
# # TODO can probably allocate this array and overwrite each time
# # would then have to only change the call to dsysv

# # TODO only write the upper/lower triangle
# _matrix = np.empty((P_ind.size, P_ind.size), dtype=np.float64)
# for i in range(P_ind.size):
#     for j in range(P_ind.size):
#         _matrix[i, j] = AtA[P_ind[i], P_ind[j]]

# _matrix = np.empty((P_ind.size, P_ind.size), dtype=np.float64)
# _indices = np.triu_indices(P_ind.size, 0)
# for i in range(_indices[0].size):
#     _matrix[_indices[0][i], _indices[1][i]] = AtA[
#         P_ind[_indices[0][i]], P_ind[_indices[1][i]]
#     ]

# _a = P.nonzero()[0]
# _matrix = np.empty((_a.size, _a.size), dtype=np.float64)
# for i in range(_a.size):
#     for j in range(_a.size):
#         _matrix[i, j] = AtA[_a[i], _a[j]]


@njit
def _ix_2d(a: NDArray[np.int32], b: NDArray[np.float64], out) -> NDArray[np.float64]:
    """
    Implements b[np.ix_(a, a)] or equivalently b[a[:, None], a[None, :]]

    Parameters
    ----------
    a : NDArray[np.int32]
        _description_
    b : NDArray[np.float64]
        _description_

    Returns
    -------
    NDArray[np.float64]
        _description_
    """

    # return b[np.ix_(a, a)]
    _a = a.nonzero()[0]
    # out = np.empty((_a.size, _a.size), dtype=np.float64)

    for i in range(_a.size):
        for j in range(_a.size):
            out[i, j] = b[_a[i], _a[j]]

    # indices = np.triu_indices(_a.size, 0)
    # for i in range(indices[0].size):
    #     out[indices[0][i], indices[1][i]] = b[_a[indices[0][i]], _a[indices[1][i]]]

    return np.asfortranarray(out[: _a.size, : _a.size])


@extending.register_jitable
def nnls_112_114(
    A,
    b,
    maxiter=None,
    atol=None,
):
    """_summary_

    Parameters
    ----------
    A : np.ndarray
        _description_
    b : np.ndarray
        _description_
    maxiter : int | None, optional
        _description_, by default None
    atol : float | None, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    ValueError
        _description_
    """
    A = np.asarray_chkfinite(A)
    b = np.asarray_chkfinite(b)

    if len(A.shape) != 2:
        raise ValueError(
            "Expected a two-dimensional array (matrix)"
            + f", but the shape of A is {A.shape}"
        )
    if len(b.shape) != 1:
        raise ValueError(
            "Expected a one-dimensional array (vector)"
            + f", but the shape of b is {b.shape}"
        )

    m, _ = A.shape

    if m != b.shape[0]:
        raise ValueError(
            "Incompatible dimensions. The first dimension of "
            + f"A is {m}, while the shape of b is {(b.shape[0], )}"
        )

    x, rnorm, mode = _nnls_112_114(A, b, maxiter, atol)
    if mode != 1:
        raise RuntimeError("Maximum number of iterations reached.")

    return x, rnorm


@njit
def _nnls_112_114(A, b, maxiter=None, tol=None):
    m, n = A.shape
    AtA = A.T @ A
    Atb = b @ A

    out = np.empty_like(AtA)

    if maxiter is None:
        maxiter = 3 * n
    if tol is None:
        tol = 10 * max(m, n) * np.spacing(1.0)

    x = np.zeros(n, dtype=np.float64)
    s = np.zeros(n, dtype=np.float64)
    P = np.zeros(n, dtype=np.bool_)

    w = Atb.copy().astype(np.float64)

    iter = 0

    while (not P.all()) and (w[~P] > tol).any():
        k = np.argmax(w * (~P))
        P[k] = True
        s[:] = 0.0
        s[P] = numba_dsysv(_ix_2d(P, AtA, out), Atb[P])

        while (iter < maxiter) and (s[P].min() < 0):
            iter += 1
            inds = P * (s < 0)
            alpha = (x[inds] / (x[inds] - s[inds])).min()
            x *= 1 - alpha
            x += alpha * s
            P[x <= tol] = False
            s[P] = numba_dsysv(_ix_2d(P, AtA, out), Atb[P])
            s[~P] = 0

        x[:] = s[:]
        w[:] = Atb - AtA @ x

    if iter == maxiter:
        return x, 0.0, -1

    return x, np.linalg.norm(A @ x - b), 1

# ---------------------------- "1.15" <= scipy.__version__ --------------------------- #

# TODO: implement a version of this function

def _nnls_115_(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    maxiter: np.int32,
) -> tuple[NDArray[np.float64], float]:
    """
    Wrapper of the nnls subroutine used in scipy.optimize.nnls from version 1.15

    Parameters
    ----------
    A : NDArray[np.float64]
        m by n matrix
    b : NDArray[np.float64]
        m vector
    maxiter : np.int32
        Maximum number of iterations

    Returns
    -------
    tuple[NDArray[np.float64], float]
        The solution vector and the residual
    """
    raise NotImplementedError("This function is not implemented yet.")