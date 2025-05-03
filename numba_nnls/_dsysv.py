"""Numba wrapper for LAPACK dsysv function from scipy.linalg.cython_lapack."""

from __future__ import annotations

from llvmlite import binding
from numba import njit, types
from numpy import empty, float64, int32
from numpy.typing import NDArray

from .utils import get_scipy_linalg_lib, ptr_from_val, val_from_ptr

_path = get_scipy_linalg_lib("cython_lapack")
binding.load_library_permanently(_path)

_ptr_int = types.CPointer(types.int32)
_ptr_dble = types.CPointer(types.float64)

_dsysv = types.ExternalFunction(
    "dsysv_",
    types.void(
        types.CPointer(types.int64),  # uplo
        _ptr_int,  # *n
        _ptr_int,  # *nrhs
        _ptr_dble,  # *a
        _ptr_int,  # *lda
        _ptr_int,  # *ipiv
        _ptr_dble,  # *b
        _ptr_int,  # *ldb
        _ptr_dble,  # *work
        _ptr_int,  # *lwork
        _ptr_int,  # *info
    ),
)


@njit
def numba_dsysv(
    A: NDArray[float64], b: NDArray[float64], uplo: str = "L"
) -> NDArray[float64]:
    """
    A numba wrapper of the dsysv function from LAPACK.

    It is equivalent to:
        scipy.linalg.solve(asfortranarray(A), b, lower=(uplo == "L"),
                            overwrite_a = True, overwrite_b = True,
                            check_finite=False, assume_a="sym",
                            transposed=False)


    A and b are modified in-place, the original values are therefore lost.
    If these are required, make a copy before calling this function.

    From the LAPACK documentation:

        DSYSV computes the solution to a real system of linear equations
            A * X = B,
        where A is an N-by-N symmetric matrix and X and B are N-by-NRHS
        matrices.

        The diagonal pivoting method is used to factor A as
            A = U * D * U**T,  if UPLO = 'U', or
            A = L * D * L**T,  if UPLO = 'L',
        where U (or L) is a product of permutation and unit upper (lower)
        triangular matrices, and D is symmetric and block diagonal with
        1-by-1 and 2-by-2 diagonal blocks.  The factored form of A is then
        used to solve the system of equations A * X = B.

    Parameters
    ----------
    A : (N, N) NDArray[float64]
        Symmetric square matrix A, only the lower/upper triangle is needed.
    b : (N, NRHS) NDArray[float64]
        Input data for the right hand side.
    uplo : str, optional
        Either "L" or "U" denoting whether the lower or upper triangle of A
        are stored, by default "L"

    Returns
    -------
    (N, NRHS) NDArray[float64]
        The solution array

    Raises
    ------
    RuntimeError
        If the LAPACK function fails whilst determining the optimal lwork
    """
    uploptr = ptr_from_val(ord(uplo))

    _n = int32(A.shape[0])

    nptr = ptr_from_val(_n)

    if b.ndim == 1:
        _b = b
        _nrhs = int32(1)
    elif b.ndim == 2:
        _nrhs = int32(b.shape[-1])
        _b = b

    nrhsptr = ptr_from_val(_nrhs)

    ipiv = empty(_n, dtype=int32)
    work = empty(1, dtype=float64)
    infoptr = ptr_from_val(int32(0))

    # Get optimal lwork for dsysv
    _dsysv(
        uploptr,
        nptr,
        nrhsptr,
        A.view(float64).ctypes,
        nptr,
        ipiv.ctypes,
        _b.view(float64).ctypes,
        nptr,
        work.ctypes,
        ptr_from_val(int32(-1)),  # -1 to get optimal lwork
        infoptr,
    )
    if val_from_ptr(infoptr):
        raise RuntimeError(
            f"LAPACK dsysv failed with error code: {val_from_ptr(infoptr)}"
        )

    lwork = int32(work[0])
    work = empty(lwork, dtype=float64)
    lworkptr = ptr_from_val(lwork)

    _dsysv(
        uploptr,
        nptr,
        nrhsptr,
        A.view(float64).ctypes,
        nptr,
        ipiv.ctypes,
        _b.view(float64).ctypes,
        nptr,
        work.ctypes,
        lworkptr,
        infoptr,
    )
    return b
