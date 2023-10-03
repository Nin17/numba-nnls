"""_summary_
"""

# pylint: disable=invalid-name

import ctypes as ct

import numba as nb
import numpy as np
from numpy.typing import NDArray

_PTR = ct.POINTER
_dble = ct.c_double
_int = ct.c_int

_ptr_dble = _PTR(_dble)
_ptr_int = _PTR(_int)

# Signature of dsysv:
# void dsysv(
# char *uplo,
# int *n,
# int *nrhs,
# d *a,
# int *lda,
# int *ipiv,
# d *b,
# int *ldb,
# d *work,
# int *lwork,
# int *info
# )

addr = nb.extending.get_cython_function_address(
    "scipy.linalg.cython_lapack", "dsysv"
)
functype = ct.CFUNCTYPE(
    None,  # return type
    ct.c_void_p,  # *uplo
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
)


_L = ct.cast("L".encode("utf-8"), ct.c_void_p).value
_U = ct.cast("U".encode("utf-8"), ct.c_void_p).value


_dsysv = functype(addr)


@nb.njit
def numba_dsysv(
    A: NDArray[np.float64], b: NDArray[np.float64], uplo: str = "L"
) -> NDArray[np.float64]:
    """
    A numba wrapper of the dsysv function from LAPACK.

    For uplo="L", it is equivalent to:
        scipy.linalg.solve(A, b, lower=True, overwrite_a = True,
                            overwrite_b = True, check_finite=False,
                            assume_a="sym", transposed=False)
    and for uplo="U", it is equivalent to:
        scipy.linalg.solve(A, b, lower=False, overwrite_a = True,
                            overwrite_b = True, check_finite=False,
                            assume_a="sym", transposed=False)

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
    A : (N, N) NDArray[np.float64]
        Symmetric square matrix A, only the lower/upper triangle is needed.
    b : (N, NRHS) NDArray[np.float64]
        Input data for the right hand side.
    uplo : str, optional
        Either "L" or "U" denoting whether the lower or upper triangle of A
        are stored, by default "L"

    Returns
    -------
    (N, NRHS) NDArray[np.float64]
        The solution array

    Raises
    ------
    RuntimeError
        If the LAPACK function fails whilst determining the optimal lwork
    """
    uplo_dict = {"U": _U, "L": _L}
    _uplo = uplo_dict[uplo]

    _n = A.shape[0]

    if b.ndim == 1:
        _nrhs = 1
    elif b.ndim == 2:
        _nrhs = b.shape[-1]

    n = np.array(_n, dtype=np.int32)
    nrhs = np.array(_nrhs, dtype=np.int32)
    lda = np.array(_n, dtype=np.int32)
    ipiv = np.empty(_n, dtype=np.int32)
    ldb = np.array(_n, dtype=np.int32)
    work = np.empty(1, dtype=np.float64)
    lwork = np.array(-1, dtype=np.int32)  # -1 to get optimal lwork
    info = np.empty(1, dtype=np.int32)

    # Get optimal lwork for dsysv
    _dsysv(
        _uplo,
        n.ctypes,
        nrhs.ctypes,
        A.view(np.float64).ctypes,
        lda.ctypes,
        ipiv.view(np.int32).ctypes,
        b.view(np.float64).ctypes,
        ldb.ctypes,
        work.ctypes,
        lwork.ctypes,
        info.ctypes,
    )

    def _check_info(info):
        if info[0] != 0:
            raise RuntimeError(
                f"LAPACK dsysv failed with error code: {info[0]}"
            )

    _check_info(info)

    _ws = int(work[0])
    lwork = np.array(_ws, dtype=np.int32)
    work = np.empty(_ws, dtype=np.float64)

    _dsysv(
        _uplo,
        n.ctypes,
        nrhs.ctypes,
        A.view(np.float64).ctypes,
        lda.ctypes,
        ipiv.view(np.int32).ctypes,
        b.view(np.float64).ctypes,
        ldb.ctypes,
        work.ctypes,
        lwork.ctypes,
        info.ctypes,
    )
    return b
