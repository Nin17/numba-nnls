# numba-nnls

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A [numba](https://numba.pydata.org) implementation of
[scipy.optimize.nnls](
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html#scipy-optimize-nnls)

## Installation

Clone the repository with:

```bash
git clone https://github.com/Nin17/nnls_numba.git
```

`cd` to the repository and install with `pip`:

```bash
pip install .
```

Conda installation coming soon.

> [!IMPORTANT]  
> On windows, you also need to install lapack: `conda install conda-forge::lapack`

## Usage

Simply install `nnls_numba` in your environment to use `scipy.optimize.nnls` in nopython numba functions:

```python
import numba as nb
import numpy as np
from scipy import optimize

rng = np.random.default_rng(69)

@nb.njit
def nnls(A, b, maxiter=-1):
    return optimize.nnls(A, b, maxiter)

A = rng.random((10, 5))
x = rng.random(5)
x[::3] *= -1
b = A @ x

print(optimize.nnls(A, b)) # (array([0., 0.07394937, 0.28001083, 0.,0.03610975]), 0.45879487313397976)
print(nnls(A, b)) # (array([0., 0.07394937, 0.28001083, 0., 0.03610975]), 0.45879487313397976)
```

The implementation used as the overload of `scipy.optimize.nnls` depends on the version
of SciPy installed. For SciPy 0.7 $\leq$ version $\leq$ 1.11 the implementation is a
wrapper of the same Fortran implementation used. For version $\gt$ 1.11, it is a Numba
compatible re-write of the python code.

You can use both implementations directly by importing from `nnls_numba`:

```python
import numpy as np
import nnls_numba

rng = np.random.default_rng(69)

A = rng.random((10, 5))
x = rng.random(5)
x[::3] *= -1
b = A @ x

# Rewrite of implementation in versions >= 1.12
print(nnls_numba.nnls_new(A, b, None, 1e-9)) # (array([0., 0.07394937, 0.28001083, 0., 0.03610975]), 0.4587948731339797)
# Wrapper of the old fortran subroutine
print(nnls_numba.nnls_old(A, b, -1)) # (array([0., 0.07394937, 0.28001083, 0., 0.03610975]), 0.45879487313397976)
```

### Performance

The Fortran code is typically slow for large problems, hence it was replaced by a python
implementation in SciPy:
[PR-18570](https://github.com/scipy/scipy/pull/18570).

The Fortran subroutine is also available without allocating the temporary work arrays
for you in the `nnls_numba.nnls_old_` function. This is useful if you require to solve
many problems of the same size as the work arrays can be allocated once and reused
(beware though that it modifies its arguments in-place hence the copies in the timings):

```python
import numba as nb
import numpy as np
import nnls_numba
from scipy import optimize

rng = np.random.default_rng(69)

K, M, N = 100_000, 10, 5

A = rng.random((K, M, N))
x = rng.random((K, N))
x[:, ::3] *= -1
b = np.einsum('ijk,ik->ij', A, x)

def many_nnls_scipy(A, b):
    output = np.empty((A.shape[0], A.shape[2]))
    for i in range(A.shape[0]):
        output[i] = optimize.nnls(A[i], b[i])[0]
    return output

@nb.njit
def many_nnls_new(A, b, maxiter=-1):
    assert A.shape[:-1] == b.shape
    assert A.ndim == 3
    output = np.empty((A.shape[0], A.shape[2]))
    for i in range(A.shape[0]):
        output[i] = nnls_numba.nnls_new(A[i], b[i], None, 1e-9)[0]
    return output

@nb.njit
def many_nnls_old(A, b, maxiter=-1):
    assert A.shape[:-1] == b.shape
    assert A.ndim == 3
    output = np.empty((A.shape[0], A.shape[2]))
    for i in range(A.shape[0]):
        output[i] = nnls_numba.nnls_old(A[i], b[i], maxiter)[0]
    return output

@nb.njit
def many_nnls_old_(A, b, maxiter=-1):
    assert A.shape[:-1] == b.shape
    assert A.ndim == 3
    m = A.shape[1]
    n = A.shape[2]
    x = np.empty((A.shape[0], n))
    rnorm = np.empty(1, dtype=np.float64)
    w = np.empty(m, dtype=np.float64)
    zz = np.empty(m, dtype=np.float64)
    index = np.empty(m, dtype=np.int32)
    mode = np.empty(1, dtype=np.int32)
    for i in range(A.shape[0]):
        nnls_numba.nnls_old_(A[i], m, n, b[i], x[i], rnorm, w, zz, index, mode, maxiter)
    return x
    
assert np.allclose(many_nnls_new(A, b), many_nnls_scipy(A, b))
assert np.allclose(many_nnls_new(A, b), many_nnls_old(A, b))
assert np.allclose(many_nnls_new(A, b), many_nnls_old_(A, b))

%timeit many_nnls_scipy(A, b)
%timeit many_nnls_new(A, b)
%timeit many_nnls_old(A, b)
%timeit many_nnls_old_(A.copy(), b.copy())
```

Output:

```text
422 ms ± 524 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
164 ms ± 1.72 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
57.9 ms ± 3.53 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
31.6 ms ± 605 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```
