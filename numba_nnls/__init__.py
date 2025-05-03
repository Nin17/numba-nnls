"""_summary_"""

__version__ = "0.1.0"

from . import scipy_overloads
from ._nnls import nnls_new, nnls_old, nnls_old_


def _init() -> None:
    from . import scipy_overloads


__all__ = ["nnls_new", "nnls_old", "nnls_old_", "scipy_overloads"]
