"""_summary_"""

__version__ = "0.1.0"

__all__ = ("nnls_007_111", "nnls_007_111_", "nnls_112_114", "scipy_overloads")

from . import scipy_overloads
from ._nnls import nnls_007_111, nnls_007_111_, nnls_112_114


def _init() -> None:
    from . import scipy_overloads
