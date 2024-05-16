"""_summary_"""

from pathlib import Path

from numba.core import cgutils, types
from numba.extending import intrinsic


@intrinsic
def ptr_from_val(typingctx, data):
    """
    https://stackoverflow.com/questions/51541302/
    how-to-wrap-a-cffi-function-in-numba-taking-pointers
    """

    def impl(context, builder, signature, args):
        ptr = cgutils.alloca_once_value(builder, args[0])
        return ptr

    sig = types.CPointer(data)(data)
    return sig, impl


@intrinsic
def val_from_ptr(typingctx, data):
    """
    https://stackoverflow.com/questions/51541302/
    how-to-wrap-a-cffi-function-in-numba-taking-pointers
    """

    def impl(context, builder, signature, args):
        val = builder.load(args[0])
        return val

    sig = data.dtype(data)
    return sig, impl


def get_extension_path(lib_name):
    """
    Modified from rocket-fft
    """
    search_path = Path(__file__).parent
    ext_path = f"{lib_name}.*"
    matches = search_path.glob(ext_path)
    try:
        return str(next(matches))
    except StopIteration:
        return None
