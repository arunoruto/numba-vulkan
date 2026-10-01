"""How NumPy ufuncs are lowered on the Vulkan target.

Numba lowers a call such as ``np.sqrt(x)`` by looking up the ufunc's
"loops" (one per combination of argument types) in a database supplied by
the target. This module builds that database, following the structure of
``numba.cuda.ufuncs``:

* arithmetic, comparison, logical and bitwise ufuncs reuse Numba's CPU
  implementations, which emit plain LLVM instructions;
* everything that needs a math library is routed to this target's
  implementation of the corresponding ``math`` function, so that ufuncs and
  ``math`` behave identically (including the float64 restrictions).

Only loops over booleans, integers, float32 and float64 are provided.
"""

import math
from functools import lru_cache

import numpy as np

from numba_vulkan import mathimpl

# Type characters of the loops that are kept from Numba's CPU database.
_REAL_CHARS = set("?bBhHiIlLqQfd")

# Ufuncs whose CPU implementation needs nothing but LLVM instructions.
_REUSED = (
    "add", "subtract", "multiply", "divide", "floor_divide", "remainder",
    "negative", "positive", "absolute", "sign", "square", "reciprocal",
    "maximum", "minimum", "fmax", "fmin",
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "logical_and", "logical_or", "logical_xor", "logical_not",
    "bitwise_and", "bitwise_or", "bitwise_xor", "invert",
    "left_shift", "right_shift",
)  # fmt: skip

# Ufuncs implemented by a ``math`` function, by number of arguments.
_UNARY = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "arcsin": math.asin, "arccos": math.acos, "arctan": math.atan,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "arcsinh": math.asinh, "arccosh": math.acosh, "arctanh": math.atanh,
    "exp": math.exp, "exp2": math.exp2, "expm1": math.expm1,
    "log": math.log, "log2": math.log2, "log10": math.log10, "log1p": math.log1p,
    "sqrt": math.sqrt, "fabs": math.fabs,
    "floor": math.floor, "ceil": math.ceil, "trunc": math.trunc,
    "degrees": math.degrees, "rad2deg": math.degrees,
    "radians": math.radians, "deg2rad": math.radians,
}  # fmt: skip
_BINARY = {
    "arctan2": math.atan2, "hypot": math.hypot, "copysign": math.copysign,
    "fmod": math.fmod, "power": math.pow,
}  # fmt: skip
_PREDICATES = {"isnan": math.isnan, "isinf": math.isinf, "isfinite": math.isfinite}


def _via_math(pyfn):
    """Loop implementation that defers to this target's ``math`` lowering.

    Parameters
    ----------
    pyfn : callable
        The ``math`` function.

    Returns
    -------
    callable
        ``impl(context, builder, sig, args)`` as expected by Numba's ufunc
        machinery.
    """

    def impl(context, builder, sig, args):
        """Generate one loop by calling the ``math`` lowering."""
        return context.get_function(pyfn, sig)(builder, args)

    return impl


def _int_power(context, builder, sig, args):
    """Integer loops of ``np.power``; see `mathimpl.int_int_power`."""
    return mathimpl.int_int_power(context, builder, sig, args)


@lru_cache
def ufunc_db():
    """Build the database of supported ufunc loops.

    Returns
    -------
    dict
        Maps each supported ufunc to a dictionary from NumPy loop signature
        (for example ``"ff->f"``) to the function generating that loop.
    """
    from numba.np import ufunc_db as cpu_db

    cpu_db._lazy_init_db()
    cpu = {ufunc.__name__: (ufunc, loops) for ufunc, loops in cpu_db._ufunc_db.items()}
    db = {}
    for name in _REUSED:
        ufunc, loops = cpu[name]
        db[ufunc] = {
            sig: fn
            for sig, fn in loops.items()
            if set(sig.replace("->", "")) <= _REAL_CHARS
        }
    for name, pyfn in _UNARY.items():
        db[getattr(np, name)] = {"f->f": _via_math(pyfn), "d->d": _via_math(pyfn)}
    for name, pyfn in _BINARY.items():
        db[getattr(np, name)] = {"ff->f": _via_math(pyfn), "dd->d": _via_math(pyfn)}
    for name, pyfn in _PREDICATES.items():
        db[getattr(np, name)] = {"f->?": _via_math(pyfn), "d->?": _via_math(pyfn)}
    for sig in cpu["power"][1]:
        if set(sig.replace("->", "")) <= set("bBhHiIlLqQ"):
            db[np.power][sig] = _int_power
    return db


def get_ufunc_info(ufunc_key):
    """Look up the loops of a ufunc.

    Parameters
    ----------
    ufunc_key : numpy.ufunc
        The ufunc.

    Returns
    -------
    dict
        Loop signature to implementation.

    Raises
    ------
    KeyError
        If the ufunc is not supported on Vulkan.
    """
    return ufunc_db()[ufunc_key]
