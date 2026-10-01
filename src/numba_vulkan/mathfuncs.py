"""``math`` functions that Vulkan lacks, written in Python.

GLSL.std.450 has no ``hypot``, ``log1p``, ``erf``, ``gamma``... and there is
no vendor math library to fall back on, as libdevice is for CUDA. The
functions here are ordinary Python implementations that Numba compiles for
the Vulkan target whenever a kernel calls the corresponding ``math``
function.

Each factory receives the Numba float type to compute in and returns the
implementation for it. All constants are created in that type, because a
Python float literal would promote the whole computation to float64.

The transcendental building blocks (``exp``, ``log``, ``sin``) are 32-bit
only on Vulkan, so most of these functions inherit that restriction; see
`numba_vulkan.mathimpl`.
"""

import math

from numba.core import types
from numba.core.typing import signature

from numba_vulkan.mathimpl import lower


def _hypot(ty):
    """Build ``math.hypot``.

    Parameters
    ----------
    ty : numba.types.Float
        Float type to compute in.

    Returns
    -------
    function
        The implementation for ``ty``.

    Notes
    -----
    The arguments are scaled by the larger magnitude, so the squares
    cannot overflow.
    """
    zero = ty(0)

    def hypot(x, y):
        # Scale by the larger magnitude so the squares cannot overflow.
        """Euclidean norm of two values."""
        a, b = abs(x), abs(y)
        big = max(a, b)
        if big == zero or math.isinf(big):
            return big
        a, b = a / big, b / big
        return big * math.sqrt(a * a + b * b)

    return hypot


def _log1p(ty):
    """Build ``math.log1p``.

    Parameters
    ----------
    ty : numba.types.Float
        Float type to compute in.

    Returns
    -------
    function
        The implementation for ``ty``.

    Notes
    -----
    Small arguments use ``log1p(x) = 2 atanh(x / (2 + x))`` with the
    series of ``atanh``. The usual correction of ``log(1 + x)`` is not
    enough here, because the ``log`` of GPU drivers is itself imprecise
    close to 1.
    """
    half, one, two = ty(0.5), ty(1), ty(2)
    c3, c5, c7, c9, c11 = ty(1 / 3), ty(1 / 5), ty(1 / 7), ty(1 / 9), ty(1 / 11)

    def log1p(x):
        """Logarithm of ``1 + x``, accurate for small ``x``."""
        if abs(x) < half:
            s = x / (two + x)
            s2 = s * s
            return (
                two
                * s
                * (one + s2 * (c3 + s2 * (c5 + s2 * (c7 + s2 * (c9 + s2 * c11)))))
            )
        return math.log(one + x)

    return log1p


def _expm1(ty):
    """Build ``math.expm1``.

    Parameters
    ----------
    ty : numba.types.Float
        Float type to compute in.

    Returns
    -------
    function
        The implementation for ``ty``.

    Notes
    -----
    Small arguments use the Taylor series of the exponential, for the same
    reason as in `_log1p`.
    """
    half, one = ty(0.5), ty(1)
    d2, d3, d4, d5, d6, d7, d8, d9 = (ty(1 / n) for n in range(2, 10))

    def expm1(x):
        """``exp(x) - 1``, accurate for small ``x``."""
        if abs(x) < half:
            inner = one + x * d7 * (one + x * d8 * (one + x * d9))
            inner = one + x * d4 * (one + x * d5 * (one + x * d6 * inner))
            return x * (one + x * d2 * (one + x * d3 * inner))
        return math.exp(x) - one

    return expm1


def _asinh(ty):
    """Build ``math.asinh``.

    Parameters
    ----------
    ty : numba.types.Float
        Float type to compute in.

    Returns
    -------
    function
        The implementation for ``ty``.
    """
    one = ty(1)

    def asinh(x):
        """Inverse hyperbolic sine, computed through ``log1p``."""
        a = abs(x)
        r = math.log1p(a + a * a / (one + math.sqrt(one + a * a)))
        return -r if x < 0 else r

    return asinh


def _acosh(ty):
    """Build ``math.acosh``.

    Parameters
    ----------
    ty : numba.types.Float
        Float type to compute in.

    Returns
    -------
    function
        The implementation for ``ty``.
    """
    one = ty(1)

    def acosh(x):
        """Inverse hyperbolic cosine."""
        return math.log(x + math.sqrt(x * x - one))

    return acosh


def _atanh(ty):
    """Build ``math.atanh``.

    Parameters
    ----------
    ty : numba.types.Float
        Float type to compute in.

    Returns
    -------
    function
        The implementation for ``ty``.
    """
    one, two, half = ty(1), ty(2), ty(0.5)

    def atanh(x):
        """Inverse hyperbolic tangent, computed through ``log1p``."""
        a = abs(x)
        r = half * math.log1p(two * a / (one - a))
        return -r if x < 0 else r

    return atanh


def _degrees(ty):
    """Build ``math.degrees``.

    Parameters
    ----------
    ty : numba.types.Float
        Float type to compute in.

    Returns
    -------
    function
        The implementation for ``ty``.
    """
    factor = ty(180 / math.pi)
    return lambda x: x * factor


def _radians(ty):
    """Build ``math.radians``.

    Parameters
    ----------
    ty : numba.types.Float
        Float type to compute in.

    Returns
    -------
    function
        The implementation for ``ty``.
    """
    factor = ty(math.pi / 180)
    return lambda x: x * factor


# Abramowitz and Stegun 7.1.26: erfc(x) = poly(t) * exp(-x^2) for x >= 0,
# with t = 1 / (1 + p x). The absolute error is below 1.5e-7, which is
# float32 accuracy; nothing better is attempted for float64.
_ERF_P = 0.3275911
_ERF_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)


def _jit(pyfunc):
    """Make a helper callable from the implementations below."""
    from numba_vulkan.dispatcher import jit

    return jit(pyfunc)


def _erfc_positive(ty):
    """Build the complementary error function for non-negative arguments.

    Parameters
    ----------
    ty : numba.types.Float
        Float type to compute in.

    Returns
    -------
    function
        The implementation for ``ty``.

    Notes
    -----
    Abramowitz and Stegun, formula 7.1.26; absolute error below 1.5e-7.
    """
    one, p = ty(1), ty(_ERF_P)
    a1, a2, a3, a4, a5 = (ty(a) for a in _ERF_A)

    def erfc_positive(x):
        """``erfc(x)`` for ``x >= 0``."""
        t = one / (one + p * x)
        poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
        return poly * math.exp(-x * x)

    return _jit(erfc_positive)


def _erf(ty):
    """Build ``math.erf``.

    Parameters
    ----------
    ty : numba.types.Float
        Float type to compute in.

    Returns
    -------
    function
        The implementation for ``ty``.
    """
    one = ty(1)
    tail = _erfc_positive(ty)

    def erf(x):
        """Error function, from the tail for the absolute value."""
        r = one - tail(abs(x))
        return -r if x < 0 else r

    return erf


def _erfc(ty):
    """Build ``math.erfc``.

    Parameters
    ----------
    ty : numba.types.Float
        Float type to compute in.

    Returns
    -------
    function
        The implementation for ``ty``.
    """
    two = ty(2)
    tail = _erfc_positive(ty)

    def erfc(x):
        """Complementary error function."""
        r = tail(abs(x))
        return two - r if x < 0 else r

    return erfc


# Lanczos approximation with g = 5 (Numerical Recipes, "gammln").
_LANCZOS = (
    76.18009172947146,
    -86.50532032941677,
    24.01409824083091,
    -1.231739572450155,
    0.1208650973866179e-2,
    -0.5395239384953e-5,
)


def _lgamma_positive(ty):
    """Build the log-gamma function for arguments of at least one half.

    Parameters
    ----------
    ty : numba.types.Float
        Float type to compute in.

    Returns
    -------
    function
        The implementation for ``ty``.

    Notes
    -----
    Lanczos approximation with g = 5 and six coefficients.
    """
    half, one, offset = ty(0.5), ty(1), ty(5.5)
    first, root = ty(1.000000000190015), ty(2.5066282746310005)
    c0, c1, c2, c3, c4, c5 = (ty(c) for c in _LANCZOS)

    def lgamma_positive(x):
        """``log(gamma(x))`` for ``x >= 0.5``."""
        tmp = x + offset
        tmp -= (x + half) * math.log(tmp)
        ser = first
        ser += c0 / (x + one)
        ser += c1 / (x + one + one)
        ser += c2 / (x + ty(3))
        ser += c3 / (x + ty(4))
        ser += c4 / (x + ty(5))
        ser += c5 / (x + ty(6))
        return -tmp + math.log(root * ser / x)

    return _jit(lgamma_positive)


def _lgamma(ty):
    """Build ``math.lgamma``.

    Parameters
    ----------
    ty : numba.types.Float
        Float type to compute in.

    Returns
    -------
    function
        The implementation for ``ty``.

    Notes
    -----
    Arguments below one half use the reflection formula.
    """
    half, one, pi = ty(0.5), ty(1), ty(math.pi)
    positive = _lgamma_positive(ty)

    def lgamma(x):
        """Logarithm of the absolute value of the gamma function."""
        if x < half:
            # Reflection formula for the left half of the real line.
            return math.log(pi / abs(math.sin(pi * x))) - positive(one - x)
        return positive(x)

    return lgamma


def _gamma(ty):
    """Build ``math.gamma``.

    Parameters
    ----------
    ty : numba.types.Float
        Float type to compute in.

    Returns
    -------
    function
        The implementation for ``ty``.

    Notes
    -----
    Computed as the exponential of the log-gamma function, with the
    reflection formula below one half.
    """
    half, one, pi = ty(0.5), ty(1), ty(math.pi)
    positive = _lgamma_positive(ty)

    def gamma(x):
        """Gamma function."""
        if x < half:
            return pi / (math.sin(pi * x) * math.exp(positive(one - x)))
        return math.exp(positive(x))

    return gamma


_UNARY = {
    math.log1p: _log1p,
    math.expm1: _expm1,
    math.asinh: _asinh,
    math.acosh: _acosh,
    math.atanh: _atanh,
    math.degrees: _degrees,
    math.radians: _radians,
    math.erf: _erf,
    math.erfc: _erfc,
    math.lgamma: _lgamma,
    math.gamma: _gamma,
}
_BINARY = {
    math.hypot: _hypot,
}


def _register(pyfn, factory, nargs):
    """Register a Python implementation as the lowering of a ``math`` function.

    Parameters
    ----------
    pyfn : callable
        The ``math`` function.
    factory : callable
        Takes a Numba float type and returns the implementation for it.
    nargs : int
        Number of arguments.
    """

    def impl(context, builder, sig, args):
        """Compile the Python implementation and call it.

        Integer arguments are converted to float64 first, as Numba's
        typing prescribes.
        """
        ty = (
            sig.return_type
            if isinstance(sig.return_type, types.Float)
            else types.float64
        )
        vals = [context.cast(builder, a, t, ty) for a, t in zip(args, sig.args)]
        inner = signature(ty, *[ty] * nargs)
        res = context.compile_internal(builder, factory(ty), inner, vals)
        return context.cast(builder, res, ty, sig.return_type)

    for kinds in ([types.Float] * nargs, [types.Integer] * nargs):
        lower(pyfn, *kinds)(impl)
    if nargs == 2:
        lower(pyfn, types.Float, types.Integer)(impl)
        lower(pyfn, types.Integer, types.Float)(impl)


for _fn, _factory in _UNARY.items():
    _register(_fn, _factory, 1)
for _fn, _factory in _BINARY.items():
    _register(_fn, _factory, 2)
