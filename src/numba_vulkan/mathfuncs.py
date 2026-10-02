"""``math`` functions that Vulkan lacks, written in Python.

These are fallbacks. Where libclc provides a function (see
`numba_vulkan.mathimpl.libclc_name`), its version is used instead; the
implementations here serve when libclc is not installed and with
``fastmath``.


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

import numpy as np
from numba.core import types
from numba.core.typing import signature

from numba_vulkan.mathimpl import call_libclc, libclc_name, lower


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


def _jit(pyfunc, fastmath=False):
    """Make a helper callable from the implementations below.

    Parameters
    ----------
    pyfunc : function
        The helper.
    fastmath : bool
        Whether its math calls use the device's built-in functions.

    Returns
    -------
    numba_vulkan.dispatcher.VulkanDispatcher
    """
    from numba_vulkan.dispatcher import jit

    return jit(pyfunc, fastmath=fastmath)


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


def _sinpi(ty):
    """Build ``sin(pi * x)`` with exact argument reduction.

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
    ``x`` is reduced to [-1/2, 1/2] with operations that are exact
    (truncation, and subtractions whose results are representable), so the
    result
    keeps its relative accuracy near the zeros at the integers, which
    ``sin(pi * x)`` loses for large ``x``. ``fmod`` would not do: Vulkan
    only requires ``OpFRem`` to be as accurate as ``x - y * trunc(x / y)``.
    """
    half, one, two, pi = ty(0.5), ty(1), ty(2), ty(math.pi)

    def sinpi(x):
        """``sin(pi * x)``."""
        r = x - two * np.trunc(x * half)
        if r > one:
            r -= two
        elif r < -one:
            r += two
        if r > half:
            r = one - r
        elif r < -half:
            r = -one - r
        return math.sin(pi * r)

    return _jit(sinpi)


# Coefficients of the gamma function after OCML (AMD's device library), as
# adopted by upstream libclc after LLVM 22 ("libclc: Improve tgamma
# handling", llvm-project#188066). That code is under Apache-2.0 WITH
# LLVM-exception, like the bundled libclc; see data/LICENSE-libclc.txt.
# For |x| < 16, after shifting x into [-1/2, 1/2] by the recurrence, gamma
# is n / (d (1 + y q(y))); beyond that, Stirling's series in 1 / |x|. Each tuple lists a polynomial's
# coefficients from the highest degree down, for Horner's scheme.
_GAMMA = {
    64: {
        "q": (
            "-0x1.aed75feec7b9ap-23", "0x1.31854a0be3cd3p-20",
            "-0x1.5037d6a97a8b7p-20", "-0x1.51d67f2cdbcfbp-16",
            "0x1.0c8ab2ac5112dp-13", "-0x1.c364ce9b5e149p-13",
            "-0x1.317113a39f929p-10", "0x1.d919c501178a3p-8",
            "-0x1.3b4af282da690p-7", "-0x1.59af103bf2cd0p-5",
            "0x1.5512320b432ccp-3", "-0x1.5815e8fa28886p-5",
            "-0x1.4fcf4026afa24p-1", "0x1.2788cfc6fb61cp-1",
        ),
        # Stirling: the series is 1 + p(1/x) / x.
        "p": (
            "-0x1.2b04c5ea74bbfp-11", "0x1.14869344f1d9bp-14",
            "0x1.9b3457156ffefp-11", "-0x1.e1427e86ee097p-13",
            "-0x1.5f7266f67c4e0p-9", "0x1.c71c71c0f96adp-9",
            "0x1.5555555555a28p-4",
        ),
        "sqrt2pi": "0x1.40d931ff62706p+1",
        # Beyond `overflow` the result is infinite; below `split` the
        # reflection is evaluated in two steps to avoid overflow, and
        # below `underflow` the result is zero.
        "overflow": "0x1.573fae561f646p+7",
        "split": "-170.5",
        "underflow": "-184.0",
    },
    32: {
        "q": (
            "0x1.d5a56ep-8", "-0x1.4dcb00p-7", "-0x1.59c03ap-5",
            "0x1.55405ap-3", "-0x1.5810f2p-5", "-0x1.4fcfd6p-1",
            "0x1.2788ccp-1",
        ),
        # Stirling: the series is p(1/x) itself.
        "p": ("0x1.96d7e4p-9", "0x1.556652p-4", "0x1.fffff8p-1"),
        "sqrt2pi": "0x1.40d932p+1",
        "overflow": "0x1.18521ep+5",
        "split": "-30.0",
        "underflow": "-41.0",
    },
}  # fmt: skip


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
    A port of the ``tgamma`` of upstream libclc after LLVM 22, which
    replaced the ``exp(lgamma(x))`` of libclc 22. That form loses precision
    as ``lgamma`` grows (up to 1800 ulp near 170) and overflows in the
    reflection for large negative arguments. It is used even where libclc
    is available, and ``fastmath`` does not apply to it.

    Upstream writes the recurrence steps as multiply-adds, ``n * y - n``
    and ``d * y + d``, which cancel badly near the poles unless they are
    fused (Mesa's copy of libclc fuses them for that reason). Here they are
    ``n * (y - 1)`` and ``d * (y + 1)``: in the range where they are used,
    ``y - 1`` and ``y + 1`` are exact, so each step rounds once, as a fused
    multiply-add would, on every device. Vulkan does not guarantee fusion;
    llvmpipe does not fuse float64.
    """
    table = _GAMMA[ty.bitwidth]
    q = tuple(ty(float.fromhex(c)) for c in table["q"])
    p = tuple(ty(float.fromhex(c)) for c in table["p"])
    stirling_has_one = ty.bitwidth == 64
    sqrt2pi = ty(float.fromhex(table["sqrt2pi"]))
    sqrtpiby2 = sqrt2pi / ty(2)
    overflow = ty(float.fromhex(table["overflow"]))
    split, underflow = ty(float(table["split"])), ty(float(table["underflow"]))
    zero, quarter, half, one = ty(0), ty(0.25), ty(0.5), ty(1)
    sixteen, inf, nan = ty(16), ty(math.inf), ty(math.nan)
    sinpi = _sinpi(ty)

    def is_integer(x):
        # also true for infinities, whose gamma is NaN on the left
        return math.isinf(x) or np.trunc(x) == x

    is_integer = _jit(is_integer)

    def times_series(g, xr, acc):
        # g times Stirling's series, rounded as upstream does
        if stirling_has_one:
            return g * (xr * acc) + g
        return g * acc

    times_series = _jit(times_series)

    def gamma(x):
        """Gamma function."""
        ax = abs(x)
        if ax < sixteen:
            y = x
            if x > zero:
                n = one
                while y > ty(2.5):
                    n = n * (y - one)
                    y = y - one
                    n = n * (y - one)
                    y = y - one
                if y > ty(1.5):
                    n = n * (y - one)
                    y = y - one
                if x >= half:
                    y = y - one
                d = x if x < half else one
            else:
                d = x
                while y < ty(-1.5):
                    d = d * (y + one)
                    y = y + one
                    d = d * (y + one)
                    y = y + one
                if y < -half:
                    d = d * (y + one)
                    y = y + one
                n = one
            acc = q[0]
            for c in q[1:]:
                acc = y * acc + c
            ret = n / (d * (y * acc) + d)
            if x == zero:
                return math.copysign(inf, x)
            if x < zero and is_integer(x):
                return nan
            return ret
        # x^(x/2 - 1/4) e^(-x/2): its square is gamma up to the series, so
        # no partial product below can overflow, in whatever order a
        # driver evaluates it (NVIDIA reorders in fastmath kernels, which
        # are not decorated NoContraction).
        h = math.pow(ax, ax * half - quarter) * math.exp(-ax * half)
        xr = one / ax
        acc = p[0]
        for c in p[1:]:
            acc = xr * acc + c
        if x > zero:
            if x > overflow:
                return inf
            return times_series(sqrt2pi * h * h, xr, acc)
        if is_integer(x) or math.isnan(x):
            return nan
        s = -x * sinpi(x)
        if x > split:
            return sqrtpiby2 / times_series(s * h * h, xr, acc)
        if x > underflow:
            return (sqrtpiby2 / times_series(h, xr, acc)) / (s * h)
        return math.copysign(zero, s)

    # Compiled without fastmath even in fastmath kernels, so that it keeps
    # libclc's exp and pow and LLVM keeps the order of operations.
    accurate = _jit(gamma)

    def gamma_call(x):
        """Gamma function."""
        return accurate(x)

    return gamma_call


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
        from_libclc = libclc_name(context, pyfn, ty)
        if from_libclc is not None:
            res = call_libclc(builder, from_libclc, vals, ty)
            return context.cast(builder, res, ty, sig.return_type)
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
