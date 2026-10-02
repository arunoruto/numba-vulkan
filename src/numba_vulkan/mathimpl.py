"""Lowering of the ``math`` module onto GLSL.std.450 via LLVM intrinsics."""

import math
import operator

from llvmlite import ir
from numba.core import cgutils, types
from numba.core.imputils import Registry

from numba_vulkan import libclc, narrowing
from numba_vulkan.errors import VulkanUnsupportedError

registry = Registry("vkmathimpl")
lower = registry.lower

# GLSL.std.450 defines these for 16/32-bit floats only.
_F32_ONLY = {
    math.sin: "llvm.sin",
    math.cos: "llvm.cos",
    math.tan: "llvm.tan",
    math.asin: "llvm.asin",
    math.acos: "llvm.acos",
    math.atan: "llvm.atan",
    math.sinh: "llvm.sinh",
    math.cosh: "llvm.cosh",
    math.tanh: "llvm.tanh",
    math.exp: "llvm.exp",
    math.log: "llvm.log",
    math.log2: "llvm.log2",
    math.log10: "llvm.log10",
    math.pow: "llvm.pow",
    math.atan2: "llvm.atan2",
    math.exp2: "llvm.exp2",
}
# ... and these for doubles as well.
_ANY_FLOAT = {
    math.sqrt: "llvm.sqrt",
    math.fabs: "llvm.fabs",
}
_BINARY = (math.pow, math.atan2)

# Name of each function in libclc (OpenCL spelling).
_LIBCLC = {
    math.sin: "sin",
    math.cos: "cos",
    math.tan: "tan",
    math.asin: "asin",
    math.acos: "acos",
    math.atan: "atan",
    math.sinh: "sinh",
    math.cosh: "cosh",
    math.tanh: "tanh",
    math.asinh: "asinh",
    math.acosh: "acosh",
    math.atanh: "atanh",
    math.exp: "exp",
    math.exp2: "exp2",
    math.expm1: "expm1",
    math.log: "log",
    math.log2: "log2",
    math.log10: "log10",
    math.log1p: "log1p",
    math.pow: "pow",
    math.atan2: "atan2",
    math.hypot: "hypot",
    math.erf: "erf",
    math.erfc: "erfc",
    math.lgamma: "lgamma",
    # math.gamma is not taken from libclc 22, whose tgamma loses precision
    # for large arguments; mathfuncs has a port of the newer upstream one.
}
# Functions libclc has, but whose code does not survive the SPIR-V backend
# yet, by (name, bit width); they fall back to this package's own versions.
# Empty at present.
_LIBCLC_BROKEN = set()
_ROUNDING = {
    math.floor: "llvm.floor",
    math.ceil: "llvm.ceil",
    math.trunc: "llvm.trunc",
}


def call_intrinsic(builder, name, args):
    """Call an LLVM floating point intrinsic.

    Parameters
    ----------
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    name : str
        Intrinsic name without type suffix, for example ``"llvm.sin"``.
    args : list of llvmlite.ir.Value
        Arguments, all of the same float type.

    Returns
    -------
    llvmlite.ir.Value
        The result, of the same type as the arguments.

    Notes
    -----
    The SPIR-V backend maps these intrinsics to GLSL.std.450 extended
    instructions.
    """
    ty = args[0].type
    fnty = ir.FunctionType(ty, [ty] * len(args))
    return builder.call(builder.module.declare_intrinsic(name, [ty], fnty), args)


def libclc_name(context, pyfn, ty):
    """Decide whether a ``math`` function is taken from libclc.

    libclc is preferred: its functions are accurate to the last digit or
    two (but see KI-31), give the same results on every device, and exist
    in double precision. With ``fastmath``, float32 functions use the device's
    built-in versions instead, which are faster but less accurate and
    differ between drivers. float64 has no built-in alternative.

    Parameters
    ----------
    context : VulkanTargetContext
        The target context; its ``fast_math`` flag is consulted.
    pyfn : callable
        The ``math`` function.
    ty : numba.types.Float
        Type the function is evaluated in.

    Returns
    -------
    str or None
        The libclc name to call, or ``None`` to use another implementation.
    """
    name = _LIBCLC.get(pyfn)
    if name is None or (name, ty.bitwidth) in _LIBCLC_BROKEN or not libclc.available():
        return None
    if ty == types.float32 and context.fast_math:
        return None
    return name


def _float_math(context, builder, pyfn, name, f32_only, sig, args):
    """Lower a ``math`` function to a libclc call or an LLVM intrinsic.

    Integer arguments are converted to float64 first, as Numba's typing
    prescribes. See `libclc_name` for how the implementation is chosen.

    Parameters
    ----------
    context : VulkanTargetContext
        The target context; its ``narrow_math`` flag is consulted.
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    pyfn : callable
        The ``math`` function, used in error messages.
    name : str
        Name of the LLVM intrinsic.
    f32_only : bool
        Whether GLSL.std.450 lacks a float64 version of the function.
    sig : numba.core.typing.Signature
        Signature the call was typed with.
    args : sequence of llvmlite.ir.Value
        Argument values.

    Returns
    -------
    llvmlite.ir.Value
        The result as a value of the float type computed in.

    Raises
    ------
    VulkanUnsupportedError
        For a float64 call of a 32-bit-only function when libclc is not
        installed and ``narrow_math`` is off.
    """
    ty = sig.return_type if isinstance(sig.return_type, types.Float) else types.float64
    vals = [context.cast(builder, a, t, ty) for a, t in zip(args, sig.args)]
    from_libclc = libclc_name(context, pyfn, ty)
    if from_libclc is not None:
        return call_libclc(builder, from_libclc, vals, ty)
    if f32_only and ty == types.float64 and not narrowing.current.floats:
        if not context.narrow_math:
            raise VulkanUnsupportedError(
                f"math.{pyfn.__name__} on float64 needs libclc, which was not found "
                "(Vulkan's own math library, GLSL.std.450, is 32-bit only). Install "
                f"libclc or set {libclc.ENV_VAR}; alternatively use float32 values, "
                "or pass narrow_math=True to compute in float32"
            )
        vals = [builder.fptrunc(v, ir.FloatType()) for v in vals]
        return builder.fpext(call_intrinsic(builder, name, vals), ir.DoubleType())
    return call_intrinsic(builder, name, vals)


# Cody and Waite's split of ln 2 (as in fdlibm): the head has 21 trailing
# zero bits, so its products with the integers used below are exact.
_LN2_HI = float.fromhex("0x1.62e42feep-1")
_LN2_LO = float.fromhex("0x1.a39ef35793c76p-33")
# Beyond this, exp overflows or underflows to zero for any reduction.
_EXP_REDUCE_LIMIT = 746.0


def reduced_exp(builder, x, scale=0):
    """``exp`` of a ``double``, reduced here before calling libclc.

    libclc's own ``float64`` ``exp`` loses precision as the argument grows
    on some devices (up to 673 ulp on llvmpipe and 208 on Intel's UHD 630
    near 700, 1 ulp on NVIDIA), while it is accurate everywhere for small
    arguments. So ``x`` is split into ``k ln 2 + r`` with ``|r| <= ln(2)/2``
    and exact arithmetic, libclc computes ``exp(r)``, and the result is
    scaled by ``2**k``, built from its bit pattern in two factors so that
    neither overflows near the ends of the range.

    Parameters
    ----------
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    x : llvmlite.ir.Value
        The argument, a ``double``.
    scale : int, optional
        Power of two to multiply the result by, exactly; ``-1`` gives
        ``exp(x) / 2`` without overflowing where that is finite.

    Returns
    -------
    llvmlite.ir.Value
        ``exp(x) * 2**scale`` as a ``double``.

    Notes
    -----
    Arguments beyond about 746 in magnitude, infinities and NaN go to
    libclc unreduced; the result is then infinite, zero or NaN anyway.
    """
    f64, i32 = ir.DoubleType(), ir.IntType(32)
    absx = call_intrinsic(builder, "llvm.fabs", [x])
    # False for NaN, which therefore takes the unreduced path.
    reduce = builder.fcmp_ordered("<=", absx, f64(_EXP_REDUCE_LIMIT))
    safe = builder.select(reduce, x, f64(0.0))
    rounding = call_intrinsic(builder, "llvm.copysign", [f64(0.5), safe])
    n = call_intrinsic(
        builder,
        "llvm.trunc",
        [builder.fadd(builder.fmul(safe, f64(1 / math.log(2))), rounding)],
    )
    r = builder.fsub(
        builder.fsub(safe, builder.fmul(n, f64(_LN2_HI))),
        builder.fmul(n, f64(_LN2_LO)),
    )
    k = builder.fptosi(n, i32)
    if scale:
        k = builder.add(k, i32(scale))
    half = builder.ashr(k, i32(1))
    scaled = libclc.call(builder, "exp", [builder.select(reduce, r, x)])
    for e in (half, builder.sub(k, half)):
        high = builder.shl(builder.add(e, i32(1023)), i32(20))
        scaled = builder.fmul(scaled, words_double(builder, i32(0), high))
    return scaled


def _branch(builder, small, small_value, large_value):
    """Choose between two values computed only on their own branch.

    Parameters
    ----------
    builder : llvmlite.ir.IRBuilder
    small : llvmlite.ir.Value
        ``i1`` condition selecting `small_value`.
    small_value, large_value : callable
        Emit the code of each alternative and return its value.

    Returns
    -------
    llvmlite.ir.Value
    """
    with builder.if_else(small) as (then, otherwise):
        with then:
            a = small_value()
            a_block = builder.block
        with otherwise:
            b = large_value()
            b_block = builder.block
    phi = builder.phi(a.type)
    phi.add_incoming(a, a_block)
    phi.add_incoming(b, b_block)
    return phi


def _cosh_sinh(sign):
    """Build ``cosh`` (``sign`` 1) or ``sinh`` (``sign`` -1) for doubles.

    From 1 on, they are ``e/2 + sign / (4 e/2)`` with ``e/2 = exp(|x|) / 2``
    from `reduced_exp`; below, libclc's own versions, which are accurate
    there, are used.
    """
    name = "cosh" if sign == 1 else "sinh"

    def lower_double(builder, x):
        f64 = ir.DoubleType()
        absx = call_intrinsic(builder, "llvm.fabs", [x])

        def large():
            half_e = reduced_exp(builder, absx, scale=-1)
            tail = builder.fdiv(f64(0.25), half_e)
            if sign == 1:
                return builder.fadd(half_e, tail)
            value = builder.fsub(half_e, tail)
            return call_intrinsic(builder, "llvm.copysign", [value, x])

        small = builder.fcmp_unordered("<", absx, f64(1.0))
        return _branch(builder, small, lambda: libclc.call(builder, name, [x]), large)

    return lower_double


def _expm1_double(builder, x):
    """``expm1`` for doubles: libclc below 1/2 in magnitude, else ``exp - 1``.

    From 1/2 on, ``exp(x) - 1`` amplifies the error of `reduced_exp` at most
    2.5-fold, and libclc's ``expm1`` loses precision for large arguments in
    the same way as its ``exp``. A zero argument is returned as it is,
    because libclc's ``expm1(-0.0)`` is ``+0.0``.
    """
    f64 = ir.DoubleType()
    absx = call_intrinsic(builder, "llvm.fabs", [x])
    small = builder.fcmp_unordered("<", absx, f64(0.5))

    def near_zero():
        value = libclc.call(builder, "expm1", [x])
        return builder.select(builder.fcmp_ordered("==", x, f64(0.0)), x, value)

    return _branch(
        builder,
        small,
        near_zero,
        lambda: builder.fsub(reduced_exp(builder, x), f64(1.0)),
    )


# erfc for 1.25 <= |x| < 28, as in fdlibm (and libclc, which copies it):
# erfc(x) = exp(-z*z - 0.5625) * exp((z - x) * (z + x) + u(t) / v(t)) / x
# with t = 1 / x**2, z = x rounded to 21 significant bits and v = 1 + t*(...).
# Coefficients from highest degree down; B below 1 / 0.35, A above.
#
# Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
# Developed at SunPro, a Sun Microsystems, Inc. business.
# Permission to use, copy, modify, and distribute this software is freely
# granted, provided that this notice is preserved.
_ERFC_SPLIT = float.fromhex("0x1.6db6dp+1")  # about 1 / 0.35
_ERFC_A = (
    (
        -4.83519191608651397019e02, -1.02509513161107724954e03,
        -6.37566443368389627722e02, -1.60636384855821916062e02,
        -1.77579549177547519889e01, -7.99283237680523006574e-01,
        -9.86494292470009928597e-03,
    ),
    (
        -2.24409524465858183362e01, 4.74528541206955367215e02,
        2.55305040643316442583e03, 3.19985821950859553908e03,
        1.53672958608443695994e03, 3.25792512996573918826e02,
        3.03380607434824582924e01,
    ),
)  # fmt: skip
_ERFC_B = (
    (
        -9.81432934416914548592e00, -8.12874355063065934246e01,
        -1.84605092906711035994e02, -1.62396669462573470355e02,
        -6.23753324503260060396e01, -1.05586262253232909814e01,
        -6.93858572707181764372e-01, -9.86494403484714822705e-03,
    ),
    (
        -6.04244152148580987438e-02, 6.57024977031928170135e00,
        1.08635005541779435134e02, 4.29008140027567833386e02,
        6.45387271733267880336e02, 4.34565877475229228821e02,
        1.37657754143519042600e02, 1.96512716674392571292e01,
    ),
)  # fmt: skip


def _horner(builder, t, coefficients):
    """Evaluate a polynomial in ``t``, coefficients from the highest degree."""
    acc = t.type(coefficients[0])
    for c in coefficients[1:]:
        acc = builder.fadd(builder.fmul(acc, t), t.type(c))
    return acc


def _erfc_double(builder, x):
    """``erfc`` for doubles, with `reduced_exp` from 1.25 to 28 in magnitude.

    libclc's version computes fdlibm's formula with its own ``exp``, whose
    argument reaches -785 there and which then loses precision on some
    devices. This evaluates the same formula with `reduced_exp`. ``z`` is
    split off by Veltkamp's method instead of masking bits; it has 21
    significant bits, so ``-z*z - 0.5625`` is exact. Elsewhere, and for NaN,
    libclc's version is used: its ``exp`` arguments are small there, or the
    result is 0 or 2.
    """
    f64 = ir.DoubleType()
    a = call_intrinsic(builder, "llvm.fabs", [x])
    ours = builder.and_(
        builder.fcmp_ordered(">=", a, f64(1.25)),
        builder.and_(
            builder.fcmp_ordered("<", x, f64(28.0)),
            builder.fcmp_ordered(">", x, f64(-6.0)),
        ),
    )

    def tail():
        t = builder.fdiv(f64(1.0), builder.fmul(a, a))
        below = builder.fcmp_ordered("<", a, f64(_ERFC_SPLIT))
        u = builder.select(
            below, _horner(builder, t, _ERFC_B[0]), _horner(builder, t, _ERFC_A[0])
        )
        v = builder.select(
            below, _horner(builder, t, _ERFC_B[1]), _horner(builder, t, _ERFC_A[1])
        )
        q = builder.fdiv(u, builder.fadd(builder.fmul(t, v), f64(1.0)))
        c = builder.fmul(a, f64(2.0**32 + 1))
        z = builder.fsub(c, builder.fsub(c, a))
        first = reduced_exp(
            builder,
            builder.fsub(builder.fneg(builder.fmul(z, z)), f64(0.5625)),
        )
        second = reduced_exp(
            builder,
            builder.fadd(builder.fmul(builder.fsub(z, a), builder.fadd(z, a)), q),
        )
        value = builder.fdiv(builder.fmul(first, second), a)
        negative = builder.fcmp_ordered("<", x, f64(0.0))
        return builder.select(negative, builder.fsub(f64(2.0), value), value)

    return _branch(builder, ours, tail, lambda: libclc.call(builder, "erfc", [x]))


# libclc functions whose float64 versions lose precision for large
# arguments on some devices (KI-31 has the measurements), and the lowering
# that replaces them. All of them depend on libclc's exp.
_EXP_FAMILY = {
    "exp": reduced_exp,
    "expm1": _expm1_double,
    "cosh": _cosh_sinh(1),
    "sinh": _cosh_sinh(-1),
    "erfc": _erfc_double,
}


def call_libclc(builder, name, args, ty):
    """Call a libclc math function, or this package's replacement for it.

    Parameters
    ----------
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    name : str
        The libclc function, as returned by `libclc_name`.
    args : list of llvmlite.ir.Value
        Arguments, of type `ty`.
    ty : numba.types.Float
        Type the function is evaluated in.

    Returns
    -------
    llvmlite.ir.Value
        The result, of type `ty`.
    """
    if name in _EXP_FAMILY and ty == types.float64 and not narrowing.current.floats:
        return _EXP_FAMILY[name](builder, args[0])
    return libclc.call(builder, name, args)


def _register(pyfn, name, f32_only):
    """Register the lowering of one float ``math`` function.

    Parameters
    ----------
    pyfn : callable
        The ``math`` function.
    name : str
        Name of the LLVM intrinsic implementing it.
    f32_only : bool
        Whether GLSL.std.450 lacks a float64 version of the function.
    """
    nargs = 2 if pyfn in _BINARY else 1

    def impl(context, builder, sig, args):
        """Lowering registered for `pyfn`; see `_float_math`."""
        return _float_math(context, builder, pyfn, name, f32_only, sig, args)

    for ty in (types.Float, types.Integer):
        lower(pyfn, *([ty] * nargs))(impl)
    if nargs == 2:
        lower(pyfn, types.Float, types.Integer)(impl)
        lower(pyfn, types.Integer, types.Float)(impl)


for _fn, _name in _F32_ONLY.items():
    _register(_fn, _name, True)
for _fn, _name in _ANY_FLOAT.items():
    _register(_fn, _name, False)


def _register_rounding(pyfn, name):
    """Register the lowering of a ``math`` function that rounds to an integer.

    Parameters
    ----------
    pyfn : callable
        ``math.floor``, ``math.ceil`` or ``math.trunc``.
    name : str
        Name of the LLVM intrinsic implementing it.
    """

    @lower(pyfn, types.Float)
    def impl(context, builder, sig, args):
        """Round a float and convert the result to the integer return type."""
        res = call_intrinsic(builder, name, list(args))
        return context.cast(builder, res, sig.args[0], sig.return_type)

    @lower(pyfn, types.Integer)
    def impl_int(context, builder, sig, args):
        """Rounding an integer only converts it to the return type."""
        return context.cast(builder, args[0], sig.args[0], sig.return_type)


for _fn, _name in _ROUNDING.items():
    _register_rounding(_fn, _name)


def _single(builder, value):
    """Convert a float64 value to float32 when compiling without float64.

    Bit manipulation must happen at the width the value will have on the
    device; see `numba_vulkan.narrowing`.
    """
    return narrowing.to_single(builder, value) if narrowing.current.floats else value


def double_words(builder, value):
    """The low and high 32-bit words of a ``double``.

    Working on words instead of a 64-bit integer keeps bit manipulation of
    ``float64`` values intact when a kernel computes with 32-bit integers
    (see `numba_vulkan.narrowing`).

    Returns
    -------
    low, high : llvmlite.ir.Value
        ``i32`` values.
    """
    i32 = ir.IntType(32)
    pair = builder.bitcast(value, ir.VectorType(i32, 2))
    return builder.extract_element(pair, i32(0)), builder.extract_element(pair, i32(1))


def words_double(builder, low, high):
    """The ``double`` with the given low and high words."""
    i32 = ir.IntType(32)
    pair = ir.Constant(ir.VectorType(i32, 2), None)
    pair = builder.insert_element(pair, low, i32(0))
    pair = builder.insert_element(pair, high, i32(1))
    return builder.bitcast(pair, ir.DoubleType())


def _classify(builder, value, kind):
    """Test a float for NaN, infinity or finiteness.

    Parameters
    ----------
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    value : llvmlite.ir.Value
        A float value.
    kind : {'nan', 'inf', 'finite'}
        The test to perform.

    Returns
    -------
    llvmlite.ir.Value
        The ``i1`` result.
    """
    # The tests work on the bit pattern. Float comparisons would be simpler,
    # but LLVM turns them into OpUnordered, which shaders may not use, and
    # drivers with fast-math enabled are free to fold them away.
    value = _single(builder, value)
    i32 = ir.IntType(32)
    if isinstance(value.type, ir.DoubleType):
        # Two 32-bit words, so that the test survives narrowing integers.
        low, high = double_words(builder, value)
        top = builder.and_(high, i32(0x7FFFFFFF))
        exponent = i32(0x7FF00000)
        fraction = builder.icmp_unsigned("!=", low, i32(0))
        at_top = builder.icmp_unsigned("==", top, exponent)
        if kind == "nan":
            above = builder.icmp_unsigned(">", top, exponent)
            return builder.or_(above, builder.and_(at_top, fraction))
        if kind == "inf":
            return builder.and_(at_top, builder.not_(fraction))
        return builder.icmp_unsigned("<", top, exponent)
    bits = i32
    exponent = bits(0xFF << 23)
    magnitude = builder.and_(builder.bitcast(value, bits), bits(0x7FFFFFFF))
    if kind == "nan":
        return builder.icmp_unsigned(">", magnitude, exponent)
    if kind == "inf":
        return builder.icmp_unsigned("==", magnitude, exponent)
    return builder.icmp_unsigned("<", magnitude, exponent)


def _register_classification(pyfn, kind):
    """Register the lowering of ``math.isnan``, ``isinf`` or ``isfinite``.

    Parameters
    ----------
    pyfn : callable
        The ``math`` function.
    kind : {'nan', 'inf', 'finite'}
        The test it performs.
    """

    @lower(pyfn, types.Float)
    def impl(context, builder, sig, args):
        """Classify a float; see `_classify`."""
        return _classify(builder, args[0], kind)

    @lower(pyfn, types.Integer)
    def impl_int(context, builder, sig, args):
        """Integers are always finite."""
        return ir.Constant(ir.IntType(1), int(kind == "finite"))


for _fn, _kind in ((math.isnan, "nan"), (math.isinf, "inf"), (math.isfinite, "finite")):
    _register_classification(_fn, _kind)


@lower(math.copysign, types.Float, types.Float)
def lower_copysign(context, builder, sig, args):
    """Lower ``math.copysign`` by combining bit patterns.

    The SPIR-V backend cannot select ``llvm.copysign``.

    Parameters
    ----------
    context : VulkanTargetContext
        The target context.
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    sig : numba.core.typing.Signature
        Signature the call was typed with.
    args : sequence of llvmlite.ir.Value
        Argument values, in the types of ``sig.args``.

    Returns
    -------
    llvmlite.ir.Value
        The magnitude of the first argument with the sign of the second.
    """
    ty = sig.return_type
    magnitude, sign = (context.cast(builder, a, t, ty) for a, t in zip(args, sig.args))
    wide = magnitude.type
    magnitude, sign = _single(builder, magnitude), _single(builder, sign)
    if isinstance(magnitude.type, ir.DoubleType):
        # Only the high words differ; see `double_words`.
        i32 = ir.IntType(32)
        low, high = double_words(builder, magnitude)
        sign_high = double_words(builder, sign)[1]
        high = builder.or_(
            builder.and_(high, i32(0x7FFFFFFF)),
            builder.and_(sign_high, i32(0x80000000)),
        )
        return words_double(builder, low, high)
    bits = ir.IntType(32)
    sign_bit = bits(1 << (bits.width - 1))
    combined = builder.or_(
        builder.and_(
            builder.bitcast(magnitude, bits), bits((1 << (bits.width - 1)) - 1)
        ),
        builder.and_(builder.bitcast(sign, bits), sign_bit),
    )
    result = builder.bitcast(combined, magnitude.type)
    return result if result.type == wide else narrowing.to_double(builder, result)


@lower(math.fmod, types.Float, types.Float)
def lower_fmod(context, builder, sig, args):
    """Lower ``math.fmod``: the remainder with the sign of the dividend.

    Parameters
    ----------
    context : VulkanTargetContext
        The target context.
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    sig : numba.core.typing.Signature
        Signature the call was typed with.
    args : sequence of llvmlite.ir.Value
        Argument values, in the types of ``sig.args``.

    Returns
    -------
    llvmlite.ir.Value
        The remainder, as a value of the return type.
    """
    vals = [
        context.cast(builder, a, t, sig.return_type) for a, t in zip(args, sig.args)
    ]
    return builder.frem(*vals)


def _power_by_squaring(builder, base, exponent, one, mul):
    """Raise a value to a non-negative integer power with multiplications.

    Parameters
    ----------
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted. On return it is
        positioned in a new block after the loop.
    base : llvmlite.ir.Value
        The base.
    exponent : llvmlite.ir.Value
        The exponent, an integer value treated as unsigned.
    one : llvmlite.ir.Constant
        The multiplicative identity in the type of ``base``.
    mul : callable
        ``builder.mul`` or ``builder.fmul``.

    Returns
    -------
    llvmlite.ir.Value
        ``base ** exponent``.
    """
    result = cgutils.alloca_once_value(builder, one)
    factor = cgutils.alloca_once_value(builder, base)
    remaining = cgutils.alloca_once_value(builder, exponent)
    cond = builder.append_basic_block("pow.cond")
    body = builder.append_basic_block("pow.body")
    done = builder.append_basic_block("pow.done")
    builder.branch(cond)
    builder.position_at_end(cond)
    left = builder.load(remaining)
    builder.cbranch(builder.icmp_unsigned("!=", left, left.type(0)), body, done)
    builder.position_at_end(body)
    current = builder.load(factor)
    odd = builder.trunc(left, ir.IntType(1))
    acc = builder.load(result)
    builder.store(builder.select(odd, mul(acc, current), acc), result)
    builder.store(mul(current, current), factor)
    builder.store(builder.lshr(left, left.type(1)), remaining)
    builder.branch(cond)
    builder.position_at_end(done)
    return builder.load(result)


# Numba's own integer-exponent power falls back to a float64 pow() call,
# which Vulkan lacks, so these are implemented with multiplications only.
def float_int_power(context, builder, sig, args):
    """Lower ``float ** int`` and ``pow(float, int)``.

    A negative exponent gives the reciprocal of the positive power.

    Parameters
    ----------
    context : VulkanTargetContext
        The target context.
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    sig : numba.core.typing.Signature
        Signature the call was typed with.
    args : sequence of llvmlite.ir.Value
        Argument values, in the types of ``sig.args``.

    Returns
    -------
    llvmlite.ir.Value
        The power, as a value of the return type.

    Notes
    -----
    Numba's implementation contains a float64 ``pow`` fallback for large
    exponents, which Vulkan lacks; this one only multiplies.
    """
    basety, expty = sig.args
    base, exponent = args
    one = base.type(1.0)
    if expty.signed:
        negative = builder.icmp_signed("<", exponent, exponent.type(0))
        exponent = builder.select(negative, builder.neg(exponent), exponent)
    res = _power_by_squaring(builder, base, exponent, one, builder.fmul)
    if expty.signed:
        res = builder.select(negative, builder.fdiv(one, res), res)
    return context.cast(builder, res, basety, sig.return_type)


def int_int_power(context, builder, sig, args):
    """Lower ``int ** int`` and ``pow(int, int)``.

    As in Numba, a negative exponent gives 0 unless the base is 1 or -1.
    Unlike Numba, ``0 ** negative`` gives 0 instead of raising.

    Parameters
    ----------
    context : VulkanTargetContext
        The target context.
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    sig : numba.core.typing.Signature
        Signature the call was typed with.
    args : sequence of llvmlite.ir.Value
        Argument values, in the types of ``sig.args``.

    Returns
    -------
    llvmlite.ir.Value
        The power, as a value of the return type.
    """
    tp = sig.return_type
    base = context.cast(builder, args[0], sig.args[0], tp)
    exponent = context.cast(builder, args[1], sig.args[1], tp)
    one = base.type(1)
    if not sig.args[1].signed:
        return _power_by_squaring(builder, base, exponent, one, builder.mul)
    # As in Numba, a negative exponent gives 0 unless the base is 1 or -1.
    negative = builder.icmp_signed("<", exponent, exponent.type(0))
    exponent = builder.select(negative, builder.neg(exponent), exponent)
    res = _power_by_squaring(builder, base, exponent, one, builder.mul)
    unit = builder.or_(
        builder.icmp_signed("==", base, one),
        builder.icmp_signed("==", base, base.type(-1)),
    )
    truncated = builder.and_(negative, builder.not_(unit))
    return builder.select(truncated, base.type(0), res)


def power_override(fn, sig):
    """Select this package's implementation of an integer-exponent power.

    Numba registers its implementations per concrete type, which entries
    in a lowering registry cannot outrank, so
    `VulkanTargetContext.get_function` asks this function first.

    Parameters
    ----------
    fn : callable or numba.types.Function
        The function being resolved.
    sig : numba.core.typing.Signature
        Signature of the call.

    Returns
    -------
    callable or None
        `float_int_power` or `int_int_power` for powers with a
        non-literal integer exponent, otherwise ``None``.
    """
    if isinstance(fn, types.Function):
        fn = fn.typing_key
    if fn not in (operator.pow, operator.ipow, pow) or len(sig.args) != 2:
        return None
    base, exponent = sig.args
    if not isinstance(exponent, types.Integer) or isinstance(exponent, types.Literal):
        return None
    if isinstance(base, types.Float):
        return float_int_power
    if isinstance(base, types.Integer):
        return int_int_power
    return None
