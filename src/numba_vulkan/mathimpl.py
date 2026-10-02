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
    math.gamma: "tgamma",
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
    two (``gamma`` excepted, see KI-30), give the same results on every
    device, and exist in double precision. With ``fastmath``, float32 functions use the device's
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
        return libclc.call(builder, from_libclc, vals)
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
