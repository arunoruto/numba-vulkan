"""Lowering of the ``math`` module onto GLSL.std.450 via LLVM intrinsics."""

import math
import operator

from llvmlite import ir
from numba.core import cgutils, types
from numba.core.imputils import Registry

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
}
# ... and these for doubles as well.
_ANY_FLOAT = {
    math.sqrt: "llvm.sqrt",
    math.fabs: "llvm.fabs",
}
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


def _float_math(context, builder, pyfn, name, f32_only, sig, args):
    """Lower a ``math`` function to an LLVM intrinsic.

    Integer arguments are converted to float64 first, as Numba's typing
    prescribes.

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
        For a float64 call of a 32-bit-only function without
        ``narrow_math``.
    """
    ty = sig.return_type if isinstance(sig.return_type, types.Float) else types.float64
    vals = [context.cast(builder, a, t, ty) for a, t in zip(args, sig.args)]
    if f32_only and ty == types.float64:
        if not context.narrow_math:
            raise VulkanUnsupportedError(
                f"math.{pyfn.__name__} on float64 is not available in Vulkan shaders "
                "(the GLSL.std.450 math library is 32-bit only); use float32 values "
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
    nargs = 2 if pyfn in (math.pow, math.atan2) else 1

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
