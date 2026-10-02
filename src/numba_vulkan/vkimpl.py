"""Lowering of Vulkan-specific functions and of array element access."""

import operator

from llvmlite import ir
from numba.core import cgutils, types
from numba.core.imputils import Registry

from numba_vulkan import stubs
from numba_vulkan.buffers import i32, load_element, store_element
from numba_vulkan.errors import VulkanUnsupportedError
from numba_vulkan.vktypes import VulkanArray, VulkanDispatcherType

registry = Registry("vkimpl")
lower = registry.lower


@registry.lower_constant(VulkanDispatcherType)
def lower_dispatcher_constant(context, builder, ty, pyval):
    # Calls are resolved at compile time; the function has no runtime value.
    """Lower a Vulkan dispatcher used as a global or closure variable.

    Parameters
    ----------
    context : VulkanTargetContext
        The target context.
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    ty : VulkanDispatcherType
        Type of the constant.
    pyval : VulkanDispatcher
        The dispatcher.

    Returns
    -------
    llvmlite.ir.Value
        A dummy value; calls are resolved at compile time.
    """
    return context.get_dummy_value()


@lower(stubs.global_id, types.IntegerLiteral)
def lower_global_id(context, builder, sig, args):
    """Lower ``global_id(axis)`` to the GlobalInvocationId built-in.

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
        The ``int32`` invocation index along the literal axis.
    """
    fnty = ir.FunctionType(i32, [i32])
    fn = builder.module.declare_intrinsic("llvm.spv.thread.id", [i32], fnty)
    return builder.call(fn, [i32(sig.args[0].literal_value)])


def buffer_element_type(context, dtype):
    """LLVM type of one element of a buffer holding ``dtype`` values.

    Parameters
    ----------
    context : VulkanTargetContext
        The target context.
    dtype : numba.types.Type
        Numba element type.

    Returns
    -------
    llvmlite.ir.Type
        ``i32`` for booleans (SPIR-V has no storable bool), otherwise
        the data type of the integer or float.

    Raises
    ------
    VulkanUnsupportedError
        If ``dtype`` is not a boolean, integer or float.
    """
    if isinstance(dtype, types.Boolean):
        return i32
    if not isinstance(dtype, (types.Integer, types.Float)):
        raise VulkanUnsupportedError(f"arrays of {dtype} are not supported on Vulkan")
    return context.get_data_type(dtype)


def _linear_index(context, builder, aryty, ary, idxty, idx):
    """Compute the flat element index of an array access.

    Negative indices wrap around as in Python. Indices are checked against
    the shape only when bounds checking is enabled for the function.

    Parameters
    ----------
    context : VulkanTargetContext
        The target context.
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    aryty : VulkanArray
        Type of the array.
    ary : llvmlite.ir.Value
        The array value (its metadata structure).
    idxty : numba.types.Type
        Type of the index: an integer or a tuple of integers.
    idx : llvmlite.ir.Value
        The index value.

    Returns
    -------
    llvmlite.ir.Value
        The row-major (C order) index as an ``intp`` value.

    Raises
    ------
    VulkanUnsupportedError
        For non-C layouts, slices, and indices that do not address a
        single element.
    """
    if aryty.layout != "C":
        raise VulkanUnsupportedError("only C-contiguous arrays are supported on Vulkan")
    if isinstance(idxty, types.BaseTuple):
        indices = cgutils.unpack_tuple(builder, idx, count=len(idxty))
        index_types = list(idxty)
    else:
        indices, index_types = [idx], [idxty]
    if len(indices) != aryty.ndim or not all(
        isinstance(t, types.Integer) for t in index_types
    ):
        raise VulkanUnsupportedError(
            f"indexing a {aryty.ndim}-d array with {idxty} is not supported on Vulkan "
            "(only full integer indexing is)"
        )
    proxy = cgutils.create_struct_proxy(aryty)(context, builder, value=ary)
    shape = cgutils.unpack_tuple(builder, proxy.shape, count=aryty.ndim)
    linear = None
    for dim, (index, ty) in enumerate(zip(indices, index_types)):
        index = context.cast(builder, index, ty, types.intp)
        if ty.signed:
            # Python-style wraparound of negative indices.
            negative = builder.icmp_signed("<", index, index.type(0))
            index = builder.select(negative, builder.add(index, shape[dim]), index)
        if context.enable_boundscheck:
            # Unsigned, so that indices that are still negative fail as well.
            outside = builder.icmp_unsigned(">=", index, shape[dim])
            with builder.if_then(outside, likely=False):
                message = (
                    f"index is out of bounds for axis {dim} of a "
                    f"{aryty.ndim}-dimensional array"
                )
                context.call_conv.return_user_exc(builder, IndexError, (message,))
        linear = (
            index
            if linear is None
            else builder.add(builder.mul(linear, shape[dim]), index)
        )
    return linear


@lower(operator.getitem, VulkanArray, types.Integer)
@lower(operator.getitem, VulkanArray, types.BaseTuple)
@lower(operator.getitem, VulkanArray, types.SliceType)
def lower_getitem(context, builder, sig, args):
    """Lower ``array[index]``.

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
        The element, as a value of the array's element type.

    Raises
    ------
    VulkanUnsupportedError
        If the index does not address a single element.
    """
    aryty, idxty = sig.args
    index = _linear_index(context, builder, aryty, args[0], idxty, args[1])
    elem = buffer_element_type(context, aryty.dtype)
    val = load_element(builder, aryty.binding, elem, index)
    if isinstance(aryty.dtype, types.Boolean):
        val = builder.icmp_unsigned("!=", val, i32(0))
    return val


@lower(operator.setitem, VulkanArray, types.Integer, types.Any)
@lower(operator.setitem, VulkanArray, types.BaseTuple, types.Any)
@lower(operator.setitem, VulkanArray, types.SliceType, types.Any)
def lower_setitem(context, builder, sig, args):
    """Lower ``array[index] = value``.

    The value is cast to the element type of the array first.

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
        A dummy value.

    Raises
    ------
    VulkanUnsupportedError
        If the index does not address a single element.
    """
    aryty, idxty, valty = sig.args
    index = _linear_index(context, builder, aryty, args[0], idxty, args[1])
    val = context.cast(builder, args[2], valty, aryty.dtype)
    if isinstance(aryty.dtype, types.Boolean):
        val = builder.zext(val, i32)
    elem = buffer_element_type(context, aryty.dtype)
    store_element(builder, aryty.binding, elem, index, val)
    return context.get_dummy_value()
