"""Lowering of Vulkan-specific functions and of array element access."""

import operator

from llvmlite import ir
from numba.core import cgutils, types
from numba.core.extending import intrinsic
from numba.core.imputils import Registry, RefType, iternext_impl
from numba.cpython import slicing

from numba_vulkan import stubs
from numba_vulkan.buffers import i32, load_element, store_element
from numba_vulkan.errors import VulkanUnsupportedError
from numba_vulkan.vktypes import (
    VulkanArray,
    VulkanArrayIterator,
    VulkanDispatcherType,
)

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


class _Selection:
    """Result of applying an index to an array: an element or a view.

    Attributes
    ----------
    offset : llvmlite.ir.Value
        Position in the buffer of the element, or of the first element of
        the view, as an ``intp`` count of elements.
    shape, strides : list of llvmlite.ir.Value
        Extents and strides (in bytes) of the view; empty for an element.
    """

    def __init__(self, offset, shape, strides):
        self.offset = offset
        self.shape = shape
        self.strides = strides


def _unpack(context, builder, aryty, ary):
    """Take an array value apart.

    Returns
    -------
    offset : llvmlite.ir.Value
        Position of the first element in the buffer, in elements.
    shape : list of llvmlite.ir.Value
        Extents.
    strides : list of llvmlite.ir.Value
        Strides in bytes.
    steps : list of llvmlite.ir.Value
        Strides in elements.
    """
    proxy = cgutils.create_struct_proxy(aryty)(context, builder, value=ary)
    shape = cgutils.unpack_tuple(builder, proxy.shape, count=aryty.ndim)
    strides = cgutils.unpack_tuple(builder, proxy.strides, count=aryty.ndim)
    itemsize = context.get_abi_sizeof(buffer_element_type(context, aryty.dtype))
    if aryty.layout == "C":
        # Contiguous: the steps follow from the shape, which lets LLVM fold
        # them when the shape is known.
        steps, step = [], ir.Constant(proxy.offset.type, 1)
        for extent in reversed(shape):
            steps.insert(0, step)
            step = builder.mul(step, extent)
    else:
        steps = [
            builder.sdiv(stride, stride.type(itemsize), flags=["exact"])
            for stride in strides
        ]
    return proxy.offset, shape, strides, steps


def _check_bounds(context, builder, index, extent, dim, ndim):
    """Raise ``IndexError`` unless ``0 <= index < extent``, if enabled."""
    if not context.enable_boundscheck:
        return
    # Unsigned, so that indices that are still negative fail as well.
    outside = builder.icmp_unsigned(">=", index, extent)
    with builder.if_then(outside, likely=False):
        message = f"index is out of bounds for axis {dim} of a {ndim}-dimensional array"
        context.call_conv.return_user_exc(builder, IndexError, (message,))


def _select(context, builder, aryty, ary, idxty, idx):
    """Apply an index made of integers, slices and ``...`` to an array.

    Negative integers wrap around as in Python. Integers are checked
    against the shape only when bounds checking is enabled for the
    function; slices are clipped to the array, as in Python.

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
        Type of the index: an integer, a slice, an ellipsis or a tuple of
        those.
    idx : llvmlite.ir.Value
        The index value.

    Returns
    -------
    _Selection
        The element or view that the index selects. Axes without an index
        are taken whole.

    Raises
    ------
    VulkanUnsupportedError
        For indices other than integers and slices, and for more indices
        than the array has axes.
    """
    if isinstance(idxty, types.BaseTuple):
        indices = cgutils.unpack_tuple(builder, idx, count=len(idxty))
        index_types = list(idxty)
    else:
        indices, index_types = [idx], [idxty]
    ellipses = [
        k for k, t in enumerate(index_types) if isinstance(t, types.EllipsisType)
    ]
    if (
        len(index_types) - len(ellipses) > aryty.ndim
        or len(ellipses) > 1
        or not all(
            isinstance(t, (types.Integer, types.SliceType, types.EllipsisType))
            for t in index_types
        )
    ):
        raise VulkanUnsupportedError(
            f"indexing a {aryty.ndim}-d array with {idxty} is not supported on Vulkan "
            "(only integers, slices and ... are)"
        )
    if ellipses:
        # An ellipsis stands for as many whole axes as are left over.
        (at,) = ellipses
        missing = aryty.ndim - (len(indices) - 1)
        indices[at : at + 1] = [None] * missing
        index_types[at : at + 1] = [None] * missing
    offset, shape, strides, steps = _unpack(context, builder, aryty, ary)
    out_shape, out_strides = [], []
    for dim, (index, ty) in enumerate(zip(indices, index_types)):
        if ty is None:
            out_shape.append(shape[dim])
            out_strides.append(strides[dim])
        elif isinstance(ty, types.Integer):
            index = context.cast(builder, index, ty, types.intp)
            if ty.signed:
                # Python-style wraparound of negative indices.
                negative = builder.icmp_signed("<", index, index.type(0))
                index = builder.select(negative, builder.add(index, shape[dim]), index)
            _check_bounds(context, builder, index, shape[dim], dim, aryty.ndim)
            offset = builder.add(offset, builder.mul(index, steps[dim]))
        else:
            piece = context.make_helper(builder, ty, index)
            slicing.guard_invalid_slice(context, builder, ty, piece)
            slicing.fix_slice(builder, piece, shape[dim])
            offset = builder.add(offset, builder.mul(piece.start, steps[dim]))
            out_shape.append(slicing.get_slice_length(builder, piece))
            out_strides.append(slicing.fix_stride(builder, piece, strides[dim]))
    # Axes without an index are taken whole.
    out_shape += shape[len(indices) :]
    out_strides += strides[len(indices) :]
    return _Selection(offset, out_shape, out_strides)


def _make_view(context, builder, viewty, selection):
    """Build the array value of a view.

    Parameters
    ----------
    context : VulkanTargetContext
        The target context.
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    viewty : VulkanArray
        Type of the view.
    selection : _Selection
        Where the view lies in the buffer.

    Returns
    -------
    llvmlite.ir.Value
    """
    intp = context.get_value_type(types.intp)
    proxy = cgutils.create_struct_proxy(viewty)(context, builder)
    nitems = intp(1)
    for extent in selection.shape:
        nitems = builder.mul(nitems, extent)
    proxy.nitems = nitems
    proxy.itemsize = intp(
        context.get_abi_sizeof(buffer_element_type(context, viewty.dtype))
    )
    proxy.shape = cgutils.pack_array(builder, selection.shape, ty=intp)
    proxy.strides = cgutils.pack_array(builder, selection.strides, ty=intp)
    proxy.offset = selection.offset
    return proxy._getvalue()


def _load(context, builder, aryty, position):
    """Read the element at a position of an array's buffer."""
    elem = buffer_element_type(context, aryty.dtype)
    val = load_element(builder, aryty.binding, elem, position)
    if isinstance(aryty.dtype, types.Boolean):
        val = builder.icmp_unsigned("!=", val, i32(0))
    return val


def _store(context, builder, aryty, position, val, valty):
    """Write a value, cast to the element type, at a position of a buffer."""
    val = context.cast(builder, val, valty, aryty.dtype)
    if isinstance(aryty.dtype, types.Boolean):
        val = builder.zext(val, i32)
    elem = buffer_element_type(context, aryty.dtype)
    store_element(builder, aryty.binding, elem, position, val)


def _position(context, builder, aryty, ary, indices):
    """Buffer position of the element at the given per-axis indices."""
    offset, _, _, steps = _unpack(context, builder, aryty, ary)
    for index, step in zip(indices, steps):
        offset = builder.add(offset, builder.mul(index, step))
    return offset


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
        The element, or a view of the same buffer if the index contains
        slices or covers only the leading axes.

    Raises
    ------
    VulkanUnsupportedError
        If the index contains anything but integers and slices.
    """
    aryty, idxty = sig.args
    selection = _select(context, builder, aryty, args[0], idxty, args[1])
    if isinstance(sig.return_type, types.Array):
        return _make_view(context, builder, sig.return_type, selection)
    return _load(context, builder, aryty, selection.offset)


@lower(operator.getitem, VulkanArray, types.EllipsisType)
def lower_getitem_ellipsis(context, builder, sig, args):
    """Lower ``array[...]``; see `lower_getitem`."""
    return lower_getitem(context, builder, sig, args)


@registry.lower_getattr(VulkanArray, "T")
def lower_transpose(context, builder, ty, value):
    """Lower ``array.T``: a view with the axes reversed.

    Parameters
    ----------
    context : VulkanTargetContext
        The target context.
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    ty : VulkanArray
        Type of the array.
    value : llvmlite.ir.Value
        The array.

    Returns
    -------
    llvmlite.ir.Value
        The transposed view; the array itself for fewer than two axes.
    """
    if ty.ndim < 2:
        return value
    offset, shape, strides, _ = _unpack(context, builder, ty, value)
    viewty = context.typing_context.resolve_getattr(ty, "T")
    selection = _Selection(offset, shape[::-1], strides[::-1])
    return _make_view(context, builder, viewty, selection)


@lower(operator.setitem, VulkanArray, types.EllipsisType, types.Any)
def lower_setitem_ellipsis(context, builder, sig, args):
    """Lower ``array[...] = value``; see `lower_setitem`."""
    return lower_setitem(context, builder, sig, args)


@lower(operator.setitem, VulkanArray, types.Integer, types.Any)
@lower(operator.setitem, VulkanArray, types.BaseTuple, types.Any)
@lower(operator.setitem, VulkanArray, types.SliceType, types.Any)
def lower_setitem(context, builder, sig, args):
    """Lower ``array[index] = value``.

    If the index selects one element, the value is cast to the element
    type and stored. If it selects a view, the view is filled with a
    scalar, or with the elements of an array of the same shape.

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
        If the index contains anything but integers and slices, if the
        value is an array of another dimensionality, or if it is an array
        in the same buffer as the target (the copy could overlap).
    """
    aryty, idxty, valty = sig.args
    selection = _select(context, builder, aryty, args[0], idxty, args[1])
    if not selection.shape:
        _store(context, builder, aryty, selection.offset, args[2], valty)
        return context.get_dummy_value()

    ndim = len(selection.shape)
    viewty = aryty.copy(ndim=ndim, layout="A")
    view = _make_view(context, builder, viewty, selection)
    if isinstance(valty, types.Array):
        if not isinstance(valty, VulkanArray) or valty.ndim != ndim:
            raise VulkanUnsupportedError(
                f"assigning {valty} to a {ndim}-d slice is not supported on Vulkan "
                "(only scalars and arrays of the same dimensionality are)"
            )
        if valty.binding == aryty.binding:
            raise VulkanUnsupportedError(
                "copying between slices of the same array is not supported on "
                "Vulkan, because the slices could overlap"
            )
        source_shape = _unpack(context, builder, valty, args[2])[1]
        for extent, wanted in zip(source_shape, selection.shape):
            with builder.if_then(
                builder.icmp_signed("!=", extent, wanted), likely=False
            ):
                context.call_conv.return_user_exc(
                    builder,
                    ValueError,
                    ("cannot assign slice from input of different size",),
                )
    intp = context.get_value_type(types.intp)
    with cgutils.loop_nest(builder, selection.shape, intp) as indices:
        if isinstance(valty, types.Array):
            source = _position(context, builder, valty, args[2], indices)
            val, itemty = _load(context, builder, valty, source), valty.dtype
        else:
            val, itemty = args[2], valty
        target = _position(context, builder, viewty, view, indices)
        _store(context, builder, aryty, target, val, itemty)
    return context.get_dummy_value()


@lower("getiter", VulkanArray)
def lower_getiter(context, builder, sig, args):
    """Lower ``iter(array)``: an iterator over the first axis.

    Parameters
    ----------
    context : VulkanTargetContext
        The target context.
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    sig : numba.core.typing.Signature
        Signature the call was typed with.
    args : sequence of llvmlite.ir.Value
        The array.

    Returns
    -------
    llvmlite.ir.Value
        The iterator.
    """
    iterator = context.make_helper(builder, sig.return_type)
    iterator.index = cgutils.alloca_once_value(
        builder, context.get_constant(types.intp, 0)
    )
    iterator.array = args[0]
    return iterator._getvalue()


@lower("iternext", VulkanArrayIterator)
@iternext_impl(RefType.BORROWED)
def lower_iternext(context, builder, sig, args, result):
    """Lower ``next(iterator)`` for an array iterator.

    Yields elements of a 1-d array and views along the first axis
    otherwise.

    Parameters
    ----------
    context : VulkanTargetContext
        The target context.
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    sig : numba.core.typing.Signature
        Signature the call was typed with.
    args : sequence of llvmlite.ir.Value
        The iterator.
    result : numba.core.imputils._IternextResult
        Receives the validity flag and the yielded value.
    """
    (iterty,) = sig.args
    aryty = iterty.array_type
    iterator = context.make_helper(builder, iterty, value=args[0])
    index = builder.load(iterator.index)
    offset, shape, strides, steps = _unpack(context, builder, aryty, iterator.array)
    valid = builder.icmp_signed("<", index, shape[0])
    result.set_valid(valid)
    with builder.if_then(valid):
        position = builder.add(offset, builder.mul(index, steps[0]))
        if aryty.ndim == 1:
            result.yield_(_load(context, builder, aryty, position))
        else:
            selection = _Selection(position, shape[1:], strides[1:])
            result.yield_(_make_view(context, builder, iterty.yield_type, selection))
        builder.store(builder.add(index, index.type(1)), iterator.index)


@intrinsic(target="vulkan")
def flat_item(typingctx, array, position):
    """Element of an array by its position in row-major order.

    Parameters
    ----------
    array : VulkanArray
        The array, of any dimensionality.
    position : int
        Position between 0 and ``array.size - 1``; it is not checked.

    Returns
    -------
    scalar
        The element.
    """
    if not isinstance(array, VulkanArray) or not isinstance(position, types.Integer):
        return None

    def codegen(context, builder, sig, args):
        """Emit the element access."""
        aryty, posty = sig.args
        k = context.cast(builder, args[1], posty, types.intp)
        offset, shape, _, steps = _unpack(context, builder, aryty, args[0])
        if aryty.layout == "C":
            return _load(context, builder, aryty, builder.add(offset, k))
        # A view: split the position into one index per axis, last first.
        for extent, step in zip(reversed(shape), reversed(steps)):
            index = builder.urem(k, extent)
            k = builder.udiv(k, extent)
            offset = builder.add(offset, builder.mul(index, step))
        return _load(context, builder, aryty, offset)

    return array.dtype(array, position), codegen
