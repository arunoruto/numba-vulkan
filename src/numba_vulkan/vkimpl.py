"""Lowering of Vulkan-specific functions and of array element access."""

import itertools
import math
import operator
import zlib

import numpy as np
from llvmlite import ir
from numba.core import cgutils, types
from numba.core.extending import intrinsic
from numba.core.imputils import RefType, Registry, iternext_impl
from numba.cpython import slicing

from numba_vulkan import narrowing, stubs, vkdecl
from numba_vulkan.buffers import (
    PRINT_BINDING,
    atomic_element,
    barrier,
    compare_and_swap,
    i32,
    load_element,
    print_formats,
    shared_sizes,
    store_element,
)
from numba_vulkan.errors import VulkanUnsupportedError
from numba_vulkan.mathimpl import double_words
from numba_vulkan.vktypes import (
    VulkanArray,
    VulkanArrayIterator,
    VulkanDispatcherType,
    VulkanExpr,
    VulkanRecord,
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
    binding : int or llvmlite.ir.Value
        Binding of the buffer; see `_binding`.
    """

    def __init__(self, offset, shape, strides, binding):
        self.offset = offset
        self.shape = shape
        self.strides = strides
        self.binding = binding


def _binding(context, builder, aryty, ary):
    """The binding of an array's buffer.

    Returns
    -------
    int or llvmlite.ir.Value
        The number in the type, or, for arrays that carry it as a value
        (kernel arguments), that value; inlining makes it a constant.
    """
    if aryty.binding is not None:
        return aryty.binding
    return cgutils.create_struct_proxy(aryty)(context, builder, value=ary).binding


def _same_buffer(context, builder, first, second):
    """Whether two arrays live in the same buffer.

    Parameters
    ----------
    first, second : tuple
        Type and value of each array.

    Returns
    -------
    bool or llvmlite.ir.Value
        A Python bool where the types decide it, otherwise an ``i1`` that
        is constant after inlining.
    """
    (ty1, v1), (ty2, v2) = first, second
    if ty1.binding is not None and ty2.binding is not None:
        return ty1.binding == ty2.binding
    a = _binding_value(context, builder, ty1, v1)
    b = _binding_value(context, builder, ty2, v2)
    return builder.icmp_unsigned("==", a, b)


def _binding_value(context, builder, aryty, ary):
    """The binding of an array's buffer as an ``i32`` value."""
    binding = _binding(context, builder, aryty, ary)
    return i32(binding) if isinstance(binding, int) else binding


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
    itemsize = context.get_abi_sizeof(storage_type(context, aryty))
    # Positions are computed in 32 bits: buffers are indexed with 32-bit
    # integers anyway, and 64-bit arithmetic is slow on GPUs.
    if aryty.layout == "C":
        # Contiguous: the steps follow from the shape, which lets LLVM fold
        # them when the shape is known.
        steps, step = [], i32(1)
        for extent in reversed(shape):
            steps.insert(0, step)
            step = builder.mul(step, _i32(builder, extent))
    else:
        steps = [
            _i32(builder, builder.sdiv(stride, stride.type(itemsize), flags=["exact"]))
            for stride in strides
        ]
    return _i32(builder, proxy.offset), shape, strides, steps


def _i32(builder, value):
    """An integer as ``i32``; positions in buffers need no more."""
    return value if value.type == i32 else builder.trunc(value, i32)


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
            offset = builder.add(offset, builder.mul(_i32(builder, index), steps[dim]))
        else:
            piece = context.make_helper(builder, ty, index)
            slicing.guard_invalid_slice(context, builder, ty, piece)
            slicing.fix_slice(builder, piece, shape[dim])
            offset = builder.add(
                offset, builder.mul(_i32(builder, piece.start), steps[dim])
            )
            out_shape.append(slicing.get_slice_length(builder, piece))
            out_strides.append(slicing.fix_stride(builder, piece, strides[dim]))
    # Axes without an index are taken whole.
    out_shape += shape[len(indices) :]
    out_strides += strides[len(indices) :]
    binding = _binding(context, builder, aryty, ary)
    return _Selection(offset, out_shape, out_strides, binding)


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
    proxy.itemsize = intp(context.get_abi_sizeof(storage_type(context, viewty)))
    proxy.shape = cgutils.pack_array(builder, selection.shape, ty=intp)
    proxy.strides = cgutils.pack_array(builder, selection.strides, ty=intp)
    proxy.offset = builder.sext(selection.offset, intp)
    proxy.binding = _binding_member(viewty, selection.binding)
    return proxy._getvalue()


def _binding_member(aryty, binding):
    """The value of the ``binding`` member of an array of type `aryty`.

    Only arrays whose type has no binding read it. The others store 0, so
    that the per-process numbers of constant arrays stay out of the IR that
    the kernel cache hashes (see `numba_vulkan.kernelcache`).
    """
    if aryty.binding is not None:
        return i32(0)
    return i32(binding) if isinstance(binding, int) else binding


def half_conversion(builder, value, target):
    """Convert between ``half`` and ``float`` behind LLVM's back.

    LLVM would fold ``fptrunc(fpext(x) * 2)`` into a ``half`` multiply,
    which needs the ``shaderFloat16`` device feature. The conversions are
    therefore placeholder calls until after optimisation (see
    `numba_vulkan.buffers.expand_buffer_access`).

    Parameters
    ----------
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    value : llvmlite.ir.Value
        A ``half`` or ``float`` value.
    target : llvmlite.ir.Type
        The other of the two types.

    Returns
    -------
    llvmlite.ir.Value
    """
    name = (
        "numba_vulkan.tohalf"
        if isinstance(target, ir.HalfType)
        else "numba_vulkan.fromhalf"
    )
    fn = builder.module.globals.get(name)
    if fn is None:
        fn = ir.Function(builder.module, ir.FunctionType(target, [value.type]), name)
        fn.attributes.add("readnone")
        fn.attributes.add("nounwind")
    return builder.call(fn, [value])


def storage_type(context, aryty):
    """LLVM type of one element of an array's buffer.

    Parameters
    ----------
    context : VulkanTargetContext
        The target context.
    aryty : VulkanArray
        The array type.

    Returns
    -------
    llvmlite.ir.Type
        ``half`` for arrays of ``float16`` values, otherwise see
        `buffer_element_type`.
    """
    if getattr(aryty, "half", False):
        return ir.HalfType()
    if isinstance(aryty.dtype, VulkanRecord):
        # Only the size matters: records are read as 32-bit words.
        return ir.ArrayType(ir.IntType(8), aryty.dtype.size)
    return buffer_element_type(context, aryty.dtype)


def _load(context, builder, aryty, position, binding):
    """Read the element at a position of an array's buffer.

    For a record array, the element is a record (see `VulkanRecord`).
    """
    if isinstance(aryty.dtype, VulkanRecord):
        proxy = cgutils.create_struct_proxy(aryty.dtype)(context, builder)
        proxy.binding = i32(binding) if isinstance(binding, int) else binding
        proxy.position = builder.mul(position, i32(aryty.dtype.size))
        return proxy._getvalue()
    elem = storage_type(context, aryty)
    val = load_element(builder, binding, elem, position)
    if isinstance(aryty.dtype, types.Boolean):
        val = builder.icmp_unsigned("!=", val, i32(0))
    elif isinstance(elem, ir.HalfType):
        val = half_conversion(builder, val, ir.FloatType())
    return val


def _store(context, builder, aryty, position, val, valty, binding):
    """Write a value, cast to the element type, at a position of a buffer.

    For a record array, the value is a record, copied field by field.
    """
    if isinstance(aryty.dtype, VulkanRecord):
        target = _load(context, builder, aryty, position, binding)
        for name in aryty.dtype.record.fields:
            field = _read_field(context, builder, valty, val, name)
            _write_field(context, builder, aryty.dtype, target, name, field)
        return
    val = context.cast(builder, val, valty, aryty.dtype)
    elem = storage_type(context, aryty)
    if isinstance(aryty.dtype, types.Boolean):
        val = builder.zext(val, i32)
    elif isinstance(elem, ir.HalfType):
        val = half_conversion(builder, val, elem)
    store_element(builder, binding, elem, position, val)


def _position(context, builder, aryty, ary, indices):
    """Buffer position of the element at the given per-axis indices."""
    offset, _, _, steps = _unpack(context, builder, aryty, ary)
    for index, step in zip(indices, steps):
        offset = builder.add(offset, builder.mul(_i32(builder, index), step))
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
    return _load(context, builder, aryty, selection.offset, selection.binding)


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
    binding = _binding(context, builder, ty, value)
    selection = _Selection(offset, shape[::-1], strides[::-1], binding)
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
        _store(
            context, builder, aryty, selection.offset, args[2], valty, selection.binding
        )
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
        shared = _same_buffer(context, builder, (valty, args[2]), (aryty, args[0]))
        if shared is not False:
            # Allowed only as a no-op: ``a[i] += x`` assigns a view to itself.
            offset, shape, _, steps = _unpack(context, builder, valty, args[2])
            _, _, _, view_steps = _unpack(context, builder, viewty, view)
            same = builder.icmp_signed("==", offset, selection.offset)
            for a, b in zip(shape + steps, selection.shape + view_steps):
                same = builder.and_(same, builder.icmp_signed("==", a, b))
            overlap = builder.not_(same)
            if shared is not True:
                overlap = builder.and_(shared, overlap)
            with builder.if_then(overlap, likely=False):
                context.call_conv.return_user_exc(
                    builder,
                    ValueError,
                    ("copying between slices of the same array, which could overlap",),
                )
            if shared is True:
                return context.get_dummy_value()
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
    if isinstance(valty, types.Array):
        source_binding = _binding(context, builder, valty, args[2])
    with cgutils.loop_nest(builder, selection.shape, intp) as indices:
        if isinstance(valty, types.Array):
            source = _position(context, builder, valty, args[2], indices)
            val = _load(context, builder, valty, source, source_binding)
            itemty = valty.dtype
        else:
            val, itemty = args[2], valty
        target = _position(context, builder, viewty, view, indices)
        _store(context, builder, aryty, target, val, itemty, selection.binding)
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
    binding = _binding(context, builder, aryty, iterator.array)
    valid = builder.icmp_signed("<", index, shape[0])
    result.set_valid(valid)
    with builder.if_then(valid):
        position = builder.add(offset, builder.mul(_i32(builder, index), steps[0]))
        if aryty.ndim == 1:
            result.yield_(_load(context, builder, aryty, position, binding))
        else:
            selection = _Selection(position, shape[1:], strides[1:], binding)
            result.yield_(_make_view(context, builder, iterty.yield_type, selection))
        builder.store(builder.add(index, index.type(1)), iterator.index)


@intrinsic(target="vulkan")
def flat_item(typingctx, array, position):
    """Element of an array by its position in row-major order.

    Parameters
    ----------
    array : VulkanArray or VulkanExpr
        The array or array expression, of any dimensionality.
    position : int
        Position between 0 and ``array.size - 1``; it is not checked.

    Returns
    -------
    scalar
        The element.
    """
    if not isinstance(array, (VulkanArray, VulkanExpr)) or not isinstance(
        position, types.Integer
    ):
        return None

    def codegen(context, builder, sig, args):
        """Emit the element access."""
        aryty, posty = sig.args
        k = context.cast(builder, args[1], posty, types.intp)
        if isinstance(aryty, VulkanExpr):
            shape = expression_shape(context, builder, aryty, args[0])
            indices = []
            for extent in reversed(shape):
                indices.insert(0, builder.urem(k, extent))
                k = builder.udiv(k, extent)
            return expression_element(context, builder, aryty, args[0], indices)
        offset, shape, _, steps = _unpack(context, builder, aryty, args[0])
        binding = _binding(context, builder, aryty, args[0])
        if aryty.layout == "C":
            position = builder.add(offset, _i32(builder, k))
            return _load(context, builder, aryty, position, binding)
        # A view: split the position into one index per axis, last first.
        for extent, step in zip(reversed(shape), reversed(steps)):
            index = builder.urem(k, extent)
            k = builder.udiv(k, extent)
            offset = builder.add(offset, builder.mul(_i32(builder, index), step))
        return _load(context, builder, aryty, offset, binding)

    return array.dtype(array, position), codegen


def _register_axis(stub, intrinsic):
    """Lower a function of a literal axis to a SPIR-V built-in variable."""

    @lower(stub, types.IntegerLiteral)
    def lower_axis(context, builder, sig, args):
        fnty = ir.FunctionType(i32, [i32])
        fn = builder.module.declare_intrinsic(intrinsic, [i32], fnty)
        return builder.call(fn, [i32(sig.args[0].literal_value)])

    lower_axis.__doc__ = f"Lower ``{stub.__name__}(axis)`` to ``{intrinsic}``."
    return lower_axis


_register_axis(stubs.local_id, "llvm.spv.thread.id.in.group")
_register_axis(stubs.group_id, "llvm.spv.group.id")


@lower(stubs.local_size, types.IntegerLiteral)
def lower_local_size(context, builder, sig, args):
    """Lower ``local_size(axis)``.

    The workgroup size is only fixed when the kernel is compiled, after its
    functions; a placeholder call stands for it until then (see
    `numba_vulkan.codegen.VulkanCodeLibrary`).

    Returns
    -------
    llvmlite.ir.Value
    """
    name = "numba_vulkan.local_size"
    fn = builder.module.globals.get(name)
    if fn is None:
        fn = ir.Function(builder.module, ir.FunctionType(i32, [i32]), name)
        fn.attributes.add("readnone")
        fn.attributes.add("nounwind")
    size = builder.call(fn, [i32(sig.args[0].literal_value)])
    # The size is only known at launch, so LLVM learns that it is positive:
    # signed arithmetic on it, such as ``local_size(0) // 2``, then needs
    # no sign checks.
    size.set_metadata("range", builder.module.add_metadata([i32(1), i32(65537)]))
    return size


@lower(stubs.num_groups, types.IntegerLiteral)
def lower_num_groups(context, builder, sig, args):
    """Lower ``num_groups(axis)``.

    Grids larger than the device allows are dispatched in parts, for which
    the ``NumWorkgroups`` built-in gives the size of the part. The size of
    the whole grid is passed as push constants instead; a placeholder call
    stands for it until the push-constant block is laid out (see
    `numba_vulkan.codegen.num_groups_members`).

    Returns
    -------
    llvmlite.ir.Value
    """
    name = "numba_vulkan.num_groups"
    fn = builder.module.globals.get(name)
    if fn is None:
        fn = ir.Function(builder.module, ir.FunctionType(i32, [i32]), name)
        fn.attributes.add("readnone")
        fn.attributes.add("nounwind")
    count = builder.call(fn, [i32(sig.args[0].literal_value)])
    count.set_metadata("range", builder.module.add_metadata([i32(1), i32(1 << 31)]))
    return count


@lower(stubs.barrier)
def lower_barrier(context, builder, sig, args):
    """Lower ``barrier()``; see `numba_vulkan.buffers.barrier`.

    Returns
    -------
    llvmlite.ir.Value
        A dummy value.
    """
    barrier(builder)
    return context.get_dummy_value()


def static_array(context, builder, aryty, shape):
    """The value of an array whose shape is known at compile time.

    Parameters
    ----------
    context : VulkanTargetContext
        The target context.
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    aryty : VulkanArray
        Type of the array.
    shape : tuple of int
        Its shape.

    Returns
    -------
    llvmlite.ir.Value
        The metadata structure, with C-contiguous strides and offset 0.
    """
    intp = context.get_value_type(types.intp)
    itemsize = context.get_abi_sizeof(storage_type(context, aryty))
    strides, step = [], itemsize
    for extent in reversed(shape):
        strides.insert(0, step)
        step *= extent
    proxy = cgutils.create_struct_proxy(aryty)(context, builder)
    proxy.nitems = intp(math.prod(shape))
    proxy.itemsize = intp(itemsize)
    proxy.shape = cgutils.pack_array(builder, [intp(n) for n in shape], ty=intp)
    proxy.strides = cgutils.pack_array(builder, [intp(n) for n in strides], ty=intp)
    proxy.offset = intp(0)
    proxy.binding = _binding_member(aryty, aryty.binding)
    return proxy._getvalue()


@lower(stubs.shared.array, types.IntegerLiteral, types.Any, types.Any)
def lower_shared_array(context, builder, sig, args):
    """Lower ``shared.array(shape, dtype)``.

    The memory is a workgroup variable named after the array's binding,
    declared when buffer accesses are expanded; here only the metadata of
    the array is built.

    Returns
    -------
    llvmlite.ir.Value
    """
    shape = sig.args[1]
    dims = (
        (shape.literal_value,)
        if isinstance(shape, types.IntegerLiteral)
        else tuple(s.literal_value for s in shape)
    )
    return static_array(context, builder, sig.return_type, dims)


@lower(stubs.local.array, types.IntegerLiteral, types.Any, types.Any)
def lower_local_array(context, builder, sig, args):
    """Lower ``local.array(shape, dtype)``; see `lower_shared_array`.

    Returns
    -------
    llvmlite.ir.Value
    """
    return lower_shared_array(context, builder, sig, args)


def _register_constructor(function, fill):
    """Lower ``np.empty`` and its relatives inside kernels."""

    def lower_constructor(context, builder, sig, args):
        aryty = sig.return_type
        array = lower_shared_array(context, builder, sig, args)
        if function is np.empty:
            return array
        if fill:
            value = context.cast(builder, args[2], sig.args[2], aryty.dtype)
        else:
            value = context.get_constant(aryty.dtype, 1 if function is np.ones else 0)
        count = context.get_constant(types.intp, shared_sizes[aryty.binding])
        with cgutils.for_range(builder, count) as loop:
            position = _i32(builder, loop.index)
            _store(context, builder, aryty, position, value, aryty.dtype, aryty.binding)
        return array

    lower_constructor.__doc__ = f"Lower ``np.{function.__name__}`` in kernels."
    count = 2 if fill else 1
    for extra in range(2):  # with and without dtype
        types_ = [types.IntegerLiteral] + [types.Any] * (count + extra)
        lower(function, *types_)(lower_constructor)


for _function in (np.empty, np.zeros, np.ones):
    _register_constructor(_function, fill=False)
_register_constructor(np.full, fill=True)


def _atomic_position(context, builder, aryty, ary, idxty, idx):
    """Binding and buffer position of the element an atomic applies to."""
    selection = _select(context, builder, aryty, ary, idxty, idx)
    return selection.binding, selection.offset


def _float_atomic(context, builder, binding, position, value, update):
    """Apply a float read-modify-write with a compare-and-swap loop.

    Parameters
    ----------
    update : callable
        Builds the new value from the current one and `value`; it may
        return ``None`` to signal that the element stays as it is.

    Returns
    -------
    llvmlite.ir.Value
        The element's previous value.
    """
    single = ir.FloatType()
    current = load_element(builder, binding, single, position)
    first = builder.bitcast(current, i32)
    start = builder.basic_block
    loop = builder.append_basic_block("atomic.loop")
    swap = builder.append_basic_block("atomic.swap")
    done = builder.append_basic_block("atomic.done")
    builder.branch(loop)
    builder.position_at_end(loop)
    bits = builder.phi(i32)
    bits.add_incoming(first, start)
    old = builder.bitcast(bits, single)
    new = builder.bitcast(update(old, value), i32)
    # Nothing to write: the element already has the result (max, min).
    builder.cbranch(builder.icmp_unsigned("==", new, bits), done, swap)
    builder.position_at_end(swap)
    seen = compare_and_swap(builder, binding, position, bits, new)
    bits.add_incoming(seen, swap)
    builder.cbranch(builder.icmp_unsigned("==", seen, bits), done, loop)
    builder.position_at_end(done)
    return old


_FLOAT_UPDATES = {
    "add": lambda b, old, v: b.fadd(old, v),
    "sub": lambda b, old, v: b.fsub(old, v),
    # Python's max and min: a NaN operand leaves the element unchanged.
    "max": lambda b, old, v: b.select(b.fcmp_ordered(">", v, old), v, old),
    "min": lambda b, old, v: b.select(b.fcmp_ordered("<", v, old), v, old),
}
_INTEGER_OPS = {
    "add": "add",
    "sub": "sub",
    "and_": "and",
    "or_": "or",
    "xor": "xor",
    "exch": "xchg",
    "max": "max",
    "min": "min",
}


def _register_atomic(stub):
    """Lower one atomic operation."""
    name = stub.__name__

    def lower_atomic(context, builder, sig, args):
        aryty, idxty, valty = sig.args
        dtype = aryty.dtype
        binding, position = _atomic_position(
            context, builder, aryty, args[0], idxty, args[1]
        )
        value = context.cast(builder, args[2], valty, dtype)
        if isinstance(dtype, types.Integer):
            if dtype.bitwidth == 64 and not narrowing.current.ints:
                raise VulkanUnsupportedError(
                    f"atomic.{name} on {dtype} is not supported: LLVM's SPIR-V "
                    "backend does not offer 64-bit integer atomics for Vulkan. Use "
                    "an int32 or uint32 array (or narrow=True)"
                )
            op = _INTEGER_OPS[name]
            if op in ("max", "min") and not dtype.signed:
                op = "u" + op
            elem = buffer_element_type(context, dtype)
            return atomic_element(builder, binding, elem, position, op, value)
        if name == "exch":
            if dtype == types.float64 and not narrowing.current.floats:
                raise VulkanUnsupportedError("atomic.exch on float64 is not supported")
            single = narrowing.to_single(builder, value)
            old = atomic_element(
                builder,
                binding,
                i32,
                position,
                "xchg",
                builder.bitcast(single, i32),
            )
            old = builder.bitcast(old, ir.FloatType())
            return narrowing.to_double(builder, old) if dtype == types.float64 else old
        if dtype == types.float64 and not narrowing.current.floats:
            raise VulkanUnsupportedError(
                f"atomic.{name} on float64 needs 64-bit compare-and-swap, which "
                "LLVM's SPIR-V backend cannot emit yet; use float32, or "
                "narrow=True"
            )
        if name in ("add", "sub") and narrowing.current.float_atomics:
            single = narrowing.to_single(builder, value)
            if name == "sub":
                single = builder.fneg(single)
            old = atomic_element(
                builder, binding, ir.FloatType(), position, "fadd", single
            )
            return narrowing.to_double(builder, old) if dtype == types.float64 else old
        update = _FLOAT_UPDATES[name]
        old = _float_atomic(
            context,
            builder,
            binding,
            position,
            narrowing.to_single(builder, value),
            lambda o, v: update(builder, o, v),
        )
        return narrowing.to_double(builder, old) if dtype == types.float64 else old

    lower_atomic.__name__ = f"lower_atomic_{name}"
    lower_atomic.__doc__ = f"Lower ``atomic.{name}(array, index, value)``."
    lower(stub, VulkanArray, types.Any, types.Any)(lower_atomic)


for _name in ("add", "sub", "max", "min", "and_", "or_", "xor", "exch"):
    _register_atomic(getattr(stubs.atomic, _name))


@lower(stubs.atomic.cas, VulkanArray, types.Any, types.Any, types.Any)
def lower_cas(context, builder, sig, args):
    """Lower ``atomic.cas(array, index, expected, value)``.

    Returns
    -------
    llvmlite.ir.Value
        The element's previous value.
    """
    aryty, idxty, expty, valty = sig.args
    dtype = aryty.dtype
    binding, position = _atomic_position(
        context, builder, aryty, args[0], idxty, args[1]
    )
    expected = context.cast(builder, args[2], expty, dtype)
    value = context.cast(builder, args[3], valty, dtype)
    if isinstance(dtype, types.Float):
        expected, value = (builder.bitcast(v, i32) for v in (expected, value))
    old = compare_and_swap(builder, binding, position, expected, value)
    return (
        builder.bitcast(old, ir.FloatType()) if isinstance(dtype, types.Float) else old
    )


# -- array expressions ----------------------------------------------------------


def _operand_shape(context, builder, ty, value):
    """Extents of an expression operand; none for a scalar."""
    if isinstance(ty, VulkanArray):
        return _unpack(context, builder, ty, value)[1]
    if isinstance(ty, VulkanExpr):
        return expression_shape(context, builder, ty, value)
    return []


def _reduced_axis(builder, opty, axis):
    """The axis of a reduction along an axis, negative ones counted back."""
    negative = builder.icmp_signed("<", axis, axis.type(0))
    return builder.select(negative, builder.add(axis, axis.type(opty.ndim)), axis)


def _operands(context, builder, ty, value):
    """The operand values of an expression."""
    proxy = cgutils.create_struct_proxy(ty)(context, builder, value=value)
    return [getattr(proxy, f"operand{k}") for k in range(len(ty.operands))]


def expression_shape(context, builder, ty, value):
    """The broadcast shape of an expression.

    Raises ``ValueError`` in the kernel if the operands cannot be
    broadcast together.

    Returns
    -------
    list of llvmlite.ir.Value
        One ``intp`` extent per dimension.
    """
    if getattr(ty.op, "reduces_axis", False):
        # A reduction along an axis: its operand's shape without that axis.
        (opty, _), (operand, axis) = ty.operands, _operands(context, builder, ty, value)
        extents = _operand_shape(context, builder, opty, operand)
        axis = _reduced_axis(builder, opty, axis)
        bad = builder.or_(
            builder.icmp_signed("<", axis, axis.type(0)),
            builder.icmp_signed(">=", axis, axis.type(opty.ndim)),
        )
        with builder.if_then(bad, likely=False):
            context.call_conv.return_user_exc(
                builder,
                ValueError,
                (f"axis is out of bounds for an array of dimension {opty.ndim}",),
            )
        return [
            builder.select(
                builder.icmp_signed("<", axis.type(d), axis), extents[d], extents[d + 1]
            )
            for d in range(len(extents) - 1)
        ]
    intp = context.get_value_type(types.intp)
    one = intp(1)
    shape = [one] * ty.ndim
    for opty, operand in zip(ty.operands, _operands(context, builder, ty, value)):
        extents = _operand_shape(context, builder, opty, operand)
        for k, extent in enumerate(extents):
            dim = ty.ndim - len(extents) + k
            current = shape[dim]
            clash = builder.and_(
                builder.icmp_signed("!=", extent, current),
                builder.and_(
                    builder.icmp_signed("!=", extent, one),
                    builder.icmp_signed("!=", current, one),
                ),
            )
            with builder.if_then(clash, likely=False):
                context.call_conv.return_user_exc(
                    builder, ValueError, ("operands could not be broadcast together",)
                )
            shape[dim] = builder.select(
                builder.icmp_signed("==", current, one), extent, current
            )
    return shape


def _operand_element(context, builder, ty, value, indices):
    """One element of an operand at the given (right-aligned) indices."""
    if isinstance(ty, VulkanExpr):
        return expression_element(
            context, builder, ty, value, indices[len(indices) - ty.ndim :]
        )
    if not isinstance(ty, VulkanArray):
        return value
    offset, shape, _, steps = _unpack(context, builder, ty, value)
    own = indices[len(indices) - ty.ndim :]
    for index, extent, step in zip(own, shape, steps):
        # Axes of extent 1 are broadcast.
        index = builder.select(
            builder.icmp_signed("==", extent, extent.type(1)), index.type(0), index
        )
        offset = builder.add(offset, builder.mul(_i32(builder, index), step))
    return _load(context, builder, ty, offset, _binding(context, builder, ty, value))


def expression_element(context, builder, ty, value, indices):
    """Compute one element of an expression.

    Parameters
    ----------
    context : VulkanTargetContext
        The target context.
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    ty : VulkanExpr
        Type of the expression.
    value : llvmlite.ir.Value
        The expression.
    indices : list of llvmlite.ir.Value
        One non-negative, in-range ``intp`` index per dimension of `ty`.

    Returns
    -------
    llvmlite.ir.Value
        The element, of type ``ty.dtype``.
    """
    if getattr(ty.op, "reduces_axis", False):
        # A reduction along an axis: a loop, written in Python by the op.
        (opty, _), (operand, axis) = ty.operands, _operands(context, builder, ty, value)
        function = ty.op.element_function(opty.ndim)
        sig = ty.dtype(opty, types.intp, *([types.intp] * len(indices)))
        axis = _reduced_axis(builder, opty, axis)
        return context.compile_internal(
            builder, function, sig, [operand, axis, *indices]
        )
    values, elements = [], []
    for opty, operand in zip(ty.operands, _operands(context, builder, ty, value)):
        values.append(_operand_element(context, builder, opty, operand, indices))
        elements.append(
            opty.dtype if isinstance(opty, (VulkanArray, VulkanExpr)) else opty
        )
    typing = context.typing_context
    fnty = typing.resolve_value_type(ty.op)
    sig = typing.resolve_function_type(fnty, tuple(elements), {})
    values = [
        context.cast(builder, v, t, a) for v, t, a in zip(values, elements, sig.args)
    ]
    result = context.get_function(fnty, sig)(builder, values)
    return context.cast(builder, result, sig.return_type, ty.dtype)


def _register_expression(op, arity):
    """Lower `op` on arrays: it only records its operands."""

    def lower_expression(context, builder, sig, args):
        proxy = cgutils.create_struct_proxy(sig.return_type)(context, builder)
        for k, value in enumerate(args):
            setattr(proxy, f"operand{k}", value)
        return proxy._getvalue()

    lower_expression.__doc__ = f"Lower ``{getattr(op, '__name__', op)}`` on arrays."
    kinds = (VulkanArray, VulkanExpr, types.Number, types.Boolean)
    for pattern in itertools.product(kinds, repeat=arity):
        if any(k in (VulkanArray, VulkanExpr) for k in pattern):
            lower(op, *pattern)(lower_expression)


for _op in vkdecl._BINARY_OPERATORS:
    _register_expression(_op, 2)
for _op in vkdecl._UNARY_OPERATORS:
    _register_expression(_op, 1)
for _ufunc in vkdecl._ufuncs():
    _register_expression(_ufunc, _ufunc.nin)


def _wrap_indices(context, builder, shape, idxty, idx):
    """Per-axis indices with negative ones wrapped around, as ``intp``."""
    if isinstance(idxty, types.BaseTuple):
        indices = cgutils.unpack_tuple(builder, idx, count=len(idxty))
        index_types = list(idxty)
    else:
        indices, index_types = [idx], [idxty]
    out = []
    for dim, (index, ty) in enumerate(zip(indices, index_types)):
        index = context.cast(builder, index, ty, types.intp)
        if ty.signed:
            negative = builder.icmp_signed("<", index, index.type(0))
            index = builder.select(negative, builder.add(index, shape[dim]), index)
        _check_bounds(context, builder, index, shape[dim], dim, len(shape))
        out.append(index)
    return out


@lower(operator.getitem, VulkanExpr, types.Integer)
@lower(operator.getitem, VulkanExpr, types.BaseTuple)
def lower_expression_getitem(context, builder, sig, args):
    """Lower ``expression[index]``: compute one element.

    Returns
    -------
    llvmlite.ir.Value
    """
    exprty, idxty = sig.args
    shape = expression_shape(context, builder, exprty, args[0])
    indices = _wrap_indices(context, builder, shape, idxty, args[1])
    return expression_element(context, builder, exprty, args[0], indices)


@registry.lower_getattr(VulkanExpr, "shape")
def lower_expression_shape(context, builder, ty, value):
    """Lower ``expression.shape``."""
    intp = context.get_value_type(types.intp)
    return cgutils.pack_array(
        builder, expression_shape(context, builder, ty, value), ty=intp
    )


@registry.lower_getattr(VulkanExpr, "size")
def lower_expression_size(context, builder, ty, value):
    """Lower ``expression.size``."""
    size = context.get_constant(types.intp, 1)
    for extent in expression_shape(context, builder, ty, value):
        size = builder.mul(size, extent)
    return size


@registry.lower_getattr(VulkanExpr, "ndim")
def lower_expression_ndim(context, builder, ty, value):
    """Lower ``expression.ndim``."""
    return context.get_constant(types.intp, ty.ndim)


@lower(len, VulkanExpr)
def lower_expression_len(context, builder, sig, args):
    """Lower ``len(expression)``."""
    return expression_shape(context, builder, sig.args[0], args[0])[0]


def _arrays_in(ty, value, context, builder, anywhere=False):
    """The array operands of an expression, recursively.

    Returns
    -------
    list of tuple
        Type and value of each array, and whether the expression reads it
        at positions other than those of its own elements (under a
        reduction along an axis).
    """
    if isinstance(ty, VulkanArray):
        return [(ty, value, anywhere)]
    if not isinstance(ty, VulkanExpr):
        return []
    anywhere = anywhere or getattr(ty.op, "reduces_axis", False)
    found = []
    for opty, operand in zip(ty.operands, _operands(context, builder, ty, value)):
        found += _arrays_in(opty, operand, context, builder, anywhere)
    return found


def _element_span(builder, offset, shape, steps):
    """The positions in its buffer that a view's elements lie between.

    Returns
    -------
    tuple of llvmlite.ir.Value
        The lowest and highest position, as 32-bit element positions like
        offsets and steps, and whether the view has no elements.
    """
    low = high = _i32(builder, offset)
    zero, empty = i32(0), cgutils.false_bit
    for extent, step in zip(shape, steps):
        extent = _i32(builder, extent)
        reach = builder.mul(builder.sub(extent, i32(1)), _i32(builder, step))
        negative = builder.icmp_signed("<", reach, zero)
        low = builder.add(low, builder.select(negative, reach, zero))
        high = builder.add(high, builder.select(negative, zero, reach))
        empty = builder.or_(empty, builder.icmp_signed("==", extent, zero))
    return low, high, empty


def assign_expression(context, builder, aryty, selection, exprty, expr):
    """Write an expression, element by element, into a view.

    Parameters
    ----------
    aryty : VulkanArray
        Type of the array written to.
    selection : _Selection
        The view of it that is written.
    exprty : VulkanExpr
        Type of the expression.
    expr : llvmlite.ir.Value
        The expression.

    Notes
    -----
    NumPy computes the right-hand side before writing. Element by element,
    that is the same only if the expression reads the written array at the
    very positions it writes, or only outside the span of positions it
    writes; anything else raises ``ValueError``.
    """
    view_shape = selection.shape
    if exprty.ndim > len(view_shape):
        raise VulkanUnsupportedError(
            f"cannot assign a {exprty.ndim}-d expression to a {len(view_shape)}-d slice"
        )
    shape = expression_shape(context, builder, exprty, expr)
    for k, extent in enumerate(shape):
        wanted = view_shape[len(view_shape) - exprty.ndim + k]
        bad = builder.and_(
            builder.icmp_signed("!=", extent, wanted),
            builder.icmp_signed("!=", extent, extent.type(1)),
        )
        with builder.if_then(bad, likely=False):
            context.call_conv.return_user_exc(
                builder,
                ValueError,
                ("cannot assign slice from input of different size",),
            )
    itemsize = context.get_abi_sizeof(storage_type(context, aryty))
    view_steps = [
        _i32(builder, builder.sdiv(stride, stride.type(itemsize), flags=["exact"]))
        for stride in selection.strides
    ]
    write_span = _element_span(builder, selection.offset, view_shape, view_steps)
    for opty, operand, anywhere in _arrays_in(exprty, expr, context, builder):
        if opty.binding is not None and isinstance(selection.binding, int):
            shared = opty.binding == selection.binding
        else:
            source = _binding_value(context, builder, opty, operand)
            target = selection.binding
            target = i32(target) if isinstance(target, int) else target
            shared = builder.icmp_unsigned("==", source, target)
        if shared is False:
            continue
        offset, op_shape, _, steps = _unpack(context, builder, opty, operand)
        if anywhere or opty.ndim != len(view_shape):
            same = cgutils.false_bit  # read elsewhere, or a different shape
        else:
            same = builder.icmp_signed("==", offset, selection.offset)
            for a, b in zip(op_shape, view_shape):
                same = builder.and_(same, builder.icmp_signed("==", a, b))
            for a, b in zip(steps, view_steps):
                same = builder.and_(same, builder.icmp_signed("==", a, b))
        # Reads whose elements all lie outside the span written are fine.
        read_span = _element_span(builder, offset, op_shape, steps)
        apart = builder.or_(
            builder.or_(read_span[2], write_span[2]),
            builder.or_(
                builder.icmp_signed("<", read_span[1], write_span[0]),
                builder.icmp_signed("<", write_span[1], read_span[0]),
            ),
        )
        overlap = builder.and_(builder.not_(same), builder.not_(apart))
        if shared is not True:
            overlap = builder.and_(shared, overlap)
        with builder.if_then(overlap, likely=False):
            context.call_conv.return_user_exc(
                builder,
                ValueError,
                (
                    "the expression reads the array it is assigned to at other positions",
                ),
            )
    viewty = aryty.copy(ndim=len(view_shape), layout="A")
    view = _make_view(context, builder, viewty, selection)
    intp = context.get_value_type(types.intp)
    with cgutils.loop_nest(builder, view_shape, intp) as indices:
        value = expression_element(
            context,
            builder,
            exprty,
            expr,
            list(indices)[len(view_shape) - exprty.ndim :],
        )
        target = _position(context, builder, viewty, view, indices)
        _store(context, builder, aryty, target, value, exprty.dtype, selection.binding)


@lower(operator.setitem, VulkanArray, types.Integer, VulkanExpr)
@lower(operator.setitem, VulkanArray, types.BaseTuple, VulkanExpr)
@lower(operator.setitem, VulkanArray, types.SliceType, VulkanExpr)
@lower(operator.setitem, VulkanArray, types.EllipsisType, VulkanExpr)
def lower_setitem_expression(context, builder, sig, args):
    """Lower ``array[index] = expression``; see `assign_expression`.

    Returns
    -------
    llvmlite.ir.Value
        A dummy value.
    """
    aryty, idxty, exprty = sig.args
    selection = _select(context, builder, aryty, args[0], idxty, args[1])
    assign_expression(context, builder, aryty, selection, exprty, args[2])
    return context.get_dummy_value()


def _register_inplace(op, plain):
    """Lower an in-place operator on an array as an element-wise update."""

    def lower_inplace(context, builder, sig, args):
        aryty, valty = sig.args
        exprty = VulkanExpr(plain, (aryty, valty), aryty.dtype, aryty.ndim)
        proxy = cgutils.create_struct_proxy(exprty)(context, builder)
        proxy.operand0, proxy.operand1 = args
        offset, shape, strides, _ = _unpack(context, builder, aryty, args[0])
        binding = _binding(context, builder, aryty, args[0])
        selection = _Selection(offset, shape, strides, binding)
        assign_expression(context, builder, aryty, selection, exprty, proxy._getvalue())
        return args[0]

    lower_inplace.__doc__ = f"Lower ``{op.__name__}`` on arrays."
    for kind in (VulkanArray, VulkanExpr, types.Number, types.Boolean):
        lower(op, VulkanArray, kind)(lower_inplace)


for _op, _plain in vkdecl._INPLACE_OPERATORS.items():
    _register_inplace(_op, _plain)


# -- print ------------------------------------------------------------------------


def _print_words(context, builder, ty, value):
    """How a value is recorded: its format code and its 32-bit words."""
    if isinstance(ty, types.Boolean):
        return "b", [builder.zext(value, i32)]
    if isinstance(ty, types.Integer):
        wide = context.cast(
            builder, value, ty, types.int64 if ty.signed else types.uint64
        )
        high = (
            builder.ashr(wide, wide.type(32))
            if ty.signed
            else builder.lshr(wide, wide.type(32))
        )
        return ("i" if ty.signed else "u"), [
            builder.trunc(wide, i32),
            builder.trunc(high, i32),
        ]
    if isinstance(ty, types.Float):
        if ty == types.float32 or narrowing.current.floats:
            single = context.cast(builder, value, ty, types.float32)
            single = narrowing.to_single(builder, single)
            return "f", [builder.bitcast(single, i32)]
        double = context.cast(builder, value, ty, types.float64)
        return "d", list(double_words(builder, double))
    raise VulkanUnsupportedError(
        f"print() in Vulkan kernels supports constant strings and numbers, not {ty}"
    )


@lower(print, types.VarArg(types.Any))
def lower_print(context, builder, sig, args):
    """Lower ``print(...)`` to a record in the kernel's print buffer.

    The buffer starts with a cursor and its capacity, in 32-bit words. A
    call reserves room for its record with an atomic addition to the cursor
    and, if it fits, writes the number of its format followed by the
    values. The host prints the records after the kernel (see
    `numba_vulkan.runtime.print_records`). String constants are part of
    the format and are not written.

    Returns
    -------
    llvmlite.ir.Value
        A dummy value.
    """
    parts, words = [], []
    for ty, value in zip(sig.args, args):
        if isinstance(ty, types.StringLiteral):
            parts.append(("s", ty.literal_value))
            continue
        code, values = _print_words(context, builder, ty, value)
        parts.append((code,))
        words += values
    form = tuple(parts)
    key = zlib.crc32(repr(form).encode()) & 0x7FFFFFFF
    print_formats[key] = form
    words = [i32(key)] + words
    start = atomic_element(builder, PRINT_BINDING, i32, i32(0), "add", i32(len(words)))
    capacity = load_element(builder, PRINT_BINDING, i32, i32(1))
    position = builder.add(start, i32(2))
    end = builder.add(position, i32(len(words)))
    with builder.if_then(builder.icmp_unsigned("<=", end, capacity)):
        for k, word in enumerate(words):
            store_element(
                builder, PRINT_BINDING, i32, builder.add(position, i32(k)), word
            )
    return context.get_dummy_value()


# -- subgroups ----------------------------------------------------------------

_GROUP_SUFFIX = {"i32": "i32", "i64": "i64", "float": "f32", "double": "f64"}


def _group_call(builder, operation, restype, args):
    """Call the placeholder of a subgroup operation.

    The function is only declared; `numba_vulkan.codegen.lower_group_operations`
    turns the call into the group instruction. It is ``convergent``, so that
    LLVM neither moves it across conditions nor duplicates it.

    Parameters
    ----------
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    operation : str
        Name after ``nv.sg.``, without the type, for example ``"fadd.incl"``.
    restype : llvmlite.ir.Type
        Result type.
    args : list of llvmlite.ir.Value
        Operands.

    Returns
    -------
    llvmlite.ir.Value
    """
    value_type = str(args[0].type) if args else ""
    suffix = _GROUP_SUFFIX.get(value_type)
    name = f"nv.sg.{operation}" + (f".{suffix}" if suffix else "")
    fn = builder.module.globals.get(name)
    if fn is None:
        fnty = ir.FunctionType(restype, [a.type for a in args])
        fn = ir.Function(builder.module, fnty, name)
        fn.attributes.add("convergent")
        fn.attributes.add("nounwind")
    return builder.call(fn, args)


def _register_subgroup_builtin(stub, intrinsic):
    """Lower a function without arguments to a SPIR-V built-in variable."""

    @lower(stub)
    def lower_builtin(context, builder, sig, args):
        fn = builder.module.globals.get(intrinsic)
        if fn is None:
            fn = ir.Function(builder.module, ir.FunctionType(i32, []), intrinsic)
        return builder.call(fn, [])

    lower_builtin.__doc__ = f"Lower ``subgroup.{stub.__name__}()`` to ``{intrinsic}``."
    return lower_builtin


for _name, _intrinsic in (
    ("size", "llvm.spv.subgroup.size"),
    ("lane", "llvm.spv.subgroup.local.invocation.id"),
    ("id", "llvm.spv.subgroup.id"),
    ("count", "llvm.spv.num.subgroups"),
):
    _register_subgroup_builtin(getattr(stubs.subgroup, _name), _intrinsic)


def _arithmetic(op, ty):
    """Name of the group instruction for an operation on values of `ty`."""
    if isinstance(ty, types.Float):
        return {"sum": "fadd", "prod": "fmul", "min": "fmin", "max": "fmax"}[op]
    signed = "s" if ty.signed else "u"
    return {
        "sum": "iadd",
        "prod": "imul",
        "min": f"{signed}min",
        "max": f"{signed}max",
    }[op]


def _register_subgroup_arithmetic(stub, op, scan):
    """Lower a reduction or scan over the subgroup."""

    @lower(stub, types.Number)
    def lower_arithmetic(context, builder, sig, args):
        operation = f"{_arithmetic(op, sig.args[0])}.{scan}"
        return _group_call(builder, operation, args[0].type, [args[0]])

    lower_arithmetic.__doc__ = f"Lower ``subgroup.{stub.__name__}(value)``."
    return lower_arithmetic


for _op in ("sum", "prod", "min", "max"):
    _register_subgroup_arithmetic(getattr(stubs.subgroup, _op), _op, "red")
    _register_subgroup_arithmetic(
        getattr(stubs.subgroup, f"inclusive_{_op}"), _op, "incl"
    )
    _register_subgroup_arithmetic(
        getattr(stubs.subgroup, f"exclusive_{_op}"), _op, "excl"
    )


def _register_subgroup_exchange(stub, operation):
    """Lower a function of a value and a lane, delta or mask."""

    @lower(stub, types.Number, types.Integer)
    def lower_exchange(context, builder, sig, args):
        lane = context.cast(builder, args[1], sig.args[1], types.uint32)
        return _group_call(builder, operation, args[0].type, [args[0], lane])

    lower_exchange.__doc__ = f"Lower ``subgroup.{stub.__name__}(value, lane)``."
    return lower_exchange


for _name, _operation in (
    ("broadcast", "broadcast"),
    ("shuffle", "shuffle"),
    ("shuffle_xor", "shufflexor"),
    ("shuffle_up", "shuffleup"),
    ("shuffle_down", "shuffledown"),
):
    _register_subgroup_exchange(getattr(stubs.subgroup, _name), _operation)


@lower(stubs.subgroup.broadcast_first, types.Number)
def lower_broadcast_first(context, builder, sig, args):
    """Lower ``subgroup.broadcast_first(value)``."""
    return _group_call(builder, "broadcastfirst", args[0].type, [args[0]])


@lower(stubs.subgroup.any, types.Boolean)
def lower_subgroup_any(context, builder, sig, args):
    """Lower ``subgroup.any(predicate)``."""
    return _group_call(builder, "any", ir.IntType(1), [args[0]])


@lower(stubs.subgroup.all, types.Boolean)
def lower_subgroup_all(context, builder, sig, args):
    """Lower ``subgroup.all(predicate)``."""
    return _group_call(builder, "all", ir.IntType(1), [args[0]])


@lower(stubs.subgroup.elect)
def lower_subgroup_elect(context, builder, sig, args):
    """Lower ``subgroup.elect()``."""
    return _group_call(builder, "elect", ir.IntType(1), [])


def _ballot(builder, predicate):
    """The ballot of a predicate as four 32-bit words."""
    return _group_call(builder, "ballot", ir.VectorType(i32, 4), [predicate])


@lower(stubs.subgroup.ballot, types.Boolean)
def lower_subgroup_ballot(context, builder, sig, args):
    """Lower ``subgroup.ballot(predicate)`` to a tuple of four words."""
    words = _ballot(builder, args[0])
    items = [builder.extract_element(words, i32(k)) for k in range(4)]
    return context.make_tuple(builder, sig.return_type, items)


@lower(stubs.subgroup.ballot_count, types.Boolean)
def lower_subgroup_ballot_count(context, builder, sig, args):
    """Lower ``subgroup.ballot_count(predicate)``."""
    return _group_call(builder, "ballotcount.red", i32, [_ballot(builder, args[0])])


# -- records ------------------------------------------------------------------


def _record_place(context, builder, recty, rec):
    """Binding and first byte of a record."""
    proxy = cgutils.create_struct_proxy(recty)(context, builder, value=rec)
    return proxy.binding, proxy.position


def _alignment(recty, offset):
    """Position of a field within its first word, if known when compiling.

    Records whose size is a multiple of four start at word boundaries, so
    the position of a field in its word follows from its offset.

    Returns
    -------
    int or None
    """
    return offset % 4 if recty.size % 4 == 0 else None


def _read_word(builder, binding, position, size, known):
    """The `size` (1 to 4) bytes at a byte position, as the low part of an i32.

    Only words that hold some of the bytes are read, so nothing beyond the
    buffer is touched. The bits above the `size` bytes are undefined.
    """
    first = builder.lshr(position, i32(2))
    low = load_element(builder, binding, i32, first)
    if known == 0:
        return low
    last = builder.lshr(builder.add(position, i32(size - 1)), i32(2))
    high = load_element(builder, binding, i32, last)
    if known is not None:
        shift = i32(8 * known)
        return builder.or_(
            builder.lshr(low, shift), builder.shl(high, i32(32 - 8 * known))
        )
    shift = builder.shl(builder.and_(position, i32(3)), i32(3))
    joined = builder.or_(
        builder.lshr(low, shift), builder.shl(high, builder.sub(i32(32), shift))
    )
    # A shift by 32 is undefined; then the bytes are the first word as it is.
    return builder.select(builder.icmp_unsigned("==", shift, i32(0)), low, joined)


def _write_word(builder, binding, position, size, bits, known):
    """Write the low `size` (1 to 4) bytes of an i32 at a byte position.

    Whole aligned words are stored. Anything else changes only its own
    bytes, with atomic ``and`` and ``or`` on the words it covers, so that
    invocations writing neighbouring fields do not undo each other.
    """
    first = builder.lshr(position, i32(2))
    if known == 0 and size == 4:
        store_element(builder, binding, i32, first, bits)
        return
    mask = i32((1 << (8 * size)) - 1)
    bits = builder.and_(bits, mask)
    if known is not None:
        shift = i32(8 * known)
    else:
        shift = builder.shl(builder.and_(position, i32(3)), i32(3))
    ones = i32(0xFFFFFFFF)
    atomic_element(
        builder, binding, i32, first, "and", builder.xor(builder.shl(mask, shift), ones)
    )
    atomic_element(builder, binding, i32, first, "or", builder.shl(bits, shift))
    if known is not None and known + size <= 4:
        return
    # The bytes that spill into the next word, if any (none for shift 0).
    last = builder.lshr(builder.add(position, i32(size - 1)), i32(2))
    back = builder.sub(i32(32), shift)
    aligned = builder.icmp_unsigned("==", shift, i32(0))
    spill_mask = builder.select(aligned, i32(0), builder.lshr(mask, back))
    spill = builder.select(aligned, i32(0), builder.lshr(bits, back))
    atomic_element(builder, binding, i32, last, "and", builder.xor(spill_mask, ones))
    atomic_element(builder, binding, i32, last, "or", spill)


def _field_place(context, builder, recty, rec, name):
    """Binding, first byte, type, size and known alignment of a field."""
    fieldty, offset = recty.field(name)
    if not isinstance(fieldty, (types.Integer, types.Float, types.Boolean)):
        raise VulkanUnsupportedError(
            f"record fields of type {fieldty} are not supported on Vulkan "
            "(only booleans, integers and floats are)"
        )
    if isinstance(fieldty, types.Float) and fieldty.bitwidth == 16:
        raise VulkanUnsupportedError("float16 record fields are not supported")
    size = recty.record.dtype.fields[name][0].itemsize
    binding, position = _record_place(context, builder, recty, rec)
    position = builder.add(position, i32(offset))
    return binding, position, fieldty, size, _alignment(recty, offset)


def _read_field(context, builder, recty, rec, name):
    """Read a field of a record.

    Returns
    -------
    llvmlite.ir.Value
        Of the field's type.
    """
    binding, position, fieldty, size, known = _field_place(
        context, builder, recty, rec, name
    )
    llty = context.get_value_type(fieldty)
    if size == 8:
        halves = ir.Constant(ir.VectorType(i32, 2), None)
        for k in range(2):
            at = builder.add(position, i32(4 * k))
            word = _read_word(builder, binding, at, 4, known)
            halves = builder.insert_element(halves, word, i32(k))
        return builder.bitcast(halves, llty)
    word = _read_word(builder, binding, position, size, known)
    if size < 4:
        word = builder.trunc(word, ir.IntType(8 * size))
    if isinstance(fieldty, types.Boolean):
        return builder.icmp_unsigned("!=", word, word.type(0))
    return builder.bitcast(word, llty) if isinstance(fieldty, types.Float) else word


def _write_field(context, builder, recty, rec, name, value):
    """Write a value of the field's type into a field of a record."""
    binding, position, fieldty, size, known = _field_place(
        context, builder, recty, rec, name
    )
    if size == 8:
        halves = builder.bitcast(value, ir.VectorType(i32, 2))
        for k in range(2):
            at = builder.add(position, i32(4 * k))
            word = builder.extract_element(halves, i32(k))
            _write_word(builder, binding, at, 4, word, known)
        return
    if isinstance(fieldty, types.Float):
        value = builder.bitcast(value, i32)
    elif value.type != i32:
        value = builder.zext(value, i32)
    _write_word(builder, binding, position, size, value, known)


@registry.lower_getattr_generic(VulkanRecord)
def lower_record_getattr(context, builder, ty, value, attr):
    """Lower ``record.field``."""
    return _read_field(context, builder, ty, value, attr)


@registry.lower_setattr_generic(VulkanRecord)
def lower_record_setattr(context, builder, sig, args, attr):
    """Lower ``record.field = value``."""
    recty, valty = sig.args
    fieldty = recty.field(attr)[0]
    value = context.cast(builder, args[1], valty, fieldty)
    _write_field(context, builder, recty, args[0], attr, value)


@lower(operator.getitem, VulkanRecord, types.StringLiteral)
@lower("static_getitem", VulkanRecord, types.StringLiteral)
def lower_record_getitem(context, builder, sig, args):
    """Lower ``record["field"]``."""
    return _read_field(
        context, builder, sig.args[0], args[0], sig.args[1].literal_value
    )


@lower(operator.setitem, VulkanRecord, types.StringLiteral, types.Any)
@lower("static_setitem", VulkanRecord, types.StringLiteral, types.Any)
def lower_record_setitem(context, builder, sig, args):
    """Lower ``record["field"] = value``."""
    recty, namety, valty = sig.args
    name = namety.literal_value
    value = context.cast(builder, args[2], valty, recty.field(name)[0])
    _write_field(context, builder, recty, args[0], name, value)
    return context.get_dummy_value()
