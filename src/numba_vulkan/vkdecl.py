"""Typing of Vulkan-specific functions."""

import math
import operator

import numpy as np
from numba.core import errors, types
from numba.core.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
    CallableTemplate,
    ConcreteTemplate,
    Registry,
    signature,
)

from numba_vulkan import stubs
from numba_vulkan.buffers import LOCAL_BASE, SHARED_BASE, shared_sizes
from numba_vulkan.vktypes import VulkanArray, VulkanExpr, VulkanRecord

registry = Registry()


@registry.register_global(stubs.global_id)
class GlobalId(AbstractTemplate):
    """Typing of `numba_vulkan.stubs.global_id`.

    Only a literal axis of 0, 1 or 2 is accepted, because the axis selects
    a component of a SPIR-V built-in at compile time.
    """

    def generic(self, args, kws):
        # Only literal axes type-check; Numba retries with literal types.
        """Type a call.

        Parameters
        ----------
        args : tuple of numba.types.Type
            Positional argument types.
        kws : dict
            Keyword argument types.

        Returns
        -------
        numba.core.typing.Signature or None
            ``int32(literal)`` for a valid literal axis, otherwise ``None``,
            which makes Numba retry with literal argument types.
        """
        if len(args) == 1 and not kws and isinstance(args[0], types.IntegerLiteral):
            if args[0].literal_value in (0, 1, 2):
                return signature(types.int32, args[0])


def _axis_template(stub):
    """Register the typing of a function of a literal axis returning int32."""

    @registry.register_global(stub)
    class AxisTemplate(GlobalId):
        __doc__ = f"Typing of `numba_vulkan.stubs.{stub.__name__}`."

    return AxisTemplate


for _stub in (stubs.local_id, stubs.group_id, stubs.local_size, stubs.num_groups):
    _axis_template(_stub)


@registry.register_global(stubs.barrier)
class Barrier(ConcreteTemplate):
    """Typing of `numba_vulkan.stubs.barrier`: no arguments, no result."""

    cases = [signature(types.none)]


def _literal_shape(shape):
    """The shape given by a literal integer or tuple of them, or ``None``."""
    if isinstance(shape, types.IntegerLiteral):
        return (shape.literal_value,)
    if isinstance(shape, types.BaseTuple) and all(
        isinstance(s, types.IntegerLiteral) for s in shape
    ):
        return tuple(s.literal_value for s in shape)
    return None


def _dtype(dtype):
    """The Numba type that a dtype argument stands for, or ``None``."""
    if isinstance(dtype, types.NumberClass):
        return dtype.instance_type
    if isinstance(dtype, types.DType):
        return dtype.dtype
    if isinstance(dtype, types.Function) and dtype.typing_key is bool:
        return types.boolean
    return None


def _static_array_type(shape, dtype, site, base):
    """The type of an array with a constant shape, or ``None``.

    Parameters
    ----------
    shape, dtype : numba.types.Type
        Types of the shape and dtype arguments.
    site : numba.types.Type
        Type of the hidden call-site argument.
    base : int
        `SHARED_BASE` or `LOCAL_BASE`.

    Returns
    -------
    VulkanArray or None
    """
    dims, dtype = _literal_shape(shape), _dtype(dtype)
    if dims is None or dtype is None or not isinstance(site, types.IntegerLiteral):
        return None
    if any(n <= 0 for n in dims):
        raise errors.TypingError("arrays in kernels need a positive constant shape")
    binding = base + site.literal_value
    shared_sizes[binding] = math.prod(dims)
    return VulkanArray(dtype, len(dims), "C", binding)


@registry.register_global(stubs.local.array)
class LocalArray(CallableTemplate):
    """Typing of ``local.array(shape, dtype)``; see `SharedArray`."""

    def generic(self):
        """Return the typer.

        Returns
        -------
        callable
        """

        def typer(_vulkan_site, shape, dtype):
            return _static_array_type(shape, dtype, _vulkan_site, LOCAL_BASE)

        return typer


def _numpy_constructor(function, fill):
    """Register the typing of ``np.empty`` and its relatives inside kernels.

    They give local arrays (see `LocalArray`). A shape that is not a
    constant is an error, because shaders cannot allocate memory.

    Parameters
    ----------
    function : callable
        ``np.empty``, ``np.zeros``, ``np.ones`` or ``np.full``.
    fill : bool
        Whether the function takes a fill value after the shape.
    """

    def check(shape, site):
        # Numba types the call twice, the second time with literals; only
        # then can a shape be told to be constant or not.
        if not isinstance(site, types.IntegerLiteral):
            return False
        if _literal_shape(shape) is None and not isinstance(shape, types.Literal):
            if isinstance(shape, (types.Integer, types.BaseTuple)):
                raise errors.TypingError(
                    f"np.{function.__name__}() in a Vulkan kernel needs a constant "
                    "shape: shaders cannot allocate memory at run time"
                )
        return True

    if fill:

        @registry.register_global(function)
        class Constructor(CallableTemplate):
            def generic(self):
                def typer(_vulkan_site, shape, fill_value, dtype=None):
                    if not check(shape, _vulkan_site):
                        return None
                    if dtype is None or isinstance(dtype, types.NoneType):
                        dtype = types.NumberClass(fill_value)
                    return _static_array_type(shape, dtype, _vulkan_site, LOCAL_BASE)

                return typer

    else:

        @registry.register_global(function)
        class Constructor(CallableTemplate):
            def generic(self):
                def typer(_vulkan_site, shape, dtype=None):
                    if not check(shape, _vulkan_site):
                        return None
                    if dtype is None or isinstance(dtype, types.NoneType):
                        dtype = types.NumberClass(types.float64)
                    return _static_array_type(shape, dtype, _vulkan_site, LOCAL_BASE)

                return typer

    Constructor.__doc__ = f"Typing of ``np.{function.__name__}`` inside kernels."
    return Constructor


for _function in (np.empty, np.zeros, np.ones):
    _numpy_constructor(_function, fill=False)
_numpy_constructor(np.full, fill=True)


@registry.register_global(stubs.shared.array)
class SharedArray(CallableTemplate):
    """Typing of ``shared.array(shape, dtype)``.

    The call carries a hidden first argument: a literal that identifies the
    call site, added by `numba_vulkan.compiler.NumberSharedArrays`. It
    becomes the "binding" of the array type, which tells element accesses
    which workgroup variable to use.
    """

    def generic(self):
        """Return the typer.

        Returns
        -------
        callable
        """

        def typer(_vulkan_site, shape, dtype):
            return _static_array_type(shape, dtype, _vulkan_site, SHARED_BASE)

        return typer


_ATOMIC_TYPES = (
    types.int32,
    types.uint32,
    types.int64,
    types.uint64,
    types.float32,
    types.float64,
)
_INTEGER_TYPES = (types.int32, types.uint32, types.int64, types.uint64)


def _atomic_template(stub, dtypes):
    """Register the typing of an atomic operation on elements of `dtypes`."""

    @registry.register_global(stub)
    class AtomicTemplate(AbstractTemplate):
        __doc__ = f"Typing of ``atomic.{stub.__name__}(array, index, value)``."

        def generic(self, args, kws):
            if kws or len(args) != 3:
                return None
            array, index, value = args
            if not _atomic_target(array, index, dtypes):
                return None
            return signature(array.dtype, array, index, array.dtype)

    return AtomicTemplate


def _atomic_target(array, index, dtypes):
    """Whether atomics apply to `array` at an index of type `index`."""
    if not isinstance(array, VulkanArray) or array.dtype not in dtypes:
        return False
    if array.half:
        raise errors.TypingError(
            "atomic operations on float16 arrays are not supported"
        )
    if array.binding is not None and array.binding >= LOCAL_BASE:
        raise errors.TypingError("atomic operations on local arrays are not supported")
    if not array.mutable:
        raise errors.TypingError("atomic operations need a writable array")
    count = len(index) if isinstance(index, types.BaseTuple) else 1
    indices = index if isinstance(index, types.BaseTuple) else (index,)
    return count == array.ndim and all(isinstance(i, types.Integer) for i in indices)


for _name in ("add", "sub", "max", "min", "exch"):
    _atomic_template(getattr(stubs.atomic, _name), _ATOMIC_TYPES)
for _name in ("and_", "or_", "xor"):
    _atomic_template(getattr(stubs.atomic, _name), _INTEGER_TYPES)


@registry.register_global(stubs.atomic.cas)
class CompareAndSwap(AbstractTemplate):
    """Typing of ``atomic.cas(array, index, expected, value)``."""

    def generic(self, args, kws):
        """Type a call; 32-bit elements only."""
        if kws or len(args) != 4:
            return None
        array, index = args[:2]
        if not _atomic_target(array, index, (types.int32, types.uint32, types.float32)):
            return None
        return signature(array.dtype, array, index, array.dtype, array.dtype)


# -- array expressions ----------------------------------------------------------

_BINARY_OPERATORS = (
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.mod,
    operator.pow,
    operator.and_,
    operator.or_,
    operator.xor,
    operator.lshift,
    operator.rshift,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
    operator.eq,
    operator.ne,
)
_UNARY_OPERATORS = (operator.neg, operator.pos, operator.invert, abs)
_INPLACE_OPERATORS = {
    operator.iadd: operator.add,
    operator.isub: operator.sub,
    operator.imul: operator.mul,
    operator.itruediv: operator.truediv,
    operator.ifloordiv: operator.floordiv,
    operator.imod: operator.mod,
    operator.ipow: operator.pow,
    operator.iand: operator.and_,
    operator.ior: operator.or_,
    operator.ixor: operator.xor,
}


def _ufuncs():
    """The NumPy ufuncs that kernels support on scalars, as functions."""
    from numba_vulkan import ufuncs

    names = (
        list(ufuncs._REUSED)
        + list(ufuncs._UNARY)
        + list(ufuncs._BINARY)
        + list(ufuncs._PREDICATES)
    )
    return [getattr(np, name) for name in dict.fromkeys(names) if hasattr(np, name)]


def is_array_like(ty):
    """Whether a type is an array or an array expression of this target."""
    return isinstance(ty, (VulkanArray, VulkanExpr))


def element_type(ty):
    """Type of an element of an array or expression; a scalar's own type."""
    return ty.dtype if is_array_like(ty) else ty


def expression_type(context, op, args):
    """The type of applying `op` element-wise to `args`, or ``None``.

    Parameters
    ----------
    context : numba.core.typing.Context
        The typing context.
    op : callable
        An operator or ufunc.
    args : tuple of numba.types.Type
        Argument types; at least one must be an array or expression, the
        others scalars.

    Returns
    -------
    VulkanExpr or None
    """
    if not any(is_array_like(a) for a in args):
        return None
    if not all(
        is_array_like(a) or isinstance(a, (types.Number, types.Boolean)) for a in args
    ):
        return None
    elements = tuple(element_type(a) for a in args)
    try:
        sig = context.resolve_function_type(
            context.resolve_value_type(op), elements, {}
        )
    except errors.TypingError:
        return None
    if sig is None or not isinstance(sig.return_type, (types.Number, types.Boolean)):
        return None
    ndim = max(a.ndim for a in args if is_array_like(a))
    return VulkanExpr(op, args, sig.return_type, ndim)


def _expression_template(op):
    """Register the typing of `op` applied to arrays as a `VulkanExpr`."""

    @registry.register_global(op)
    class ExpressionTemplate(AbstractTemplate):
        __doc__ = f"Typing of ``{getattr(op, '__name__', op)}`` on arrays."

        def generic(self, args, kws):
            if kws:
                return None
            result = expression_type(self.context, op, args)
            return None if result is None else signature(result, *args)

    return ExpressionTemplate


for _op in _BINARY_OPERATORS + _UNARY_OPERATORS + tuple(_ufuncs()):
    _expression_template(_op)


def _inplace_template(op, plain):
    """Register the typing of an in-place operator on a writable array."""

    @registry.register_global(op)
    class InplaceTemplate(AbstractTemplate):
        __doc__ = f"Typing of ``{op.__name__}`` on arrays: an element-wise update."

        def generic(self, args, kws):
            if kws or len(args) != 2 or not isinstance(args[0], VulkanArray):
                return None
            if expression_type(self.context, plain, args) is None:
                return None
            if not args[0].mutable:
                raise errors.TypingError("cannot modify a read-only array")
            return signature(args[0], *args)

    return InplaceTemplate


for _op, _plain in _INPLACE_OPERATORS.items():
    _inplace_template(_op, _plain)


@registry.register_global(operator.getitem)
class ExpressionGetItem(AbstractTemplate):
    """Typing of ``expression[index]``: one element, for a full index."""

    def generic(self, args, kws):
        """Type a call; only integer indices for every axis are accepted."""
        if len(args) != 2 or not isinstance(args[0], VulkanExpr):
            return None
        expr, index = args
        indices = index if isinstance(index, types.BaseTuple) else (index,)
        if len(indices) != expr.ndim or not all(
            isinstance(i, types.Integer) for i in indices
        ):
            raise errors.TypingError(
                f"an array expression can only be indexed with {expr.ndim} integers"
            )
        return signature(expr.dtype, *args)


@registry.register_global(operator.setitem)
class ExpressionSetItem(AbstractTemplate):
    """Typing of ``array[index] = expression``."""

    def generic(self, args, kws):
        """Type a call."""
        if len(args) != 3 or not isinstance(args[2], VulkanExpr):
            return None
        if not isinstance(args[0], VulkanArray):
            return None
        if not args[0].mutable:
            raise errors.TypingError("cannot modify a read-only array")
        return signature(types.none, *args)


@registry.register_attr
class ExpressionAttributes(AttributeTemplate):
    """Attributes of array expressions: ``shape``, ``size``, ``ndim``."""

    key = VulkanExpr

    def resolve_shape(self, ty):
        return types.UniTuple(types.intp, ty.ndim)

    def resolve_size(self, ty):
        return types.intp

    def resolve_ndim(self, ty):
        return types.intp


@registry.register_global(len)
class ExpressionLen(AbstractTemplate):
    """Typing of ``len(expression)``."""

    def generic(self, args, kws):
        """Type a call."""
        if len(args) == 1 and isinstance(args[0], VulkanExpr) and args[0].ndim:
            return signature(types.intp, *args)
        return None


@registry.register_global(operator.getitem)
@registry.register_global(operator.setitem)
class FancyIndexing(AbstractTemplate):
    """Reject indexing with arrays (masks, index lists) with a clear message."""

    def generic(self, args, kws):
        """Raise for an array or expression used as an index."""
        if len(args) < 2 or not isinstance(args[0], VulkanArray):
            return None
        index = args[1]
        parts = index if isinstance(index, types.BaseTuple) else (index,)
        if any(isinstance(p, (types.Array, VulkanExpr)) for p in parts):
            raise errors.TypingError(
                "indexing with arrays (boolean masks, lists of indices) is not "
                "supported in Vulkan kernels: it would create an array of run-time "
                "size. Loop over the elements instead."
            )
        return None


# -- subgroups ----------------------------------------------------------------


def _subgroup_value(ty):
    """Whether subgroups can combine or exchange values of a type."""
    return isinstance(ty, (types.Integer, types.Float)) and ty.bitwidth in (32, 64)


def _subgroup_template(stub, typer):
    """Register the typing of a subgroup function.

    Parameters
    ----------
    stub : function
        The function in `numba_vulkan.stubs.subgroup`.
    typer : callable
        Takes the argument types and returns the result type, or ``None``
        if they do not fit.
    """

    @registry.register_global(stub)
    class SubgroupTemplate(AbstractTemplate):
        __doc__ = f"Typing of ``subgroup.{stub.__name__}``."

        def generic(self, args, kws):
            if kws:
                return None
            result = typer(*args)
            unsupported = args and isinstance(args[0], types.Number)
            if result is None and unsupported and not _subgroup_value(args[0]):
                raise errors.TypingError(
                    f"subgroup.{stub.__name__}() takes 32- or 64-bit integers "
                    f"or floats, not {args[0]}"
                )
            return None if result is None else signature(result, *args)

    return SubgroupTemplate


def _no_arguments(result):
    """A typer for functions without arguments."""
    return lambda *args: None if args else result


def _same(*args):
    """The type of a single value argument."""
    return args[0] if len(args) == 1 and _subgroup_value(args[0]) else None


def _with_lane(*args):
    """The type of the value of a value-and-lane call."""
    if len(args) == 2 and _subgroup_value(args[0]):
        return args[0] if isinstance(args[1], types.Integer) else None
    return None


def _predicate(result):
    """A typer for functions of one boolean."""

    def typer(*args):
        if len(args) == 1 and isinstance(args[0], types.Boolean):
            return result
        return None

    return typer


for _name in ("size", "lane", "id", "count"):
    _subgroup_template(getattr(stubs.subgroup, _name), _no_arguments(types.int32))
for _op in ("sum", "prod", "min", "max"):
    for _name in (_op, f"inclusive_{_op}", f"exclusive_{_op}"):
        _subgroup_template(getattr(stubs.subgroup, _name), _same)
_subgroup_template(stubs.subgroup.broadcast_first, _same)
for _name in ("broadcast", "shuffle", "shuffle_xor", "shuffle_up", "shuffle_down"):
    _subgroup_template(getattr(stubs.subgroup, _name), _with_lane)
_subgroup_template(stubs.subgroup.any, _predicate(types.boolean))
_subgroup_template(stubs.subgroup.all, _predicate(types.boolean))
_subgroup_template(stubs.subgroup.elect, _no_arguments(types.boolean))
_subgroup_template(stubs.subgroup.ballot, _predicate(types.UniTuple(types.uint32, 4)))
_subgroup_template(stubs.subgroup.ballot_count, _predicate(types.uint32))


# -- records ------------------------------------------------------------------


def _field(record, name):
    """Type of a field of a record, or a ``TypingError`` naming the fields."""
    if name not in record.record.fields:
        raise errors.TypingError(
            f"the record has no field '{name}', only {', '.join(record.record.fields)}"
        )
    return record.field(name)[0]


@registry.register_attr
class RecordAttributes(AttributeTemplate):
    """Typing of ``record.field``, to read and to assign."""

    key = VulkanRecord

    def generic_resolve(self, record, attr):
        """The type of a field.

        Returns
        -------
        numba.types.Type or None
        """
        if attr in record.record.fields:
            return record.field(attr)[0]
        return None


@registry.register_global(operator.getitem)
class RecordGetItem(AbstractTemplate):
    """Typing of ``record["field"]``."""

    def generic(self, args, kws):
        """Type a call."""
        record = len(args) == 2 and isinstance(args[0], VulkanRecord)
        if record and isinstance(args[1], types.StringLiteral):
            return signature(_field(args[0], args[1].literal_value), *args)
        return None


@registry.register
class RecordStaticGetItem(AbstractTemplate):
    """Typing of ``record["field"]`` with a constant name."""

    key = "static_getitem"

    def generic(self, args, kws):
        """Type a call."""
        if (
            len(args) == 2
            and isinstance(args[0], VulkanRecord)
            and isinstance(args[1], str)
        ):
            name = types.literal(args[1])
            return signature(_field(args[0], args[1]), args[0], name)
        return None


@registry.register_global(operator.setitem)
class RecordSetItem(AbstractTemplate):
    """Typing of ``record["field"] = value``."""

    def generic(self, args, kws):
        """Type a call."""
        record = len(args) == 3 and isinstance(args[0], VulkanRecord)
        if not (record and isinstance(args[1], types.StringLiteral)):
            return None
        field = _field(args[0], args[1].literal_value)
        if self.context.can_convert(args[2], field) is None:
            return None
        return signature(types.none, args[0], args[1], field)


@registry.register
class RecordStaticSetItem(AbstractTemplate):
    """Typing of ``record["field"] = value`` with a constant name."""

    key = "static_setitem"

    def generic(self, args, kws):
        """Type a call."""
        record = len(args) == 3 and isinstance(args[0], VulkanRecord)
        if not (record and isinstance(args[1], str)):
            return None
        field = _field(args[0], args[1])
        if self.context.can_convert(args[2], field) is None:
            return None
        return signature(types.none, args[0], types.literal(args[1]), field)
