"""Reductions over arrays, written in Python on top of element access.

Numba's own ``sum``, ``min``... walk the array through its data pointer,
which arrays on this target do not have. The versions here are overloads
for the ``vulkan`` target that read the elements by position instead (see
`numba_vulkan.vkimpl.flat_item`), so they work for arrays and views of any
dimensionality. They reduce over all elements, or, with a constant
``axis``, along one axis: kernels cannot allocate the resulting array, so
``a.sum(axis=0)`` is an array expression (see `AxisReduction`), whose
elements are computed where they are read.

Each is available as a method (``a.sum()``) and as the NumPy function
(``np.sum(a)``).
"""

import functools
from dataclasses import dataclass

import numpy as np
from numba.core import cgutils, errors, types
from numba.core.extending import intrinsic, overload, overload_method

from numba_vulkan.vkimpl import flat_item
from numba_vulkan.vktypes import VulkanArray, VulkanExpr

TARGET = "vulkan"


def _accumulator(dtype):
    """Type in which sums and products of ``dtype`` elements are formed.

    As in NumPy, integers and booleans widen to the platform integer.

    Parameters
    ----------
    dtype : numba.types.Type
        Element type.

    Returns
    -------
    numba.types.Type
    """
    if isinstance(dtype, types.Boolean):
        return types.intp
    if isinstance(dtype, types.Integer):
        return types.intp if dtype.signed else types.uintp
    return dtype


def _check_whole_axis(a, axis):
    """Raise unless `axis` is None, or a valid axis of a 1-d array."""


@overload(_check_whole_axis, target=TARGET)
def _check_whole_axis_impl(a, axis):
    """Build `_check_whole_axis`: reductions over all elements take it."""
    if axis is None or isinstance(axis, (types.NoneType, types.Omitted)):
        return lambda a, axis: None

    def impl(a, axis):
        if axis != 0 and axis != -1:
            raise ValueError("axis is out of bounds for an array of dimension 1")

    return impl


def _sum(a):
    """Build ``sum``: the sum of all elements."""
    acc = _accumulator(a.dtype)

    def impl(a, axis=None):
        _check_whole_axis(a, axis)
        total = acc(0)
        for k in range(a.size):
            total += flat_item(a, k)
        return total

    return impl


def _prod(a):
    """Build ``prod``: the product of all elements."""
    acc = _accumulator(a.dtype)

    def impl(a, axis=None):
        _check_whole_axis(a, axis)
        total = acc(1)
        for k in range(a.size):
            total *= flat_item(a, k)
        return total

    return impl


def _mean(a):
    """Build ``mean``: float64 for integers, the element type otherwise."""
    acc = a.dtype if isinstance(a.dtype, types.Float) else types.float64

    def impl(a, axis=None):
        _check_whole_axis(a, axis)
        total = acc(0)
        for k in range(a.size):
            total += flat_item(a, k)
        return total / acc(a.size)

    return impl


def _min(a):
    """Build ``min``; raises ``ValueError`` for an empty array."""

    def impl(a, axis=None):
        _check_whole_axis(a, axis)
        if a.size == 0:
            raise ValueError("zero-size array has no minimum")
        best = flat_item(a, 0)
        for k in range(1, a.size):
            value = flat_item(a, k)
            # written so that a NaN, once found, is kept, as in NumPy
            if value < best or value != value:  # noqa: PLR0124 - NaN, for any type
                best = value
        return best

    return impl


def _max(a):
    """Build ``max``; raises ``ValueError`` for an empty array."""

    def impl(a, axis=None):
        _check_whole_axis(a, axis)
        if a.size == 0:
            raise ValueError("zero-size array has no maximum")
        best = flat_item(a, 0)
        for k in range(1, a.size):
            value = flat_item(a, k)
            if value > best or value != value:  # noqa: PLR0124 - NaN, for any type
                best = value
        return best

    return impl


def _argmin(a):
    """Build ``argmin``: row-major position of the first minimum."""

    def impl(a, axis=None):
        _check_whole_axis(a, axis)
        if a.size == 0:
            raise ValueError("attempt to get argmin of an empty sequence")
        best, where = flat_item(a, 0), 0
        for k in range(1, a.size):
            value = flat_item(a, k)
            if value < best:
                best, where = value, k
        return where

    return impl


def _argmax(a):
    """Build ``argmax``: row-major position of the first maximum."""

    def impl(a, axis=None):
        _check_whole_axis(a, axis)
        if a.size == 0:
            raise ValueError("attempt to get argmax of an empty sequence")
        best, where = flat_item(a, 0), 0
        for k in range(1, a.size):
            value = flat_item(a, k)
            if value > best:
                best, where = value, k
        return where

    return impl


def _any(a):
    """Build ``any``: whether some element is non-zero."""

    def impl(a, axis=None):
        _check_whole_axis(a, axis)
        for k in range(a.size):
            if flat_item(a, k):
                return True
        return False

    return impl


def _all(a):
    """Build ``all``: whether every element is non-zero."""

    def impl(a, axis=None):
        _check_whole_axis(a, axis)
        for k in range(a.size):
            if not flat_item(a, k):
                return False
        return True

    return impl


@dataclass(frozen=True)
class AxisReduction:
    """The operation of an array expression that reduces along an axis.

    ``a.sum(axis=k)`` is a `numba_vulkan.vktypes.VulkanExpr` with this as
    its operation, and ``a`` and the axis as its operands; the axis is a
    run-time value, as in Numba's ``sum(axis=...)`` on the CPU, since the
    number of dimensions of the result does not depend on it. Its shape is
    that of ``a`` without the axis; each element is a loop along the axis,
    written in Python and compiled where the element is read (see
    `numba_vulkan.vkimpl.expression_element`).

    Parameters
    ----------
    name : str
        The reduction: a key of ``_REDUCTIONS``.
    acc : numba.types.Type
        Type in which sums, products and means are formed.
    """

    name: str
    acc: types.Type
    # Tells numba_vulkan.vkimpl how to compute the shape and the elements.
    reduces_axis = True

    @property
    def __name__(self):
        """Name used in the expression's type name."""
        return f"{self.name}(axis)"

    def element_function(self, ndim):
        """The function that computes one element.

        Parameters
        ----------
        ndim : int
            Dimensions of the operand.

        Returns
        -------
        function
            Takes the operand, the axis (made non-negative, and checked
            when the shape was computed) and one index per dimension of
            the result.
        """
        return _element_function(self, ndim)


# Loops that compute one element of a reduction along an axis; ``{item}``
# reads the element at position ``k`` along it, ``ACC`` is the accumulator
# type.
_ELEMENT_BODIES = {
    "sum": """
    total = ACC(0)
    for k in range(n):
        total += {item}
    return total""",
    "prod": """
    total = ACC(1)
    for k in range(n):
        total *= {item}
    return total""",
    "mean": """
    total = ACC(0)
    for k in range(n):
        total += {item}
    return total / ACC(n)""",
    "min": """
    if n == 0:
        raise ValueError("zero-size array has no minimum")
    k = 0
    best = {item}
    for k in range(1, n):
        value = {item}
        if value < best or value != value:  # noqa: PLR0124 - NaN
            best = value
    return best""",
    "max": """
    if n == 0:
        raise ValueError("zero-size array has no maximum")
    k = 0
    best = {item}
    for k in range(1, n):
        value = {item}
        if value > best or value != value:  # noqa: PLR0124 - NaN
            best = value
    return best""",
    "argmin": """
    if n == 0:
        raise ValueError("attempt to get argmin of an empty sequence")
    k = 0
    best, where = {item}, 0
    for k in range(1, n):
        value = {item}
        if value < best:
            best, where = value, k
    return where""",
    "argmax": """
    if n == 0:
        raise ValueError("attempt to get argmax of an empty sequence")
    k = 0
    best, where = {item}, 0
    for k in range(1, n):
        value = {item}
        if value > best:
            best, where = value, k
    return where""",
    "any": """
    for k in range(n):
        if {item}:
            return True
    return False""",
    "all": """
    for k in range(n):
        if not {item}:
            return False
    return True""",
}


@functools.cache
def _element_function(op, ndim):
    """Write and define the function of `AxisReduction.element_function`."""
    outer = [f"i{d}" for d in range(ndim - 1)]
    position = []
    for d in range(ndim):
        # Index along operand axis d: an index of the result before the
        # reduced axis, k on it, and the next index of the result after it.
        if d == 0:
            position.append("(k if axis == 0 else i0)")
        elif d == ndim - 1:
            position.append(f"(k if axis == {d} else i{d - 1})")
        else:
            position.append(
                f"(i{d} if axis > {d} else (k if axis == {d} else i{d - 1}))"
            )
    body = _ELEMENT_BODIES[op.name].format(item=f"a[{', '.join(position)}]")
    source = f"def element(a, axis, {', '.join(outer)}):\n    n = a.shape[axis]{body}\n"
    scope = {"ACC": op.acc}
    exec(source, scope)  # noqa: S102 - generated from the table above
    return scope["element"]


def _result_type(name, dtype):
    """Element type of a reduction of `dtype` elements."""
    if name in ("sum", "prod"):
        return _accumulator(dtype)
    if name == "mean":
        return dtype if isinstance(dtype, types.Float) else types.float64
    if name in ("argmin", "argmax"):
        return types.intp
    if name in ("any", "all"):
        return types.boolean
    return dtype


@functools.cache
def _axis_reducer(name):
    """An intrinsic that makes the expression of a reduction along an axis."""

    @intrinsic(target=TARGET)
    def reduce_along(typingctx, a, axis):
        acc = _result_type("mean" if name == "mean" else "sum", a.dtype)
        op = AxisReduction(name, acc)
        dtype = _result_type(name, a.dtype)
        exprty = VulkanExpr(op, (a, types.intp), dtype, a.ndim - 1)

        def codegen(context, builder, sig, args):
            proxy = cgutils.create_struct_proxy(sig.return_type)(context, builder)
            proxy.operand0 = args[0]
            proxy.operand1 = context.cast(builder, args[1], sig.args[1], types.intp)
            return proxy._getvalue()

        return exprty(a, axis), codegen

    return reduce_along


_REDUCTIONS = {
    "sum": (_sum, [np.sum]),
    "prod": (_prod, [np.prod]),
    "mean": (_mean, [np.mean]),
    "min": (_min, [np.min, min]),
    "max": (_max, [np.max, max]),
    "argmin": (_argmin, [np.argmin]),
    "argmax": (_argmax, [np.argmax]),
    "any": (_any, [np.any]),
    "all": (_all, [np.all]),
}


def _register(name, factory, functions):
    """Register a reduction as an array method and for its functions.

    Parameters
    ----------
    name : str
        Name of the array method.
    factory : callable
        Builds the implementation for an array type.
    functions : list of callable
        Functions that mean the same when called with one array.
    """

    def typer(a, axis=None):
        """Select the implementation for arrays and expressions of this target."""
        if not isinstance(a, (VulkanArray, VulkanExpr)):
            return None
        if axis is None or isinstance(axis, (types.NoneType, types.Omitted)):
            return factory(a)
        if not isinstance(axis, types.Integer):
            raise errors.TypingError(f"{name}(axis=...) takes an integer axis")
        ndim = a.ndim
        if ndim == 1:
            return factory(a)  # a scalar, as in NumPy; it checks the axis
        reduce_along = _axis_reducer(name)

        def impl(a, axis=None):
            # Checked where the shape is computed: raising here would make
            # the array of the expression depend on the branch taken.
            return reduce_along(a, axis)

        return impl

    def builtin_typer(a):
        """``min(a)`` and ``max(a)`` on an array: over all elements."""
        if isinstance(a, (VulkanArray, VulkanExpr)):
            return factory(a)  # its axis keeps its default
        return None

    overload_method(VulkanArray, name, target=TARGET)(typer)
    overload_method(VulkanExpr, name, target=TARGET)(typer)
    for function in functions:
        if function in (min, max):
            overload(function, target=TARGET, strict=False)(builtin_typer)
        else:
            overload(function, target=TARGET)(typer)


for _name, (_factory, _functions) in _REDUCTIONS.items():
    _register(_name, _factory, _functions)


@overload(np.dot, target=TARGET)
def _dot(a, b):
    """Build ``np.dot`` for two 1-d arrays."""
    if not (
        isinstance(a, (VulkanArray, VulkanExpr))
        and isinstance(b, (VulkanArray, VulkanExpr))
    ):
        return None
    if a.ndim != 1 or b.ndim != 1:
        return None
    acc = np.result_type(str(a.dtype), str(b.dtype)).type

    def impl(a, b):
        if a.shape[0] != b.shape[0]:
            raise ValueError("incompatible array sizes for np.dot(a, b)")
        total = acc(0)
        for k in range(a.shape[0]):
            total += flat_item(a, k) * flat_item(b, k)
        return total

    return impl
