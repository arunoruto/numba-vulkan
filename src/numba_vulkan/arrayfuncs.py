"""Reductions over arrays, written in Python on top of element access.

Numba's own ``sum``, ``min``... walk the array through its data pointer,
which arrays on this target do not have. The versions here are overloads
for the ``vulkan`` target that read the elements by position instead (see
`numba_vulkan.vkimpl.flat_item`), so they work for arrays and views of any
dimensionality. They reduce over all elements; ``axis`` is not supported.

Each is available as a method (``a.sum()``) and as the NumPy function
(``np.sum(a)``).
"""

import numpy as np
from numba.core import types
from numba.core.extending import overload, overload_method

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


def _sum(a):
    """Build ``sum``: the sum of all elements."""
    acc = _accumulator(a.dtype)

    def impl(a):
        total = acc(0)
        for k in range(a.size):
            total += flat_item(a, k)
        return total

    return impl


def _prod(a):
    """Build ``prod``: the product of all elements."""
    acc = _accumulator(a.dtype)

    def impl(a):
        total = acc(1)
        for k in range(a.size):
            total *= flat_item(a, k)
        return total

    return impl


def _mean(a):
    """Build ``mean``: float64 for integers, the element type otherwise."""
    acc = a.dtype if isinstance(a.dtype, types.Float) else types.float64

    def impl(a):
        total = acc(0)
        for k in range(a.size):
            total += flat_item(a, k)
        return total / acc(a.size)

    return impl


def _min(a):
    """Build ``min``; raises ``ValueError`` for an empty array."""

    def impl(a):
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

    def impl(a):
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

    def impl(a):
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

    def impl(a):
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

    def impl(a):
        for k in range(a.size):
            if flat_item(a, k):
                return True
        return False

    return impl


def _all(a):
    """Build ``all``: whether every element is non-zero."""

    def impl(a):
        for k in range(a.size):
            if not flat_item(a, k):
                return False
        return True

    return impl


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

    def typer(a):
        """Select the implementation for arrays and expressions of this target."""
        if isinstance(a, (VulkanArray, VulkanExpr)):
            return factory(a)
        return None

    overload_method(VulkanArray, name, target=TARGET)(typer)
    overload_method(VulkanExpr, name, target=TARGET)(typer)
    for function in functions:
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
