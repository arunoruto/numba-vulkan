"""Typing of Vulkan-specific functions."""

import math

from numba.core import errors, types
from numba.core.typing.templates import (
    AbstractTemplate,
    CallableTemplate,
    ConcreteTemplate,
    Registry,
    signature,
)

from numba_vulkan import stubs
from numba_vulkan.buffers import SHARED_BASE, shared_sizes
from numba_vulkan.vktypes import VulkanArray

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


@registry.register_global(stubs.shared.array)
class SharedArray(CallableTemplate):
    """Typing of ``shared.array(shape, dtype)``.

    The call carries a third, hidden argument: a literal that identifies the
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

        def typer(shape, dtype, _vulkan_site=None):
            dims, dtype = _literal_shape(shape), _dtype(dtype)
            if dims is None or dtype is None:
                return None
            if not isinstance(_vulkan_site, types.IntegerLiteral):
                return None
            if any(n <= 0 for n in dims):
                raise errors.TypingError("shared arrays need a positive constant shape")
            binding = SHARED_BASE + _vulkan_site.literal_value
            shared_sizes[binding] = math.prod(dims)
            return VulkanArray(dtype, len(dims), "C", binding)

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
