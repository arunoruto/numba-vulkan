"""Numba types specific to the Vulkan target."""

from numba.core import types


class VulkanArray(types.Array):
    """An array backed by the storage buffer at a fixed descriptor binding.

    Vulkan shaders have no general pointers, so the buffer an array lives
    in is part of its type rather than of its runtime value. A function
    taking arrays is therefore compiled once per combination of bindings.

    Parameters
    ----------
    dtype : numba.types.Type
        Element type.
    ndim : int
        Number of dimensions.
    layout : {'C', 'A'}
        Memory layout: ``'C'`` for contiguous arrays, ``'A'`` for views
        with arbitrary strides.
    binding : int
        Descriptor binding of the buffer holding the data.
    readonly : bool, optional
        Whether the array is immutable.
    aligned : bool, optional
        Whether the data is aligned.
    half : bool, optional
        Whether the buffer holds ``float16`` values; the elements are then
        ``float32`` to the kernel.

    Attributes
    ----------
    binding : int
        Descriptor binding of the buffer holding the data.
    half : bool
        As given.
    """

    def __init__(
        self, dtype, ndim, layout, binding, readonly=False, aligned=True, half=False
    ):
        self.binding = binding
        self.half = half
        storage = ", float16" if half else ""
        name = f"vkarray({dtype}{storage}, {ndim}d, {layout}, binding={binding})"
        super().__init__(
            dtype, ndim, layout, readonly=readonly, name=name, aligned=aligned
        )

    def copy(self, dtype=None, ndim=None, layout=None, readonly=None):
        """Return a copy of this type with some properties replaced.

        Unlike `numba.types.Array.copy`, the result keeps the binding, so
        types that Numba derives from an array still refer to its buffer.

        Parameters
        ----------
        dtype : numba.types.Type, optional
            New element type.
        ndim : int, optional
            New number of dimensions.
        layout : str, optional
            New memory layout.
        readonly : bool, optional
            New mutability.

        Returns
        -------
        VulkanArray
        """
        return VulkanArray(
            dtype=self.dtype if dtype is None else dtype,
            ndim=self.ndim if ndim is None else ndim,
            layout=self.layout if layout is None else layout,
            binding=self.binding,
            readonly=not self.mutable if readonly is None else readonly,
            aligned=self.aligned,
            half=self.half,
        )

    @property
    def iterator_type(self):
        """Type of ``iter(array)``.

        Returns
        -------
        VulkanArrayIterator
        """
        return VulkanArrayIterator(self)

    @property
    def key(self):
        """Identity of the type: Numba's array key plus the binding.

        Returns
        -------
        tuple
        """
        return super().key + (self.binding, self.half)


class HalfArray(types.Array):
    """Type of a ``float16`` array argument before it is given a binding.

    Its elements are ``float32``; see `VulkanArray`.
    """

    def __init__(self, ndim, layout, readonly=False):
        super().__init__(
            types.float32,
            ndim,
            layout,
            readonly=readonly,
            name=f"halfarray({ndim}d, {layout})",
        )

    def copy(self, dtype=None, ndim=None, layout=None, readonly=None):
        """Return a copy with some properties replaced (but not the dtype)."""
        return HalfArray(
            self.ndim if ndim is None else ndim,
            self.layout if layout is None else layout,
            not self.mutable if readonly is None else readonly,
        )

    @property
    def key(self):
        """Identity of the type."""
        return super().key + ("half",)


class VulkanArrayIterator(types.ArrayIterator):
    """Iterator over the first axis of a `VulkanArray`.

    A type of its own, so that iteration can be lowered without the data
    pointer that Numba's array iterator relies on.
    """


class VulkanDispatcherType(types.Dispatcher):
    """Type of a Vulkan-compiled function used inside another one.

    It exists so that constants of this type can be lowered to a dummy
    value: calls are resolved at compile time, and the default lowering of
    `numba.types.Dispatcher` would embed a host address.
    """


class VulkanExpr(types.Type):
    """An array expression that is evaluated element by element.

    Kernels cannot allocate memory, so ``a * 2`` or ``np.sqrt(a) + b`` on
    arrays does not compute a new array. Its value is the operation and its
    operands; an element is computed when it is read, by indexing, a
    reduction, iteration or an assignment to a slice.

    Parameters
    ----------
    op : callable
        The operator or ufunc, such as ``operator.add`` or ``np.sqrt``.
    operands : tuple of numba.types.Type
        Types of the operands: arrays, expressions or scalars.
    dtype : numba.types.Type
        Type of an element of the result.
    ndim : int
        Number of dimensions of the result, after broadcasting.

    Attributes
    ----------
    op, operands, dtype, ndim
        As given.
    """

    def __init__(self, op, operands, dtype, ndim):
        self.op = op
        self.operands = tuple(operands)
        self.dtype = dtype
        self.ndim = ndim
        name = getattr(op, "__name__", str(op))
        super().__init__(f"vkexpr({name}, {', '.join(map(str, operands))})")

    @property
    def key(self):
        """Identity of the type: the operation and the operand types.

        Returns
        -------
        tuple
        """
        return self.op, self.operands
