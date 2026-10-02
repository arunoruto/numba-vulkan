"""``@vectorize`` and ``@guvectorize`` for the Vulkan target.

Numba's decorators look up the class that builds a ufunc in a registry per
target; this module registers ``"vulkan"`` there, so that ::

    @numba.vectorize(["float32(float32, float32)"], target="vulkan")
    def add(a, b):
        return a + b

compiles the scalar function for Vulkan and returns a ufunc-like object
that runs it over whole arrays on the device. As with ``target="cuda"``,
the result is not a real `numpy.ufunc`: it is called directly, takes NumPy
arrays, device arrays and scalars, broadcasts them as NumPy does, and
returns a NumPy array, or a device array when any input is one.

Each call runs one kernel, generated from a template, in which every
invocation computes one element (for ``vectorize``) or one loop iteration
of the core function (for ``guvectorize``). The kernels are compiled per
signature and array dimensionality and cached like any other kernel.

Without explicit signatures, the types are taken from the arguments of
each call, with Python scalars treated as NumPy 2 does (``x * 2.0`` keeps
the type of a ``float32`` array ``x``).
"""

import numpy as np
from numba.core import sigutils, types
from numba.np import numpy_support
from numba.np.ufunc.decorators import GUVectorize, Vectorize
from numba.np.ufunc.sigparse import parse_signature

from numba_vulkan import runtime
from numba_vulkan.dispatcher import jit
from numba_vulkan.stubs import global_id

TARGET = "vulkan"


# -- arguments ----------------------------------------------------------------


class _Argument:
    """One argument of a call, as it will be handed to the kernel.

    Attributes
    ----------
    value : numpy.ndarray, DeviceArray or scalar
        The value.
    weak : bool
        Whether it is a Python scalar, whose type adapts to the others.
    """

    def __init__(self, value):
        self.weak = isinstance(value, (bool, int, float, complex))
        if isinstance(value, runtime.DeviceArray):
            self.value = value
        elif isinstance(value, np.ndarray) or not np.isscalar(value):
            self.value = np.asarray(value)
        else:
            self.value = value

    @property
    def is_array(self):
        """Whether the argument is an array (on the host or a device)."""
        return isinstance(self.value, (np.ndarray, runtime.DeviceArray))

    @property
    def shape(self):
        """Shape of the argument; ``()`` for a scalar."""
        return self.value.shape if self.is_array else ()

    @property
    def dtype(self):
        """Element type of an array or type of a NumPy scalar."""
        return np.asarray(self.value).dtype if not self.is_array else self.value.dtype

    def converted(self, dtype):
        """The value with elements of the given type.

        Host arrays and scalars are converted on the host; a device array
        of another type goes through the host as well.
        """
        dtype = np.dtype(dtype)
        if isinstance(self.value, runtime.DeviceArray):
            if self.value.dtype == dtype:
                return self.value
            host = self.value.copy_to_host().astype(dtype)
            return runtime.to_device(host, self.value.device)
        if self.is_array:
            return self.value if self.value.dtype == dtype else self.value.astype(dtype)
        return dtype.type(self.value)


def _natural_types(arguments):
    """Element types of the arguments, for functions without signatures.

    Python scalars take the type NumPy would give them next to the array
    arguments.
    """
    strong = [a.dtype for a in arguments if not a.weak]
    out = []
    for argument in arguments:
        if argument.weak:
            out.append(np.result_type(*strong, argument.value))
        else:
            out.append(argument.dtype)
    return out


def _accepts(argument, dtype):
    """Whether an argument may be converted to `dtype` for a signature."""
    dtype = np.dtype(dtype)
    if argument.weak:
        value = argument.value
        if isinstance(value, bool):
            return True
        if isinstance(value, int):
            return dtype.kind in "iufc" and (dtype.kind != "u" or value >= 0)
        if isinstance(value, float):
            return dtype.kind in "fc"
        return dtype.kind == "c"
    return np.can_cast(argument.dtype, dtype, "safe")


def _select(signatures, arguments, name):
    """The first signature whose argument types the arguments fit.

    Parameters
    ----------
    signatures : list of tuple
        ``(argument dtypes, signature)`` pairs.
    arguments : list of _Argument
        The input arguments.
    name : str
        Name of the function, for the error message.

    Returns
    -------
    tuple
        The matching pair.

    Raises
    ------
    TypeError
        If no signature fits.
    """
    for dtypes, signature in signatures:
        if all(_accepts(a, d) for a, d in zip(arguments, dtypes)):
            return dtypes, signature
    given = ", ".join(
        str(a.dtype) if not a.weak else type(a.value).__name__ for a in arguments
    )
    raise TypeError(f"'{name}' has no signature for arguments of type ({given})")


def _device_of(arguments, device):
    """The device to run on: the given one, or that of a device array."""
    if device is not None:
        return runtime.get_device(device)
    for argument in arguments:
        if isinstance(argument.value, runtime.DeviceArray):
            return argument.value.device
    return runtime.get_device()


def _on_device(arguments):
    """Whether results should be device arrays: any input is one."""
    return any(isinstance(a.value, runtime.DeviceArray) for a in arguments)


def _allocate(shape, dtype, device, on_device):
    """An output array, on the device or on the host."""
    if on_device:
        return runtime.device_array(shape, dtype, device)
    return np.empty(shape, dtype=dtype)


def _compile_source(source, name, scope):
    """Turn generated kernel source into a Vulkan kernel."""
    namespace = dict(scope)
    exec(compile(source, f"<{name}>", "exec"), namespace)  # noqa: S102
    return jit(namespace[name])


def _broadcast_index(array, array_ndim, index_names):
    """Index expression that broadcasts an array against the loop indices.

    The array's axes are aligned with the last of `index_names`; an axis of
    extent 1 is always read at position 0.
    """
    offset = len(index_names) - array_ndim
    parts = [
        f"({index_names[offset + d]} if {array}.shape[{d}] != 1 else 0)"
        for d in range(array_ndim)
    ]
    return ", ".join(parts)


def _unravel(lines, flat, shape_of, ndim, prefix):
    """Append code that splits a flat position into per-axis indices."""
    names = [f"{prefix}{d}" for d in range(ndim)]
    if ndim:
        lines.append(f"        __r__ = {flat}")
        for d in reversed(range(1, ndim)):
            lines.append(f"        {names[d]} = __r__ % {shape_of}.shape[{d}]")
            lines.append(f"        __r__ = __r__ // {shape_of}.shape[{d}]")
        lines.append(f"        {names[0]} = __r__")
    return names


# -- vectorize ----------------------------------------------------------------


class VulkanVectorize:
    """Builder of an element-wise ufunc, used by ``vectorize(target="vulkan")``.

    Parameters
    ----------
    func : function
        The scalar function.
    identity : object, optional
        Accepted for compatibility; reductions are not supported.
    cache : bool, optional
        Accepted for compatibility; kernels are always cached on disk.
    targetoptions : dict, optional
        Options for `numba_vulkan.jit`, such as ``fastmath``, and
        ``device`` for the default device.
    """

    def __init__(self, func, identity=None, cache=False, targetoptions=None):
        self.pyfunc = func
        self.options = dict(targetoptions or {})
        self.signatures = []

    def add(self, sig):
        """Add a signature such as ``"float32(float32, float32)"``.

        Parameters
        ----------
        sig : str or numba.core.typing.Signature
            The signature.
        """
        args, restype = sigutils.normalize_signature(sig)
        self.signatures.append((args, restype))

    def disable_compile(self):
        """Accepted for compatibility: with signatures, only those are used."""

    def build_ufunc(self):
        """Create the ufunc.

        Returns
        -------
        VulkanUFunc
        """
        return VulkanUFunc(self.pyfunc, self.signatures, self.options)


class VulkanUFunc:
    """An element-wise function over arrays, run on a Vulkan device.

    Created by ``numba.vectorize(..., target="vulkan")``.

    Parameters
    ----------
    pyfunc : function
        The scalar function.
    signatures : list of tuple
        ``(argument types, return type)`` pairs; empty to compile for the
        types of each call.
    options : dict
        Options for `numba_vulkan.jit`, and ``device``.

    Attributes
    ----------
    nin, nout : int
        Number of inputs and outputs.

    Examples
    --------
    >>> @numba.vectorize(["float32(float32, float32)"], target="vulkan")
    ... def hypot(a, b):
    ...     return math.sqrt(a * a + b * b)
    >>> hypot(np.float32([3, 5]), np.float32(4))
    array([5.     , 6.40312], dtype=float32)
    """

    def __init__(self, pyfunc, signatures, options):
        self.pyfunc = pyfunc
        self.__name__ = pyfunc.__name__
        self.__doc__ = pyfunc.__doc__
        options = dict(options)
        self._device = options.pop("device", None)
        self._core = jit(**options)(pyfunc)
        self._options = options
        self.nin = pyfunc.__code__.co_argcount
        self.nout = 1
        self._signatures = [
            (
                [numpy_support.as_dtype(t) for t in args],
                (args, restype),
            )
            for args, restype in signatures
        ]
        self._kernels = {}

    @property
    def types(self):
        """The signatures in NumPy's notation, such as ``"ff->f"``.

        Returns
        -------
        list of str
        """
        out = []
        for dtypes, (_, restype) in self._signatures:
            ret = numpy_support.as_dtype(restype)
            out.append("".join(d.char for d in dtypes) + "->" + ret.char)
        return out

    def __repr__(self):
        return f"<Vulkan ufunc '{self.__name__}'>"

    def __call__(self, *args, out=None, device=None):
        """Apply the function element-wise.

        Parameters
        ----------
        *args : array_like, DeviceArray or scalar
            The inputs; they are broadcast against each other.
        out : numpy.ndarray or DeviceArray, optional
            Where to write the result; it must have the broadcast shape.
        device : int, str or Device, optional
            Device to run on. By default that of a device array among the
            inputs, or the selected device.

        Returns
        -------
        numpy.ndarray, DeviceArray or scalar
            The result; a device array if any input is one, a NumPy scalar
            if all inputs are scalars. `out` itself when given.

        Raises
        ------
        TypeError
            For a wrong number of inputs, or types no signature accepts.
        """
        if len(args) != self.nin:
            raise TypeError(
                f"'{self.__name__}' takes {self.nin} inputs, got {len(args)}"
            )
        arguments = [_Argument(a) for a in args]
        placed = arguments + ([_Argument(out)] if out is not None else [])
        target = _device_of(placed, device if device is not None else self._device)
        shape = np.broadcast_shapes(*(a.shape for a in arguments))

        if self._signatures:
            dtypes, (argtypes, restype) = _select(
                self._signatures, arguments, self.__name__
            )
        else:
            dtypes = _natural_types(arguments)
            argtypes = tuple(numpy_support.from_dtype(d) for d in dtypes)
            restype = None
        values = [a.converted(d) for a, d in zip(arguments, dtypes)]
        # Compiling the core first fixes its return type for the kernel.
        cres = self._core.compile_device(argtypes, restype)
        result_dtype = numpy_support.as_dtype(cres.signature.return_type)

        on_device = _on_device(arguments) or isinstance(out, runtime.DeviceArray)
        if out is None and not shape and not on_device:
            # Only scalars: a scalar result, as from a NumPy ufunc.
            return self(*args, out=np.empty((), result_dtype), device=target)[()]
        if out is None:
            out = _allocate(shape, result_dtype, target, on_device)
        elif tuple(out.shape) != shape:
            raise ValueError(
                f"out has shape {tuple(out.shape)}, but the result has {shape}"
            )
        if isinstance(out, np.ndarray) and not out.flags.c_contiguous:
            result = self(*args, device=target)
            out[...] = result
            return out

        # Inputs that cover the whole result are passed as one dimension.
        flat = all(not a.is_array or a.shape == shape for a in arguments)
        if flat:
            values = [
                v.reshape(-1) if a.is_array else v for a, v in zip(arguments, values)
            ]
            target_out = out.reshape(-1)
        else:
            target_out = out
        kinds = tuple(
            (len(v.shape) if a.is_array else None) for a, v in zip(arguments, values)
        )
        kernel = self._kernel(kinds, len(target_out.shape))
        # At least one invocation, which does nothing for an empty result.
        count = max(int(np.prod(shape, dtype=np.int64)), 1)
        kernel.forall(count, device=target)(*values, target_out)
        return out

    def _kernel(self, kinds, out_ndim):
        """The kernel for inputs of the given dimensionalities.

        Parameters
        ----------
        kinds : tuple of int or None
            Number of dimensions of each input; ``None`` for a scalar.
        out_ndim : int
            Number of dimensions of the result.

        Returns
        -------
        VulkanDispatcher
        """
        key = (kinds, out_ndim)
        if key not in self._kernels:
            names = [f"a{k}" for k in range(len(kinds))]
            lines = [
                f"def __vectorized__({', '.join(names)}, __out__):",
                "    __i__ = __global_id__(0)",
                "    if __i__ < __out__.size:",
            ]
            if out_ndim == 1 and all(k in (None, 1) for k in kinds):
                index_names = ["__i__"]
            else:
                index_names = _unravel(lines, "__i__", "__out__", out_ndim, "__j")
            items = []
            for name, ndim in zip(names, kinds):
                if ndim is None:
                    items.append(name)
                else:
                    items.append(
                        f"{name}[{_broadcast_index(name, ndim, index_names)}]"
                        if ndim
                        else f"{name}[()]"
                    )
            target = f"__out__[{', '.join(index_names)}]" if out_ndim else "__out__[()]"
            lines.append(f"        {target} = __core__({', '.join(items)})")
            scope = {"__global_id__": global_id, "__core__": self._core}
            self._kernels[key] = _compile_source(
                "\n".join(lines) + "\n", "__vectorized__", scope
            )
            self._kernels[key].__name__ = f"{self.__name__}_vectorized"
        return self._kernels[key]


# -- guvectorize ---------------------------------------------------------------


class VulkanGUVectorize:
    """Builder of a generalized ufunc, used by ``guvectorize(target="vulkan")``.

    Parameters
    ----------
    func : function
        The core function; it writes its results into its output arguments.
    signature : str
        The layout, such as ``"(n),(n)->()"``.
    identity, cache, writable_args : optional
        Accepted for compatibility.
    targetoptions : dict, optional
        Options for `numba_vulkan.jit`, and ``device``.
    """

    def __init__(
        self,
        func,
        signature,
        identity=None,
        cache=False,
        targetoptions=None,
        writable_args=(),
    ):
        self.pyfunc = func
        self.layout = signature
        self.options = dict(targetoptions or {})
        self.options.pop("is_dynamic", None)
        self.signatures = []

    def add(self, sig):
        """Add a signature such as ``"void(float32[:], float32[:])"``.

        Parameters
        ----------
        sig : str or numba.core.typing.Signature
            The signature.
        """
        args, _ = sigutils.normalize_signature(sig)
        self.signatures.append(args)

    def disable_compile(self):
        """Accepted for compatibility: with signatures, only those are used."""

    def build_ufunc(self):
        """Create the generalized ufunc.

        Returns
        -------
        VulkanGUFunc
        """
        return VulkanGUFunc(self.pyfunc, self.layout, self.signatures, self.options)


class VulkanGUFunc:
    """A generalized ufunc run on a Vulkan device.

    Created by ``numba.guvectorize(..., target="vulkan")``. The core
    function is called once per element of the loop dimensions, with views
    of the core dimensions of each argument.

    Parameters
    ----------
    pyfunc : function
        The core function.
    layout : str
        The layout, such as ``"(n),(n)->()"``.
    signatures : list of tuple
        Argument types of each signature; empty to compile for the types of
        each call, which then requires the outputs to be passed.
    options : dict
        Options for `numba_vulkan.jit`, and ``device``.

    Attributes
    ----------
    nin, nout : int
        Number of inputs and outputs.

    Examples
    --------
    >>> @numba.guvectorize(["void(float32[:], float32[:], float32[:])"],
    ...                    "(n),(n)->()", target="vulkan")
    ... def dot(a, b, out):
    ...     out[0] = np.dot(a, b)
    >>> dot(np.ones((4, 3), np.float32), np.arange(3, dtype=np.float32))
    array([3., 3., 3., 3.], dtype=float32)
    """

    def __init__(self, pyfunc, layout, signatures, options):
        self.pyfunc = pyfunc
        self.__name__ = pyfunc.__name__
        self.__doc__ = pyfunc.__doc__
        options = dict(options)
        self._device = options.pop("device", None)
        self._core = jit(**options)(pyfunc)
        self.layout = layout
        self._inputs, self._outputs = parse_signature(layout)
        self.nin, self.nout = len(self._inputs), len(self._outputs)
        if pyfunc.__code__.co_argcount != self.nin + self.nout:
            raise TypeError(
                f"'{self.__name__}' takes {pyfunc.__code__.co_argcount} arguments, "
                f"but the layout '{layout}' has {self.nin + self.nout}"
            )
        self._signatures = []
        for argtypes in signatures:
            dtypes = [
                numpy_support.as_dtype(t.dtype if isinstance(t, types.Array) else t)
                for t in argtypes
            ]
            arrays = [isinstance(t, types.Array) for t in argtypes]
            self._signatures.append((dtypes, (tuple(argtypes), arrays)))
        self._kernels = {}

    def __repr__(self):
        return f"<Vulkan gufunc '{self.__name__}' {self.layout}>"

    def __call__(self, *args, out=None, device=None):
        """Apply the core function over the loop dimensions.

        Parameters
        ----------
        *args : array_like, DeviceArray or scalar
            The inputs, optionally followed by the outputs.
        out : array or tuple of arrays, optional
            The outputs, instead of passing them positionally.
        device : int, str or Device, optional
            Device to run on. By default that of a device array among the
            arguments, or the selected device.

        Returns
        -------
        array or tuple of arrays
            The outputs: NumPy arrays, or device arrays if any input is one.

        Raises
        ------
        TypeError
            For a wrong number of arguments, types no signature accepts, or
            missing outputs of a function without signatures.
        ValueError
            If core dimensions disagree or the outputs have the wrong shape.
        """
        if out is not None:
            args = args + (tuple(out) if isinstance(out, (tuple, list)) else (out,))
        if len(args) not in (self.nin, self.nin + self.nout):
            raise TypeError(
                f"'{self.__name__}' takes {self.nin} inputs and optionally "
                f"{self.nout} outputs, got {len(args)} arguments"
            )
        inputs = [_Argument(a) for a in args[: self.nin]]
        outputs = [_Argument(a) for a in args[self.nin :]]
        target = _device_of(
            inputs + outputs, device if device is not None else self._device
        )

        if self._signatures:
            dtypes, (_, arrays) = _select(self._signatures, inputs, self.__name__)
        elif not outputs:
            raise TypeError(
                f"'{self.__name__}' has no signatures, so its outputs must be "
                "passed: the output types cannot be inferred"
            )
        else:
            dtypes = _natural_types(inputs) + [o.dtype for o in outputs]
            # As on the CPU: scalar inputs for "()", arrays for outputs.
            arrays = [bool(core) for core in self._inputs] + [True] * self.nout

        # Bind the core dimensions and find the loop shape.
        sizes, loops = {}, []
        for k, (argument, core) in enumerate(zip(inputs, self._inputs)):
            loops.append(self._bind(argument.shape, core, sizes, k))
        loop_shape = np.broadcast_shapes(*loops)
        for core in self._outputs:
            missing = [d for d in core if d not in sizes]
            if missing:
                raise ValueError(
                    f"the output dimension {missing[0]} of '{self.__name__}' is not "
                    "given by any input"
                )

        on_device = _on_device(inputs + outputs)
        values = [a.converted(d) for a, d in zip(inputs, dtypes)]
        if outputs:
            out_values = []
            for k, (argument, core) in enumerate(zip(outputs, self._outputs)):
                want = loop_shape + tuple(sizes[d] for d in core)
                if tuple(argument.shape) != want:
                    raise ValueError(
                        f"output {k} has shape {tuple(argument.shape)}, expected {want}"
                    )
                if argument.dtype != dtypes[self.nin + k]:
                    raise TypeError(
                        f"output {k} has type {argument.dtype}, expected "
                        f"{dtypes[self.nin + k]}"
                    )
                out_values.append(argument.value)
        else:
            out_values = [
                _allocate(
                    loop_shape + tuple(sizes[d] for d in core),
                    dtypes[self.nin + k],
                    target,
                    on_device,
                )
                for k, core in enumerate(self._outputs)
            ]
        results = list(out_values)

        # Array arguments for a "()" core get a trailing axis of length 1.
        cores = list(self._inputs) + list(self._outputs)
        passed = []
        for value, core, array in zip(values + out_values, cores, arrays):
            if not core and array:
                if not isinstance(value, (np.ndarray, runtime.DeviceArray)):
                    value = np.asarray(value)
                value = value.reshape(tuple(value.shape) + (1,))
                core = ("",)
            passed.append((value, len(core)))
        for k, (value, _) in enumerate(passed[self.nin :]):
            if isinstance(value, np.ndarray) and not value.flags.c_contiguous:
                raise ValueError(f"output {k} must be C-contiguous")

        kinds = tuple(
            (len(v.shape) - c)
            if isinstance(v, (np.ndarray, runtime.DeviceArray))
            else None
            for v, c in passed
        )
        kernel = self._kernel(kinds, len(loop_shape))
        count = int(np.prod(loop_shape, dtype=np.int64))
        if count:
            kernel.forall(count, device=target)(*(v for v, _ in passed))
        return results[0] if self.nout == 1 else tuple(results)

    def _bind(self, shape, core, sizes, position):
        """Record the core dimensions of an argument; return its loop shape."""
        if len(shape) < len(core):
            raise ValueError(
                f"argument {position} of '{self.__name__}' needs at least "
                f"{len(core)} dimensions, but has {len(shape)}"
            )
        loop = tuple(shape[: len(shape) - len(core)])
        for name, extent in zip(core, shape[len(loop) :]):
            if sizes.setdefault(name, extent) != extent:
                raise ValueError(
                    f"dimension {name} of '{self.__name__}' is {sizes[name]} in one "
                    f"argument and {extent} in another"
                )
        return loop

    def _kernel(self, kinds, loop_ndim):
        """The kernel for arguments with the given numbers of loop dimensions.

        Parameters
        ----------
        kinds : tuple of int or None
            Number of loop dimensions of each argument; ``None`` for a
            scalar.
        loop_ndim : int
            Number of loop dimensions of the call.

        Returns
        -------
        VulkanDispatcher
        """
        key = (kinds, loop_ndim)
        if key not in self._kernels:
            names = [f"a{k}" for k in range(len(kinds))]
            first_out = names[self.nin]
            count = (
                " * ".join(f"{first_out}.shape[{d}]" for d in range(loop_ndim)) or "1"
            )
            lines = [
                f"def __gufunc__({', '.join(names)}):",
                "    __i__ = __global_id__(0)",
                f"    if __i__ < {count}:",
            ]
            index_names = _unravel(lines, "__i__", first_out, loop_ndim, "__j")
            items = []
            for name, ndim in zip(names, kinds):
                if not ndim:  # a scalar, or an array of core dimensions only
                    items.append(name)
                else:
                    items.append(f"{name}[{_broadcast_index(name, ndim, index_names)}]")
            lines.append(f"        __core__({', '.join(items)})")
            scope = {"__global_id__": global_id, "__core__": self._core}
            self._kernels[key] = _compile_source(
                "\n".join(lines) + "\n", "__gufunc__", scope
            )
            self._kernels[key].__name__ = f"{self.__name__}_gufunc"
        return self._kernels[key]


Vectorize.target_registry[TARGET] = VulkanVectorize
GUVectorize.target_registry[TARGET] = VulkanGUVectorize
