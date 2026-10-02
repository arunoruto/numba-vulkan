"""Dispatcher and ``jit`` decorator of the Vulkan target."""

import functools
import os
import warnings

import numpy as np
from numba import typeof
from numba.core import types, typing, utils
from numba.np import numpy_support
from numba.core.target_extension import (
    dispatcher_registry,
    jit_registry,
    target_registry,
)

from numba_vulkan import narrowing, runtime
from numba_vulkan.buffers import STATUS_INDEX, arg_binding
from numba_vulkan.compiler import compile_kernel, compile_vulkan
from numba_vulkan.errors import VulkanPrecisionWarning, VulkanUnsupportedError
from numba_vulkan.target import TARGET_NAME, exception_table, vulkan_target
from numba_vulkan.vktypes import VulkanArray, VulkanDispatcherType


class VulkanDispatcher:
    """A Python function compiled lazily for Vulkan.

    It is launched as a kernel through `forall`, and it can be called
    from other Vulkan-compiled functions, in which case it is inlined.
    Instances are normally created with `jit`.

    Parameters
    ----------
    py_func : function
        The Python function.
    targetoptions : dict, optional
        Options; ``fastmath``, ``narrow_math`` and ``boundscheck`` are
        used.

    Attributes
    ----------
    py_func : function
        The Python function.
    fastmath : bool
        Whether speed is preferred over accuracy; see `jit`.
    narrow_math : bool
        Whether float64 transcendental functions are computed in float32
        when libclc is unavailable.
    boundscheck : bool
        Whether array indices are checked.
    narrow : bool or None
        Whether 64-bit types are narrowed to 32 bits: always, never, or
        (``None``) on devices that lack them; see `jit`.
    overloads : dict
        Compile results by tuple of argument types.

    Examples
    --------
    >>> import numpy as np
    >>> import numba_vulkan as nv
    >>> @nv.jit
    ... def double(x, out):
    ...     i = nv.global_id(0)
    ...     if i < x.shape[0]:
    ...         out[i] = 2 * x[i]
    >>> x = np.arange(4, dtype=np.float32)
    >>> out = np.zeros_like(x)
    >>> double.forall(x.size)(x, out)
    >>> out
    array([0., 2., 4., 6.], dtype=float32)
    """

    targetdescr = vulkan_target

    @property
    def overloads(self):
        """Compile results by tuple of argument types.

        Functions compiled without float64 (see `numba_vulkan.narrowing`)
        are kept apart from the ordinary ones; the property gives those of
        the mode being compiled in.

        Returns
        -------
        dict
        """
        return self._overloads.setdefault(narrowing.current.floats, {})

    _can_compile = True

    def __init__(self, py_func, targetoptions=None):
        self.py_func = py_func
        self.targetoptions = dict(targetoptions or {})
        self.narrow_math = bool(self.targetoptions.get("narrow_math", False))
        self.fastmath = bool(self.targetoptions.get("fastmath", False))
        self.boundscheck = bool(self.targetoptions.get("boundscheck", False))
        self.narrow = self.targetoptions.get("narrow")
        if self.narrow is None and os.environ.get("NUMBA_VULKAN_NARROW", "0") != "0":
            self.narrow = True
        self._overloads = {}
        self._kernels = {}
        # Kernels by the argument types of a launch, before binding.
        self._launched = {}
        self._compiling = 0
        functools.update_wrapper(self, py_func)

    def __repr__(self):
        return f"VulkanDispatcher({self.py_func.__qualname__})"

    # -- interface used by Numba's typing and lowering ----------------------

    @property
    def _numba_type_(self):
        """Numba type of this object when it appears in compiled code.

        Returns
        -------
        numba_vulkan.vktypes.VulkanDispatcherType
        """
        return VulkanDispatcherType(self)

    @property
    def is_compiling(self):
        """Whether a compilation of this function is in progress.

        Numba's type inference uses this to detect recursion.

        Returns
        -------
        bool
        """
        return self._compiling > 0

    @property
    def nopython_signatures(self):
        """Signatures of all specialisations compiled so far.

        Returns
        -------
        list of numba.core.typing.Signature
        """
        return [cres.signature for cres in self.overloads.values()]

    def compile_device(self, args, return_type=None):
        """Compile for ``args`` as a function callable from other kernels.

        Parameters
        ----------
        args : tuple of numba.types.Type
            Argument types.
        return_type : numba.types.Type, optional
            Return type; inferred when omitted.

        Returns
        -------
        VulkanCompileResult
        """
        args = tuple(types.unliteral(a) for a in args)
        if args not in self.overloads:
            self._compiling += 1
            try:
                cres = compile_vulkan(
                    self.py_func,
                    return_type,
                    args,
                    self.narrow_math,
                    self.fastmath,
                    self.boundscheck,
                )
            except KeyError as exc:
                # Numba's own array code asks for the data pointer, which
                # arrays on this target do not have.
                if "field named 'data'" not in str(exc):
                    raise
                raise VulkanUnsupportedError(
                    f"'{self.py_func.__name__}' uses an array operation that needs "
                    "direct access to memory, which Vulkan shaders do not have. "
                    "Indexing, slicing, iteration and the reductions listed in "
                    "the documentation are supported."
                ) from None
            finally:
                self._compiling -= 1
            self.overloads[args] = cres
            cres.target_context.insert_user_function(
                cres.entry_point, cres.fndesc, [cres.library]
            )
        return self.overloads[args]

    def fold_argument_types(self, args, kws):
        """Turn keyword, default and star arguments into positional types.

        Parameters
        ----------
        args : sequence of numba.types.Type
            Positional argument types of a call.
        kws : dict
            Keyword argument types of the call.

        Returns
        -------
        pysig : inspect.Signature
            Signature of the Python function.
        args : tuple of numba.types.Type
            One type per parameter of the Python function.
        """
        pysig = utils.pysignature(self.py_func)
        args = typing.fold_arguments(
            pysig,
            args,
            kws,
            lambda index, param, value: value,
            lambda index, param, default: types.Omitted(default),
            lambda index, param, values: types.StarArgTuple(values),
        )
        return pysig, args

    def get_call_template(self, args, kws):
        """Provide the typing template for a call from compiled code.

        Compiles the specialisation for the argument types if necessary.

        Parameters
        ----------
        args : sequence of numba.types.Type
            Positional argument types.
        kws : dict
            Keyword argument types.

        Returns
        -------
        template : type
            A concrete template holding the compiled signatures.
        pysig : inspect.Signature
            Signature of the Python function.
        args : tuple of numba.types.Type
            Folded argument types.
        kws : dict
            Always empty; keywords were folded into ``args``.
        """
        pysig, args = self.fold_argument_types(args, kws)
        self.compile_device(tuple(args))
        name = self.py_func.__name__
        template = typing.make_concrete_template(
            f"CallTemplate({name})", key=name, signatures=self.nopython_signatures
        )
        return template, pysig, args, {}

    def get_overload(self, sig):
        """Key identifying the compiled specialisation for a signature.

        Parameters
        ----------
        sig : numba.core.typing.Signature or tuple of numba.types.Type
            A signature or its argument types.

        Returns
        -------
        int
            The entry point of the `VulkanCompileResult`.

        Raises
        ------
        KeyError
            If the specialisation has not been compiled.
        """
        args = sig.args if isinstance(sig, typing.Signature) else sig
        return self.overloads[tuple(types.unliteral(a) for a in args)].entry_point

    def get_compile_result(self, sig):
        """Compile for a signature and return the result.

        Parameters
        ----------
        sig : numba.core.typing.Signature
            The signature.

        Returns
        -------
        VulkanCompileResult
        """
        return self.compile_device(sig.args, sig.return_type)

    # -- kernel launch --------------------------------------------------------

    def compile(self, argtypes, ndim=1, mode=narrowing.Mode()):
        """Compile (or fetch) the kernel specialisation for ``argtypes``.

        Parameters
        ----------
        argtypes : tuple of numba.types.Type
            Argument types; arrays are bound to consecutive storage buffers.
        ndim : int
            Dimensionality of the dispatch grid.
        mode : numba_vulkan.narrowing.Mode
            The 64-bit types the kernel must do without.

        Returns
        -------
        CompiledKernel
        """
        bound = []
        for index, ty in enumerate(argtypes):
            if isinstance(ty, types.Array):
                ty = VulkanArray(
                    ty.dtype,
                    ty.ndim,
                    ty.layout,
                    arg_binding(index),
                    readonly=not ty.mutable,
                )
            bound.append(ty)
        key = (tuple(bound), ndim, mode)
        if key not in self._kernels:
            with narrowing.using(mode):
                cres = self.compile_device(key[0])
                kernel = compile_kernel(cres, ndim, exact=not self.fastmath)
            if kernel.narrowed.floats and self.narrow is None:
                warnings.warn(
                    f"kernel '{self.py_func.__name__}' uses float64, which the "
                    "device does not support; it is computed in float32 instead. "
                    "Pass narrow=True to @jit to accept this silently.",
                    VulkanPrecisionWarning,
                    stacklevel=4,
                )
            self._kernels[key] = kernel
        return self._kernels[key]

    def forall(self, extent, device=None):
        """Bind a dispatch grid, like ``numba.cuda``'s ``kernel.forall``.

        Parameters
        ----------
        extent : int or tuple of int
            Number of invocations along each of up to three axes. It is
            rounded up to whole workgroups, so kernels must bounds-check
            :func:`numba_vulkan.stubs.global_id`.
        device : int, str or None
            Device to run on; defaults to the selected device.

        Returns
        -------
        callable
            Call it with the kernel arguments to run.
        """
        extent = (extent,) if np.isscalar(extent) else tuple(extent)
        if not 1 <= len(extent) <= 3:
            raise ValueError("the dispatch grid must have 1 to 3 dimensions")

        def launch(*args):
            """Run the kernel over the bound grid with the given arguments."""
            return self._launch(extent, device, args)

        return launch

    def __call__(self, *args):
        """Reject a direct call; kernels need a dispatch grid.

        Raises
        ------
        TypeError
            Always, pointing at `forall`.
        """
        name = self.py_func.__name__
        raise TypeError(
            f"kernel '{name}' needs a dispatch grid: use {name}.forall(n)(...)"
        )

    def _launch(self, extent, device, args):
        """Compile for the given arguments and run on a device.

        Parameters
        ----------
        extent : tuple of int
            Number of invocations along each axis of the grid.
        device : int, str or None
            Device to run on.
        args : tuple
            Kernel arguments: NumPy arrays, device arrays and scalars.

        Raises
        ------
        ValueError
            If a device array is on another device.
        Exception
            Whatever an invocation of the kernel raised; see `_raise`.
        VulkanSupportError
            If the device lacks a capability the kernel needs.

        Notes
        -----
        Scalars are passed as one-element buffers. Arrays are converted on
        the host where the device stores them differently: booleans as
        int32, 64-bit types as 32-bit ones where the kernel is narrowed,
        and arrays that are not C-contiguous as contiguous copies.
        """
        target = runtime.get_device(device)
        if self.narrow is None:
            mode = target.mode
        else:
            mode = narrowing.Mode(floats=self.narrow, ints=self.narrow)
        argtypes, hosts, shapes, staged = [], [], [], []
        for arg in args:
            if isinstance(arg, runtime.DeviceArray):
                if arg._stored != narrowing.stored_dtype(arg.dtype, mode):
                    raise ValueError(
                        f"a {arg.dtype} device array on {arg.device.info.name} holds "
                        f"{arg._stored} elements, which does not match a kernel "
                        f"compiled with narrow={self.narrow}"
                    )
                argtypes.append(_device_array_type(arg.dtype, arg.ndim))
                shapes.extend(arg.shape)
                hosts.append(arg)
            elif isinstance(arg, np.ndarray):
                argtypes.append(typeof(arg).copy(layout="C", readonly=False))
                shapes.extend(arg.shape)
                stored = narrowing.stored_dtype(arg.dtype, mode)
                if stored != arg.dtype or not arg.flags.c_contiguous:
                    # Converted on the host: booleans are int32 on the device
                    # (SPIR-V has no storable bool), 64-bit types are 32-bit
                    # where the kernel is narrowed, and strided or
                    # Fortran-ordered arrays travel as contiguous copies.
                    staged.append((len(hosts), arg))
                    arg = np.ascontiguousarray(arg, dtype=stored)
                hosts.append(arg)
            else:
                ty = typeof(arg)
                argtypes.append(ty)
                stored = narrowing.stored_dtype(np.dtype(str(ty)), mode)
                hosts.append(np.array([arg], dtype=stored))
        key = (tuple(argtypes), len(extent), mode)
        kernel = self._launched.get(key)
        if kernel is None:
            kernel = self._launched[key] = self.compile(argtypes, len(extent), mode)
        groups = [1, 1, 1]
        for axis, n in enumerate(extent):
            groups[axis] = -(-int(n) // kernel.local_size[axis])
        if 0 in groups:
            return
        # Element 0 receives the status of the kernel, the shapes follow.
        meta = np.array([0, *shapes], dtype=np.int32)
        target.run(kernel, tuple(groups), [meta, *hosts])
        for index, original in staged:
            if arg_binding(index) in kernel.written_bindings:
                copy = hosts[index]
                original[...] = copy != 0 if original.dtype == np.bool_ else copy
        if meta[STATUS_INDEX]:
            self._raise(int(meta[STATUS_INDEX]))

    def _raise(self, code):
        """Raise the exception that a kernel reported.

        Parameters
        ----------
        code : int
            Status code left by the kernel.

        Raises
        ------
        Exception
            The exception registered for `code`, with a note saying where
            in the kernel it was raised.
        """
        exc, exc_args, location = exception_table.get_exception(code)
        if exc is None:
            exc, exc_args = RuntimeError, ("exception re-raised in a kernel",)
        error = exc(*(exc_args or ()))
        where = f"raised in Vulkan kernel '{self.py_func.__name__}'"
        if location:
            where += f", in {location[0]} at {location[1]}:{location[2]}"
        error.add_note(where)
        raise error


@functools.lru_cache(maxsize=None)
def _device_array_type(dtype, ndim):
    """Numba type of a device array."""
    return types.Array(numpy_support.from_dtype(dtype), ndim, "C")


def jit(
    pyfunc=None,
    *,
    fastmath=False,
    narrow_math=False,
    boundscheck=False,
    narrow=None,
    **options,
):
    """Compile a Python function for Vulkan.

    Parameters
    ----------
    pyfunc : function, optional
        The function. Kernels must return ``None`` and write into arrays;
        functions called from kernels may return values.
    fastmath : bool
        Trade accuracy for speed: float32 math functions use the device's
        built-in versions instead of libclc, and the driver may reassociate
        and contract float arithmetic. Results then differ between devices.
    narrow_math : bool
        Without libclc installed, evaluate float64 transcendental functions
        in float32 precision instead of rejecting them.
    boundscheck : bool
        Check array indices in this function and raise ``IndexError`` from
        the launch when one is out of bounds. Without it, such accesses
        read or corrupt unrelated memory of the buffer or are ignored,
        depending on the driver. The environment variable
        ``NUMBA_BOUNDSCHECK=1`` turns the check on everywhere.
    narrow : bool or None
        Whether a kernel computes with 32-bit floats and integers where its
        code says ``float64`` and ``int64``, which Numba uses for Python
        literals and all index arithmetic. ``None``, the default, narrows
        on devices without 64-bit types and warns when that affects
        floats; ``True`` always narrows, without a warning; ``False`` never
        does, so the kernel fails on such devices. Arrays of 64-bit
        elements are converted on the host. Only the setting of the kernel
        matters, not that of the functions it calls.
    **options
        Accepted for compatibility with Numba's generic ``jit`` and ignored.

    Returns
    -------
    VulkanDispatcher
    """
    options["narrow_math"] = narrow_math
    options["fastmath"] = fastmath
    options["boundscheck"] = boundscheck
    options["narrow"] = narrow
    if pyfunc is None:
        return lambda f: VulkanDispatcher(f, options)
    return VulkanDispatcher(pyfunc, options)


_target = target_registry[TARGET_NAME]
jit_registry[_target] = jit
dispatcher_registry[_target] = VulkanDispatcher
