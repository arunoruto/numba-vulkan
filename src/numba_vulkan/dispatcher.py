"""Dispatcher and ``jit`` decorator of the Vulkan target."""

import functools

import numpy as np
from numba import typeof
from numba.core import types, typing, utils
from numba.np import numpy_support
from numba.core.target_extension import (
    dispatcher_registry,
    jit_registry,
    target_registry,
)

from numba_vulkan import runtime
from numba_vulkan.buffers import arg_binding
from numba_vulkan.compiler import compile_kernel, compile_vulkan
from numba_vulkan.target import TARGET_NAME, vulkan_target
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
        Options; ``fastmath`` and ``narrow_math`` are used.

    Attributes
    ----------
    py_func : function
        The Python function.
    fastmath : bool
        Whether speed is preferred over accuracy; see `jit`.
    narrow_math : bool
        Whether float64 transcendental functions are computed in float32
        when libclc is unavailable.
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
    _can_compile = True

    def __init__(self, py_func, targetoptions=None):
        self.py_func = py_func
        self.targetoptions = dict(targetoptions or {})
        self.narrow_math = bool(self.targetoptions.get("narrow_math", False))
        self.fastmath = bool(self.targetoptions.get("fastmath", False))
        self.overloads = {}
        self._kernels = {}
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
                    self.py_func, return_type, args, self.narrow_math, self.fastmath
                )
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

    def compile(self, argtypes, ndim=1):
        """Compile (or fetch) the kernel specialisation for ``argtypes``.

        Parameters
        ----------
        argtypes : tuple of numba.types.Type
            Argument types; arrays are bound to consecutive storage buffers.
        ndim : int
            Dimensionality of the dispatch grid.

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
        key = (tuple(bound), ndim)
        if key not in self._kernels:
            cres = self.compile_device(key[0])
            self._kernels[key] = compile_kernel(cres, ndim, exact=not self.fastmath)
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
            Kernel arguments: C-contiguous NumPy arrays, device arrays and
            scalars.

        Raises
        ------
        ValueError
            If an array is not C-contiguous, or if a device array is on
            another device.
        VulkanSupportError
            If the device lacks a capability the kernel needs.

        Notes
        -----
        Scalars are passed as one-element buffers. Boolean arrays are
        converted to and from int32 on the host.
        """
        argtypes, hosts, shapes, staged = [], [], [], []
        for arg in args:
            if isinstance(arg, runtime.DeviceArray):
                dtype = numpy_support.from_dtype(arg.dtype)
                argtypes.append(types.Array(dtype, arg.ndim, "C"))
                shapes.extend(arg.shape)
                hosts.append(arg)
            elif isinstance(arg, np.ndarray):
                if not arg.flags.c_contiguous:
                    raise ValueError("only C-contiguous arrays are supported")
                argtypes.append(typeof(arg).copy(readonly=False))
                shapes.extend(arg.shape)
                if arg.dtype == np.bool_:
                    # Booleans are int32 on the device (SPIR-V has no
                    # storable bool), so they are converted on the host.
                    staged.append((len(hosts), arg))
                    arg = arg.astype(np.int32)
                hosts.append(arg)
            else:
                ty = typeof(arg)
                argtypes.append(ty)
                store = np.int32 if isinstance(ty, types.Boolean) else str(ty)
                hosts.append(np.array([arg], dtype=store))
        kernel = self.compile(argtypes, ndim=len(extent))
        groups = [1, 1, 1]
        for axis, n in enumerate(extent):
            groups[axis] = -(-int(n) // kernel.local_size[axis])
        if 0 in groups:
            return
        meta = np.array(shapes, dtype=np.int32)
        runtime.get_device(device).run(kernel, tuple(groups), [meta, *hosts])
        for index, original in staged:
            if arg_binding(index) in kernel.written_bindings:
                original[...] = hosts[index] != 0


def jit(pyfunc=None, *, fastmath=False, narrow_math=False, **options):
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
    **options
        Accepted for compatibility with Numba's generic ``jit`` and ignored.

    Returns
    -------
    VulkanDispatcher
    """
    options["narrow_math"] = narrow_math
    options["fastmath"] = fastmath
    if pyfunc is None:
        return lambda f: VulkanDispatcher(f, options)
    return VulkanDispatcher(pyfunc, options)


_target = target_registry[TARGET_NAME]
jit_registry[_target] = jit
dispatcher_registry[_target] = VulkanDispatcher
