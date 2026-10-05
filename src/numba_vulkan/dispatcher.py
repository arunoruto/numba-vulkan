"""Dispatcher and ``jit`` decorator of the Vulkan target."""

import dataclasses
import functools
import inspect
import os
import re
import struct
import warnings

import numpy as np
from numba import typeof
from numba.core import errors, types, typing, utils
from numba.core.target_extension import (
    dispatcher_registry,
    jit_registry,
    target_registry,
)
from numba.np import numpy_support

from numba_vulkan import kernelcache, narrowing, runtime
from numba_vulkan.buffers import STATUS_INDEX, arg_binding, print_formats
from numba_vulkan.codegen import CompiledKernel
from numba_vulkan.compiler import compile_kernel, compile_vulkan
from numba_vulkan.errors import (
    VulkanPerformanceWarning,
    VulkanPrecisionWarning,
    VulkanUnsupportedError,
)
from numba_vulkan.target import TARGET_NAME, exception_table, vulkan_target
from numba_vulkan.vktypes import (
    HalfArray,
    VulkanArray,
    VulkanDispatcherType,
    VulkanRecord,
)


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

        Functions compiled for different device features (see
        `numba_vulkan.narrowing.Mode`) are kept apart; the property gives
        those of the mode being compiled in.

        Returns
        -------
        dict
        """
        return self._overloads.setdefault(narrowing.current, {})

    _can_compile = True

    def __init__(self, py_func, targetoptions=None):
        self.py_func = py_func
        self.targetoptions = dict(targetoptions or {})
        self.narrow_math = bool(self.targetoptions.get("narrow_math", False))
        self.fastmath = bool(self.targetoptions.get("fastmath", False))
        self.boundscheck = bool(self.targetoptions.get("boundscheck", False))
        self.error_model = self.targetoptions.get("error_model", "numpy")
        if self.error_model not in ("numpy", "python"):
            raise ValueError(
                f"error_model must be 'numpy' or 'python', not {self.error_model!r}"
            )
        self.narrow = self.targetoptions.get("narrow")
        if self.narrow not in (None, True, False, "ints", "floats"):
            raise ValueError(
                f"narrow must be True, False, 'ints' or 'floats', not {self.narrow!r}"
            )
        if self.narrow is None and os.environ.get("NUMBA_VULKAN_NARROW", "0") != "0":
            self.narrow = True
        self.cache = bool(self.targetoptions.get("cache", False))
        self._overloads = {}
        self._kernels = {}
        # Kernels by the argument types of a launch, before binding.
        self._launched = {}
        # Launch plans by argument signature; see _launch.
        self._plans = {}
        self._compiling = 0
        self._uncachable_warned = False
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
                    self.error_model,
                )
            except errors.TypingError as exc:
                # Numba's array constructors ask for an allocator.
                if "_allocate" not in str(exc):
                    raise
                raise VulkanUnsupportedError(
                    f"'{self.py_func.__name__}' creates an array whose size is only "
                    "known at run time, which Vulkan shaders cannot do. Arrays of a "
                    "constant shape (np.zeros(4), nv.local.array) and array "
                    "expressions, which are computed element by element, work."
                ) from None
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

    def compile(
        self,
        argtypes,
        ndim=1,
        mode=narrowing.Mode(),  # noqa: B008 - immutable
        local_size=None,
        aliases=(),
    ):
        """Compile (or fetch) the kernel specialisation for ``argtypes``.

        Parameters
        ----------
        argtypes : tuple of numba.types.Type
            Argument types; arrays are bound to consecutive storage buffers.
        ndim : int
            Dimensionality of the dispatch grid.
        mode : numba_vulkan.narrowing.Mode
            The 64-bit types the kernel must do without.
        local_size : tuple of int, optional
            Workgroup size; by default 64 invocations, shaped by `ndim`.
        aliases : tuple of tuple
            ``(index, earlier)`` for array arguments that share the buffer
            of an earlier one; see `numba_vulkan.compiler.compile_kernel`.

        Returns
        -------
        CompiledKernel
        """
        bound = []
        for index, ty in enumerate(argtypes):
            if isinstance(ty, types.Array) and isinstance(ty.dtype, types.Record):
                ty = ty.copy(dtype=_record_type(ty.dtype))
            if isinstance(ty, types.Array):
                # The binding is a value of the array (see vktypes), so
                # functions called with it are compiled once for all.
                ty = VulkanArray(
                    ty.dtype,
                    ty.ndim,
                    ty.layout,
                    None,
                    readonly=not ty.mutable,
                    half=isinstance(ty, HalfArray),
                )
            bound.append(ty)
        local_size = _shape3(local_size) if local_size else None
        key = (tuple(bound), ndim, mode, *((aliases,) if aliases else ()))
        if key not in self._kernels:
            name = self._cache_name(key) if self.cache else None
            kernel = self._load(name) if name else None
            if kernel is None:
                counted = exception_table.counted
                with narrowing.using(mode):
                    cres = self.compile_device(key[0])
                    kernel = compile_kernel(
                        cres, ndim, exact=not self.fastmath, aliases=aliases
                    )
                # Exceptions with counted codes differ between processes.
                if name and exception_table.counted == counted:
                    kernelcache.store_kernel(
                        name,
                        {
                            "kernel": kernel,
                            "exceptions": exception_table.entries(),
                            "prints": dict(print_formats),
                        },
                    )
            self._warn(kernel, bound)
            self._kernels[key] = kernel
        kernel = self._kernels[key]
        if local_size is None or local_size == kernel.local_size:
            return kernel
        # The workgroup size is a specialization constant: other sizes share
        # the module, and only get a pipeline of their own.
        variant = (key, local_size)
        if variant not in self._kernels:
            self._kernels[variant] = dataclasses.replace(kernel, local_size=local_size)
        return self._kernels[variant]

    def _cache_name(self, key):
        """Name of the cache entry of a specialisation, if it can be cached.

        Parameters
        ----------
        key : tuple
            Bound argument types, grid dimensionality and narrowing mode.

        Returns
        -------
        str or None
            ``None`` if the kernel or a function it calls has no source
            file, or the cache is off (``NUMBA_VULKAN_CACHE=0``).
        """
        if not kernelcache.enabled():
            return None
        stamps = [kernelcache.source_stamp(d.py_func) for d in self._dependencies()]
        if None in stamps:
            if not self._uncachable_warned:
                self._uncachable_warned = True
                warnings.warn(
                    f"kernel '{self.py_func.__name__}' or a function it calls has no "
                    "source file, so cache=True has no effect",
                    errors.NumbaWarning,
                    stacklevel=4,
                )
            return None
        bound, ndim, mode = key
        options = (
            self.fastmath,
            self.narrow_math,
            self.boundscheck,
            self.error_model,
            os.environ.get("NUMBA_BOUNDSCHECK"),
        )
        return kernelcache.kernel_key(
            stamps, tuple(map(repr, bound)), ndim, tuple(mode), options
        )

    def _dependencies(self):
        """This dispatcher and the ``@nv.jit`` functions its code refers to.

        Followed through globals, attributes of modules and closures, and
        recursively.

        Returns
        -------
        list of VulkanDispatcher
        """
        found, work, seen = [], [self], set()
        while work:
            dispatcher = work.pop()
            if id(dispatcher) in seen:
                continue
            seen.add(id(dispatcher))
            found.append(dispatcher)
            function = dispatcher.py_func
            names = _code_names(function.__code__)
            scope = dict(function.__globals__)
            for name, cell in zip(
                function.__code__.co_freevars, function.__closure__ or ()
            ):
                try:
                    scope[name] = cell.cell_contents
                except ValueError:  # an empty cell
                    pass
            for name in names:
                value = scope.get(name)
                values = [value]
                if inspect.ismodule(value):
                    values += [getattr(value, other, None) for other in names]
                work += [v for v in values if isinstance(v, VulkanDispatcher)]
        return found

    def _load(self, name):
        """A cached kernel, or ``None`` if there is none that fits.

        The exceptions and ``print`` formats it uses are registered again.
        """
        entry = kernelcache.load_kernel(name)
        if entry is None or not isinstance(entry.get("kernel"), CompiledKernel):
            return None
        if not exception_table.restore(entry.get("exceptions", {})):
            return None
        print_formats.update(entry.get("prints", {}))
        return entry["kernel"]

    def _warn(self, kernel, argtypes):
        """Point out float64 that costs precision or speed, once per kernel.

        Parameters
        ----------
        kernel : CompiledKernel
            The kernel just compiled.
        argtypes : list of numba.types.Type
            Its argument types.
        """
        if not narrowing.WARNINGS:
            return
        name = self.py_func.__name__
        silence = "Set NUMBA_VULKAN_WARNINGS=0 to silence this."
        if kernel.narrowed.floats and self.narrow not in (True, "floats"):
            warnings.warn(
                f"kernel '{name}' uses float64, which the device does not support; "
                "it is computed in float32 instead. Pass narrow=True to @jit to "
                f"accept this. {silence}",
                VulkanPrecisionWarning,
                stacklevel=5,
            )
        elif not kernel.mode.floats and _FLOAT64_ARITHMETIC.search(kernel.llvm_ir):
            explicit = any(_holds_float64(getattr(t, "dtype", t)) for t in argtypes)
            hint = (
                ""
                if explicit
                else " None of its arguments is float64, so the cause is probably "
                "a Python float such as 0.5, which Numba types as float64; write "
                "np.float32(0.5) to stay in float32."
            )
            warnings.warn(
                f"kernel '{name}' computes with float64, which most GPUs run at a "
                f"small fraction of the float32 speed.{hint} {silence}",
                VulkanPerformanceWarning,
                stacklevel=5,
            )

    def forall(self, extent, device=None, local_size=None, stream=None):
        """Bind a dispatch grid, like ``numba.cuda``'s ``kernel.forall``.

        Parameters
        ----------
        extent : int or tuple of int
            Number of invocations along each of up to three axes. It is
            rounded up to whole workgroups, so kernels must bounds-check
            :func:`numba_vulkan.stubs.global_id`.
        device : int, str or None
            Device to run on; defaults to the selected device.
        local_size : int or tuple of int, optional
            Workgroup size. By default 64 invocations: ``(64,)``,
            ``(8, 8)`` or ``(4, 4, 4)`` depending on the grid.
        stream : numba_vulkan.runtime.Stream, optional
            Enqueue the launch on a stream; see
            `numba_vulkan.runtime.Stream`. NumPy arrays are copied on the
            stream and hold the results after it is synchronized.

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
            return self._launch(
                args, device, extent=extent, local_size=local_size, stream=stream
            )

        return launch

    def __getitem__(self, config):
        """Bind a launch configuration, like ``kernel[blocks, threads]`` in CUDA.

        Parameters
        ----------
        config : tuple
            ``(groups, local_size)`` or ``(groups, local_size, device)``:
            the number of workgroups and the workgroup size, each an int or
            a tuple of up to three ints. Unlike `forall`, the grid is
            exactly ``groups * local_size`` invocations. The third element
            may also be a `numba_vulkan.runtime.Stream`, as in CUDA.

        Returns
        -------
        callable
            Call it with the kernel arguments to run.
        """
        if not isinstance(config, tuple) or len(config) not in (2, 3):
            raise TypeError(
                "use kernel[groups, local_size] or kernel[groups, local_size, device]"
            )
        groups, local_size = config[:2]
        device = config[2] if len(config) == 3 else None
        stream = None
        if isinstance(device, runtime.Stream):
            stream, device = device, device.device
        groups = (groups,) if np.isscalar(groups) else tuple(groups)
        local = (local_size,) if np.isscalar(local_size) else tuple(local_size)
        if not (1 <= len(groups) <= 3 and 1 <= len(local) <= 3):
            raise ValueError("groups and local_size must have 1 to 3 dimensions")

        def launch(*args):
            """Run the kernel with the bound configuration."""
            return self._launch(
                args, device, groups=groups, local_size=local, stream=stream
            )

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
            f"kernel '{name}' needs a dispatch grid: use {name}.forall(n)(...) or "
            f"{name}[groups, local_size](...)"
        )

    def _launch(
        self, args, device, extent=None, groups=None, local_size=None, stream=None
    ):
        """Compile for the given arguments and run on a device.

        Parameters
        ----------
        args : tuple
            Kernel arguments: NumPy arrays, device arrays and scalars.
        device : int, str or None
            Device to run on.
        extent : tuple of int, optional
            Number of invocations along each axis, rounded up to whole
            workgroups.
        groups : tuple of int, optional
            Number of workgroups along each axis, instead of `extent`.
        local_size : tuple of int, optional
            Workgroup size.
        stream : numba_vulkan.runtime.Stream, optional
            Enqueue on a stream instead.

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
        Scalars and the extents of arrays are passed as push constants, or,
        if they do not fit, scalars as one-element buffers and extents in
        the buffer at binding 0 (see `numba_vulkan.compiler`). Arrays are
        converted on the host where the device stores them differently:
        booleans as int32, 64-bit types as 32-bit ones where the kernel is
        narrowed, and arrays that are not C-contiguous as contiguous copies.
        """
        if stream is not None:
            if device is not None and runtime.get_device(device) is not stream.device:
                raise ValueError("the stream belongs to another device")
            target = stream.device
        else:
            target = runtime.get_device(device)
        # A launch plan: the kernel and the stored type of each scalar,
        # for arguments of a signature seen before. It skips typing.
        signature = _signature(args)
        if local_size is not None and type(local_size) is not tuple:
            local_size = (local_size,) if np.isscalar(local_size) else tuple(local_size)
        if signature is not None:
            plan_key = (
                signature,
                target,
                len(extent if extent is not None else groups),
                local_size,
            )
            plan = self._plans.get(plan_key)
            if plan is not None:
                kernel, scalars = plan
                hosts, shapes = [], []
                for arg, stored in zip(args, scalars):
                    if stored is None:
                        shapes.extend(arg.shape)
                        if not (arg._offset == 0 and arg.is_contiguous):
                            shapes.extend((arg._offset, *arg._steps))
                        hosts.append(arg)
                    else:
                        hosts.append(narrowing.convert(np.array([arg]), stored))
                self._launch_kernel(
                    kernel, target, hosts, shapes, extent, groups, stream
                )
                return
        mode = target.mode
        if self.narrow is not None:
            floats = self.narrow in (True, "floats") or (
                self.narrow == "ints" and mode.floats
            )
            ints = self.narrow in (True, "ints") or (
                self.narrow == "floats" and mode.ints
            )
            mode = mode._replace(floats=floats, ints=ints)
        argtypes, hosts, shapes, staged, convert = [], [], [], [], []
        on_host, host_arrays, scalars = False, [], []
        for arg in args:
            if isinstance(arg, runtime.DeviceArray):
                if arg._stored != narrowing.stored_dtype(arg.dtype, mode):
                    raise ValueError(
                        f"a {arg.dtype} device array on {arg.device.info.name} holds "
                        f"{arg._stored} elements, which does not match a kernel "
                        f"compiled with narrow={self.narrow}"
                    )
                # A view that starts elsewhere or has gaps passes its first
                # element and steps after its extents (see compiler).
                plain = arg._offset == 0 and arg.is_contiguous
                layout = "C" if plain else "A"
                # Records are typed as stored: narrowing moves their fields.
                dtype = arg._stored if arg.dtype.names else arg.dtype
                argtypes.append(_device_array_type(dtype, arg.ndim, layout))
                shapes.extend(arg.shape)
                if not plain:
                    shapes.extend((arg._offset, *arg._steps))
                hosts.append(arg)
                scalars.append(None)
            elif isinstance(arg, np.ndarray):
                on_host = True
                stored = narrowing.stored_dtype(arg.dtype, mode)
                if arg.dtype == np.float16:
                    argtypes.append(HalfArray(arg.ndim, "C"))
                elif arg.dtype.names:
                    # Records are typed as stored: narrowing moves their fields.
                    argtypes.append(_device_array_type(stored, arg.ndim))
                else:
                    argtypes.append(typeof(arg).copy(layout="C", readonly=False))
                shapes.extend(arg.shape)
                if stored != arg.dtype or not arg.flags.c_contiguous:
                    # Converted on the host: booleans are int32 on the device
                    # (SPIR-V has no storable bool), 64-bit types are 32-bit
                    # where the kernel is narrowed, and strided or
                    # Fortran-ordered arrays travel as contiguous copies.
                    # Conversion waits for the kernel, which tells whether
                    # the array is read at all.
                    staged.append((len(hosts), arg))
                    convert.append((len(hosts), arg, stored))
                host_arrays.append((len(hosts), arg, stored))
                hosts.append(arg)
            else:
                ty = typeof(arg)
                argtypes.append(ty)
                stored = narrowing.stored_dtype(np.dtype(str(ty)), mode)
                hosts.append(narrowing.convert(np.array([arg]), stored))
                scalars.append(stored)
        ndim = len(extent if extent is not None else groups)
        if local_size is not None:
            local_size = (local_size,) if np.isscalar(local_size) else tuple(local_size)
            ndim = max(ndim, len(local_size))
        aliases = _aliases(args)
        key = (tuple(argtypes), ndim, mode, local_size, aliases)
        kernel = self._launched.get(key)
        if kernel is None:
            kernel = self.compile(argtypes, ndim, mode, local_size, aliases)
            self._launched[key] = kernel
        if signature is not None:
            self._plans[plan_key] = (kernel, tuple(scalars))
        self._launch_kernel(
            kernel,
            target,
            hosts,
            shapes,
            extent,
            groups,
            stream,
            staged,
            convert,
            on_host,
            host_arrays,
        )

    def _launch_kernel(
        self,
        kernel,
        target,
        hosts,
        shapes,
        extent,
        groups,
        stream,
        staged=(),
        convert=(),
        on_host=False,
        host_arrays=(),
    ):
        """Run a compiled kernel: the second half of `_launch`.

        Parameters
        ----------
        kernel : CompiledKernel
            The kernel.
        target : numba_vulkan.runtime.Device
            The device.
        hosts : list
            One per argument: device and NumPy arrays, and scalars as
            one-element arrays of their stored type.
        shapes : list of int
            The extents of the arrays, and offsets and steps of views.
        extent, groups : tuple of int or None
            The grid, in invocations or in workgroups; see `_launch`.
        stream : numba_vulkan.runtime.Stream or None
            The stream, if any.
        staged : list of tuple
            Index in `hosts` and the NumPy array passed, for NumPy arrays
            the device stores differently: written back after the kernel.
        convert : list of tuple
            Index, NumPy array and stored type, for those to convert.
        on_host : bool
            Whether any argument is a NumPy array.
        host_arrays : list of tuple
            Index, NumPy array and stored type of every NumPy array.
        """
        if extent is not None:
            groups = [
                -(-int(n) // kernel.local_size[axis]) for axis, n in enumerate(extent)
            ]
        groups = _shape3(groups)
        for index, arg, stored in convert:
            check = arg_binding(index) in kernel.read_bindings
            hosts[index] = narrowing.convert(arg, stored, check)
        if 0 in groups:
            return
        push = _pack(kernel, hosts, shapes, groups) if kernel.push_format else b""
        for kind, index in kernel.push_sources:
            if kind == "arg":
                hosts[index] = None  # no buffer
        if kernel.args_pushed:
            # Element 0 receives the status of the kernel.
            meta = np.zeros(1, dtype=np.int32)
        else:
            # Element 0 receives the status of the kernel, the shapes follow.
            meta = np.array([0, *shapes], dtype=np.int32)
        if stream is not None:
            if kernel.print_binding is not None:
                raise TypeError("kernels launched on a stream cannot print")
            self._launch_stream(
                stream, kernel, tuple(groups), meta, hosts, push, host_arrays
            )
            return
        if _ASYNC and not on_host and kernel.print_binding is None:
            # Nothing to copy back: do not wait. Exceptions are reported by
            # the next synchronisation.
            target.launch(kernel, tuple(groups), [meta, *hosts], push)
            return
        target.run(kernel, tuple(groups), [meta, *hosts], push)
        for index, original in staged:
            if arg_binding(index) in kernel.written_bindings:
                copy = hosts[index]
                original[...] = copy != 0 if original.dtype == np.bool_ else copy
        if meta[STATUS_INDEX]:
            self._raise(int(meta[STATUS_INDEX]))

    def _launch_stream(self, stream, kernel, groups, meta, hosts, push, host_arrays):
        """Enqueue a launch on a stream, with copies for its host arrays.

        Parameters
        ----------
        stream : numba_vulkan.runtime.Stream
            The stream.
        kernel : CompiledKernel
            The kernel.
        groups : tuple of int
            The size of the grid in workgroups.
        meta : numpy.ndarray
            The status word, and the shapes without push constants.
        hosts : list
            The arguments as passed to the device (see `_launch`).
        push : bytes
            The push constants.
        host_arrays : list of tuple
            Index in `hosts`, the NumPy array passed and its stored type,
            for every NumPy array argument.

        Notes
        -----
        As in ``numba.cuda``, NumPy arrays are copied to the device on the
        stream before the kernel, and those the kernel writes back after
        it; they hold the results after ``stream.synchronize()``.
        """
        hosts = list(hosts)
        back = []
        for index, original, stored in host_arrays:
            if hosts[index] is None:
                continue
            on_device = runtime.DeviceArray(
                stream.device, original.shape, stored, _stored=stored
            )
            on_device.copy_to_device(hosts[index], stream=stream)
            hosts[index] = on_device
            if arg_binding(index) in kernel.written_bindings:
                back.append((on_device, original))
        stream.device.launch_stream(stream, kernel, groups, [meta, *hosts], push)
        for on_device, original in back:
            on_device._copy_out_stream(original, stream, cast=True)
        if not _ASYNC:
            stream.synchronize()

    def _raise(self, code):
        """Raise the exception that a kernel reported; see `raise_kernel_error`."""
        raise_kernel_error(self.py_func.__name__, code)


def raise_kernel_error(name, code, asynchronous=False):
    """Raise the exception that a kernel reported.

    Parameters
    ----------
    name : str
        Name of the kernel.
    code : int
        Status code left by the kernel.
    asynchronous : bool
        Whether the kernel was launched without waiting for it, so that
        the exception is raised later, by a synchronisation.

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
    where = f"raised in Vulkan kernel '{name}'"
    if location:
        where += f", in {location[0]} at {location[1]}:{location[2]}"
    if asynchronous:
        where += (
            ", by a launch since the last synchronisation (set "
            "NUMBA_VULKAN_SYNC=1 to check every launch at once)"
        )
    error.add_note(where)
    raise error


# Launches on device arrays return before the kernel has finished, unless
# NUMBA_VULKAN_SYNC=1.
_ASYNC = os.environ.get("NUMBA_VULKAN_SYNC", "0") == "0"


_FLOAT64_ARITHMETIC = re.compile(
    r"= (?:fadd|fsub|fmul|fdiv|frem|fneg)\b[^\n]*\bdouble\b|"
    r"call [^\n]*double @(?:_Z\d+\w+d\b|llvm\.\w+\.f64)"
)


def _pack(kernel, hosts, shapes, groups):
    """Pack the push constants of a launch.

    Parameters
    ----------
    kernel : CompiledKernel
        The kernel, which has push constants.
    hosts : list
        The arguments, scalars as one-element arrays of their stored type.
    shapes : list of int
        The extents of all array arguments, in order.
    groups : tuple of int
        The size of the grid in workgroups.

    Returns
    -------
    bytes
    """
    values = [
        hosts[k].item()
        if kind == "arg"
        else (shapes[k] if kind == "shape" else groups[k])
        for kind, k in kernel.push_sources
    ]
    return struct.pack(kernel.push_format, *values)


def _code_names(code):
    """The global and attribute names a code object and its nested ones use."""
    names = set(code.co_names)
    for const in code.co_consts:
        if inspect.iscode(const):
            names |= _code_names(const)
    return names


# Scalar types whose Numba type follows from the Python type alone.
_SCALARS = (float, bool, np.bool_, np.number)


def _signature(args):
    """What decides the kernel and the layout of a launch's arguments.

    Parameters
    ----------
    args : tuple
        The arguments of a launch.

    Returns
    -------
    tuple or None
        Per argument: the element type, stored type, dimensions and
        plainness of a device array, or the type of a scalar; then the
        arguments that share a buffer (`_aliases`). None if an argument is
        anything else (NumPy arrays among them), for which launches take
        the full path.
    """
    signature = []
    for arg in args:
        kind = type(arg)
        if kind is runtime.DeviceArray:
            signature.append(
                (
                    arg.dtype,
                    arg._stored,
                    len(arg.shape),
                    arg._offset == 0 and arg.is_contiguous,
                )
            )
        elif kind is int:
            if not -(1 << 63) <= arg < 1 << 63:
                return None  # typed as uint64
            signature.append(kind)
        elif isinstance(arg, _SCALARS):
            signature.append(kind)
        else:
            return None
    return (*signature, _aliases(args))


def _aliases(args):
    """Device array arguments that share the buffer of an earlier one.

    Parameters
    ----------
    args : tuple
        The arguments of a launch.

    Returns
    -------
    tuple of tuple
        ``(index, earlier)`` per such argument; see
        `numba_vulkan.compiler.compile_kernel`.
    """
    first, aliases = {}, []
    for index, arg in enumerate(args):
        if type(arg) is runtime.DeviceArray:
            earlier = first.setdefault(id(arg._buffer), index)
            if earlier != index:
                aliases.append((index, earlier))
    return tuple(aliases)


def _shape3(shape):
    """A shape of up to three extents, padded with ones to three."""
    if type(shape) is tuple and len(shape) == 3 and all(type(n) is int for n in shape):
        return shape
    shape = (shape,) if np.isscalar(shape) else tuple(int(n) for n in shape)
    if not 1 <= len(shape) <= 3:
        raise ValueError(f"expected 1 to 3 extents, got {shape}")
    return shape + (1,) * (3 - len(shape))


@functools.cache
def _holds_float64(ty):
    """Whether a type is float64 or a record with a float64 field."""
    if isinstance(ty, VulkanRecord):
        ty = ty.record
    if isinstance(ty, types.Record):
        return any(ty.typeof(name) == types.float64 for name in ty.fields)
    return ty == types.float64


def _record_type(record):
    """The Vulkan record type for a Numba record type.

    Raises
    ------
    VulkanUnsupportedError
        For fields other than booleans, integers and floats of 32 or 64
        bits (nested records, arrays, ``float16``, complex numbers).
    """
    for name in record.fields:
        fieldty = record.typeof(name)
        ok = isinstance(fieldty, (types.Boolean, types.Integer)) or (
            isinstance(fieldty, types.Float) and fieldty.bitwidth in (32, 64)
        )
        if not ok:
            raise VulkanUnsupportedError(
                f"record field '{name}' of type {fieldty} is not supported on "
                "Vulkan: fields must be booleans, integers, float32 or float64"
            )
    return VulkanRecord(record)


@functools.cache
def _device_array_type(dtype, ndim, layout="C"):
    """Numba type of a device array; layout ``"A"`` for views with gaps."""
    if dtype == np.float16:
        return HalfArray(ndim, layout)
    return types.Array(numpy_support.from_dtype(dtype), ndim, layout)


def jit(
    pyfunc=None,
    *,
    fastmath=False,
    narrow_math=False,
    boundscheck=False,
    error_model="numpy",
    narrow=None,
    cache=False,
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
    error_model : {'numpy', 'python'}
        What a division by zero does: ``'numpy'`` (the default, as in
        ``numba.cuda``) gives NumPy's result, ``'python'`` raises
        ``ZeroDivisionError`` from the launch, as Numba does on the CPU.
        The check costs a comparison per division.
    narrow : bool, {'ints', 'floats'} or None
        Whether a kernel computes with 32-bit integers and floats where its
        code says ``int64`` and ``float64``, which Numba uses for Python
        literals and all index arithmetic. ``None``, the default, narrows
        integers always, because 64-bit integer arithmetic is slow on GPUs
        (unless ``NUMBA_VULKAN_INT64=1``), and floats only on devices
        without ``float64``, with a warning. ``"ints"`` narrows only the
        integers, also on such devices; ``"floats"`` narrows only the
        floats; ``True`` narrows both, without a warning; ``False`` neither,
        so the kernel fails on devices without 64-bit types. Arrays of
        64-bit elements are converted on the host, and integers that do not
        fit raise ``OverflowError``. Only the setting of the kernel matters,
        not that of the functions it calls.
    cache : bool
        Keep compiled kernels on disk, so that later processes skip Numba's
        compilation, as with Numba's ``cache=True``. A kernel is compiled
        again when its source file or that of an ``@nv.jit`` function it
        calls changes. Changes to other functions it uses (``@overload``,
        functions of other targets) and to the values of global arrays are
        not noticed, as in Numba. Functions without a source file
        (interactive, ``exec``) are not cached.
    **options
        Accepted for compatibility with Numba's generic ``jit`` and ignored.

    Returns
    -------
    VulkanDispatcher
    """
    options["narrow_math"] = narrow_math
    options["fastmath"] = fastmath
    options["boundscheck"] = boundscheck
    options["error_model"] = error_model
    options["narrow"] = narrow
    options["cache"] = cache
    if pyfunc is None:
        return lambda f: VulkanDispatcher(f, options)
    return VulkanDispatcher(pyfunc, options)


_target = target_registry[TARGET_NAME]
jit_registry[_target] = jit
dispatcher_registry[_target] = VulkanDispatcher
