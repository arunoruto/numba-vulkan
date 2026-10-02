"""The Vulkan target, registered with Numba's target extension API."""

from functools import cached_property

import llvmlite.binding as llvm
from numba.core import datamodel, itanium_mangler, typing
from numba.core.base import BaseContext, _wrap_impl
from numba.core.callconv import MinimalCallConv
from numba.core.compiler import Flags
from numba.core.descriptors import TargetDescriptor
from numba.core.dispatcher import Dispatcher
from numba.core.options import TargetOptions
from numba.core.target_extension import GPU, target_registry

from numba_vulkan import codegen
from numba_vulkan.errors import VulkanUnsupportedError
from numba_vulkan.models import vulkan_data_manager

TARGET_NAME = "vulkan"


class Vulkan(GPU):
    """Marks the Vulkan target in Numba's target hierarchy.

    It derives from Numba's generic ``GPU`` target, so overloads registered
    for ``target="gpu"`` or ``target="generic"`` apply to it as well.
    """


target_registry[TARGET_NAME] = Vulkan


class VulkanTypingContext(typing.BaseContext):
    """Typing context: which functions and operations type-check on Vulkan."""

    def load_additional_registries(self):
        """Install the typing registries beyond Numba's built-in one.

        These are the Vulkan-specific declarations and Numba's ``math``,
        ``cmath``, NumPy and enum declarations.
        """
        from numba.core.typing import cmathdecl, enumdecl, mathdecl, npydecl

        from numba_vulkan import vkdecl

        self.install_registry(vkdecl.registry)
        self.install_registry(mathdecl.registry)
        self.install_registry(cmathdecl.registry)
        self.install_registry(npydecl.registry)
        self.install_registry(enumdecl.registry)

    def resolve_value_type(self, val):
        """Type a Python value used as a global or closure variable.

        A function compiled for the CPU (``numba.njit``) is replaced by a
        Vulkan dispatcher for the same Python function, so that calling it
        from a kernel recompiles it for Vulkan.

        Parameters
        ----------
        val : object
            The value.

        Returns
        -------
        numba.types.Type
        """
        from numba_vulkan.dispatcher import VulkanDispatcher

        # A CPU-jitted function called from a kernel is recompiled for Vulkan.
        if isinstance(val, Dispatcher):
            try:
                val = val.__vulkan_dispatcher
            except AttributeError:
                disp = VulkanDispatcher(val.py_func)
                val.__vulkan_dispatcher = disp
                val = disp
        return super().resolve_value_type(val)


class VulkanCallConv(MinimalCallConv):
    """Calling convention of functions compiled for Vulkan.

    Numba's minimal convention: the return value is written through a
    pointer argument and the function returns a status code. It never
    reaches the shader, because everything is inlined.
    """

    pass


class VulkanTargetContext(BaseContext):
    """Target context: how typed operations are lowered to LLVM IR.

    Parameters
    ----------
    typingctx : VulkanTypingContext
        The typing context.
    target : str, optional
        Name of the target in Numba's registry.

    Attributes
    ----------
    narrow_math : bool
        Whether float64 transcendental functions are evaluated in float32
        when libclc is unavailable. Set per compiled function through
        `subtarget`.
    fast_math : bool
        Whether float32 math uses the device's built-in functions instead of
        libclc. Set per compiled function through `subtarget`.
    """

    implement_powi_as_math_call = True
    strict_alignment = True
    # Evaluate float64 transcendental functions in float32 (set per function).
    narrow_math = False
    # Prefer the device's built-in float32 math over libclc (set per function).
    fast_math = False

    def __init__(self, typingctx, target=TARGET_NAME):
        super().__init__(typingctx, target)
        self.data_model_manager = vulkan_data_manager.chain(datamodel.default_manager)

    @property
    def enable_boundscheck(self):
        """Bounds checking is always off; shaders cannot raise exceptions.

        Returns
        -------
        bool
        """
        return False

    def create_module(self, name):
        """Create an empty LLVM module for this target.

        Parameters
        ----------
        name : str
            Module name.

        Returns
        -------
        llvmlite.ir.Module
        """
        return self._internal_codegen._create_empty_module(name)

    def init(self):
        """Create the code generator. Called once by Numba's base context."""
        self._internal_codegen = codegen.VulkanCodegen("numba.vulkan.jit")

    def load_additional_registries(self):
        # Imported for their side effect of filling Numba's builtin registry.
        """Install the lowering registries beyond Numba's built-in one.

        Importing Numba's implementation modules fills its built-in registry
        with number, tuple, range and array metadata operations. The Vulkan
        registries add array element access, ``global_id`` and ``math``.
        """
        from numba.cpython import (  # noqa: F401
            enumimpl,
            iterators,
            numbers,
            rangeobj,
            slicing,
            tupleobj,
        )
        from numba.np import arrayobj  # noqa: F401

        from numba.np import npyimpl

        from numba_vulkan import mathfuncs, mathimpl, vkimpl  # noqa: F401

        self.install_registry(npyimpl.registry)
        self.install_registry(vkimpl.registry)
        self.install_registry(mathimpl.registry)

    def get_function(self, fn, sig, _firstcall=True):
        """Find the implementation of a function for a signature.

        Parameters
        ----------
        fn : callable or numba.types.Type
            The function.
        sig : numba.core.typing.Signature
            Signature of the call.
        _firstcall : bool, optional
            Used internally by Numba.

        Returns
        -------
        callable
            ``impl(builder, args)`` emitting the call.

        Notes
        -----
        Integer-exponent powers are intercepted here; see
        `numba_vulkan.mathimpl.power_override`.
        """
        from numba_vulkan import mathimpl

        # Numba registers these per concrete type, which a registry cannot
        # outrank, so they are intercepted here.
        impl = mathimpl.power_override(fn, sig)
        if impl is not None:
            return _wrap_impl(impl, self, sig)
        return super().get_function(fn, sig, _firstcall)

    def _compile_subroutine_no_cache(self, builder, impl, sig, locals=None, flags=None):
        """Compile a Python helper used by a lowering implementation.

        Numba's default flags select the Python error model, which adds a
        zero-division check and an early exit to every division. Shaders
        cannot raise, and the extra exits produce control flow that LLVM's
        SPIR-V structurizer mishandles, so the NumPy error model is used.

        Parameters
        ----------
        builder : llvmlite.ir.IRBuilder
            Builder of the calling function.
        impl : function
            The Python helper.
        sig : numba.core.typing.Signature
            Signature to compile it for.
        locals : dict, optional
            Type annotations for local variables.
        flags : numba.core.compiler.Flags, optional
            Compiler flags; defaults suited to this target when omitted.

        Returns
        -------
        numba.core.compiler.CompileResult
        """
        if flags is None:
            flags = Flags()
            flags.error_model = "numpy"
        return super()._compile_subroutine_no_cache(builder, impl, sig, locals, flags)

    def get_ufunc_info(self, ufunc_key):
        """Look up how a NumPy ufunc is lowered.

        Parameters
        ----------
        ufunc_key : numpy.ufunc
            The ufunc.

        Returns
        -------
        dict
            Loop signature to implementation; see `numba_vulkan.ufuncs`.

        Raises
        ------
        KeyError
            If the ufunc is not supported on Vulkan.
        """
        from numba_vulkan import ufuncs

        return ufuncs.get_ufunc_info(ufunc_key)

    def make_constant_array(self, builder, aryty, arr):
        """Reject NumPy arrays used as global constants.

        Numba would emit the data as an LLVM global and index it through a
        pointer, which the SPIR-V backend cannot translate.

        Parameters
        ----------
        builder : llvmlite.ir.IRBuilder
            Builder positioned where the code is emitted.
        aryty : numba.types.Array
            Type of the array.
        arr : numpy.ndarray
            The array.

        Raises
        ------
        VulkanUnsupportedError
            Always.
        """
        raise VulkanUnsupportedError(
            "global NumPy arrays cannot be used inside Vulkan kernels yet; "
            "pass the array as an argument instead"
        )

    def codegen(self):
        """The code generator of this context.

        Returns
        -------
        numba_vulkan.codegen.VulkanCodegen
        """
        return self._internal_codegen

    @cached_property
    def target_data(self):
        """LLVM data layout of the SPIR-V target, used for type sizes.

        Returns
        -------
        llvmlite.binding.TargetData
        """
        return llvm.create_target_data(self._internal_codegen._data_layout)

    @cached_property
    def call_conv(self):
        """The calling convention of compiled functions.

        Returns
        -------
        VulkanCallConv
        """
        return VulkanCallConv(self)

    def mangler(self, name, argtypes, *, abi_tags=(), uid=None):
        """Mangle a function name with its argument types.

        Parameters
        ----------
        name : str
            Qualified function name.
        argtypes : sequence of numba.types.Type
            Argument types.
        abi_tags : sequence of str, optional
            ABI tags to embed.
        uid : int, optional
            Unique identifier of the function.

        Returns
        -------
        str
            The symbol name, in Itanium C++ ABI style.
        """
        return itanium_mangler.mangle(name, argtypes, abi_tags=abi_tags, uid=uid)


class VulkanTargetOptions(TargetOptions):
    """Options accepted by the target; none beyond Numba's defaults."""

    pass


class VulkanTarget(TargetDescriptor):
    """Target descriptor: gives access to the typing and target contexts.

    The contexts are created on first use, so importing the package does
    not initialise LLVM.

    Parameters
    ----------
    name : str
        Name of the target in Numba's registry.
    """

    options = VulkanTargetOptions

    def __init__(self, name):
        self._typingctx = None
        self._targetctx = None
        super().__init__(name)

    @property
    def typing_context(self):
        """The shared typing context.

        Returns
        -------
        VulkanTypingContext
        """
        if self._typingctx is None:
            self._typingctx = VulkanTypingContext()
        return self._typingctx

    @property
    def target_context(self):
        """The shared target context.

        Returns
        -------
        VulkanTargetContext
        """
        if self._targetctx is None:
            self._targetctx = VulkanTargetContext(self.typing_context)
        return self._targetctx


vulkan_target = VulkanTarget(TARGET_NAME)
