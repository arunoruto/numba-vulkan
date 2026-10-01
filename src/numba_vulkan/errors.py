"""Exceptions raised by the Vulkan target."""

from numba.core import errors


class VulkanUnsupportedError(errors.NumbaError):
    """A kernel uses something Vulkan compute shaders cannot express.

    Raised while compiling, for example for slices, recursion or float64
    transcendental functions. It is a `numba.core.errors.NumbaError`, so
    Numba adds the source location of the offending statement.
    """


class SpirvCodegenError(errors.NumbaError):
    """LLVM's SPIR-V backend could not translate a kernel.

    Raised when the backend aborts, when the generated module fails the
    built-in sanity check, or when ``spirv-val`` rejects it (with
    ``NUMBA_VULKAN_VALIDATE=1``). It usually points at a gap in this
    package rather than at a mistake in the kernel.
    """


class VulkanSupportError(RuntimeError):
    """A device lacks a capability that a kernel needs.

    Raised when a kernel is launched, for example on a device without
    float64 support, or when no usable Vulkan device exists.
    """
