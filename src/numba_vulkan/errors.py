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


class VulkanPrecisionWarning(UserWarning):
    """A kernel computes in float32 where its code says float64.

    Issued once per kernel when it is compiled for a device without
    float64 support; see `numba_vulkan.narrowing`.
    """


class VulkanPerformanceWarning(UserWarning):
    """A kernel computes with float64, which is slow on most GPUs.

    Issued once per kernel when it is compiled. ``NUMBA_VULKAN_WARNINGS=0``
    silences it, as does ``warnings.filterwarnings("ignore",
    category=nv.VulkanPerformanceWarning)``.
    """


class VulkanValidationWarning(UserWarning):
    """Vulkan's validation layer found a problem in how a device is used.

    Issued only with ``NUMBA_VULKAN_DEBUG=1``, for every warning and error
    the layer reports; `numba_vulkan.runtime.validation_messages` returns
    them as well.
    """


class VulkanSupportError(RuntimeError):
    """A device lacks a capability that a kernel needs.

    Raised when a kernel is launched, for example on a device without
    float64 support, or when no usable Vulkan device exists.
    """
