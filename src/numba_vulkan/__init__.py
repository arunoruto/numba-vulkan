"""Proof of concept: a Vulkan compute target for Numba."""

from numba_vulkan.codegen import CompiledKernel
from numba_vulkan.dispatcher import VulkanDispatcher, jit
from numba_vulkan.errors import (
    SpirvCodegenError,
    VulkanPerformanceWarning,
    VulkanPrecisionWarning,
    VulkanSupportError,
    VulkanUnsupportedError,
)
from numba_vulkan.runtime import (
    DeviceArray,
    device_array,
    device_array_like,
    get_device,
    list_devices,
    select_device,
    synchronize,
    to_device,
)
from numba_vulkan.stubs import (
    atomic,
    barrier,
    global_id,
    group_id,
    local,
    local_id,
    local_size,
    num_groups,
    shared,
    syncthreads,
)

__all__ = [
    "CompiledKernel",
    "DeviceArray",
    "SpirvCodegenError",
    "VulkanDispatcher",
    "VulkanPerformanceWarning",
    "VulkanPrecisionWarning",
    "VulkanSupportError",
    "VulkanUnsupportedError",
    "atomic",
    "barrier",
    "device_array",
    "device_array_like",
    "get_device",
    "global_id",
    "group_id",
    "jit",
    "list_devices",
    "local_id",
    "local",
    "local_size",
    "num_groups",
    "select_device",
    "shared",
    "synchronize",
    "syncthreads",
    "to_device",
]

# Registers target="vulkan" with numba.vectorize and numba.guvectorize.
from numba_vulkan import vectorizers  # noqa: E402, F401
