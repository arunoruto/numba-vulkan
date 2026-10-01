"""Proof of concept: a Vulkan compute target for Numba."""

from numba_vulkan.codegen import CompiledKernel
from numba_vulkan.dispatcher import VulkanDispatcher, jit
from numba_vulkan.errors import (
    SpirvCodegenError,
    VulkanSupportError,
    VulkanUnsupportedError,
)
from numba_vulkan.runtime import get_device, list_devices, select_device
from numba_vulkan.stubs import global_id

__all__ = [
    "CompiledKernel",
    "SpirvCodegenError",
    "VulkanDispatcher",
    "VulkanSupportError",
    "VulkanUnsupportedError",
    "get_device",
    "global_id",
    "jit",
    "list_devices",
    "select_device",
]
