"""NUMBA_VULKAN_DEBUG: Vulkan's validation layer, and what it found."""

import os
import subprocess
import sys
import warnings

import numpy as np
import pytest
import vulkan as vk

import numba_vulkan as nv
from numba_vulkan import codegen, runtime


def _layer_installed():
    layers = {p.layerName for p in vk.vkEnumerateInstanceLayerProperties()}
    return runtime._VALIDATION_LAYER in layers


@pytest.mark.skipif(not _layer_installed(), reason="validation layer not installed")
def test_debug_mode_reports_what_the_layer_finds():
    # The instance is created once per process, so this runs in a new one.
    script = """
import vulkan as vk, numba_vulkan as nv
from numba_vulkan import runtime
device = nv.get_device()
info = vk.VkBufferCreateInfo(
    sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, size=0,
    usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
)
try:
    vk.vkCreateBuffer(device.handle, info, None)
except Exception:
    pass
print(runtime.validation_messages())
"""
    env = dict(os.environ, NUMBA_VULKAN_DEBUG="1")
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout
    assert "'error'" in out and "size" in out


HAZARD = """
import vulkan as vk, numba_vulkan as nv
from numba_vulkan import runtime
dev = nv.get_device()
d = dev.handle
info = vk.VkBufferCreateInfo(
    sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, size=256,
    usage=vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
)
buf = vk.vkCreateBuffer(d, info, None)
req = vk.vkGetBufferMemoryRequirements(d, buf)
kind = next(
    i for i in range(dev.memory.memoryTypeCount) if req.memoryTypeBits & (1 << i)
)
mem = vk.vkAllocateMemory(d, vk.VkMemoryAllocateInfo(
    sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    allocationSize=req.size, memoryTypeIndex=kind), None)
vk.vkBindBufferMemory(d, buf, mem, 0)
pool = vk.vkCreateCommandPool(d, vk.VkCommandPoolCreateInfo(
    sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    queueFamilyIndex=dev.family), None)
cmd = vk.vkAllocateCommandBuffers(d, vk.VkCommandBufferAllocateInfo(
    sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, commandPool=pool,
    level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1))[0]
vk.vkBeginCommandBuffer(cmd, vk.VkCommandBufferBeginInfo(
    sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO))
vk.vkCmdFillBuffer(cmd, buf, 0, 256, 1)
vk.vkCmdFillBuffer(cmd, buf, 0, 256, 2)  # no barrier: write after write
vk.vkEndCommandBuffer(cmd)
vk.vkQueueSubmit(dev.queue, 1, [vk.VkSubmitInfo(
    sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO, commandBufferCount=1,
    pCommandBuffers=[cmd])], None)
vk.vkQueueWaitIdle(dev.queue)
print(runtime.validation_messages())
"""


@pytest.mark.skipif(not _layer_installed(), reason="validation layer not installed")
@pytest.mark.parametrize(("level", "found"), [("1", False), ("2", True)])
def test_level_2_checks_synchronization(level, found):
    env = dict(os.environ, NUMBA_VULKAN_DEBUG=level)
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", HAZARD],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout
    assert ("WRITE_AFTER_WRITE" in out) == found


def test_debug_levels(monkeypatch):
    for value, level in (("", 0), ("0", 0), ("1", 1), ("2", 2), ("yes", 1)):
        monkeypatch.setenv(runtime.DEBUG_ENV_VAR, value)
        assert runtime.debug_level() == level


def test_debug_mode_without_the_layer_warns(monkeypatch):
    monkeypatch.setattr(vk, "vkEnumerateInstanceLayerProperties", list)
    with pytest.warns(nv.VulkanValidationWarning, match="not installed"):
        assert runtime._debug_setup() == ([], [])


def test_validation_messages_can_be_cleared(monkeypatch):
    monkeypatch.setattr(runtime, "_messages", [("warning", "a"), ("error", "b")])
    assert runtime.validation_messages(clear=True) == [("warning", "a"), ("error", "b")]
    assert runtime.validation_messages() == []


def _copy(x, out):
    i = nv.global_id(0)
    if i < x.size:
        out[i] = x[i] + x[i]


@pytest.mark.parametrize(
    ("dtype", "storage8"), [(np.int8, True), (np.uint8, True), (np.float32, False)]
)
def test_8_bit_buffers_declare_their_storage_capability(run, dtype, storage8):
    kernel = nv.jit(_copy)
    x = np.arange(16).astype(dtype)
    out = np.zeros_like(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run(kernel, x.size, x, out)
    np.testing.assert_array_equal(out, (x + x).astype(dtype))
    compiled = list(kernel._kernels.values())[-1]
    assert ("storage8" in compiled.capabilities) == storage8
    assert codegen.storage8_capability(compiled.spirv) == compiled.spirv
