"""Minimal Vulkan compute runtime: devices, device arrays and kernel launch."""

import itertools
import math
import os
import warnings
import weakref
from dataclasses import dataclass, field

import numpy as np
import vulkan as vk

from numba_vulkan import narrowing
from numba_vulkan.buffers import print_formats
from numba_vulkan.errors import VulkanSupportError

_DEVICE_TYPES = {
    vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: "discrete",
    vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: "integrated",
    vk.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: "virtual",
    vk.VK_PHYSICAL_DEVICE_TYPE_CPU: "cpu",
}
_PREFERENCE = ["discrete", "integrated", "virtual", "other", "cpu"]
_API_VERSION = vk.VK_MAKE_VERSION(1, 2, 0)


@dataclass
class DeviceInfo:
    """A Vulkan physical device.

    Attributes
    ----------
    index : int
        Position in `list_devices`.
    name : str
        Device name reported by the driver.
    kind : {'discrete', 'integrated', 'virtual', 'cpu', 'other'}
        Device type.
    float64, int64, int16, int8 : bool
        Whether shaders may use the respective type.
    handle : object
        The ``VkPhysicalDevice`` handle.
    int64_atomics : bool
        Whether shaders may apply atomic operations to 64-bit integers.
    float16 : bool
        Whether shaders may compute with ``float16`` values.
    storage16 : bool
        Whether buffers may hold 16-bit values (``float16`` arrays).
    float32_atomic_add : bool
        Whether shaders may add to ``float32`` values atomically, in
        buffers and in shared memory (``VK_EXT_shader_atomic_float``).
    max_local_size : tuple of int
        Largest workgroup extent along each axis.
    max_local_invocations : int
        Largest number of invocations in one workgroup.
    max_groups : tuple of int
        Largest number of workgroups along each axis of a dispatch.
    max_shared_memory : int
        Bytes of workgroup-shared memory available to a kernel.
    """

    index: int
    name: str
    kind: str
    float64: bool
    int64: bool
    int16: bool
    int8: bool
    handle: object
    int64_atomics: bool = False
    float16: bool = False
    storage16: bool = False
    float32_atomic_add: bool = False
    max_local_size: tuple = (128, 128, 64)
    max_local_invocations: int = 128
    max_groups: tuple = (65535, 65535, 65535)
    max_shared_memory: int = 16384

    def __repr__(self):
        return f"<{self.index}: {self.name} ({self.kind})>"


_instance = None
_infos = None
_devices = {}
_current = None


def _get_instance():
    """The process-wide Vulkan instance, created on first use.

    Returns
    -------
    object
        The ``VkInstance`` handle.
    """
    global _instance
    if _instance is None:
        app = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="numba-vulkan",
            applicationVersion=1,
            pEngineName="numba-vulkan",
            engineVersion=1,
            apiVersion=_API_VERSION,
        )
        info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, pApplicationInfo=app
        )
        _instance = vk.vkCreateInstance(info, None)
    return _instance


def _vulkan11_features(handle):
    """The Vulkan 1.1 features of a device.

    Parameters
    ----------
    handle : object
        A ``VkPhysicalDevice`` handle.

    Returns
    -------
    object
        The filled ``VkPhysicalDeviceVulkan11Features`` structure.
    """
    feats = vk.VkPhysicalDeviceVulkan11Features(
        sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES
    )
    feats2 = vk.VkPhysicalDeviceFeatures2(
        sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, pNext=feats
    )
    vk.vkGetPhysicalDeviceFeatures2(handle, feats2)
    return feats


def _vulkan12_features(handle):
    """The Vulkan 1.2 features of a device.

    Parameters
    ----------
    handle : object
        A ``VkPhysicalDevice`` handle.

    Returns
    -------
    object
        The filled ``VkPhysicalDeviceVulkan12Features`` structure.
    """
    feats = vk.VkPhysicalDeviceVulkan12Features(
        sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
    )
    feats2 = vk.VkPhysicalDeviceFeatures2(
        sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, pNext=feats
    )
    vk.vkGetPhysicalDeviceFeatures2(handle, feats2)
    return feats


_FLOAT_ATOMICS = "VK_EXT_shader_atomic_float"


def _float_atomic_features(handle):
    """The float atomic features of a device, if it has the extension.

    Parameters
    ----------
    handle : object
        A ``VkPhysicalDevice`` handle.

    Returns
    -------
    object or None
        The filled ``VkPhysicalDeviceShaderAtomicFloatFeaturesEXT``
        structure, or ``None`` without ``VK_EXT_shader_atomic_float``.
    """
    names = {
        e.extensionName for e in vk.vkEnumerateDeviceExtensionProperties(handle, None)
    }
    if _FLOAT_ATOMICS not in names:
        return None
    feats = vk.VkPhysicalDeviceShaderAtomicFloatFeaturesEXT(
        sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT
    )
    feats2 = vk.VkPhysicalDeviceFeatures2(
        sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, pNext=feats
    )
    vk.vkGetPhysicalDeviceFeatures2(handle, feats2)
    return feats


def list_devices(refresh=False):
    """All Vulkan 1.2+ devices that expose a compute queue.

    Parameters
    ----------
    refresh : bool
        Ask the Vulkan loader again instead of returning the devices found
        by the first call.

    Returns
    -------
    list of DeviceInfo
    """
    global _infos
    if _infos is not None and not refresh:
        return list(_infos)
    found = []
    for handle in vk.vkEnumeratePhysicalDevices(_get_instance()):
        props = vk.vkGetPhysicalDeviceProperties(handle)
        if props.apiVersion < _API_VERSION:
            continue
        feats = vk.vkGetPhysicalDeviceFeatures(handle)
        feats12 = _vulkan12_features(handle)
        feats11 = _vulkan11_features(handle)
        float_atomics = _float_atomic_features(handle)
        limits = props.limits
        found.append(
            DeviceInfo(
                index=len(found),
                name=props.deviceName,
                kind=_DEVICE_TYPES.get(props.deviceType, "other"),
                float64=bool(feats.shaderFloat64),
                int64=bool(feats.shaderInt64),
                int16=bool(feats.shaderInt16),
                int8=bool(feats12.shaderInt8),
                handle=handle,
                int64_atomics=bool(
                    feats12.shaderBufferInt64Atomics
                    and feats12.shaderSharedInt64Atomics
                ),
                float16=bool(feats12.shaderFloat16),
                storage16=bool(feats11.storageBuffer16BitAccess),
                float32_atomic_add=bool(
                    float_atomics is not None
                    and float_atomics.shaderBufferFloat32AtomicAdd
                    and float_atomics.shaderSharedFloat32AtomicAdd
                ),
                max_local_size=tuple(limits.maxComputeWorkGroupSize),
                max_local_invocations=limits.maxComputeWorkGroupInvocations,
                max_groups=tuple(limits.maxComputeWorkGroupCount),
                max_shared_memory=limits.maxComputeSharedMemorySize,
            )
        )
    _infos = found
    return list(found)


class Device:
    """A logical device with one compute queue and a pipeline cache.

    Parameters
    ----------
    info : DeviceInfo
        The physical device to open.

    Attributes
    ----------
    info : DeviceInfo
        The physical device.
    handle : object
        The ``VkDevice`` handle.
    queue : object
        The compute queue.
    mode : numba_vulkan.narrowing.Mode
        The 64-bit types the device lacks.
    pool_limit : int
        Number of bytes of released buffers kept for reuse; set initially
        from the environment variable ``NUMBA_VULKAN_POOL_MB`` (default
        1024).

    Notes
    -----
    A device is not thread-safe: launch kernels and transfer arrays from
    one thread at a time.
    """

    def __init__(self, info):
        self.info = info
        phys = info.handle
        families = vk.vkGetPhysicalDeviceQueueFamilyProperties(phys)
        self.family = next(
            i for i, f in enumerate(families) if f.queueFlags & vk.VK_QUEUE_COMPUTE_BIT
        )
        queue_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=self.family,
            queueCount=1,
            pQueuePriorities=[1.0],
        )
        features = vk.VkPhysicalDeviceFeatures(
            shaderFloat64=info.float64,
            shaderInt64=info.int64,
            shaderInt16=info.int16,
        )
        extensions, chain = [], {}
        if info.float32_atomic_add:
            chain["pNext"] = vk.VkPhysicalDeviceShaderAtomicFloatFeaturesEXT(
                sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
                shaderBufferFloat32AtomicAdd=True,
                shaderSharedFloat32AtomicAdd=True,
            )
            extensions.append(_FLOAT_ATOMICS)
        features12 = vk.VkPhysicalDeviceVulkan12Features(
            sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            shaderInt8=info.int8,
            shaderBufferInt64Atomics=info.int64_atomics,
            shaderSharedInt64Atomics=info.int64_atomics,
            shaderFloat16=info.float16,
            **chain,
        )
        features11 = vk.VkPhysicalDeviceVulkan11Features(
            sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
            storageBuffer16BitAccess=info.storage16,
            pNext=features12,
        )
        create = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext=features11,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_info],
            pEnabledFeatures=features,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions or None,
        )
        self.handle = vk.vkCreateDevice(phys, create, None)
        self.queue = vk.vkGetDeviceQueue(self.handle, self.family, 0)
        self.memory = vk.vkGetPhysicalDeviceMemoryProperties(phys)
        pool = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=self.family,
        )
        self.command_pool = vk.vkCreateCommandPool(self.handle, pool, None)
        # One command buffer for transfers and one for kernels, so that the
        # recording of a kernel launch can be submitted again unchanged.
        self._transfer_commands, self._launch_commands = vk.vkAllocateCommandBuffers(
            self.handle,
            vk.VkCommandBufferAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                commandPool=self.command_pool,
                level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=2,
            ),
        )
        self._begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        )
        self._submit_info = {
            id(cmd): [
                vk.VkSubmitInfo(
                    sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    commandBufferCount=1,
                    pCommandBuffers=[cmd],
                )
            ]
            for cmd in (self._transfer_commands, self._launch_commands)
        }
        # What the kernel command buffer currently holds: (pipeline, groups).
        self._recorded = None
        # Asynchronous launches not known to have finished, oldest first,
        # and buffers released while they may still use them.
        self._pending = []
        self._deferred = []
        self._submitted = self._retired = 0
        self._pipelines = {}
        # 64-bit types this device cannot use and kernels must do without.
        self.mode = narrowing.Mode(
            floats=not info.float64,
            ints=not info.int64,
            float_atomics=info.float32_atomic_add,
        )
        # Released buffers by (size, mappable), kept for reuse.
        self._free = {}
        self._pooled = 0
        self.pool_limit = int(os.environ.get("NUMBA_VULKAN_POOL_MB", "1024")) << 20

    # -- kernels ---------------------------------------------------------

    def check_support(self, kernel):
        """Check that this device can run a kernel.

        Parameters
        ----------
        kernel : numba_vulkan.codegen.CompiledKernel
            The compiled kernel.

        Raises
        ------
        VulkanSupportError
            If the device lacks a capability the kernel needs.
        """
        missing = [
            c for c in sorted(kernel.capabilities) if not getattr(self.info, c, False)
        ]
        if missing:
            raise VulkanSupportError(
                f"kernel '{kernel.name}' needs {', '.join(missing)} support, which "
                f"{self.info.name} does not provide"
            )
        info, local = self.info, kernel.local_size
        if math.prod(local) > info.max_local_invocations or any(
            n > limit for n, limit in zip(local, info.max_local_size)
        ):
            raise VulkanSupportError(
                f"kernel '{kernel.name}' has workgroups of {local}, but {info.name} "
                f"allows at most {info.max_local_invocations} invocations and "
                f"{info.max_local_size} per axis"
            )
        if kernel.shared_bytes > info.max_shared_memory:
            raise VulkanSupportError(
                f"kernel '{kernel.name}' uses {kernel.shared_bytes} bytes of shared "
                f"memory, but {info.name} provides {info.max_shared_memory}"
            )

    def _pipeline(self, kernel):
        """Create (or fetch) the compute pipeline of a kernel.

        Parameters
        ----------
        kernel : numba_vulkan.codegen.CompiledKernel
            The compiled kernel.

        Returns
        -------
        _Pipeline

        Raises
        ------
        VulkanSupportError
            If the device lacks a capability the kernel needs.
        """
        key = id(kernel)
        if key in self._pipelines:
            return self._pipelines[key]
        self.check_support(kernel)
        dev = self.handle
        module = vk.vkCreateShaderModule(
            dev,
            vk.VkShaderModuleCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(kernel.spirv),
                pCode=kernel.spirv,
            ),
            None,
        )
        bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=i,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            )
            for i in range(kernel.num_bindings)
        ]
        set_layout = vk.vkCreateDescriptorSetLayout(
            dev,
            vk.VkDescriptorSetLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                bindingCount=len(bindings),
                pBindings=bindings,
            ),
            None,
        )
        layout = vk.vkCreatePipelineLayout(
            dev,
            vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1,
                pSetLayouts=[set_layout],
            ),
            None,
        )
        stage = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=module,
            pName="main",
        )
        pipeline = vk.vkCreateComputePipelines(
            dev,
            vk.VK_NULL_HANDLE,
            1,
            [
                vk.VkComputePipelineCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                    # allows grids beyond the device's limits, see _dispatch
                    flags=_DISPATCH_BASE,
                    stage=stage,
                    layout=layout,
                )
            ],
            None,
        )[0]
        pool = vk.vkCreateDescriptorPool(
            dev,
            vk.VkDescriptorPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                maxSets=1,
                poolSizeCount=1,
                pPoolSizes=[
                    vk.VkDescriptorPoolSize(
                        type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        descriptorCount=len(bindings),
                    )
                ],
            ),
            None,
        )
        desc_set = vk.vkAllocateDescriptorSets(
            dev,
            vk.VkDescriptorSetAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptorPool=pool,
                descriptorSetCount=1,
                pSetLayouts=[set_layout],
            ),
        )[0]
        # The kernel is kept alive alongside its pipeline so id() stays unique.
        # Constant arrays are uploaded once and live as long as the pipeline.
        constants = []
        for binding in sorted(kernel.constants):
            data = kernel.constants[binding].reshape(-1).view(np.uint8)
            buffer = self._acquire(max(data.size, 4), host=False)
            self._upload(buffer, data)
            constants.append((buffer, max(data.size, 4)))
        self._pipelines[key] = _Pipeline(
            pipeline, layout, desc_set, constants, (set_layout, pool, module, kernel)
        )
        return self._pipelines[key]

    # -- buffers ---------------------------------------------------------

    def _memory_type(self, allowed, host):
        """Choose a memory type for a buffer.

        Parameters
        ----------
        allowed : int
            Bit mask of memory types the buffer may use.
        host : bool
            Whether the memory must be mappable. Mappable memory prefers
            CPU-cached types, because reading results back from uncached
            (write-combined) memory is very slow. Other memory prefers
            types local to the device, and among those mappable and cached
            ones, which integrated GPUs and CPU devices offer.

        Returns
        -------
        index : int
            Index of the memory type.
        mappable : bool
            Whether the memory can be mapped.

        Raises
        ------
        VulkanSupportError
            If the device offers no suitable memory.
        """
        local = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        visible = (
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        )
        cached = visible | vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        wanted = (cached, visible) if host else (local | cached, local, cached, visible)
        for want in wanted:
            for i in range(self.memory.memoryTypeCount):
                flags = self.memory.memoryTypes[i].propertyFlags
                if not allowed & (1 << i) or flags & want != want:
                    continue
                # Plain device memory must not be the small mappable window
                # that discrete GPUs expose.
                if want == local and flags & visible:
                    continue
                return i, flags & visible == visible
        raise VulkanSupportError(f"{self.info.name} has no suitable memory")

    def _new_buffer(self, nbytes, host):
        """Create a storage buffer.

        Parameters
        ----------
        nbytes : int
            Size of the buffer in bytes; must be positive.
        host : bool
            Whether the buffer must be mappable.

        Returns
        -------
        _Buffer
        """
        dev = self.handle
        handle = vk.vkCreateBuffer(
            dev,
            vk.VkBufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size=nbytes,
                usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            ),
            None,
        )
        req = vk.vkGetBufferMemoryRequirements(dev, handle)
        type_index, mappable = self._memory_type(req.memoryTypeBits, host)
        memory = vk.vkAllocateMemory(
            dev,
            vk.VkMemoryAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize=req.size,
                memoryTypeIndex=type_index,
            ),
            None,
        )
        vk.vkBindBufferMemory(dev, handle, memory, 0)
        view = None
        if mappable:
            # The mapping is kept for the lifetime of the buffer.
            mapped = vk.vkMapMemory(dev, memory, 0, nbytes, 0)
            view = np.frombuffer(mapped, dtype=np.uint8, count=nbytes)
        return _Buffer(handle, memory, nbytes, host, view)

    def _acquire(self, nbytes, host):
        """Get a buffer of at least `nbytes` bytes, reusing a released one.

        Parameters
        ----------
        nbytes : int
            Number of bytes needed.
        host : bool
            Whether the buffer must be mappable.

        Returns
        -------
        _Buffer
        """
        size = _pool_size(nbytes)
        free = self._free.get((size, host))
        if free:
            self._pooled -= size
            return free.pop()
        try:
            return self._new_buffer(size, host)
        except (vk.VkErrorOutOfDeviceMemory, vk.VkErrorOutOfHostMemory):
            self.trim(0)
            return self._new_buffer(size, host)

    def _release(self, buffer):
        """Return a buffer to the pool for reuse.

        While asynchronous launches are pending, the buffer could still be
        in use by one of them; it joins the pool when they have finished.
        """
        if self._pending:
            self._deferred.append((buffer, self._submitted))
            return
        self._pool(buffer)

    def _pool(self, buffer):
        """Put a buffer that nothing uses any more into the pool."""
        self._free.setdefault((buffer.nbytes, buffer.host), []).append(buffer)
        self._pooled += buffer.nbytes
        if self._pooled > self.pool_limit:
            self.trim(self.pool_limit // 2)

    def trim(self, limit=0):
        """Free pooled buffers until at most `limit` bytes stay pooled.

        Buffers of arrays that are no longer in use are kept for reuse, up
        to `pool_limit` bytes; this releases them to the system.

        Parameters
        ----------
        limit : int
            Number of bytes that may remain pooled.
        """
        # Largest first: those are the least likely to be asked for again.
        for key in sorted(self._free, reverse=True):
            free = self._free[key]
            while free and self._pooled > limit:
                buffer = free.pop()
                self._pooled -= buffer.nbytes
                vk.vkDestroyBuffer(self.handle, buffer.handle, None)
                vk.vkFreeMemory(self.handle, buffer.memory, None)

    def _submit(self, record, cmd=None):
        """Run a command buffer and wait for completion.

        Parameters
        ----------
        record : callable or None
            Called with the command buffer to fill it. ``None`` submits
            the buffer as it was recorded last.
        cmd : object, optional
            The command buffer; the one for transfers by default.
        """
        cmd = self._transfer_commands if cmd is None else cmd
        if record is not None:
            vk.vkBeginCommandBuffer(cmd, self._begin_info)
            _barrier(cmd)
            record(cmd)
            vk.vkEndCommandBuffer(cmd)
        vk.vkQueueSubmit(self.queue, 1, self._submit_info[id(cmd)], vk.VK_NULL_HANDLE)
        vk.vkQueueWaitIdle(self.queue)
        # Waiting for the queue to be idle finished every pending launch.
        self._retire(len(self._pending))

    def _upload(self, buffer, data):
        """Copy host bytes into a buffer.

        Parameters
        ----------
        buffer : _Buffer
            The destination.
        data : numpy.ndarray
            One-dimensional ``uint8`` array, no larger than the buffer.
        """
        if not data.size:
            return
        if buffer.view is not None:
            buffer.view[: data.size] = data
            return
        staging = self._acquire(data.size, host=True)
        try:
            staging.view[: data.size] = data
            self._submit(lambda cmd: _copy(cmd, staging, buffer, data.size))
        finally:
            self._release(staging)

    def _download(self, buffer, out):
        """Copy the contents of a buffer into host bytes.

        Parameters
        ----------
        buffer : _Buffer
            The source.
        out : numpy.ndarray
            One-dimensional ``uint8`` array to fill, no larger than the
            buffer.
        """
        if not out.size:
            return
        if buffer.view is not None:
            out[:] = buffer.view[: out.size]
            return
        staging = self._acquire(out.size, host=True)
        try:
            self._submit(lambda cmd: _copy(cmd, buffer, staging, out.size))
            out[:] = staging.view[: out.size]
        finally:
            self._release(staging)

    # -- asynchronous launches ----------------------------------------------

    MAX_PENDING = 32

    def launch(self, kernel, groups, arrays):
        """Start ``kernel`` over ``groups`` workgroups without waiting for it.

        Used for kernels that cannot raise and whose arrays are all device
        arrays, so that nothing has to be read back. A launch takes a slot
        of the kernel: a descriptor set, a recorded command buffer, a fence
        and buffers for the small host arrays. A slot is reused unchanged by
        a later launch with the same device arrays, argument sizes and grid,
        which then only copies the argument values and submits. Every launch
        begins with a memory barrier, so it sees the writes of earlier ones.

        Parameters
        ----------
        kernel : CompiledKernel
            The compiled kernel.
        groups : tuple of int
            Workgroup counts along x, y and z.
        arrays : list of numpy.ndarray or DeviceArray
            One per binding: device arrays, and small host arrays (shapes,
            scalars) whose values are copied at once.

        Raises
        ------
        ValueError
            If a device array belongs to another device.
        """
        if len(self._pending) >= self.MAX_PENDING:
            # Retire half at once: waiting has a fixed cost per call.
            self._retire(self.MAX_PENDING // 2)
        state = self._pipeline(kernel)
        parts = []
        for array in arrays:
            if isinstance(array, DeviceArray):
                if array.device is not self:
                    raise ValueError(
                        f"the array is on {array.device.info.name}, but the "
                        f"kernel runs on {self.info.name}"
                    )
                parts.append((array._buffer.serial, array._nbytes))
            else:
                parts.append((None, array.nbytes))
        key = (tuple(parts), groups)
        slot = next((s for s in state.free_slots if s.key == key), None)
        if slot is not None:
            state.free_slots.remove(slot)
        else:
            slot = self._build_slot(state, key, arrays, groups)
        for buffer, array in zip(
            slot.host_buffers, (a for a in arrays if not isinstance(a, DeviceArray))
        ):
            buffer.view[: array.nbytes] = array.reshape(-1).view(np.uint8)
        vk.vkQueueSubmit(self.queue, 1, slot.submit, slot.fence)
        self._submitted += 1
        self._pending.append((slot, state))

    def _build_slot(self, state, key, arrays, groups):
        """Set up a slot for an asynchronous launch; see `launch`."""
        # The pool of descriptor sets holds MAX_PENDING; beyond that, the
        # oldest unused slot is rebuilt.
        if state.slots >= self.MAX_PENDING and state.free_slots:
            old = state.free_slots.pop(0)
            for buffer in old.host_buffers:
                self._pool(buffer)
            desc_set, cmd, fence = old.desc_set, old.cmd, old.fence
        else:
            state.slots += 1
            desc_set = self._async_set(state)
            cmd = vk.vkAllocateCommandBuffers(
                self.handle,
                vk.VkCommandBufferAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                    commandPool=self.command_pool,
                    level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    commandBufferCount=1,
                ),
            )[0]
            fence = vk.vkCreateFence(
                self.handle,
                vk.VkFenceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO),
                None,
            )
        buffers, host_buffers = [], []
        for array in arrays:
            if isinstance(array, DeviceArray):
                buffers.append((array._buffer, max(array._nbytes, 4)))
            else:
                buffer = self._acquire(max(array.nbytes, 4), host=True)
                host_buffers.append(buffer)
                buffers.append((buffer, max(array.nbytes, 4)))
        buffers += state.constants
        writes = [
            vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=desc_set,
                dstBinding=i,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[
                    vk.VkDescriptorBufferInfo(
                        buffer=buffer.handle, offset=0, range=nbytes
                    )
                ],
            )
            for i, (buffer, nbytes) in enumerate(buffers)
        ]
        vk.vkUpdateDescriptorSets(self.handle, len(writes), writes, 0, None)
        vk.vkBeginCommandBuffer(
            cmd,
            vk.VkCommandBufferBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
            ),
        )
        _barrier(cmd)
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, state.pipeline)
        vk.vkCmdBindDescriptorSets(
            cmd,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            state.layout,
            0,
            1,
            [desc_set],
            0,
            None,
        )
        _dispatch(cmd, groups, self.info.max_groups)
        vk.vkEndCommandBuffer(cmd)
        submit = [
            vk.VkSubmitInfo(
                sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                commandBufferCount=1,
                pCommandBuffers=[cmd],
            )
        ]
        return _Slot(key, desc_set, cmd, fence, submit, host_buffers)

    def _async_set(self, state):
        """A new descriptor set of a kernel for asynchronous launches."""
        if state.async_pool is None:
            count = len(state.keep[3].argtypes) + 1 + len(state.constants)
            state.async_pool = vk.vkCreateDescriptorPool(
                self.handle,
                vk.VkDescriptorPoolCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                    maxSets=self.MAX_PENDING,
                    poolSizeCount=1,
                    pPoolSizes=[
                        vk.VkDescriptorPoolSize(
                            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            descriptorCount=count * self.MAX_PENDING,
                        )
                    ],
                ),
                None,
            )
        return vk.vkAllocateDescriptorSets(
            self.handle,
            vk.VkDescriptorSetAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptorPool=state.async_pool,
                descriptorSetCount=1,
                pSetLayouts=[state.keep[0]],
            ),
        )[0]

    def _retire(self, count):
        """Wait for the oldest `count` pending launches and recycle them."""
        done, self._pending = self._pending[:count], self._pending[count:]
        if done:
            fences = [slot.fence for slot, _ in done]
            vk.vkWaitForFences(
                self.handle, len(fences), fences, vk.VK_TRUE, 0xFFFFFFFFFFFFFFFF
            )
            vk.vkResetFences(self.handle, len(fences), fences)
        for slot, state in done:
            state.free_slots.append(slot)
        self._retired += len(done)
        # Buffers released while launches were pending join the pool once
        # every launch submitted before their release has finished.
        keep = []
        for buffer, submitted in self._deferred:
            if submitted <= self._retired:
                self._pool(buffer)
            else:
                keep.append((buffer, submitted))
        self._deferred = keep

    def synchronize(self):
        """Wait until every kernel launched on this device has finished."""
        if self._pending:
            self._retire(len(self._pending))

    def run(self, kernel, groups, arrays):
        """Dispatch ``kernel`` over ``groups`` workgroups.

        Parameters
        ----------
        kernel : CompiledKernel
            The compiled kernel.
        groups : tuple of int
            Workgroup counts along x, y and z.
        arrays : list of numpy.ndarray or DeviceArray
            One array per binding. C-contiguous host arrays are uploaded,
            and those the kernel writes to are copied back afterwards;
            device arrays are used in place.

        Raises
        ------
        ValueError
            If a device array belongs to another device.
        """
        self.synchronize()
        state = self._pipeline(kernel)
        buffers, transient = [], []
        try:
            for array in arrays:
                if isinstance(array, DeviceArray):
                    if array.device is not self:
                        raise ValueError(
                            f"the array is on {array.device.info.name}, but the "
                            f"kernel runs on {self.info.name}"
                        )
                    buffers.append((array._buffer, max(array._nbytes, 4)))
                    continue
                buffer = self._acquire(max(array.nbytes, 4), host=True)
                transient.append(buffer)
                buffers.append((buffer, max(array.nbytes, 4)))
                self._upload(buffer, array.reshape(-1).view(np.uint8))
            buffers += state.constants
            if kernel.print_binding is not None:
                if state.print_buffer is None:
                    state.print_buffer = self._acquire(
                        PRINT_BUFFER_WORDS * 4, host=True
                    )
                header = np.array([0, PRINT_BUFFER_WORDS], dtype=np.uint32)
                state.print_buffer.view[:8] = header.view(np.uint8)
                buffers.append((state.print_buffer, PRINT_BUFFER_WORDS * 4))

            # Repeated launches mostly see the same buffers, and then neither
            # the descriptor set nor the recorded commands have to change.
            bound = tuple((buffer.serial, nbytes) for buffer, nbytes in buffers)
            if bound != state.bound:
                writes = [
                    vk.VkWriteDescriptorSet(
                        sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        dstSet=state.desc_set,
                        dstBinding=i,
                        descriptorCount=1,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        pBufferInfo=[
                            vk.VkDescriptorBufferInfo(
                                buffer=buffer.handle, offset=0, range=nbytes
                            )
                        ],
                    )
                    for i, (buffer, nbytes) in enumerate(buffers)
                ]
                vk.vkUpdateDescriptorSets(self.handle, len(writes), writes, 0, None)
                state.bound = bound
                self._recorded = None  # updating a set invalidates its users

            def record(cmd):
                """Bind the kernel and dispatch it."""
                vk.vkCmdBindPipeline(
                    cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, state.pipeline
                )
                vk.vkCmdBindDescriptorSets(
                    cmd,
                    vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                    state.layout,
                    0,
                    1,
                    [state.desc_set],
                    0,
                    None,
                )
                _dispatch(cmd, groups, self.info.max_groups)

            fresh = self._recorded != (id(state), groups)
            self._recorded = None  # stays unset if recording fails
            self._submit(record if fresh else None, self._launch_commands)
            self._recorded = (id(state), groups)

            for binding, (array, (buffer, _)) in enumerate(zip(arrays, buffers)):
                if isinstance(array, DeviceArray):
                    continue
                if binding in kernel.written_bindings:
                    self._download(buffer, array.reshape(-1).view(np.uint8))
            if kernel.print_binding is not None:
                print_records(state.print_buffer.view.view(np.uint32))
        finally:
            # In reverse, so that the next launch gets the same buffers at
            # the same bindings from the pool.
            for buffer in reversed(transient):
                self._release(buffer)


_DISPATCH_BASE = 0x10  # VK_PIPELINE_CREATE_DISPATCH_BASE_BIT


def _dispatch(cmd, groups, limit):
    """Record a dispatch, split into several where the grid exceeds `limit`.

    Each part starts at a base workgroup, so ``global_id`` and ``group_id``
    are those of the whole grid; ``num_groups`` is that of the part.

    Parameters
    ----------
    cmd : object
        The command buffer.
    groups : tuple of int
        Workgroups along x, y and z.
    limit : tuple of int
        The device's largest number of workgroups along each axis.
    """
    if all(n <= m for n, m in zip(groups, limit)):
        vk.vkCmdDispatch(cmd, *groups)
        return
    gx, gy, gz = groups
    lx, ly, lz = limit
    for z in range(0, gz, lz):
        for y in range(0, gy, ly):
            for x in range(0, gx, lx):
                vk.vkCmdDispatchBase(
                    cmd, x, y, z, min(lx, gx - x), min(ly, gy - y), min(lz, gz - z)
                )


PRINT_BUFFER_WORDS = int(os.environ.get("NUMBA_VULKAN_PRINT_WORDS", 1 << 18))


def print_records(words):
    """Print what a kernel's ``print`` calls recorded.

    Parameters
    ----------
    words : numpy.ndarray
        The print buffer as ``uint32``: cursor, capacity, then records;
        see `numba_vulkan.vkimpl.lower_print`.
    """
    used, capacity = int(words[0]), int(words[1])
    position, end = 2, min(used + 2, capacity)
    while position < end:
        form = print_formats.get(int(words[position]))
        position += 1
        if form is None:
            break
        items = []
        for part in form:
            code = part[0]
            if code == "s":
                items.append(part[1])
            elif code == "b":
                items.append(bool(words[position]))
                position += 1
            elif code == "f":
                items.append(words[position : position + 1].view(np.float32)[0])
                position += 1
            else:
                pair = words[position : position + 2].copy()
                kind = {"i": np.int64, "u": np.uint64, "d": np.float64}[code]
                value = pair.view(kind)[0]
                items.append(float(value) if code == "d" else int(value))
                position += 2
        print(*items)
    if used + 2 > capacity:
        warnings.warn(
            f"the print buffer of {capacity} words overflowed; some output was "
            "lost (NUMBA_VULKAN_PRINT_WORDS sets its size)",
            stacklevel=4,
        )


def _barrier(cmd):
    """Record a barrier after which earlier writes are visible to the device.

    It covers compute shaders and transfers, in this and earlier
    submissions to the queue.
    """
    stages = vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | vk.VK_PIPELINE_STAGE_TRANSFER_BIT
    barrier = vk.VkMemoryBarrier(
        sType=vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT | vk.VK_ACCESS_TRANSFER_WRITE_BIT,
        dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT
        | vk.VK_ACCESS_SHADER_WRITE_BIT
        | vk.VK_ACCESS_TRANSFER_READ_BIT
        | vk.VK_ACCESS_TRANSFER_WRITE_BIT,
    )
    vk.vkCmdPipelineBarrier(cmd, stages, stages, 0, 1, [barrier], 0, None, 0, None)


@dataclass
class _Slot:
    """What one asynchronous launch of a kernel needs; see `Device.launch`.

    Attributes
    ----------
    key : tuple
        The device arrays (by serial number), argument sizes and grid it
        was set up for.
    desc_set, cmd, fence : object
        Its descriptor set, recorded command buffer and fence.
    submit : list
        The ``VkSubmitInfo`` for the command buffer.
    host_buffers : list of _Buffer
        Mapped buffers for the host arrays, in argument order.
    """

    key: tuple
    desc_set: object
    cmd: object
    fence: object
    submit: list
    host_buffers: list


@dataclass
class _Pipeline:
    """A kernel as set up on a device.

    Attributes
    ----------
    pipeline, layout, desc_set : object
        The ``VkPipeline``, ``VkPipelineLayout`` and ``VkDescriptorSet``
        handles. The set has one storage buffer per binding.
    constants : list of tuple
        Buffer and size in bytes of each constant array of the kernel.
    keep : tuple
        Objects that must live as long as the pipeline.
    bound : tuple or None
        Serial number and size of the buffer at each binding of the
        descriptor set, as last written.
    print_buffer : _Buffer or None
        Where the kernel's ``print`` calls write, if it has any.
    free_slots : list of _Slot
        Slots for asynchronous launches that are not in use.
    slots : int
        Number of slots created.
    async_pool : object or None
        The ``VkDescriptorPool`` those sets come from.
    """

    pipeline: object
    layout: object
    desc_set: object
    constants: list
    keep: tuple
    bound: tuple = None
    print_buffer: object = None
    free_slots: list = field(default_factory=list)
    slots: int = 0
    async_pool: object = None


@dataclass
class _Buffer:
    """A storage buffer with its memory.

    Attributes
    ----------
    handle : object
        The ``VkBuffer`` handle.
    memory : object
        The ``VkDeviceMemory`` handle bound to it.
    nbytes : int
        Size in bytes.
    host : bool
        Whether the buffer was requested as mappable.
    view : numpy.ndarray or None
        The mapped contents as bytes, or ``None`` if the memory cannot be
        mapped.
    serial : int
        A number that identifies the buffer for good; unlike ``id()`` it is
        not reused after the buffer has been destroyed.
    """

    handle: object
    memory: object
    nbytes: int
    host: bool
    view: object
    serial: int = field(default_factory=itertools.count().__next__)


def _pool_size(nbytes):
    """Round a buffer size up, so that released buffers fit later requests.

    Sizes are rounded to a power of two up to 64 KiB and to a multiple of
    64 KiB beyond.
    """
    if nbytes <= 1 << 16:
        return max(256, 1 << (nbytes - 1).bit_length())
    return -(-nbytes // (1 << 16)) * (1 << 16)


def _copy(cmd, source, target, nbytes):
    """Record a copy of the first `nbytes` bytes between two buffers."""
    region = vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=nbytes)
    vk.vkCmdCopyBuffer(cmd, source.handle, target.handle, 1, [region])


class DeviceArray:
    """An array that lives on a Vulkan device.

    Kernels use device arrays in place, without the copies to and from the
    device that NumPy arrays need on every call. Create them with
    `to_device`, `device_array` or `device_array_like`.

    Parameters
    ----------
    device : Device
        The device that holds the array.
    shape : tuple of int
        Shape of the array.
    dtype : numpy.dtype
        Element type. Boolean arrays are stored as ``int32`` on the device,
        and 64-bit types as their 32-bit counterparts on devices that lack
        them (see `numba_vulkan.narrowing`).

    Attributes
    ----------
    device : Device
        The device that holds the array.
    shape : tuple of int
        Shape of the array.
    dtype : numpy.dtype
        Element type.

    Examples
    --------
    >>> x = nv.to_device(np.arange(8, dtype=np.float32))
    >>> out = nv.device_array_like(x)
    >>> kernel.forall(8)(x, out)
    >>> out.copy_to_host()
    """

    def __init__(self, device, shape, dtype):
        self.device = device
        self.shape = tuple(int(n) for n in shape)
        self.dtype = np.dtype(dtype)
        self._stored = narrowing.stored_dtype(self.dtype, device.mode)
        self._nbytes = self.size * self._stored.itemsize
        self._buffer = device._acquire(max(self._nbytes, 4), host=False)
        # The buffer returns to the device's pool when the array is dropped.
        self._finalizer = weakref.finalize(self, device._release, self._buffer)

    @property
    def ndim(self):
        """Number of dimensions."""
        return len(self.shape)

    @property
    def size(self):
        """Number of elements."""
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def nbytes(self):
        """Size of the elements in bytes, as they are stored on the host."""
        return self.size * self.dtype.itemsize

    def __len__(self):
        if not self.shape:
            raise TypeError("len() of a zero-dimensional array")
        return self.shape[0]

    def __repr__(self):
        return (
            f"<DeviceArray shape={self.shape} dtype={self.dtype} "
            f"on {self.device.info.name}>"
        )

    def reshape(self, *shape):
        """Give the array a new shape without copying.

        Parameters
        ----------
        *shape : int or tuple of int
            The new shape, with as many elements as the array has; one
            extent may be ``-1``, as in NumPy.

        Returns
        -------
        DeviceArray
            An array that shares the data with this one.

        Raises
        ------
        ValueError
            If the number of elements differs.
        """
        if len(shape) == 1 and not np.isscalar(shape[0]):
            shape = tuple(shape[0])
        shape = np.empty(self.size, dtype=np.uint8).reshape(shape).shape
        view = object.__new__(DeviceArray)
        view.device, view.shape, view.dtype = self.device, shape, self.dtype
        view._stored, view._nbytes = self._stored, self._nbytes
        view._buffer = self._buffer
        # The original owns the buffer; the view keeps it alive.
        view._base = getattr(self, "_base", None) or self
        return view

    def ravel(self):
        """The array as one dimension, without copying.

        Returns
        -------
        DeviceArray
        """
        return self.reshape(-1)

    def copy_to_device(self, array):
        """Overwrite the contents with those of a host array.

        Parameters
        ----------
        array : array_like
            Values of the same shape; converted to the element type.

        Returns
        -------
        DeviceArray
            The array itself.

        Raises
        ------
        ValueError
            If the shapes differ.
        """
        self.device.synchronize()
        array = np.asarray(array, dtype=self._stored, order="C")
        if array.shape != self.shape:
            raise ValueError(f"cannot copy shape {array.shape} into {self.shape}")
        self.device._upload(self._buffer, array.reshape(-1).view(np.uint8))
        return self

    def copy_to_host(self, out=None):
        """Copy the contents to a NumPy array.

        Parameters
        ----------
        out : numpy.ndarray, optional
            C-contiguous array of the same shape and element type to fill.
            A new array is created if omitted.

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        ValueError
            If `out` does not match the array.
        """
        self.device.synchronize()
        if out is None:
            out = np.empty(self.shape, dtype=self.dtype)
        elif (
            out.shape != self.shape
            or out.dtype != self.dtype
            or not out.flags.c_contiguous
        ):
            raise ValueError(
                f"out must be a C-contiguous {self.dtype} array of shape {self.shape}"
            )
        if self._stored != self.dtype:
            stored = np.empty(self.shape, dtype=self._stored)
            self.device._download(self._buffer, stored.reshape(-1).view(np.uint8))
            out[...] = stored != 0 if self.dtype == np.bool_ else stored
        else:
            self.device._download(self._buffer, out.reshape(-1).view(np.uint8))
        return out

    def __array__(self, dtype=None, copy=None):
        """Convert to a NumPy array, which copies from the device."""
        out = self.copy_to_host()
        return out if dtype is None else out.astype(dtype)


def to_device(array, device=None):
    """Copy a NumPy array to a device.

    Parameters
    ----------
    array : array_like
        The values.
    device : int, str, DeviceInfo, Device or None
        The device; see `get_device`. Defaults to the selected device.

    Returns
    -------
    DeviceArray
    """
    array = np.asarray(array)
    return device_array(array.shape, array.dtype, device).copy_to_device(array)


def device_array(shape, dtype=np.float64, device=None):
    """Create an array on a device without initialising it.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the array.
    dtype : numpy.dtype
        Element type.
    device : int, str, DeviceInfo, Device or None
        The device; see `get_device`. Defaults to the selected device.

    Returns
    -------
    DeviceArray
    """
    shape = (shape,) if np.isscalar(shape) else tuple(shape)
    if not isinstance(device, Device):
        device = get_device(device)
    return DeviceArray(device, shape, dtype)


def device_array_like(array, device=None):
    """Create an uninitialised device array with the shape and type of another.

    Parameters
    ----------
    array : numpy.ndarray or DeviceArray
        The array to imitate.
    device : int, str, DeviceInfo, Device or None
        The device. Defaults to the device of `array` if it is a device
        array, and to the selected device otherwise.

    Returns
    -------
    DeviceArray
    """
    if device is None and isinstance(array, DeviceArray):
        device = array.device
    return device_array(array.shape, array.dtype, device)


def get_device(which=None):
    """Return a :class:`Device`.

    Parameters
    ----------
    which : int, str, DeviceInfo, Device or None
        Index into :func:`list_devices`, a case-insensitive substring of the
        device name, or ``None`` for the current default.

    Returns
    -------
    Device
    """
    if isinstance(which, Device):
        return which
    if which is None and _current is not None:
        return _current
    if which is None:
        which = os.environ.get("NUMBA_VULKAN_DEVICE")
    devices = list_devices()
    if not devices:
        raise VulkanSupportError("no Vulkan 1.2 device with a compute queue found")
    if isinstance(which, DeviceInfo):
        info = devices[which.index]
    elif which is None:
        info = min(devices, key=lambda d: _PREFERENCE.index(d.kind))
    elif isinstance(which, int) or which.isdigit():
        info = devices[int(which)]
    else:
        matches = [d for d in devices if which.lower() in d.name.lower()]
        if not matches:
            raise VulkanSupportError(f"no Vulkan device matching '{which}': {devices}")
        info = matches[0]
    if info.index not in _devices:
        _devices[info.index] = Device(info)
    return _devices[info.index]


def synchronize(device=None):
    """Wait until launched kernels have finished.

    Launches whose arrays are all device arrays return before the kernel has
    run, as in ``numba.cuda``. Copying a device array waits on its own, so
    this is mainly needed to measure time.

    Parameters
    ----------
    device : int, str, DeviceInfo, Device or None
        The device to wait for; every device that was used if ``None``.
    """
    targets = [get_device(device)] if device is not None else list(_devices.values())
    for target in targets:
        target.synchronize()


def select_device(which):
    """Make ``which`` the default device for subsequent kernel calls.

    Parameters
    ----------
    which : int, str or DeviceInfo
        See :func:`get_device`.

    Returns
    -------
    Device
    """
    global _current
    _current = None
    _current = get_device(which)
    return _current
