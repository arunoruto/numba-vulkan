"""Minimal Vulkan compute runtime: run a SPIR-V kernel over host arrays."""

import os
from dataclasses import dataclass

import numpy as np
import vulkan as vk

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
    """

    index: int
    name: str
    kind: str
    float64: bool
    int64: bool
    int16: bool
    int8: bool
    handle: object

    def __repr__(self):
        return f"<{self.index}: {self.name} ({self.kind})>"


_instance = None
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


def _int8_feature(handle):
    """Whether a device supports 8-bit integers in shaders.

    Parameters
    ----------
    handle : object
        A ``VkPhysicalDevice`` handle.

    Returns
    -------
    bool
    """
    feats = vk.VkPhysicalDeviceVulkan12Features(
        sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
    )
    feats2 = vk.VkPhysicalDeviceFeatures2(
        sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, pNext=feats
    )
    vk.vkGetPhysicalDeviceFeatures2(handle, feats2)
    return bool(feats.shaderInt8)


def list_devices():
    """All Vulkan 1.2+ devices that expose a compute queue.

    Returns
    -------
    list of DeviceInfo
    """
    found = []
    for handle in vk.vkEnumeratePhysicalDevices(_get_instance()):
        props = vk.vkGetPhysicalDeviceProperties(handle)
        if props.apiVersion < _API_VERSION:
            continue
        feats = vk.vkGetPhysicalDeviceFeatures(handle)
        found.append(
            DeviceInfo(
                index=len(found),
                name=props.deviceName,
                kind=_DEVICE_TYPES.get(props.deviceType, "other"),
                float64=bool(feats.shaderFloat64),
                int64=bool(feats.shaderInt64),
                int16=bool(feats.shaderInt16),
                int8=_int8_feature(handle),
                handle=handle,
            )
        )
    return found


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
        features12 = vk.VkPhysicalDeviceVulkan12Features(
            sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            shaderInt8=info.int8,
        )
        create = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext=features12,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_info],
            pEnabledFeatures=features,
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
        self._pipelines = {}

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

    def _pipeline(self, kernel):
        """Create (or fetch) the compute pipeline of a kernel.

        Parameters
        ----------
        kernel : numba_vulkan.codegen.CompiledKernel
            The compiled kernel.

        Returns
        -------
        pipeline : object
            The ``VkPipeline`` handle.
        layout : object
            The ``VkPipelineLayout`` handle.
        set_layout : object
            The ``VkDescriptorSetLayout`` handle, with one storage buffer per
            binding.

        Raises
        ------
        VulkanSupportError
            If the device lacks a capability the kernel needs.
        """
        key = id(kernel)
        if key in self._pipelines:
            return self._pipelines[key][:3]
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
                    stage=stage,
                    layout=layout,
                )
            ],
            None,
        )[0]
        # The kernel is kept alive alongside its pipeline so id() stays unique.
        self._pipelines[key] = (pipeline, layout, set_layout, module, kernel)
        return pipeline, layout, set_layout

    # -- buffers ---------------------------------------------------------

    def _host_memory_type(self, allowed):
        """Choose a mappable memory type, preferring CPU-cached memory.

        Reading results back from uncached (write-combined) memory is very
        slow on discrete GPUs.

        Parameters
        ----------
        allowed : int
            Bit mask of memory types the buffer may use.

        Returns
        -------
        int
            Index of the memory type.

        Raises
        ------
        VulkanSupportError
            If the device offers no host-visible memory for the buffer.
        """
        visible = (
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        )
        for wanted in (visible | vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT, visible):
            for i in range(self.memory.memoryTypeCount):
                flags = self.memory.memoryTypes[i].propertyFlags
                if allowed & (1 << i) and flags & wanted == wanted:
                    return i
        raise VulkanSupportError(f"{self.info.name} has no host-visible memory")

    def _create_buffer(self, nbytes):
        """Create a storage buffer backed by host-visible memory.

        Parameters
        ----------
        nbytes : int
            Size of the buffer in bytes; must be positive.

        Returns
        -------
        buffer : object
            The ``VkBuffer`` handle.
        memory : object
            The ``VkDeviceMemory`` handle bound to it.
        """
        dev = self.handle
        buffer = vk.vkCreateBuffer(
            dev,
            vk.VkBufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size=nbytes,
                usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            ),
            None,
        )
        req = vk.vkGetBufferMemoryRequirements(dev, buffer)
        type_index = self._host_memory_type(req.memoryTypeBits)
        memory = vk.vkAllocateMemory(
            dev,
            vk.VkMemoryAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize=req.size,
                memoryTypeIndex=type_index,
            ),
            None,
        )
        vk.vkBindBufferMemory(dev, buffer, memory, 0)
        return buffer, memory

    def run(self, kernel, groups, host_arrays):
        """Dispatch ``kernel`` over ``groups`` workgroups.

        Parameters
        ----------
        kernel : CompiledKernel
            The compiled kernel.
        groups : tuple of int
            Workgroup counts along x, y and z.
        host_arrays : list of numpy.ndarray
            One C-contiguous array per binding. They are uploaded, and the
            buffers the kernel writes to are copied back afterwards.
        """
        dev = self.handle
        pipeline, layout, set_layout = self._pipeline(kernel)
        buffers = []
        pool = None
        try:
            for array in host_arrays:
                nbytes = max(array.nbytes, 4)
                buffer, memory = self._create_buffer(nbytes)
                buffers.append((buffer, memory, nbytes))
                if array.nbytes:
                    mapped = vk.vkMapMemory(dev, memory, 0, nbytes, 0)
                    target = np.frombuffer(mapped, dtype=np.uint8, count=array.nbytes)
                    target[:] = array.reshape(-1).view(np.uint8)
                    vk.vkUnmapMemory(dev, memory)

            pool = vk.vkCreateDescriptorPool(
                dev,
                vk.VkDescriptorPoolCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                    maxSets=1,
                    poolSizeCount=1,
                    pPoolSizes=[
                        vk.VkDescriptorPoolSize(
                            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            descriptorCount=len(buffers),
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
            writes = [
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=desc_set,
                    dstBinding=i,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[
                        vk.VkDescriptorBufferInfo(buffer=buffer, offset=0, range=nbytes)
                    ],
                )
                for i, (buffer, _, nbytes) in enumerate(buffers)
            ]
            vk.vkUpdateDescriptorSets(dev, len(writes), writes, 0, None)

            cmd = vk.vkAllocateCommandBuffers(
                dev,
                vk.VkCommandBufferAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                    commandPool=self.command_pool,
                    level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    commandBufferCount=1,
                ),
            )[0]
            vk.vkBeginCommandBuffer(
                cmd,
                vk.VkCommandBufferBeginInfo(
                    sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                    flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                ),
            )
            vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
            vk.vkCmdBindDescriptorSets(
                cmd,
                vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                layout,
                0,
                1,
                [desc_set],
                0,
                None,
            )
            vk.vkCmdDispatch(cmd, *groups)
            vk.vkEndCommandBuffer(cmd)
            submit = vk.VkSubmitInfo(
                sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                commandBufferCount=1,
                pCommandBuffers=[cmd],
            )
            vk.vkQueueSubmit(self.queue, 1, [submit], vk.VK_NULL_HANDLE)
            vk.vkQueueWaitIdle(self.queue)
            vk.vkFreeCommandBuffers(dev, self.command_pool, 1, [cmd])

            for binding, (array, (_, memory, _)) in enumerate(
                zip(host_arrays, buffers)
            ):
                if binding not in kernel.written_bindings or array.nbytes == 0:
                    continue
                mapped = vk.vkMapMemory(dev, memory, 0, array.nbytes, 0)
                array[...] = np.frombuffer(mapped, dtype=array.dtype).reshape(
                    array.shape
                )
                vk.vkUnmapMemory(dev, memory)
        finally:
            if pool is not None:
                vk.vkDestroyDescriptorPool(dev, pool, None)
            for buffer, memory, _ in buffers:
                vk.vkDestroyBuffer(dev, buffer, None)
                vk.vkFreeMemory(dev, memory, None)


def get_device(which=None):
    """Return a :class:`Device`.

    Parameters
    ----------
    which : int, str, DeviceInfo or None
        Index into :func:`list_devices`, a case-insensitive substring of the
        device name, or ``None`` for the current default.

    Returns
    -------
    Device
    """
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
