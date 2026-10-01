"""Minimal Vulkan compute runtime: devices, device arrays and kernel launch."""

import os
import weakref
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
        self._command_buffer = vk.vkAllocateCommandBuffers(
            self.handle,
            vk.VkCommandBufferAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                commandPool=self.command_pool,
                level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1,
            ),
        )[0]
        self._pipelines = {}
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
        desc_set : object
            The ``VkDescriptorSet`` of the kernel, with one storage buffer
            per binding; it is rewritten on every launch.

        Raises
        ------
        VulkanSupportError
            If the device lacks a capability the kernel needs.
        """
        key = id(kernel)
        if key in self._pipelines:
            return self._pipelines[key][:4]
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
        self._pipelines[key] = (
            pipeline, layout, desc_set, constants, set_layout, pool, module, kernel
        )  # fmt: skip
        return pipeline, layout, desc_set, constants

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
        """Return a buffer to the pool for reuse."""
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

    def _submit(self, record):
        """Record commands with `record`, run them and wait for completion.

        Parameters
        ----------
        record : callable
            Called with the command buffer to fill.
        """
        cmd = self._command_buffer
        vk.vkBeginCommandBuffer(
            cmd,
            vk.VkCommandBufferBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            ),
        )
        record(cmd)
        vk.vkEndCommandBuffer(cmd)
        submit = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[cmd],
        )
        vk.vkQueueSubmit(self.queue, 1, [submit], vk.VK_NULL_HANDLE)
        vk.vkQueueWaitIdle(self.queue)

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
        dev = self.handle
        pipeline, layout, desc_set, constants = self._pipeline(kernel)
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
            buffers += constants

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
            vk.vkUpdateDescriptorSets(dev, len(writes), writes, 0, None)

            def record(cmd):
                """Bind the kernel and dispatch it."""
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

            self._submit(record)

            for binding, (array, (buffer, _)) in enumerate(zip(arrays, buffers)):
                if isinstance(array, DeviceArray):
                    continue
                if binding in kernel.written_bindings:
                    self._download(buffer, array.reshape(-1).view(np.uint8))
        finally:
            for buffer in transient:
                self._release(buffer)


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
    """

    handle: object
    memory: object
    nbytes: int
    host: bool
    view: object


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
        Element type. Boolean arrays are stored as ``int32`` on the device.

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
        self._stored = np.dtype(np.int32) if self.dtype == np.bool_ else self.dtype
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
        if self.dtype == np.bool_:
            stored = np.empty(self.shape, dtype=self._stored)
            self.device._download(self._buffer, stored.reshape(-1).view(np.uint8))
            out[...] = stored != 0
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
