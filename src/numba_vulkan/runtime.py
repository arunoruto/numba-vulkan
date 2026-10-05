"""Minimal Vulkan compute runtime: devices, device arrays and kernel launch."""

import contextlib
import ctypes
import functools
import itertools
import math
import os
import struct
import sys
import warnings
import weakref
from dataclasses import dataclass, field

import numpy as np

try:
    import vulkan as vk
except OSError as exc:  # the `vulkan` package could not open the loader
    hint = (
        " On macOS, install MoltenVK and the loader (`brew install molten-vk "
        "vulkan-loader`) and make the loader findable, for example with "
        "`export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`, or install "
        "the LunarG Vulkan SDK."
        if sys.platform == "darwin"
        else " Install your distribution's Vulkan loader (libvulkan1 on "
        "Debian and Ubuntu) and a driver."
    )
    raise ImportError(f"numba-vulkan needs a Vulkan loader: {exc}.{hint}") from exc

from numba_vulkan import narrowing
from numba_vulkan.buffers import META_BINDING, print_formats
from numba_vulkan.codegen import check_spirv, preserve_nan
from numba_vulkan.errors import VulkanSupportError, VulkanValidationWarning

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
    storage8 : bool
        Whether buffers may hold 8-bit values (``int8`` and ``uint8``
        arrays).
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
    nan_preserve : tuple of int
        Float widths for which shaders can require NaN, infinity and signed
        zero to be kept (``SignedZeroInfNanPreserve``).
    timestamp_bits : int
        Valid bits of the timestamps the compute queue writes; 0 if it
        writes none (see `event`).
    timestamp_period : float
        Nanoseconds per timestamp tick.
    subgroup_size : int
        Invocations per subgroup (``subgroup.size()`` in kernels).
    subgroup_basic, subgroup_vote, subgroup_arithmetic, subgroup_ballot, \
subgroup_shuffle, subgroup_shuffle_relative : bool
        Which classes of subgroup operations compute shaders may use; see
        `numba_vulkan.stubs.subgroup`.
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
    storage8: bool = False
    float32_atomic_add: bool = False
    max_local_size: tuple = (128, 128, 64)
    max_local_invocations: int = 128
    max_groups: tuple = (65535, 65535, 65535)
    max_shared_memory: int = 16384
    nan_preserve: tuple = ()
    timestamp_bits: int = 0
    timestamp_period: float = 1.0
    subgroup_size: int = 1
    subgroup_basic: bool = False
    subgroup_vote: bool = False
    subgroup_arithmetic: bool = False
    subgroup_ballot: bool = False
    subgroup_shuffle: bool = False
    subgroup_shuffle_relative: bool = False

    def __repr__(self):
        return f"<{self.index}: {self.name} ({self.kind})>"


_instance = None
_infos = None
_devices = {}
_current = None
# The default device for each value of NUMBA_VULKAN_DEVICE: launches without
# a device ask for it every time.
_defaults = {}


DEBUG_ENV_VAR = "NUMBA_VULKAN_DEBUG"
_VALIDATION_LAYER = "VK_LAYER_KHRONOS_validation"
_DEBUG_UTILS = "VK_EXT_debug_utils"
# Warnings and errors from the validation layer, as (severity, message).
_messages = []
# The messenger has to live as long as the instance.
_messenger = None


def debug_level():
    """How much ``NUMBA_VULKAN_DEBUG`` asks Vulkan's validation layer to check.

    Returns
    -------
    int
        0: nothing (the default); 1: the core checks of API usage; 2: also
        synchronization, that is missing barriers between commands, which
        is slower.
    """
    value = os.environ.get(DEBUG_ENV_VAR, "0").strip() or "0"
    return int(value) if value.isdigit() else 1


def debug_enabled():
    """Whether ``NUMBA_VULKAN_DEBUG`` asks for Vulkan's validation layer.

    Returns
    -------
    bool
    """
    return debug_level() > 0


def validation_messages(clear=False):
    """What the validation layer reported so far (``NUMBA_VULKAN_DEBUG=1``).

    Parameters
    ----------
    clear : bool, optional
        Forget the messages after returning them.

    Returns
    -------
    list of tuple
        ``(severity, message)`` pairs, severity ``"warning"`` or ``"error"``,
        oldest first.
    """
    found = list(_messages)
    if clear:
        _messages.clear()
    return found


def _on_validation_message(severity, types, data, user_data):
    """Debug messenger callback: record a message and warn about it.

    Called from C, so it must not raise: an exception, also from a warning
    turned into one, is dropped after the message has been recorded.
    """
    try:
        level = (
            "error"
            if severity & vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
            else "warning"
        )
        text = vk.ffi.string(data.pMessage).decode(errors="replace")
        _messages.append((level, text))
        warnings.warn(f"Vulkan validation {level}: {text}", VulkanValidationWarning)
    except Exception:  # noqa: BLE001, S110
        pass
    return vk.VK_FALSE


def _debug_setup():
    """Layers and extensions for ``NUMBA_VULKAN_DEBUG``, as far as installed.

    Returns
    -------
    layers, extensions : list of str
    """
    layers = {p.layerName for p in vk.vkEnumerateInstanceLayerProperties()}
    extensions = {
        e.extensionName for e in vk.vkEnumerateInstanceExtensionProperties(None)
    }
    if _VALIDATION_LAYER not in layers:
        warnings.warn(
            f"{DEBUG_ENV_VAR} is set, but Vulkan's validation layer "
            f"({_VALIDATION_LAYER}) is not installed; nothing will be checked",
            VulkanValidationWarning,
            stacklevel=3,
        )
        return [], []
    return [_VALIDATION_LAYER], [_DEBUG_UTILS] if _DEBUG_UTILS in extensions else []


def _create_messenger(instance):
    """Register `_on_validation_message` for warnings and errors."""
    global _messenger
    severities = (
        vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
        | vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
    )
    kinds = (
        vk.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
        | vk.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
        | vk.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT
    )
    create = vk.vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT")
    _messenger = create(
        instance,
        vk.VkDebugUtilsMessengerCreateInfoEXT(
            sType=vk.VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            messageSeverity=severities,
            messageType=kinds,
            pfnUserCallback=_on_validation_message,
        ),
        None,
    )


def _get_instance():
    """The process-wide Vulkan instance, created on first use.

    With ``NUMBA_VULKAN_DEBUG=1`` it enables Vulkan's validation layer and
    reports its findings through `validation_messages` and
    `numba_vulkan.errors.VulkanValidationWarning`; ``NUMBA_VULKAN_DEBUG=2``
    adds synchronization validation (see `debug_level`).

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
        # Portability drivers, MoltenVK on macOS among them, are only listed
        # when the instance asks for them.
        available = {
            e.extensionName for e in vk.vkEnumerateInstanceExtensionProperties(None)
        }
        extensions, flags = [], 0
        if _PORTABILITY_ENUMERATION in available:
            extensions.append(_PORTABILITY_ENUMERATION)
            flags |= vk.VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
        layers, debug_extensions = _debug_setup() if debug_enabled() else ([], [])
        extensions += debug_extensions
        features = None
        if layers and debug_level() >= 2:
            features = vk.VkValidationFeaturesEXT(
                sType=vk.VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
                enabledValidationFeatureCount=1,
                pEnabledValidationFeatures=[
                    vk.VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT
                ],
            )
        info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pNext=features,
            flags=flags,
            pApplicationInfo=app,
            enabledLayerCount=len(layers),
            ppEnabledLayerNames=layers or None,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions or None,
        )
        _instance = vk.vkCreateInstance(info, None)
        if debug_extensions:
            _create_messenger(_instance)
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


def _vulkan_properties(handle):
    """The Vulkan 1.1 and 1.2 properties of a device.

    Parameters
    ----------
    handle : object
        A ``VkPhysicalDevice`` handle.

    Returns
    -------
    props11, props12 : object
        The filled ``VkPhysicalDeviceVulkan11Properties`` and
        ``VkPhysicalDeviceVulkan12Properties`` structures.
    """
    props12 = vk.VkPhysicalDeviceVulkan12Properties(
        sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES
    )
    props11 = vk.VkPhysicalDeviceVulkan11Properties(
        sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES,
        pNext=props12,
    )
    props2 = vk.VkPhysicalDeviceProperties2(
        sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, pNext=props11
    )
    vk.vkGetPhysicalDeviceProperties2(handle, props2)
    return props11, props12


def _nan_preserving_widths(props12):
    """Float widths for which shaders can ask to keep NaN, infinity and -0.

    Returns
    -------
    tuple of int
        Out of 32 and 64, from ``shaderSignedZeroInfNanPreserveFloat*``.
    """
    return tuple(
        width
        for width, kept in (
            (32, props12.shaderSignedZeroInfNanPreserveFloat32),
            (64, props12.shaderSignedZeroInfNanPreserveFloat64),
        )
        if kept
    )


# Subgroup operations by the DeviceInfo attribute that says they are
# supported (VkSubgroupFeatureFlagBits).
_SUBGROUP_FEATURES = {
    "subgroup_basic": 0x1,
    "subgroup_vote": 0x2,
    "subgroup_arithmetic": 0x4,
    "subgroup_ballot": 0x8,
    "subgroup_shuffle": 0x10,
    "subgroup_shuffle_relative": 0x20,
}


def _subgroup_features(props11):
    """The subgroup operations compute shaders of a device may use.

    Returns
    -------
    dict of str to bool
        By the names in `_SUBGROUP_FEATURES`; all false unless compute
        shaders support subgroup operations at all.
    """
    compute = props11.subgroupSupportedStages & vk.VK_SHADER_STAGE_COMPUTE_BIT
    operations = props11.subgroupSupportedOperations if compute else 0
    return {name: bool(operations & bit) for name, bit in _SUBGROUP_FEATURES.items()}


_FLOAT_ATOMICS = "VK_EXT_shader_atomic_float"
_PORTABILITY_ENUMERATION = "VK_KHR_portability_enumeration"
# Implementations of Vulkan on top of other APIs (MoltenVK) offer this, and
# then it must be enabled.
_PORTABILITY_SUBSET = "VK_KHR_portability_subset"


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
        props11, props12 = _vulkan_properties(handle)
        limits = props.limits
        families = vk.vkGetPhysicalDeviceQueueFamilyProperties(handle)
        compute = next(
            (f for f in families if f.queueFlags & vk.VK_QUEUE_COMPUTE_BIT), None
        )
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
                storage8=bool(feats12.storageBuffer8BitAccess),
                float32_atomic_add=bool(
                    float_atomics is not None
                    and float_atomics.shaderBufferFloat32AtomicAdd
                    and float_atomics.shaderSharedFloat32AtomicAdd
                ),
                max_local_size=tuple(limits.maxComputeWorkGroupSize),
                max_local_invocations=limits.maxComputeWorkGroupInvocations,
                max_groups=tuple(limits.maxComputeWorkGroupCount),
                max_shared_memory=limits.maxComputeSharedMemorySize,
                timestamp_bits=compute.timestampValidBits if compute else 0,
                timestamp_period=limits.timestampPeriod,
                nan_preserve=_nan_preserving_widths(props12),
                subgroup_size=props11.subgroupSize,
                **_subgroup_features(props11),
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
        How kernels are compiled for the device by default: with 32-bit
        integers, and with 32-bit floats if it lacks ``float64``.
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
        # Streams (see `Stream`) run kernels on a second compute queue where
        # the family has one, and copies on a queue of a family that only
        # transfers, which is a separate copy engine on discrete GPUs.
        compute_queues = min(families[self.family].queueCount, 2)
        graphics_or_compute = vk.VK_QUEUE_GRAPHICS_BIT | vk.VK_QUEUE_COMPUTE_BIT
        self.transfer_family = next(
            (
                i
                for i, f in enumerate(families)
                if f.queueFlags & vk.VK_QUEUE_TRANSFER_BIT
                and not f.queueFlags & graphics_or_compute
            ),
            None,
        )
        queue_infos = [
            vk.VkDeviceQueueCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                queueFamilyIndex=self.family,
                queueCount=compute_queues,
                pQueuePriorities=[1.0] * compute_queues,
            )
        ]
        if self.transfer_family is not None:
            queue_infos.append(
                vk.VkDeviceQueueCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    queueFamilyIndex=self.transfer_family,
                    queueCount=1,
                    pQueuePriorities=[1.0],
                )
            )
        features = vk.VkPhysicalDeviceFeatures(
            shaderFloat64=info.float64,
            shaderInt64=info.int64,
            shaderInt16=info.int16,
        )
        extensions, chain = [], {}
        offered = {
            e.extensionName for e in vk.vkEnumerateDeviceExtensionProperties(phys, None)
        }
        if _PORTABILITY_SUBSET in offered:
            extensions.append(_PORTABILITY_SUBSET)
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
            storageBuffer8BitAccess=info.storage8,
            shaderBufferInt64Atomics=info.int64_atomics,
            shaderSharedInt64Atomics=info.int64_atomics,
            shaderFloat16=info.float16,
            timelineSemaphore=True,  # for streams; required by Vulkan 1.2
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
            queueCreateInfoCount=len(queue_infos),
            pQueueCreateInfos=queue_infos,
            pEnabledFeatures=features,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions or None,
        )
        self.handle = vk.vkCreateDevice(phys, create, None)
        self.queue = vk.vkGetDeviceQueue(self.handle, self.family, 0)
        self.stream_queue = vk.vkGetDeviceQueue(
            self.handle, self.family, compute_queues - 1
        )
        if self.transfer_family is not None:
            self.transfer_queue = vk.vkGetDeviceQueue(
                self.handle, self.transfer_family, 0
            )
        else:
            self.transfer_family, self.transfer_queue = self.family, self.stream_queue
        # Buffers are used by both families without transfers of ownership.
        self._families = sorted({self.family, self.transfer_family})
        self.memory = vk.vkGetPhysicalDeviceMemoryProperties(phys)
        pool = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=self.family,
        )
        self.command_pool = vk.vkCreateCommandPool(self.handle, pool, None)
        self._pools = {self.family: self.command_pool}
        if self.transfer_family != self.family:
            self._pools[self.transfer_family] = vk.vkCreateCommandPool(
                self.handle,
                vk.VkCommandPoolCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                    flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                    queueFamilyIndex=self.transfer_family,
                ),
                None,
            )
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
        # Asynchronous launches: the command buffer being recorded, those
        # submitted but not known to have finished (oldest first), and
        # finished ones for reuse. Each is numbered; buffers released while
        # launches may still use them wait for the batch of that number.
        self._open = None
        self._pending = []
        self._spare = []
        self._seq = self._retired_seq = 0
        self._deferred = []
        # Exceptions of kernels launched asynchronously, as (kernel, code).
        self._errors = []
        # Streams created on this device (see `Stream`), and those with work
        # in flight, which are kept alive until it has finished.
        self._streams = weakref.WeakSet()
        self._busy_streams = set()
        self._dead_semaphores = []
        # Pinned host arrays by the address of their memory; see
        # `pinned_array`.
        self._pinned = {}
        # Timestamp queries for events, used in turn; created when needed.
        self._queries = None
        self._next_query = 0
        self._pipelines = {}
        # 64-bit types this device cannot use and kernels must do without.
        # How kernels are compiled for this device unless they say otherwise.
        self.mode = narrowing.Mode(
            floats=not info.float64,
            ints=narrowing.NARROW_INTS or not info.int64,
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
        code = kernel.spirv
        if kernel.exact and self.info.nan_preserve:
            # Without this, drivers may assume that there are no NaNs:
            # NVIDIA's turns ``y < x ? y : x``, Python's min(), into an
            # instruction that ignores NaN.
            code = preserve_nan(code, self.info.nan_preserve)
            check_spirv(code)
        module = vk.vkCreateShaderModule(
            dev,
            vk.VkShaderModuleCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(code),
                pCode=code,
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
            if i not in kernel.unbound
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
        ranges = [
            vk.VkPushConstantRange(
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                offset=0,
                size=kernel.push_size,
            )
        ]
        layout = vk.vkCreatePipelineLayout(
            dev,
            vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1,
                pSetLayouts=[set_layout],
                pushConstantRangeCount=1 if kernel.push_size else 0,
                pPushConstantRanges=ranges if kernel.push_size else None,
            ),
            None,
        )
        # The workgroup size is a specialization constant of the module
        # (see numba_vulkan.codegen.specialize_local_size).
        sizes = struct.pack("<3I", *kernel.local_size)
        specialization = vk.VkSpecializationInfo(
            mapEntryCount=3,
            pMapEntries=[
                vk.VkSpecializationMapEntry(constantID=k, offset=4 * k, size=4)
                for k in range(3)
            ],
            dataSize=len(sizes),
            pData=vk.ffi.from_buffer(sizes),
        )
        stage = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=module,
            pName="main",
            pSpecializationInfo=specialization,
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
                **_sharing(self._families),
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
        if _stream_pending(buffer):
            # The stream gives it back once it has passed its last use.
            buffer.stream._released.append(buffer)
            return
        if self._pending or self._open:
            self._deferred.append((buffer, self._seq))
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
        self._flush()  # in order after asynchronous launches
        if record is not None:
            vk.vkBeginCommandBuffer(cmd, self._begin_info)
            _barrier(cmd)
            record(cmd)
            vk.vkEndCommandBuffer(cmd)
        if self._streams:
            self._submit_after_streams(cmd, vk.VK_NULL_HANDLE)
        else:
            submit = self._submit_info[id(cmd)]
            vk.vkQueueSubmit(self.queue, 1, submit, vk.VK_NULL_HANDLE)
        vk.vkQueueWaitIdle(self.queue)
        # Waiting for the queue to be idle finished every pending launch.
        self._retire(len(self._pending))

    def _upload(self, buffer, data, offset=0):
        """Copy host bytes into a buffer.

        Parameters
        ----------
        buffer : _Buffer
            The destination.
        data : numpy.ndarray
            One-dimensional ``uint8`` array that fits behind `offset`.
        offset : int
            Position in the buffer, in bytes.
        """
        if not data.size:
            return
        if buffer.view is not None:
            buffer.view[offset : offset + data.size] = data
            return
        staging = self._acquire(data.size, host=True)
        try:
            staging.view[: data.size] = data
            self._submit(
                lambda cmd: _copy(cmd, staging, buffer, data.size, target_offset=offset)
            )
        finally:
            self._release(staging)

    def _download(self, buffer, out, offset=0):
        """Copy the contents of a buffer into host bytes.

        Parameters
        ----------
        buffer : _Buffer
            The source.
        out : numpy.ndarray
            One-dimensional ``uint8`` array to fill from `offset` on.
        offset : int
            Position in the buffer, in bytes.
        """
        if not out.size:
            return
        if buffer.view is not None:
            out[:] = buffer.view[offset : offset + out.size]
            return
        staging = self._acquire(out.size, host=True)
        try:
            self._submit(
                lambda cmd: _copy(cmd, buffer, staging, out.size, source_offset=offset)
            )
            out[:] = staging.view[: out.size]
        finally:
            self._release(staging)

    # -- asynchronous launches ----------------------------------------------

    # Launches recorded into one command buffer before it is submitted, and
    # command buffers submitted but not known to have finished.
    MAX_BATCH = 16
    MAX_IN_FLIGHT = 4
    # Descriptor sets per kernel for asynchronous launches.
    MAX_SLOTS = 32

    def launch(self, kernel, groups, arrays, push=b""):
        """Start ``kernel`` over ``groups`` workgroups without waiting for it.

        Used for launches whose arrays are all device arrays, so that
        nothing has to be read back. Launches are recorded into a command
        buffer, which is submitted at once if the device has nothing else
        to do, and otherwise when it holds `MAX_BATCH` launches or at the
        next synchronisation, whichever comes first. A barrier precedes
        each launch, so it sees the writes of earlier ones.

        A launch uses a slot of the kernel: a descriptor set and buffers
        for its small host arrays. Launches with the same buffers and the
        same host arrays share one, so repeated launches neither write
        descriptors nor copy data; their scalars and shapes are push
        constants, recorded with each launch.

        An exception raised by the kernel is reported by the next
        `synchronize`, which happens before any copy from or to the device
        and before any launch that waits.

        Parameters
        ----------
        kernel : CompiledKernel
            The compiled kernel.
        groups : tuple of int
            Workgroup counts along x, y and z.
        arrays : list of numpy.ndarray, DeviceArray or None
            One per binding: device arrays, and small host arrays (the
            status and, for kernels without push constants, shapes and
            scalars). ``None`` for the bindings in ``kernel.unbound``.
        push : bytes, optional
            The kernel's push constants, ``kernel.push_size`` bytes.

        Raises
        ------
        ValueError
            If a device array belongs to another device.
        """
        state = self._pipeline(kernel)
        parts, host = [], []
        for array in arrays:
            if isinstance(array, DeviceArray):
                if array.device is not self:
                    raise ValueError(
                        f"the array is on {array.device.info.name}, but the "
                        f"kernel runs on {self.info.name}"
                    )
                if array._buffer.stream is not None:
                    _settle(array._buffer)
                parts.append((array._buffer.serial, array._nbytes))
            elif array is None:
                parts.append(None)
            else:
                parts.append(array.nbytes)
                host.append(array.tobytes())
        key = (tuple(parts), tuple(host))
        slot = state.slots.get(key)
        if slot is None:
            slot = self._new_slot(state, key, arrays)
        batch = self._open or self._open_batch()
        cmd = batch.cmd
        _barrier(cmd)
        compute = vk.VK_PIPELINE_BIND_POINT_COMPUTE
        if batch.state is not state:
            _lib.vkCmdBindPipeline(cmd, compute, state.pipeline)
            batch.state, batch.desc_set = state, None
        if batch.desc_set is not slot.desc_set:
            _lib.vkCmdBindDescriptorSets(
                cmd, compute, state.layout, 0, 1, slot.sets, 0, _ffi.NULL
            )
            batch.desc_set = slot.desc_set
        if push:
            _lib.vkCmdPushConstants(
                cmd,
                state.layout,
                vk.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                len(push),
                _ffi.from_buffer(push),
            )
        _dispatch(cmd, groups, self.info.max_groups)
        slot.used = batch.seq
        for array in arrays:
            if isinstance(array, DeviceArray):
                array._buffer.used = batch.seq
        if slot.status is not None and slot not in batch.checks:
            batch.checks.append(slot)
        batch.launches += 1
        if batch.launches >= self.MAX_BATCH or not self._busy():
            self._flush()

    def _new_slot(self, state, key, arrays, extra=()):
        """Set up a slot for asynchronous launches; see `launch`.

        `extra` holds ``(buffer, nbytes)`` for the bindings after the
        constant arrays: the print buffer of a launch on a stream.
        """
        if len(state.slots) >= self.MAX_SLOTS:
            # The descriptor pool is full: reuse the set of the slot used
            # longest ago, once no launch can use it any more.
            oldest = min(state.slots.values(), key=lambda s: s.used)
            if oldest.used > self._retired_seq:
                self.synchronize(report=False)
            if oldest.stream is not None:
                oldest.stream._wait(oldest.stream_value)
            del state.slots[oldest.key]
            for buffer in oldest.host_buffers:
                self._pool(buffer)
            desc_set = oldest.desc_set
        else:
            desc_set = self._async_set(state)
        buffers, host_buffers = [], []
        for array in arrays:
            if isinstance(array, DeviceArray):
                buffers.append((array._buffer, _words(array._nbytes)))
            elif array is None:
                buffers.append(None)
            else:
                buffer = self._acquire(max(array.nbytes, 4), host=True)
                buffer.view[: array.nbytes] = array.reshape(-1).view(np.uint8)
                host_buffers.append(buffer)
                buffers.append((buffer, _words(array.nbytes)))
        buffers += state.constants
        buffers += extra
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
            for i, (buffer, nbytes) in _bound(buffers)
        ]
        vk.vkUpdateDescriptorSets(self.handle, len(writes), writes, 0, None)
        kernel = state.keep[3]
        writes_status = META_BINDING in kernel.written_bindings and isinstance(
            arrays[META_BINDING], np.ndarray
        )
        slot = _Slot(
            key,
            desc_set,
            _ffi.new("VkDescriptorSet[1]", [desc_set]),
            host_buffers,
            status=host_buffers[0] if writes_status else None,
            kernel=kernel,
        )
        state.slots[key] = slot
        return slot

    def _async_set(self, state):
        """A new descriptor set of a kernel for asynchronous launches."""
        if state.async_pool is None:
            kernel = state.keep[3]
            count = kernel.num_bindings - len(kernel.unbound)
            state.async_pool = vk.vkCreateDescriptorPool(
                self.handle,
                vk.VkDescriptorPoolCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                    maxSets=self.MAX_SLOTS,
                    poolSizeCount=1,
                    pPoolSizes=[
                        vk.VkDescriptorPoolSize(
                            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            descriptorCount=count * self.MAX_SLOTS,
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

    def _open_batch(self):
        """Start recording a command buffer for asynchronous launches."""
        if len(self._pending) >= self.MAX_IN_FLIGHT:
            # All at once if all have finished: retiring has a fixed cost.
            self._retire(1 if self._busy() else len(self._pending))
        if self._spare:
            batch = self._spare.pop()
        else:
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
            # cffi frees what nothing refers to, so the batch keeps the
            # array of command buffers that the submit info points to.
            commands = _ffi.new("VkCommandBuffer[1]", [cmd])
            submit = _ffi.new(
                "VkSubmitInfo*",
                {
                    "sType": vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    "commandBufferCount": 1,
                    "pCommandBuffers": commands,
                },
            )
            batch = _Batch(cmd, fence, (submit, commands))
        self._seq += 1
        batch.seq, batch.launches, batch.checks = self._seq, 0, []
        batch.state = batch.desc_set = None
        _check(_lib.vkBeginCommandBuffer(batch.cmd, _BEGIN_INFO))
        self._open = batch
        return batch

    def _flush(self):
        """Submit the launches recorded so far."""
        batch, self._open = self._open, None
        if batch is None:
            return
        _check(_lib.vkEndCommandBuffer(batch.cmd))
        if self._streams:
            self._submit_after_streams(batch.cmd, batch.fence)
        else:
            _check(_lib.vkQueueSubmit(self.queue, 1, batch.submit[0], batch.fence))
        self._pending.append(batch)

    def _submit_after_streams(self, cmd, fence):
        """Submit to the default queue after what streams have finished.

        Buffers that streams used, and that the host knows them to be done
        with, may be used here. The order is already right, since the host
        waited for the streams; waiting for the values they have reached
        states it on the device as well, for tools that check it (such as
        the validation layer's synchronization checks).
        """
        streams = [stream for stream in self._streams if stream._done]
        timeline = vk.VkTimelineSemaphoreSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            waitSemaphoreValueCount=len(streams),
            pWaitSemaphoreValues=[stream._done for stream in streams] or None,
        )
        submit = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext=timeline,
            waitSemaphoreCount=len(streams),
            pWaitSemaphores=[stream._semaphore for stream in streams] or None,
            pWaitDstStageMask=[vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT] * len(streams)
            or None,
            commandBufferCount=1,
            pCommandBuffers=[cmd],
        )
        vk.vkQueueSubmit(self.queue, 1, [submit], fence)

    def _busy(self):
        """Whether submitted launches are still running."""
        return bool(
            self._pending
            and _lib.vkGetFenceStatus(self.handle, self._pending[-1].fence)
        )  # VK_NOT_READY rather than VK_SUCCESS

    def _retire(self, count):
        """Wait for the oldest `count` submitted batches and recycle them.

        Kernels that raised are recorded, for `synchronize` to report.
        """
        done, self._pending = self._pending[:count], self._pending[count:]
        if not done:
            return
        fences = _ffi.new("VkFence[]", [batch.fence for batch in done])
        _check(
            _lib.vkWaitForFences(
                self.handle, len(done), fences, vk.VK_TRUE, 0xFFFFFFFFFFFFFFFF
            )
        )
        _check(_lib.vkResetFences(self.handle, len(done), fences))
        for batch in done:
            for slot in batch.checks:
                # A slot used again later is checked when that finishes.
                if slot.used == batch.seq:
                    code = int(slot.status.view[:4].view(np.int32)[0])
                    if code:
                        self._errors.append((slot.kernel, code))
                        slot.status.view[:4] = 0
        self._spare.extend(done)
        self._retired_seq = done[-1].seq
        if self._dead_semaphores and not self._pending and self._open is None:
            for semaphore in self._dead_semaphores:
                vk.vkDestroySemaphore(self.handle, semaphore, None)
            self._dead_semaphores = []
        # Buffers released while launches were pending join the pool once
        # every launch recorded before their release has finished.
        keep = []
        for buffer, seq in self._deferred:
            if seq <= self._retired_seq:
                self._pool(buffer)
            else:
                keep.append((buffer, seq))
        self._deferred = keep

    def synchronize(self, report=True):
        """Wait until every kernel launched on this device has finished.

        Parameters
        ----------
        report : bool
            Whether to raise the exception of a kernel that raised one
            since the last synchronisation.

        Raises
        ------
        Exception
            The first exception that a kernel launched asynchronously
            raised since then; see `numba_vulkan.dispatcher`.
        """
        self._flush()
        self._retire(len(self._pending))
        for stream in list(self._streams):
            stream._wait(stream.value)
            self._errors += stream._errors
            stream._errors = []
        if report:
            self._report()

    def wait_for(self, buffer):
        """Wait until no launch uses a buffer any more.

        Launches that do not use it may go on running; with memory that the
        host can map, copies to and from the buffer then overlap with them.

        Parameters
        ----------
        buffer : _Buffer
            The buffer.

        Raises
        ------
        Exception
            As `synchronize` does.
        """
        # Whatever is recorded runs meanwhile.
        self._flush()
        _settle(buffer)
        if buffer.used > self._retired_seq:
            self._retire(sum(batch.seq <= buffer.used for batch in self._pending))
        self._report()

    def _report(self):
        """Raise the first exception of a kernel launched asynchronously."""
        if self._errors:
            (kernel, code), self._errors = self._errors[0], []
            from numba_vulkan.dispatcher import raise_kernel_error

            raise_kernel_error(kernel.name, code, asynchronous=True)

    def launch_stream(self, stream, kernel, groups, arrays, push=b""):
        """Enqueue ``kernel`` on a `Stream`; see `launch` for the arguments.

        The arrays must be device arrays (and the host arrays that hold the
        status and, without push constants, shapes and scalars). What the
        kernel prints is printed once the stream has passed the launch.
        """
        state = self._pipeline(kernel)
        parts, host, buffers = [], [], []
        for array in arrays:
            if isinstance(array, DeviceArray):
                if array.device is not self:
                    raise ValueError(
                        f"the array is on {array.device.info.name}, but the "
                        f"kernel runs on {self.info.name}"
                    )
                parts.append((array._buffer.serial, array._nbytes))
                buffers.append(array._buffer)
            elif array is None:
                parts.append(None)
            else:
                parts.append(array.nbytes)
                host.append(array.tobytes())
        extra, keep, after = (), [], None
        if kernel.print_binding is not None:
            # A print buffer per launch, printed once the stream is past it.
            printed = self._acquire(PRINT_BUFFER_WORDS * 4, host=True)
            header = np.array([0, PRINT_BUFFER_WORDS], dtype=np.uint32)
            printed.view[:8] = header.view(np.uint8)
            extra, keep = [(printed, PRINT_BUFFER_WORDS * 4)], [printed]
            parts.append(("print", printed.serial))

            def after():
                print_records(printed.view.view(np.uint32))

        key = (tuple(parts), tuple(host))
        slot = state.slots.get(key)
        if slot is None:
            slot = self._new_slot(state, key, arrays, extra)
        stream._submit(
            self.family,
            lambda cmd: self._record_dispatch(cmd, state, slot.desc_set, groups, push),
            buffers,
            keep=keep,
            after=after,
            checks=[slot] if slot.status is not None else (),
        )
        slot.stream, slot.stream_value = stream, stream.value

    def _unpin(self, start):
        """Give the buffer of a dropped pinned array back."""
        _, buffer = self._pinned.pop(start)
        self._release(buffer)

    def _record_dispatch(self, cmd, state, desc_set, groups, push):
        """Record binding a kernel, its push constants and its dispatch.

        Parameters
        ----------
        cmd : object
            The command buffer, being recorded.
        state : _Pipeline
            The kernel.
        desc_set : object
            Its descriptor set.
        groups : tuple of int
            Workgroups along x, y and z.
        push : bytes
            Its push constants; empty if it has none.
        """
        compute = vk.VK_PIPELINE_BIND_POINT_COMPUTE
        _lib.vkCmdBindPipeline(cmd, compute, state.pipeline)
        sets = _ffi.new("VkDescriptorSet[1]", [desc_set])
        _lib.vkCmdBindDescriptorSets(
            cmd, compute, state.layout, 0, 1, sets, 0, _ffi.NULL
        )
        if push:
            _lib.vkCmdPushConstants(
                cmd,
                state.layout,
                vk.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                len(push),
                _ffi.from_buffer(push),
            )
        _dispatch(cmd, groups, self.info.max_groups)

    def run(self, kernel, groups, arrays, push=b""):
        """Dispatch ``kernel`` over ``groups`` workgroups.

        Parameters
        ----------
        kernel : CompiledKernel
            The compiled kernel.
        groups : tuple of int
            Workgroup counts along x, y and z.
        arrays : list of numpy.ndarray, DeviceArray or None
            One array per binding. C-contiguous host arrays are uploaded,
            and those the kernel writes to are copied back afterwards;
            device arrays are used in place. ``None`` for the bindings in
            ``kernel.unbound``.
        push : bytes, optional
            The kernel's push constants, ``kernel.push_size`` bytes.

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
                    _settle(array._buffer)
                    buffers.append((array._buffer, _words(array._nbytes)))
                    continue
                if array is None:
                    buffers.append(None)
                    continue
                buffer = self._acquire(max(array.nbytes, 4), host=True)
                transient.append(buffer)
                buffers.append((buffer, _words(array.nbytes)))
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
            bound = tuple(b and (b[0].serial, b[1]) for b in buffers)
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
                    for i, (buffer, nbytes) in _bound(buffers)
                ]
                vk.vkUpdateDescriptorSets(self.handle, len(writes), writes, 0, None)
                state.bound = bound
                self._recorded = None  # updating a set invalidates its users

            def record(cmd):
                """Bind the kernel and dispatch it."""
                self._record_dispatch(cmd, state, state.desc_set, groups, push)

            fresh = self._recorded != (id(state), groups, push)
            self._recorded = None  # stays unset if recording fails
            self._submit(record if fresh else None, self._launch_commands)
            self._recorded = (id(state), groups, push)

            for binding, (array, (buffer, _)) in _bound(zip(arrays, buffers)):
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


def _sharing(families):
    """Sharing mode of buffers used by the given queue families."""
    if len(families) == 1:
        return {"sharingMode": vk.VK_SHARING_MODE_EXCLUSIVE}
    return {
        "sharingMode": vk.VK_SHARING_MODE_CONCURRENT,
        "queueFamilyIndexCount": len(families),
        "pQueueFamilyIndices": families,
    }


def _words(nbytes):
    """Bytes of a buffer that a descriptor covers: whole 32-bit words.

    Record arrays are read and written as words (see vkimpl), and may end
    in the middle of one. Buffers come from the pool in sizes of at least
    256 bytes, so the rounded range lies within them.
    """
    return max(-(-nbytes // 4) * 4, 4)


def _host_arrays(arrays):
    """The host arrays among the arrays of a launch, in order."""
    return (a for a in arrays if a is not None and not isinstance(a, DeviceArray))


def _bound(items):
    """Number the bindings of a launch, leaving out those without a buffer.

    Parameters
    ----------
    items : iterable
        One item per binding; ``None``, or a pair whose second element
        is ``None``, for a binding that has no buffer.

    Yields
    ------
    tuple
        The binding and its item.
    """
    for binding, item in enumerate(items):
        if item is not None and not (isinstance(item, tuple) and item[1] is None):
            yield binding, item


def _dispatch(cmd, groups, limit):
    """Record a dispatch, split into several where the grid exceeds `limit`.

    Each part starts at a base workgroup, so ``global_id`` and ``group_id``
    are those of the whole grid; ``num_groups`` comes from push constants.

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
        _lib.vkCmdDispatch(cmd, *groups)
        return
    gx, gy, gz = groups
    lx, ly, lz = limit
    for z in range(0, gz, lz):
        for y in range(0, gy, ly):
            for x in range(0, gx, lx):
                vk.vkCmdDispatchBase(
                    cmd, x, y, z, min(lx, gx - x), min(ly, gy - y), min(lz, gz - z)
                )


PRINT_BUFFER_WORDS = int(os.environ.get("NUMBA_VULKAN_PRINT_WORDS", str(1 << 18)))


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
    _lib.vkCmdPipelineBarrier(
        cmd, stages, stages, 0, 1, _MEMORY_BARRIER, 0, _ffi.NULL, 0, _ffi.NULL
    )


# The C functions behind the ``vulkan`` package, for the commands recorded
# per launch; the wrappers of the package convert every argument anew and
# take several times as long. The structures they take are built once,
# which costs more than recording them.
_lib, _ffi = vk.lib, vk.ffi
_MEMORY_BARRIER = _ffi.new(
    "VkMemoryBarrier*",
    {
        "sType": vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        "srcAccessMask": vk.VK_ACCESS_SHADER_WRITE_BIT
        | vk.VK_ACCESS_TRANSFER_WRITE_BIT,
        "dstAccessMask": vk.VK_ACCESS_SHADER_READ_BIT
        | vk.VK_ACCESS_SHADER_WRITE_BIT
        | vk.VK_ACCESS_TRANSFER_READ_BIT
        | vk.VK_ACCESS_TRANSFER_WRITE_BIT,
    },
)
_BEGIN_INFO = _ffi.new(
    "VkCommandBufferBeginInfo*",
    {"sType": vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO},
)


def _check(result):
    """Raise for a ``VkResult`` other than success.

    Parameters
    ----------
    result : int
        What a Vulkan function returned.

    Raises
    ------
    vulkan.VkError
        If `result` is not ``VK_SUCCESS``.
    """
    if result != vk.VK_SUCCESS:
        raise vk.VkError(f"Vulkan call failed with VkResult {result}")


@dataclass
class _Slot:
    """Descriptor set and host buffers for asynchronous launches of a kernel.

    Attributes
    ----------
    key : tuple
        The buffers (by serial number) and the contents of the host arrays
        it was set up for.
    desc_set : object
        Its descriptor set.
    sets : object
        The set as a C array, for binding it.
    host_buffers : list of _Buffer
        Mapped buffers holding the host arrays, in argument order.
    status : _Buffer or None
        The buffer the kernel leaves its error status in, if it can raise.
    kernel : CompiledKernel
        The kernel.
    used : int
        Number of the last batch that uses it.
    """

    key: tuple
    desc_set: object
    sets: object
    host_buffers: list
    status: object = None
    kernel: object = None
    used: int = 0
    stream: object = None
    stream_value: int = 0


@dataclass
class _Batch:
    """A command buffer of asynchronous launches; see `Device.launch`.

    Attributes
    ----------
    cmd, fence : object
        The command buffer and the fence its submission signals.
    submit : tuple
        Its ``VkSubmitInfo`` and the array of command buffers it points to.
    seq : int
        Its number, counting up per device.
    launches : int
        Number of launches recorded into it.
    checks : list of _Slot
        Slots whose error status is checked when it has finished.
    state, desc_set : object
        Pipeline and descriptor set last bound, to skip binding them again.
    """

    cmd: object
    fence: object
    submit: object
    seq: int = 0
    launches: int = 0
    checks: list = field(default_factory=list)
    state: object = None
    desc_set: object = None


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
    slots : dict
        Slots for asynchronous launches, by their key.
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
    slots: dict = field(default_factory=dict)
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
    used : int
        Number of the last batch of asynchronous launches that uses it.
    """

    handle: object
    memory: object
    nbytes: int
    host: bool
    view: object
    serial: int = field(default_factory=itertools.count().__next__)
    used: int = 0
    stream: object = None
    stream_value: int = 0


def _pool_size(nbytes):
    """Round a buffer size up, so that released buffers fit later requests.

    Sizes are rounded to a power of two up to 64 KiB and to a multiple of
    64 KiB beyond.
    """
    if nbytes <= 1 << 16:
        return max(256, 1 << (nbytes - 1).bit_length())
    return -(-nbytes // (1 << 16)) * (1 << 16)


def _copy(cmd, source, target, nbytes, source_offset=0, target_offset=0):
    """Record a copy of `nbytes` bytes between two buffers."""
    region = vk.VkBufferCopy(
        srcOffset=source_offset, dstOffset=target_offset, size=nbytes
    )
    vk.vkCmdCopyBuffer(cmd, source.handle, target.handle, 1, [region])


# Views with more runs of adjacent elements than this are copied on a stream
# by a kernel: drivers record and run copy regions slowly, ~1 us each.
_KERNEL_RUNS = 1024

_WORD_KERNELS = []


def _word_kernels():
    """Kernels that gather and scatter runs of 4-byte words.

    Returns
    -------
    gather, scatter : VulkanDispatcher
        Called with all words of a buffer, the packed words, the first
        word of each run and the words per run.
    """
    if not _WORD_KERNELS:
        from numba_vulkan import stubs
        from numba_vulkan.dispatcher import jit

        @jit
        def gather(whole, packed, starts, length):
            i = stubs.global_id(0)
            if i < packed.shape[0]:
                packed[i] = whole[starts[i // length] + i % length]

        @jit
        def scatter(whole, packed, starts, length):
            i = stubs.global_id(0)
            if i < packed.shape[0]:
                whole[starts[i // length] + i % length] = packed[i]

        _WORD_KERNELS.extend((gather, scatter))
    return _WORD_KERNELS


def _advanced(key):
    """Whether an index uses advanced indexing (arrays, lists, booleans)."""
    parts = key if isinstance(key, tuple) else (key,)
    return any(
        isinstance(part, (list, np.ndarray, DeviceArray, bool, np.bool_))
        for part in parts
    )


def _copy_regions(cmd, source, target, regions):
    """Record a copy of an array of ``VkBufferCopy`` between two buffers."""
    _lib.vkCmdCopyBuffer(cmd, source.handle, target.handle, len(regions), regions)


class DeviceArray:
    """An array that lives on a Vulkan device.

    Kernels use device arrays in place, without the copies to and from the
    device that NumPy arrays need on every call. Create them with
    `to_device`, `device_array` or `device_array_like`.

    Indexing works as on NumPy arrays, without advanced indexing: an
    integer for every axis reads or writes one element, and slices,
    integers for some axes, ``...`` and ``None`` give a *view* that shares
    the memory, which kernels accept like any device array. ``.T``,
    `transpose` and `reshape` give views as well (`reshape` only where the
    strides allow, as NumPy without copying).

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
    >>> out[2:6].copy_to_host()     # a view of four elements
    >>> out[0] = 1                  # writes one element
    """

    def __init__(self, device, shape, dtype, _stored=None):
        self.device = device
        self.shape = tuple(int(n) for n in shape)
        self.dtype = np.dtype(dtype)
        # Launches give the stored type of the kernel's mode.
        self._stored = (
            np.dtype(_stored)
            if _stored is not None
            else narrowing.stored_dtype(self.dtype, device.mode)
        )
        # Bytes of the buffer that kernels may address; views share them.
        self._nbytes = self.size * self._stored.itemsize
        self._buffer = device._acquire(max(self._nbytes, 4), host=False)
        # Position of the first element and steps between elements along
        # each axis, in elements of the buffer.
        self._offset = 0
        self._steps = _c_steps(self.shape)
        # The buffer returns to the device's pool when the array is dropped.
        self._finalizer = weakref.finalize(self, device._release, self._buffer)

    def _view(self, shape, offset, steps):
        """An array on the same buffer with another shape and layout."""
        view = object.__new__(DeviceArray)
        view.device, view.dtype, view._stored = self.device, self.dtype, self._stored
        view.shape, view._offset, view._steps = tuple(shape), offset, tuple(steps)
        view._nbytes, view._buffer = self._nbytes, self._buffer
        # The original owns the buffer; the view keeps it alive.
        view._base = getattr(self, "_base", None) or self
        return view

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

    @property
    def strides(self):
        """Steps between elements along each axis in bytes, as in NumPy."""
        return tuple(step * self.dtype.itemsize for step in self._steps)

    @property
    def is_contiguous(self):
        """Whether the elements lie one after the other in C order.

        Returns
        -------
        bool
        """
        # Asked for by every launch; the layout of an array never changes.
        contiguous = self.__dict__.get("_contiguous")
        if contiguous is None:
            contiguous = self._contiguous = all(
                step == want or extent == 1
                for extent, step, want in zip(
                    self.shape, self._steps, _c_steps(self.shape)
                )
            )
        return contiguous

    @property
    def T(self):
        """The array with its axes reversed, as a view."""
        return self.transpose()

    def __len__(self):
        if not self.shape:
            raise TypeError("len() of a zero-dimensional array")
        return self.shape[0]

    def __repr__(self):
        return (
            f"<DeviceArray shape={self.shape} dtype={self.dtype} "
            f"on {self.device.info.name}>"
        )

    def transpose(self, *axes):
        """Permute the axes, as a view.

        Parameters
        ----------
        *axes : int or tuple of int
            The new order of the axes; reversed by default.

        Returns
        -------
        DeviceArray
        """
        if len(axes) == 1 and not np.isscalar(axes[0]):
            axes = tuple(axes[0])
        axes = axes or tuple(reversed(range(self.ndim)))
        if sorted(a % self.ndim for a in axes) != list(range(self.ndim)):
            raise ValueError(f"axes {axes} do not match an array of {self.ndim}")
        axes = [a % self.ndim for a in axes]
        return self._view(
            [self.shape[a] for a in axes],
            self._offset,
            [self._steps[a] for a in axes],
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
            If the number of elements differs, or if the layout of the
            array does not allow the shape without copying; `copy` first.
        """
        if len(shape) == 1 and not np.isscalar(shape[0]):
            shape = tuple(shape[0])
        shape = _resolve_shape(tuple(int(n) for n in shape), self.size)
        steps = _reshaped_steps(self.shape, self._steps, shape)
        if steps is None:
            raise ValueError(
                f"a device array with strides {self.strides} cannot take the "
                f"shape {shape} without copying; call .copy() first"
            )
        return self._view(shape, self._offset, steps)

    def ravel(self):
        """The array as one dimension, a view where the layout allows it.

        Returns
        -------
        DeviceArray
        """
        steps = _reshaped_steps(self.shape, self._steps, (self.size,))
        return self.reshape(-1) if steps is not None else self.copy().reshape(-1)

    def copy(self):
        """A contiguous copy on the same device.

        Returns
        -------
        DeviceArray
        """
        return to_device(self.copy_to_host(), self.device)

    def _span(self):
        """First and last-but-one element the array covers in the buffer."""
        low = self._offset + sum(
            min(0, (n - 1) * step) for n, step in zip(self.shape, self._steps)
        )
        high = self._offset + sum(
            max(0, (n - 1) * step) for n, step in zip(self.shape, self._steps)
        )
        return low, high + 1

    def _elements(self, span, low):
        """The elements as a strided view into the host copy of their span."""
        size = self._stored.itemsize
        return np.lib.stride_tricks.as_strided(
            span[self._offset - low :],
            shape=self.shape,
            strides=[step * size for step in self._steps],
        )

    def _runs(self):
        """The runs of adjacent elements of the array, in C order.

        Returns
        -------
        starts : numpy.ndarray
            Position of the first element of each run in the buffer, in
            elements.
        length : int
            Number of elements in each run.
        """
        dims = [(n, step) for n, step in zip(self.shape, self._steps) if n != 1]
        length = 1
        while dims and dims[-1][1] == length:
            length *= dims.pop()[0]
        starts = np.full(1, self._offset, dtype=np.int64)
        for n, step in dims:
            starts = (starts[:, None] + np.arange(n, dtype=np.int64) * step).ravel()
        return starts, length

    def _regions(self, host_offset, to_device):
        """Copy regions between the array and its elements packed in C order.

        Parameters
        ----------
        host_offset : int
            Byte position of the packed elements in their buffer.
        to_device : bool
            Whether the copy goes from the packed elements to the array.

        Returns
        -------
        cdata
            An array of ``VkBufferCopy``, one per run of adjacent elements.
        """
        starts, length = self._runs()
        size = self._stored.itemsize
        packed = host_offset + np.arange(len(starts), dtype=np.uint64) * (length * size)
        regions = _ffi.new("VkBufferCopy[]", len(starts))
        table = np.frombuffer(_ffi.buffer(regions), dtype=np.uint64).reshape(-1, 3)
        device = starts.astype(np.uint64) * size
        table[:, 0] = packed if to_device else device
        table[:, 1] = device if to_device else packed
        table[:, 2] = length * size
        return regions

    def _word_runs(self):
        """The runs of the array in 4-byte words, if a kernel copies them.

        Returns
        -------
        tuple or None
            Positions of the first word of each run in the buffer, and
            words per run, for views with more than ``_KERNEL_RUNS`` runs
            of elements whose size is a multiple of 4 bytes; otherwise None,
            and the runs are copied as regions.
        """
        words, rest = divmod(self._stored.itemsize, 4)
        if rest:
            return None
        starts, length = self._runs()
        if len(starts) <= _KERNEL_RUNS:
            return None
        return starts * words, length * words

    def _copy_words(self, packed, runs, stream, to_view):
        """Enqueue a kernel that copies between the runs and packed words.

        Parameters
        ----------
        packed : DeviceArray
            Contiguous ``uint32`` array, the elements in C order.
        runs : tuple
            See `_word_runs`.
        stream : Stream
            The stream.
        to_view : bool
            Whether the copy goes from `packed` to the array.
        """
        starts, length = runs
        whole = self._buffer_words()
        first = to_device(starts.astype(np.uint32), self.device, stream=stream)
        gather, scatter = _word_kernels()
        kernel = scatter if to_view else gather
        kernel.forall(packed.size, stream=stream)(whole, packed, first, length)

    def copy_to_device(self, array, stream=None):
        """Overwrite the contents with those of a host array.

        Parameters
        ----------
        array : array_like
            Values of the same shape; converted to the element type.
        stream : Stream, optional
            Enqueue the copy on a stream and return at once; see `Stream`.
            The array (if it is not pinned, its converted copy) must not
            change before the stream has done the copy.

        Returns
        -------
        DeviceArray
            The array itself.

        Raises
        ------
        ValueError
            If the shapes differ.
        Exception
            One that a kernel launched asynchronously raised; see
            `Device.synchronize`.
        """
        if stream is not None:
            return self._copy_in_stream(array, stream)
        self.device.wait_for(self._buffer)
        array = narrowing.convert(np.asarray(array), self._stored)
        if array.shape != self.shape:
            raise ValueError(f"cannot copy shape {array.shape} into {self.shape}")
        if not self.size:
            return self
        size = self._stored.itemsize
        if self.is_contiguous:
            data = np.ascontiguousarray(array).reshape(-1).view(np.uint8)
            self.device._upload(self._buffer, data, self._offset * size)
            return self
        # A view with gaps: its span is read, changed and written back.
        low, high = self._span()
        span = np.empty(high - low, dtype=self._stored)
        self.device._download(self._buffer, span.view(np.uint8), low * size)
        self._elements(span, low)[...] = array
        self.device._upload(self._buffer, span.view(np.uint8), low * size)
        return self

    def copy_to_host(self, out=None, stream=None):
        """Copy the contents to a NumPy array.

        Parameters
        ----------
        out : numpy.ndarray, optional
            C-contiguous array of the same shape and element type to fill.
            A new array is created if omitted.
        stream : Stream, optional
            Enqueue the copy on a stream and return at once; `out` holds the
            data after ``stream.synchronize()``.

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        ValueError
            If `out` does not match the array.
        Exception
            One that a kernel launched asynchronously raised; see
            `Device.synchronize`.
        """
        if stream is not None:
            return self._copy_out_stream(out, stream)
        self.device.wait_for(self._buffer)
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
        if not self.size:
            return out
        size = self._stored.itemsize
        if self.is_contiguous:
            if self._stored == self.dtype:
                stored = out
            else:
                stored = np.empty(self.shape, dtype=self._stored)
            data = stored.reshape(-1).view(np.uint8)
            self.device._download(self._buffer, data, self._offset * size)
        else:
            low, high = self._span()
            span = np.empty(high - low, dtype=self._stored)
            self.device._download(self._buffer, span.view(np.uint8), low * size)
            stored = self._elements(span, low)
        if stored is not out:
            out[...] = stored != 0 if self.dtype == np.bool_ else stored
        return out

    def _stream_check(self, stream):
        """Check that a copy can go on a stream; its size in bytes."""
        if stream.device is not self.device:
            raise ValueError(
                f"the stream belongs to {stream.device.info.name}, the array to "
                f"{self.device.info.name}"
            )
        return self.size * self._stored.itemsize

    def _copy_in_stream(self, array, stream):
        """`copy_to_device` on a stream."""
        nbytes = self._stream_check(stream)
        array = np.asarray(array)
        if array.shape != self.shape:
            raise ValueError(f"cannot copy shape {array.shape} into {self.shape}")
        device = self.device
        source, offset = (None, 0)
        if array.dtype == self._stored:
            source, offset = _pinned_place(device, array)
        keep = []
        if source is None:
            data = narrowing.convert(array, self._stored)
            source = device._acquire(max(nbytes, 4), host=True)
            source.view[:nbytes] = data.reshape(-1).view(np.uint8)
            keep.append(source)
        if nbytes:
            runs = self._word_runs()
            if runs is None:
                # Views with gaps are copied run by run.
                target = self._buffer
                regions = self._regions(offset, to_device=True)
                record = functools.partial(
                    _copy_regions, source=source, target=target, regions=regions
                )
            else:
                packed = DeviceArray(device, (nbytes // 4,), np.uint32)
                target = packed._buffer
                record = functools.partial(
                    _copy,
                    source=source,
                    target=target,
                    nbytes=nbytes,
                    source_offset=offset,
                )
            stream._submit(
                device.transfer_family,
                record,
                [target] + ([] if keep else [source]),
                keep=keep,
            )
            if runs is not None:
                self._copy_words(packed, runs, stream, to_view=True)
        else:
            for buffer in keep:
                device._release(buffer)
        return self

    def _copy_out_stream(self, out, stream, cast=False):
        """`copy_to_host` on a stream; `cast` lets `out` have another type."""
        nbytes = self._stream_check(stream)
        if out is None:
            out = np.empty(self.shape, dtype=self.dtype)
        elif out.shape != self.shape or (out.dtype != self.dtype and not cast):
            raise ValueError(f"out must be a {self.dtype} array of shape {self.shape}")
        if not nbytes:
            return out
        device = self.device
        target, offset, after, keep = None, 0, None, []
        if self._stored == out.dtype:
            target, offset = _pinned_place(device, out)
        if target is None:
            staging = target = device._acquire(max(nbytes, 4), host=True)
            keep.append(staging)
            stored, shape = self._stored, self.shape

            def after():
                data = staging.view[:nbytes].view(stored).reshape(shape)
                out[...] = data != 0 if out.dtype == np.bool_ else data

        runs = self._word_runs()
        if runs is None:
            # Views with gaps are copied run by run.
            source = self._buffer
            regions = self._regions(offset, to_device=False)
            record = functools.partial(
                _copy_regions, source=source, target=target, regions=regions
            )
        else:
            packed = DeviceArray(device, (nbytes // 4,), np.uint32)
            self._copy_words(packed, runs, stream, to_view=False)
            source = packed._buffer
            record = functools.partial(
                _copy,
                source=source,
                target=target,
                nbytes=nbytes,
                target_offset=offset,
            )
        stream._submit(
            device.transfer_family,
            record,
            [source] + ([] if keep else [target]),
            keep=keep,
            after=after,
        )
        return out

    def _index(self, key):
        """The view that a basic index selects.

        Returns
        -------
        DeviceArray
            Zero-dimensional where the index names one element.

        Raises
        ------
        IndexError
            For indices out of bounds, as NumPy raises it.
        TypeError
            For indices of other types; advanced indexing (arrays, lists,
            booleans) is handled by `__getitem__` and `__setitem__`.
        """
        key = key if isinstance(key, tuple) else (key,)
        for part in key:
            if not (
                part is None
                or part is Ellipsis
                or isinstance(part, slice)
                or (isinstance(part, (int, np.integer)) and not isinstance(part, bool))
            ):
                raise TypeError(
                    "device arrays are indexed with integers, slices, ..., None, "
                    f"index arrays and boolean masks, not {type(part).__name__}"
                )
        if Ellipsis not in key:
            key = (*key, Ellipsis)  # a view even where all axes are indexed
        # NumPy computes the view on a stand-in with one-byte elements, so
        # that its strides and data offset count elements; nothing is read.
        stand_in = np.lib.stride_tricks.as_strided(
            np.zeros(1, np.uint8), shape=self.shape, strides=self._steps
        )
        view = stand_in[key]
        moved = view.__array_interface__["data"][0]
        moved -= stand_in.__array_interface__["data"][0]
        return self._view(view.shape, self._offset + moved, view.strides)

    def _positions(self, key):
        """Positions in the buffer of the elements an advanced index selects.

        NumPy applies the index to zero-copy broadcast views of each axis's
        indices, so the full rules of advanced indexing apply and memory
        grows with the result only.

        Returns
        -------
        numpy.ndarray
            One position per element of the result, in its shape.

        Raises
        ------
        IndexError
            As NumPy raises it.
        """
        key = key if isinstance(key, tuple) else (key,)
        key = tuple(
            part.copy_to_host() if isinstance(part, DeviceArray) else part
            for part in key
        )
        positions = self._offset
        for axis, (extent, step) in enumerate(zip(self.shape, self._steps)):
            along = np.arange(extent, dtype=np.int64).reshape(
                [-1 if d == axis else 1 for d in range(self.ndim)]
            )
            positions = positions + np.broadcast_to(along, self.shape)[key] * step
        return np.asarray(positions, dtype=np.int64)

    def _word_access(self, positions):
        """Word kernels' arguments for elements at given positions, if any.

        Returns
        -------
        tuple or None
            All words of the buffer as an array, the first word of each
            element, and words per element; None for elements whose size
            is not a multiple of 4 bytes.
        """
        words, rest = divmod(self._stored.itemsize, 4)
        if rest:
            return None
        first = (positions.reshape(-1) * words).astype(np.uint32)
        return self._buffer_words(), to_device(first, self.device), words

    def _buffer_words(self):
        """All words of the array's buffer, as a ``uint32`` array.

        For an array created contiguous, these are its elements' words.
        """
        words = self._view((self._nbytes // 4,), 0, (1,))
        words.dtype = words._stored = np.dtype(np.uint32)
        return words

    def __getitem__(self, key):
        """An element, a view, or a copy for advanced indexing.

        Basic indices (integers, slices, ``...``, ``None``) give views, as
        described for the class. Index arrays, lists and boolean masks
        (NumPy or device arrays) give a new array with the selected
        elements, gathered on the device, as NumPy gives a copy.

        Returns
        -------
        scalar or DeviceArray
        """
        if _advanced(key):
            positions = self._positions(key)
            out = DeviceArray(
                self.device, positions.shape, self.dtype, _stored=self._stored
            )
            access = self._word_access(positions)
            if access is None:
                out.copy_to_device(self.copy_to_host()[key])
            elif out.size:
                whole, first, words = access
                gather, _ = _word_kernels()
                packed = out._buffer_words()
                gather.forall(packed.size, device=self.device)(
                    whole, packed, first, words
                )
            return out.copy_to_host()[()] if out.ndim == 0 else out
        view = self._index(key)
        if view.ndim == 0 and not (isinstance(key, tuple) and None in key):
            return view.copy_to_host()[()]
        return view

    def __setitem__(self, key, value):
        """Write an element, or a value or array into the selected elements.

        `value` is broadcast to the selected shape, as in NumPy. With
        advanced indexing (see `__getitem__`) the values are scattered on
        the device; where an element is selected more than once, which of
        its values it ends up with is undefined.
        """
        if _advanced(key):
            positions = self._positions(key)
            values = np.broadcast_to(np.asarray(value), positions.shape)
            access = self._word_access(positions)
            if access is None:
                host = self.copy_to_host()
                host[key] = values
                self.copy_to_device(host)
            elif positions.size:
                whole, first, words = access
                packed = DeviceArray(
                    self.device, positions.shape, self.dtype, _stored=self._stored
                ).copy_to_device(values)
                _, scatter = _word_kernels()
                words_of = packed._buffer_words()
                scatter.forall(words_of.size, device=self.device)(
                    whole, words_of, first, words
                )
            return
        view = self._index(key)
        values = np.broadcast_to(np.asarray(value, dtype=self.dtype), view.shape)
        view.copy_to_device(values)

    def __array__(self, dtype=None, copy=None):
        """Convert to a NumPy array, which copies from the device."""
        out = self.copy_to_host()
        return out if dtype is None else out.astype(dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """Compute NumPy ufuncs on the device.

        Calls of ufuncs with one result, and their ``reduce`` along
        ``axis=None`` (or ``axis=0`` of a 1-d array), run as kernels; see
        `numba_vulkan.vectorizers`. Operators (``a + b``, ``a < b``, ...)
        and `sum`, `min` and `max` go through here as well. The result is
        a device array, or a scalar for a reduction.

        Returns
        -------
        DeviceArray, scalar or NotImplemented
            ``NotImplemented`` for other methods and options, which makes
            NumPy raise a ``TypeError``.
        """
        if ufunc.nout != 1 or kwargs.keys() - {"axis"}:
            return NotImplemented
        if out is not None:
            if len(out) != 1:
                return NotImplemented
            out = out[0]
        device_ufunc = _device_ufunc(ufunc)
        if method == "__call__" and "axis" not in kwargs:
            return device_ufunc(*inputs, out=out)
        if method == "reduce" and ufunc.nin == 2 and out is None and len(inputs) == 1:
            return device_ufunc.reduce(inputs[0], axis=kwargs.get("axis", 0))
        return NotImplemented

    def sum(self):
        """Sum of all elements, computed on the device.

        Returns
        -------
        scalar
        """
        return np.add.reduce(self, axis=None)

    def min(self):
        """Smallest element, computed on the device.

        Returns
        -------
        scalar
        """
        return np.minimum.reduce(self, axis=None)

    def max(self):
        """Largest element, computed on the device.

        Returns
        -------
        scalar
        """
        return np.maximum.reduce(self, axis=None)


def _operator(ufunc, reflected=False):
    """A binary operator method of DeviceArray that applies `ufunc`."""
    if reflected:
        return lambda self, other: ufunc(other, self)
    return lambda self, other: ufunc(self, other)


for _name, _ufunc in {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "truediv": np.true_divide,
    "floordiv": np.floor_divide,
    "mod": np.remainder,
    "pow": np.power,
    "and": np.bitwise_and,
    "or": np.bitwise_or,
    "xor": np.bitwise_xor,
}.items():
    setattr(DeviceArray, f"__{_name}__", _operator(_ufunc))
    setattr(DeviceArray, f"__r{_name}__", _operator(_ufunc, reflected=True))
for _name, _ufunc in {
    "lt": np.less,
    "le": np.less_equal,
    "gt": np.greater,
    "ge": np.greater_equal,
    "eq": np.equal,
    "ne": np.not_equal,
}.items():
    setattr(DeviceArray, f"__{_name}__", _operator(_ufunc))
DeviceArray.__neg__ = lambda self: np.negative(self)
DeviceArray.__abs__ = lambda self: np.absolute(self)
DeviceArray.__invert__ = lambda self: np.invert(self)
# Equality is elementwise, so device arrays cannot be hashed.
DeviceArray.__hash__ = None

_DEVICE_UFUNCS = {}


def _device_ufunc(ufunc):
    """A Vulkan ufunc that applies a NumPy ufunc, created once per ufunc."""
    if ufunc not in _DEVICE_UFUNCS:
        from numba import vectorize  # imports this module

        names = ", ".join(f"x{k}" for k in range(ufunc.nin))
        scope = {"np": np}
        exec(  # noqa: S102
            f"def {ufunc.__name__}({names}):\n"
            f"    return np.{ufunc.__name__}({names})\n",
            scope,
        )
        _DEVICE_UFUNCS[ufunc] = vectorize(target="vulkan", identity=ufunc.identity)(
            scope[ufunc.__name__]
        )
    return _DEVICE_UFUNCS[ufunc]


def _c_steps(shape):
    """Steps between elements of a C-contiguous array, in elements."""
    steps, step = [], 1
    for extent in reversed(shape):
        steps.insert(0, step)
        step *= max(extent, 1)
    return tuple(steps)


def _resolve_shape(shape, size):
    """A shape with its ``-1`` replaced, checked against the element count.

    Raises
    ------
    ValueError
        If the shape does not fit `size` elements.
    """
    unknown = [k for k, n in enumerate(shape) if n == -1]
    known = int(np.prod([n for n in shape if n != -1], dtype=np.int64))
    if len(unknown) > 1 or any(n < -1 for n in shape):
        raise ValueError(f"invalid shape {shape}")
    if unknown:
        if known == 0 or size % known:
            raise ValueError(f"cannot reshape {size} elements into {shape}")
        shape = tuple(size // known if n == -1 else n for n in shape)
    if int(np.prod(shape, dtype=np.int64)) != size:
        raise ValueError(f"cannot reshape {size} elements into {shape}")
    return shape


def _reshaped_steps(shape, steps, new):
    """Steps that give an array a new shape without copying, if any exist.

    A port of NumPy's ``_attempt_nocopy_reshape``: runs of axes that are
    contiguous with each other may be split and merged.

    Parameters
    ----------
    shape, steps : tuple of int
        The current shape and steps, in elements.
    new : tuple of int
        The wanted shape, with as many elements.

    Returns
    -------
    tuple of int or None
        The steps for `new`, or ``None`` if it needs a copy.
    """
    if 0 in shape or 0 in new:
        return _c_steps(new)
    old = [(n, step) for n, step in zip(shape, steps) if n != 1]
    dims, strides = [n for n, _ in old], [step for _, step in old]
    out = [0] * len(new)
    oi, oj, ni, nj = 0, 1, 0, 1
    while ni < len(new) and oi < len(dims):
        np_, op = new[ni], dims[oi]
        while np_ != op:
            if np_ < op:
                np_ *= new[nj]
                nj += 1
            else:
                op *= dims[oj]
                oj += 1
        for ok in range(oi, oj - 1):
            if strides[ok] != dims[ok + 1] * strides[ok + 1]:
                return None
        out[nj - 1] = strides[oj - 1]
        for nk in range(nj - 1, ni, -1):
            out[nk - 1] = out[nk] * new[nk]
        ni, nj, oi, oj = nj, nj + 1, oj, oj + 1
    last = out[ni - 1] if ni >= 1 else 1
    for nk in range(ni, len(new)):
        out[nk] = last
    return tuple(out)


def to_device(array, device=None, stream=None):
    """Copy a NumPy array to a device.

    Parameters
    ----------
    array : array_like
        The values.
    device : int, str, DeviceInfo, Device or None
        The device; see `get_device`. Defaults to the selected device, or
        that of `stream`.
    stream : Stream, optional
        Enqueue the copy on a stream and return at once; see `Stream`.

    Returns
    -------
    DeviceArray
    """
    array = np.asarray(array)
    if stream is not None and device is None:
        device = stream.device
    target = device_array(array.shape, array.dtype, device)
    return target.copy_to_device(array, stream=stream)


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
    if type(which) is int and which in _devices:  # the common case, quickly
        return _devices[which]
    if which is None:
        which = os.environ.get("NUMBA_VULKAN_DEVICE")
        if which not in _defaults:
            _defaults[which] = _find_device(which)
        return _defaults[which]
    return _find_device(which)


def _find_device(which):
    """`get_device` for an index, name, `DeviceInfo` or ``None``."""
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
        device = _devices[info.index] = Device(info)
        # The probe launches a kernel on the device, so it runs once the
        # device is registered.
        from numba_vulkan import probes

        device.mode = device.mode._replace(**probes.workarounds(device))
    return _devices[info.index]


def _stream_pending(buffer):
    """Whether a stream may still use a buffer."""
    stream = buffer.stream
    return stream is not None and buffer.stream_value > stream._done


def _settle(buffer):
    """Wait until no stream uses a buffer any more."""
    if _stream_pending(buffer):
        buffer.stream._wait(buffer.stream_value)


@dataclass
class _StreamOp:
    """One submission of a stream; see `Stream`.

    Attributes
    ----------
    value : int
        The value of the stream's semaphore once it has finished.
    cmd, family : object
        Its command buffer and the queue family it was submitted to.
    keep : list of _Buffer
        Staging buffers it uses, pooled when it has finished.
    after : callable or None
        Runs when it has finished: the host's part of a copy to the host.
    checks : list of _Slot
        Slots whose error status is checked when it has finished.
    """

    value: int
    cmd: object
    family: int
    keep: list
    after: object = None
    checks: list = field(default_factory=list)


class Stream:
    """An ordered sequence of copies and launches on a device.

    Work on different streams can run at the same time: copies to and from
    the device on a copy engine, kernels on the device, as with streams in
    ``numba.cuda``. Work on one stream runs in the order it was enqueued.
    Create streams with `stream`, and pass them as ``stream=`` to
    `to_device`, `DeviceArray.copy_to_device` and
    `DeviceArray.copy_to_host`, and as the third element of
    ``kernel[groups, local_size, stream]`` or as ``forall(n, stream=...)``.

    All of these return at once. Data copied to the host is there after
    `synchronize`; an exception a kernel raised is raised by it. Copies
    from and to arrays created with `pinned_array` go directly between that
    memory and the device; other host arrays go through a staging buffer,
    and the data of a copy to the host is copied out of it by `synchronize`.
    NumPy arrays passed to a launch on a stream are copied to the device
    before it and, if the kernel writes them, back after it; what a kernel
    prints is printed by `synchronize` as well. Views with
    gaps are copied run by run of adjacent elements, or with many runs by
    a kernel that gathers or scatters them on the device.

    Each stream has a timeline semaphore: every submission waits for the
    previous one of its stream, and for the streams that last used its
    buffers. Copies go to a queue of a transfer-only queue family where the
    device has one, kernels to a second compute queue where there is one.
    Work enqueued without a stream waits on the host for streams that still
    use its buffers, and the other way round.

    Parameters
    ----------
    device : Device
        The device.
    """

    def __init__(self, device):
        self.device = device
        kind = vk.VkSemaphoreTypeCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            semaphoreType=vk.VK_SEMAPHORE_TYPE_TIMELINE,
            initialValue=0,
        )
        self._semaphore = vk.vkCreateSemaphore(
            device.handle,
            vk.VkSemaphoreCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, pNext=kind
            ),
            None,
        )
        # The value the last submission signals, and one known to be reached.
        self.value = 0
        self._done = 0
        self._ops = []
        self._spare = {}
        self._released = []
        self._errors = []
        device._streams.add(self)
        # Only an idle stream can be dropped (see _submit), so its semaphore
        # is no longer used then.
        # Submissions without a stream may still wait for the semaphore when
        # the stream is dropped; the device destroys it once they are done.
        cleanup = weakref.finalize(
            self, device._dead_semaphores.append, self._semaphore
        )
        # At exit the process lets go of everything; the device may be gone.
        cleanup.atexit = False

    def __repr__(self):
        return f"<Stream on {self.device.info.name}>"

    def query(self):
        """Whether all work enqueued on the stream has finished.

        Returns
        -------
        bool
        """
        value = _ffi.new("uint64_t[1]")
        _check(
            _lib.vkGetSemaphoreCounterValue(self.device.handle, self._semaphore, value)
        )
        self._done = max(self._done, value[0])
        self._retire()
        return self._done >= self.value

    def synchronize(self):
        """Wait until all work enqueued on the stream has finished.

        Raises
        ------
        Exception
            The first exception that a kernel launched on the stream raised
            since the last synchronisation.
        """
        self._wait(self.value)
        if self._errors:
            (kernel, code), self._errors = self._errors[0], []
            from numba_vulkan.dispatcher import raise_kernel_error

            raise_kernel_error(kernel.name, code, asynchronous=True)

    @contextlib.contextmanager
    def auto_synchronize(self):
        """Wait for the stream when the block ends, as in ``numba.cuda``.

        Yields
        ------
        Stream
            The stream itself.
        """
        yield self
        self.synchronize()

    def _wait(self, value):
        """Wait until the stream has finished its work up to a value."""
        if value > self._done:
            info = vk.VkSemaphoreWaitInfo(
                sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                semaphoreCount=1,
                pSemaphores=[self._semaphore],
                pValues=[value],
            )
            vk.vkWaitSemaphores(self.device.handle, info, 0xFFFFFFFFFFFFFFFF)
            self._done = value
        self._retire()

    def _retire(self):
        """Finish the submissions that the stream has passed."""
        done = [op for op in self._ops if op.value <= self._done]
        if not done:
            return
        self._ops = [op for op in self._ops if op.value > self._done]
        device = self.device
        for op in done:
            if op.after is not None:
                op.after()
            for slot in op.checks:
                if slot.stream is self and slot.stream_value > op.value:
                    continue  # used again later; checked then
                code = int(slot.status.view[:4].view(np.int32)[0])
                if code:
                    self._errors.append((slot.kernel, code))
                    slot.status.view[:4] = 0
            for buffer in op.keep:
                device._release(buffer)
            self._spare.setdefault(op.family, []).append(op.cmd)
        released, self._released = self._released, []
        for buffer in released:
            device._release(buffer)
        if not self._ops:
            device._busy_streams.discard(self)

    def _submit(self, family, record, buffers, keep=(), after=None, checks=()):
        """Enqueue one command buffer on the stream.

        Parameters
        ----------
        family : int
            Queue family: the device's compute or transfer family.
        record : callable
            Records the commands into a command buffer.
        buffers : list of _Buffer
            The buffers it reads or writes.
        keep, after, checks
            See `_StreamOp`.
        """
        device = self.device
        # Work enqueued without a stream that uses the buffers goes first.
        for buffer in buffers:
            if buffer.used > device._retired_seq:
                device._flush()
                device._retire(
                    sum(batch.seq <= buffer.used for batch in device._pending)
                )
        waits = {}
        if self.value:
            waits[self] = self.value
        # The streams that last used the buffers, also those known to be done
        # with them, so that the order is stated on the device as well.
        for buffer in [*buffers, *keep]:
            other = buffer.stream
            if other is not None and other is not self:
                waits[other] = max(waits.get(other, 0), buffer.stream_value)
        spare = self._spare.get(family)
        if spare:
            cmd = spare.pop()
        else:
            cmd = vk.vkAllocateCommandBuffers(
                device.handle,
                vk.VkCommandBufferAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                    commandPool=device._pools[family],
                    level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    commandBufferCount=1,
                ),
            )[0]
        _check(_lib.vkBeginCommandBuffer(cmd, _BEGIN_INFO))
        record(cmd)
        _check(_lib.vkEndCommandBuffer(cmd))
        self.value += 1
        semaphores = [stream._semaphore for stream in waits]
        timeline = vk.VkTimelineSemaphoreSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            waitSemaphoreValueCount=len(waits),
            pWaitSemaphoreValues=list(waits.values()) or None,
            signalSemaphoreValueCount=1,
            pSignalSemaphoreValues=[self.value],
        )
        submit = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext=timeline,
            waitSemaphoreCount=len(waits),
            pWaitSemaphores=semaphores or None,
            pWaitDstStageMask=[vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT] * len(waits)
            or None,
            commandBufferCount=1,
            pCommandBuffers=[cmd],
            signalSemaphoreCount=1,
            pSignalSemaphores=[self._semaphore],
        )
        queue = (
            device.stream_queue if family == device.family else device.transfer_queue
        )
        vk.vkQueueSubmit(queue, 1, [submit], vk.VK_NULL_HANDLE)
        for buffer in [*buffers, *keep]:
            buffer.stream, buffer.stream_value = self, self.value
        self._ops.append(
            _StreamOp(self.value, cmd, family, list(keep), after, list(checks))
        )
        device._busy_streams.add(self)
        if len(self._ops) > 64:
            self.query()  # recycle what has finished


def stream(device=None):
    """Create a `Stream` on a device, like ``numba.cuda.stream``.

    Parameters
    ----------
    device : int, str, Device or None
        The device; the selected one by default.

    Returns
    -------
    Stream

    Examples
    --------
    >>> s = nv.stream()
    >>> d = nv.to_device(chunk, stream=s)        # returns at once
    >>> kernel[groups, 64, s](d, out)            # after the copy, on the device
    >>> host = out.copy_to_host(stream=s)        # valid after synchronize
    >>> s.synchronize()
    """
    return Stream(device if isinstance(device, Device) else get_device(device))


def pinned_array(shape, dtype=np.float64, device=None):
    """Allocate a host array that a device can copy to and from directly.

    Like ``numba.cuda.pinned_array``: the memory belongs to the device and
    is mapped into the process, so copies on a `Stream` need no staging
    buffer and run while the host goes on. Use it for the host side of data
    streamed in chunks.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the array.
    dtype : numpy.dtype
        Element type.
    device : int, str, Device or None
        The device; the selected one by default.

    Returns
    -------
    numpy.ndarray
        Uninitialised, C-contiguous.
    """
    device = device if isinstance(device, Device) else get_device(device)
    dtype = np.dtype(dtype)
    shape = (shape,) if np.isscalar(shape) else tuple(int(n) for n in shape)
    nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    buffer = device._acquire(max(nbytes, 4), host=True)
    start = buffer.view.ctypes.data
    # A ctypes object over the mapped memory is the base of every view of
    # the array, so its finalizer runs once none is left.
    holder = (ctypes.c_ubyte * max(nbytes, 1)).from_address(start)
    array = np.frombuffer(holder, dtype=np.uint8, count=nbytes)
    device._pinned[start] = (start + max(nbytes, 1), buffer)
    weakref.finalize(holder, device._unpin, start)
    return array.view(dtype).reshape(shape)


def _pinned_place(device, array):
    """The pinned buffer and offset holding a host array's data, if any."""
    if not device._pinned or not array.flags.c_contiguous:
        return None, 0
    start = array.__array_interface__["data"][0]
    for base, (end, buffer) in device._pinned.items():
        if base <= start and start + array.nbytes <= end:
            return buffer, start - base
    return None, 0


class Event:
    """A point in the work submitted to a device, like a CUDA event.

    `record` marks the point after all work launched so far; the device
    writes the time at which it gets there. `elapsed_time` gives the time
    between two events on the device, which excludes the cost of launching
    and waiting in Python, as long as the device has work: as with CUDA's
    events, time in which it waits for the host between two events counts.
    On a `Stream`, ``record(stream)`` marks the point after the work
    enqueued on that stream instead. Create events with `event`.

    Parameters
    ----------
    device : Device
        The device.

    Raises
    ------
    VulkanSupportError
        If the device's compute queue writes no timestamps.
    """

    QUERIES = 1024

    def __init__(self, device):
        if not device.info.timestamp_bits:
            raise VulkanSupportError(f"{device.info.name} cannot measure time")
        self.device = device
        self._query = None
        self._seq = None
        # The stream it was recorded on, and the value it waits for there.
        self._stream = None
        self._value = 0

    def record(self, stream=None):
        """Mark the point after all work launched on the device so far.

        Parameters
        ----------
        stream : Stream, optional
            Mark the point after the work enqueued on this stream so far
            instead, like ``cuda.event().record(stream)``.

        Returns
        -------
        Event
            The event itself.

        Raises
        ------
        ValueError
            If the stream belongs to another device.
        """
        device = self.device
        if stream is not None and stream.device is not device:
            raise ValueError("the stream belongs to another device")
        if device._queries is None:
            device._queries = vk.vkCreateQueryPool(
                device.handle,
                vk.VkQueryPoolCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                    queryType=vk.VK_QUERY_TYPE_TIMESTAMP,
                    queryCount=self.QUERIES,
                ),
                None,
            )
        # Queries are used in turn; an event recorded QUERIES events ago is
        # overwritten.
        self._query = query = device._next_query
        device._next_query = (device._next_query + 1) % self.QUERIES
        if stream is not None:
            pool = device._queries

            def record(cmd):
                _lib.vkCmdResetQueryPool(cmd, pool, query, 1)
                _lib.vkCmdWriteTimestamp(
                    cmd, vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, query
                )

            stream._submit(device.family, record, [])
            self._stream, self._value, self._seq = stream, stream.value, None
            return self
        self._stream = None
        batch = device._open or device._open_batch()
        _lib.vkCmdResetQueryPool(batch.cmd, device._queries, self._query, 1)
        _lib.vkCmdWriteTimestamp(
            batch.cmd,
            vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            device._queries,
            self._query,
        )
        # Not submitted on its own: it goes with the next launch, so that a
        # start event and the work after it share a submission.
        self._seq = batch.seq
        return self

    def synchronize(self):
        """Wait until the device has passed the event."""
        if self._stream is not None:
            self._stream._wait(self._value)
            return
        if self._seq is None:
            raise RuntimeError("the event was not recorded")
        device = self.device
        device._flush()
        if self._seq > device._retired_seq:
            device._retire(sum(b.seq <= self._seq for b in device._pending))

    def _ticks(self):
        """The timestamp the device wrote."""
        self.synchronize()
        value = _ffi.new("uint64_t[1]")
        _check(
            _lib.vkGetQueryPoolResults(
                self.device.handle,
                self.device._queries,
                self._query,
                1,
                8,
                value,
                8,
                vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT,
            )
        )
        return value[0]

    def elapsed_time(self, end):
        """Milliseconds the device took from this event to `end`.

        Parameters
        ----------
        end : Event
            A later event on the same device.

        Returns
        -------
        float
        """
        if end.device is not self.device:
            raise ValueError("the events belong to different devices")
        bits = self.device.info.timestamp_bits
        ticks = (end._ticks() - self._ticks()) & ((1 << bits) - 1)
        return ticks * self.device.info.timestamp_period / 1e6


def event(device=None):
    """Create an `Event` on a device.

    Parameters
    ----------
    device : int, str, Device or None
        The device; the selected one by default.

    Returns
    -------
    Event

    Examples
    --------
    >>> start, end = nv.event(), nv.event()
    >>> start.record()
    >>> kernel.forall(n)(x, out)      # device arrays: does not wait
    >>> end.record()
    >>> start.elapsed_time(end)       # milliseconds on the device
    """
    return Event(device if isinstance(device, Device) else get_device(device))


def event_elapsed_time(start, end):
    """Milliseconds between two events, as ``numba.cuda`` names it.

    Returns
    -------
    float
    """
    return start.elapsed_time(end)


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
