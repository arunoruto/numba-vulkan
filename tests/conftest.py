import os
import shutil
import subprocess
import sys

import pytest

import numba_vulkan as nv

# Lets tests import the fuzzer, which is a script next to them.
sys.path.insert(0, os.path.dirname(__file__))

_DEVICES = nv.list_devices()

# The tests are about the compiler, so nothing is taken from the kernel cache;
# tests/test_kernelcache.py turns it on for itself.
os.environ.setdefault("NUMBA_VULKAN_CACHE", "0")

# Invalid SPIR-V can crash a driver, so every kernel is validated first.
if shutil.which("spirv-val") is not None:
    os.environ["NUMBA_VULKAN_VALIDATE"] = "1"


@pytest.fixture(params=_DEVICES, ids=[d.name for d in _DEVICES])
def device(request):
    """Every Vulkan device on this machine, including CPU rasterisers."""
    return request.param.index


@pytest.fixture(autouse=True)
def _no_validation_messages():
    """With ``NUMBA_VULKAN_DEBUG=1``, fail tests the validation layer objects to.

    Messages are attributed to the test during which they arrive; work that
    a test leaves running can report in the next one.
    """
    from numba_vulkan import runtime

    if not runtime.debug_enabled():
        yield
        return
    runtime.validation_messages(clear=True)
    yield
    found = runtime.validation_messages(clear=True)
    if found:
        pytest.fail(
            "Vulkan validation layer:\n"
            + "\n".join(f"{level}: {text}" for level, text in found),
            pytrace=False,
        )


@pytest.fixture(autouse=True)
def _needs_float64(request):
    """Skip a test marked ``float64`` on a device without float64.

    Such devices, Apple GPUs among them, compute float64 kernels in float32
    (see numba_vulkan.narrowing), so float64 precision and the errors about
    float64 cannot be expected there. The device is the test's ``device``
    or, for tests without one, the selected device.

    ``NUMBA_VULKAN_TEST_FLOAT64`` overrides the check: ``1`` runs these
    tests on every device (to see how far float32 is off), ``0`` skips
    them on every device.
    """
    if request.node.get_closest_marker("float64") is None:
        return
    forced = os.environ.get("NUMBA_VULKAN_TEST_FLOAT64")
    if forced == "1":
        return
    if forced == "0":
        pytest.skip("NUMBA_VULKAN_TEST_FLOAT64=0")
    if "device" in request.fixturenames:
        info = _DEVICES[request.getfixturevalue("device")]
    else:
        info = nv.get_device().info
    if not info.float64:
        pytest.skip(f"{info.name} has no float64")


@pytest.fixture
def run(device):
    """Launch a kernel on the device, skipping if it lacks a capability."""

    def launch(kernel, extent, *args):
        try:
            kernel.forall(extent, device=device)(*args)
        except nv.VulkanSupportError as exc:
            pytest.skip(str(exc))

    return launch


@pytest.fixture
def validate(tmp_path):
    """Check a compiled kernel with spirv-val when it is installed."""

    def check(compiled):
        if shutil.which("spirv-val") is None:
            pytest.skip("spirv-val not available")
        path = tmp_path / "kernel.spv"
        path.write_bytes(compiled.spirv)
        res = subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.2", str(path)],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, res.stderr

    return check
