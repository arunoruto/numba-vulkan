import os
import shutil
import subprocess

import pytest

import numba_vulkan as nv

_DEVICES = nv.list_devices()

# Invalid SPIR-V can crash a driver, so every kernel is validated first.
if shutil.which("spirv-val") is not None:
    os.environ["NUMBA_VULKAN_VALIDATE"] = "1"


@pytest.fixture(params=_DEVICES, ids=[d.name for d in _DEVICES])
def device(request):
    """Every Vulkan device on this machine, including CPU rasterisers."""
    return request.param.index


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
