"""Run both benchmark suites and store the results with a description of the machine.

The results go to ``benchmarks/results/<user>/<UTC timestamp>.json``, one file
per run, so that results from different people, machines and versions can be
collected in the repository and compared (``benchmarks/report.py`` draws the
charts in the documentation).

    uv run --group bench python benchmarks/collect.py --user arunoruto
    uv run python benchmarks/collect.py --user me --machine laptop --quick

The file holds the ``--user`` handle (for example a GitHub name), the
``--machine`` label (by default the short host name), and the hardware and
software models and versions. Nothing else identifies the machine or its
user; pass ``--machine`` to choose another label.

File format (``"schema": 1``)
-----------------------------
``schema``, ``user``, ``machine``, ``date`` (UTC, ISO 8601)
``git``: ``commit``, ``branch``, ``dirty`` (uncommitted changes)
``software``: package versions, Python, operating system, libclc, and
the LLVM version of ``llc`` if it replaces llvmlite's backend
``hardware``: ``cpu`` (``model``, ``threads``), ``memory_gib``, ``vulkan`` (one
entry per device: name, kind, vendor and device ID, driver, API version,
float64 support, workarounds applied by numba-vulkan), ``cuda``
``settings``: the parameters of each suite
``results``: one record per suite, workload and backend:

    suite        "apps" (bench.py) or "kernels" (kernels.py)
    workload     e.g. "mandelbrot", "matmul"
    description  problem size, e.g. "2048x2048, 200 iterations"
    backend      "cpu", "vulkan" or "cuda"
    device       device name, or null for the CPU
    variant      e.g. "parallel", "numpy arrays", "device arrays",
                 "32-bit integers"
    label        the backend as the scripts print it
    first_s      the first call in seconds, including compilation (the kernel
                 cache is off during a run), or null
    samples_s    the timed calls after it, in seconds, end to end
    enqueue_s    optional (kernels suite): the same calls until they
                 returned, before the device finished
    device_s     optional (kernels suite): the time the device spent between
                 events recorded before and after each call
    check        agreement with the CPU result, as text
"""

import argparse
import datetime
import json
import os
import pathlib
import platform
import re
import socket
import subprocess
import sys
from importlib import metadata

import vulkan as vk

import numba_vulkan as nv
from numba_vulkan import codegen, libclc

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bench
import kernels

SCHEMA = 1
ROOT = pathlib.Path(__file__).parent.parent
RESULTS = pathlib.Path(__file__).parent / "results"
# Handles and labels become path components.
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
PACKAGES = ("numba-vulkan", "numba", "llvmlite", "numpy", "numba-cuda", "vulkan")


def _git(*args):
    """Output of a git command in the repository, or ``None``."""
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_info():
    """The commit the benchmarks ran on."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def _sysctl(name):
    """A value from macOS's ``sysctl``, or ``None``."""
    try:
        return subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _os_name():
    """Operating system and its version."""
    if sys.platform == "darwin":
        return f"macOS {platform.mac_ver()[0]} ({platform.machine()})"
    name = f"{platform.system()} {platform.release()}"
    try:
        with open("/etc/os-release") as fh:
            fields = dict(line.rstrip("\n").split("=", 1) for line in fh if "=" in line)
        name += f" ({fields.get('PRETTY_NAME', '').strip(chr(34))})"
    except OSError:
        pass
    return name


def software_info():
    """Versions of everything that shapes the results."""
    info = {"python": platform.python_version(), "os": _os_name()}
    for package in PACKAGES:
        try:
            info[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            info[package] = None
    info["libclc"] = libclc.version()
    info["llc"] = _llc_version()
    return info


def _llc_version():
    """LLVM version of the ``llc`` that replaces llvmlite's backend, if any.

    Returns
    -------
    str or None
        ``None`` when llvmlite's own backend translates the kernels.
    """
    llc = codegen.emitter.llc
    if llc is None:
        return None
    try:
        out = subprocess.run(
            [llc, "--version"], capture_output=True, text=True, check=False
        ).stdout
    except OSError:
        return "unknown"
    found = re.search(r"LLVM version (\S+)", out)
    return found.group(1) if found else "unknown"


def _cpu_model():
    """The CPU's model name."""
    if sys.platform == "darwin":
        return _sysctl("machdep.cpu.brand_string") or platform.machine()
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _memory_gib():
    """Installed memory in GiB, or ``None``."""
    if sys.platform == "darwin":
        size = _sysctl("hw.memsize")
        return round(int(size) / 2**30, 1) if size and size.isdigit() else None
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 2**20, 1)
    except OSError:
        pass
    return None


def _driver(handle):
    """Driver name and version string of a Vulkan device."""
    driver = vk.VkPhysicalDeviceDriverProperties(
        sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES
    )
    props = vk.VkPhysicalDeviceProperties2(
        sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, pNext=driver
    )
    vk.vkGetPhysicalDeviceProperties2(handle, props)
    return (
        vk.ffi.string(driver.driverName).decode(),
        vk.ffi.string(driver.driverInfo).decode(),
    )


def _version(packed):
    """A Vulkan API version as text."""
    return f"{packed >> 22}.{(packed >> 12) & 0x3FF}.{packed & 0xFFF}"


def vulkan_devices():
    """The Vulkan devices and how numba-vulkan compiles for them."""
    devices = []
    for info in nv.list_devices():
        props = vk.vkGetPhysicalDeviceProperties(info.handle)
        name, version = _driver(info.handle)
        mode = nv.get_device(info.index).mode
        devices.append(
            {
                "name": info.name,
                "kind": info.kind,
                "vendor_id": f"{props.vendorID:04x}",
                "device_id": f"{props.deviceID:04x}",
                "driver": name,
                "driver_info": version,
                "driver_version": f"{props.driverVersion:x}",
                "api_version": _version(props.apiVersion),
                "float64": info.float64,
                "float32_atomic_add": info.float32_atomic_add,
                "workarounds": [
                    w for w in ("soft_fma", "soft_rounding") if getattr(mode, w)
                ],
            }
        )
    return devices


def cuda_devices():
    """The CUDA device numba-cuda uses, if any."""
    if not bench.HAVE_CUDA:
        return []
    from numba import cuda

    device = cuda.get_current_device()
    name = device.name.decode() if isinstance(device.name, bytes) else device.name
    entry = {
        "name": name,
        "compute_capability": ".".join(map(str, device.compute_capability)),
    }
    try:
        from numba.cuda.cudadrv import driver

        major, minor = driver.driver.get_version()
        entry["driver_cuda_version"] = f"{major}.{minor}"
    except Exception:  # noqa: BLE001, S110 (the version is optional)
        pass
    return [entry]


def hardware_info():
    """CPU, memory and accelerators."""
    return {
        "cpu": {"model": _cpu_model(), "threads": os.cpu_count()},
        "memory_gib": _memory_gib(),
        "vulkan": vulkan_devices(),
        "cuda": cuda_devices(),
    }


def collect(user, machine, quick=False, verbose=True):
    """Run both suites and return the content of a results file.

    The on-disk kernel cache is turned off, so that first calls include
    compilation.
    """
    os.environ["NUMBA_VULKAN_CACHE"] = "0"
    apps = {"size": 512, "maxiter": 100, "repeat": 3} if quick else {
        "size": 2048, "maxiter": 200, "repeat": 5}  # fmt: skip
    kern = {"size": 1 << 20, "matrix": 256, "repeat": 3} if quick else {
        "size": 1 << 24, "matrix": 1024, "repeat": 10}  # fmt: skip
    date = datetime.datetime.now(datetime.UTC).replace(microsecond=0)
    results = bench.run(**apps, verbose=verbose)
    results += kernels.run(**kern, verbose=verbose)
    return {
        "schema": SCHEMA,
        "user": user,
        "machine": machine,
        "date": date.isoformat().replace("+00:00", "Z"),
        "git": git_info(),
        "software": software_info(),
        "hardware": hardware_info(),
        "settings": {
            "apps": apps,
            "kernels": kern,
            "quick": quick,
            "kernel_cache": False,
        },
        "results": results,
    }


def _default_machine():
    """The host name up to the first dot, with unusual characters replaced."""
    name = socket.gethostname().split(".")[0]
    return re.sub(r"[^A-Za-z0-9._-]", "-", name).strip("-._") or None


def _default_user():
    """The user handle from the environment or git, or ``None``."""
    return os.environ.get("NUMBA_VULKAN_BENCH_USER") or _git("config", "github.user")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--user",
        default=_default_user(),
        help="your handle, e.g. GitHub name (default: $NUMBA_VULKAN_BENCH_USER "
        "or `git config github.user`)",
    )
    parser.add_argument(
        "--machine",
        default=os.environ.get("NUMBA_VULKAN_BENCH_MACHINE") or _default_machine(),
        help="a label for this machine (default: $NUMBA_VULKAN_BENCH_MACHINE or "
        "the host name)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="small problem sizes, for trying it out"
    )
    parser.add_argument(
        "--output", type=pathlib.Path, help="write here instead of benchmarks/results/"
    )
    opts = parser.parse_args()
    for flag in ("user", "machine"):
        value = getattr(opts, flag)
        if not value or not NAME.match(value):
            parser.error(f"--{flag} must be given as letters, digits, '.', '_' or '-'")
    data = collect(opts.user, opts.machine, opts.quick)
    path = opts.output or (
        RESULTS / opts.user / (data["date"].replace(":", "") + ".json")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
