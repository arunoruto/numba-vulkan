"""Behaviour of a device that its reported features do not reveal.

Two kinds of ``float64`` behaviour make libclc and kernels go wrong on some
drivers, and both are found out by running a small kernel on each device:

* Vulkan lets a driver compute ``Fma`` as a multiplication followed by an
  addition, rounding twice. libclc assumes a fused operation; without it,
  its ``sin``, ``cos`` and ``tan`` lose all accuracy beyond small
  arguments. llvmpipe does not fuse. Kernels for such a device are
  compiled with a software version
  (`numba_vulkan.legalize.emulate_fma64`).
* llvmpipe (Mesa 26.1) gets ``Trunc`` and ``RoundEven`` of ``double``
  values wrong in its vectorised code (values that differ between
  invocations); the scalar path is correct. Kernels for such a device
  compute both from ``Floor`` (`numba_vulkan.legalize.emulate_rounding64`).

The probe therefore gives each invocation its own operands. Its results
depend only on the driver, so they are kept in ``probes.json`` in the
cache directory (``~/.cache/numba-vulkan``), per device and driver
version, unless ``NUMBA_VULKAN_CACHE=0``.

The environment variables ``NUMBA_VULKAN_SOFT_FMA`` and
``NUMBA_VULKAN_SOFT_ROUNDING`` override the probe: ``1`` applies the
workaround on every device, ``0`` on none.
"""

import json
import os
import tempfile
import warnings

import numpy as np
import vulkan as vk
from llvmlite import ir
from numba.core.extending import intrinsic

from numba_vulkan import kernelcache, libclc

FMA_ENV_VAR = "NUMBA_VULKAN_SOFT_FMA"
ROUNDING_ENV_VAR = "NUMBA_VULKAN_SOFT_ROUNDING"
# Part of the key of stored results; changes whenever the probe does.
_VERSION = "1"
_RESULTS = "probes.json"
# Rows: a, b and c, a value to truncate and one to round, one column per
# invocation.
# a * b + c with a = 1 + 2**-30, b = 1 - 2**-30, c = -1 is -2**-60 when the
# operation is fused, and 0 when the product is rounded first; the values
# differ slightly between columns so that drivers cannot treat them as
# uniform. 2**24 + k + 1.25 must truncate to 2**24 + k + 1, and k + 0.5
# round to the even neighbour.
_LANES = 16
_k = np.arange(_LANES)
_INPUT = np.stack(
    [
        1 + 2.0**-30 + _k * 2.0**-40,
        1 - 2.0**-30 - _k * 2.0**-40,
        np.full(_LANES, -1.0),
        2.0**24 + _k + 1.25,
        _k + 0.5,
    ]
)


@intrinsic(target="vulkan")
def _fma(typingctx, a, b, c):
    """``llvm.fma`` on three floats of the same type."""

    def codegen(context, builder, signature, args):
        """Call the intrinsic."""
        ty = args[0].type
        fnty = ir.FunctionType(ty, [ty] * 3)
        fn = builder.module.declare_intrinsic("llvm.fma", [ty], fnty)
        return builder.call(fn, list(args))

    return a(a, b, c), codegen


_kernel = None


def _probe_kernel():
    """The probe kernel, compiled on first use."""
    global _kernel
    if _kernel is None:
        from numba_vulkan.dispatcher import jit
        from numba_vulkan.stubs import global_id

        def probe(x, out):
            i = global_id(0)
            if i < x.shape[1]:
                out[0, i] = _fma(x[0, i], x[1, i], x[2, i])
                out[1, i] = np.trunc(x[3, i])
                out[2, i] = round(x[4, i])

        _kernel = jit(probe)
    return _kernel


def _forced(name):
    """The value an environment variable forces, or ``None``."""
    value = os.environ.get(name)
    return value == "1" if value in ("0", "1") else None


def _driver_key(device):
    """Identify a device and its driver version, for stored results."""
    props = vk.vkGetPhysicalDeviceProperties(device.info.handle)
    return (
        f"{_VERSION} {props.vendorID:x}:{props.deviceID:x} "
        f"driver {props.driverVersion:x} {props.deviceName}"
    )


def _load():
    """Stored probe results by driver key; empty if there are none."""
    try:
        with open(os.path.join(libclc.cache_directory(), _RESULTS)) as fh:
            results = json.load(fh)
    except (OSError, ValueError):
        return {}
    return results if isinstance(results, dict) else {}


def _store(key, result):
    """Add one result to the stored ones; failures are ignored."""
    directory = libclc.cache_directory()
    try:
        os.makedirs(directory, exist_ok=True)
        results = _load()
        results[key] = result
        # Written under another name first: another process may be reading.
        with tempfile.NamedTemporaryFile("w", dir=directory, delete=False) as fh:
            json.dump(results, fh, indent=1, sort_keys=True)
        os.replace(fh.name, os.path.join(directory, _RESULTS))
    except OSError:
        pass


def _measure(device):
    """Run the probe on a device.

    Returns
    -------
    dict
        ``soft_fma`` and ``soft_rounding``.
    """
    out = np.zeros((3, _LANES))
    with warnings.catch_warnings():
        # The probe computes with float64 on purpose.
        warnings.simplefilter("ignore")
        _probe_kernel().forall(_LANES, device=device)(_INPUT, out)
    exact = _INPUT[0] * _INPUT[1] + _INPUT[2]  # rounded, as on the CPU
    return {
        "soft_fma": bool((out[0] == exact).any()),
        "soft_rounding": bool(
            (out[1] != np.trunc(_INPUT[3])).any()
            or (out[2] != np.round(_INPUT[4])).any()
        ),
    }


def workarounds(device):
    """Which ``float64`` workarounds kernels for a device need.

    Parameters
    ----------
    device : numba_vulkan.runtime.Device
        The device, which must not yet use any of them.

    Returns
    -------
    dict
        ``soft_fma`` and ``soft_rounding``, as for
        `numba_vulkan.narrowing.Mode`.
    """
    forced = {
        "soft_fma": _forced(FMA_ENV_VAR),
        "soft_rounding": _forced(ROUNDING_ENV_VAR),
    }
    if None not in forced.values() or not device.info.float64:
        return {name: bool(value) for name, value in forced.items()}
    measured = None
    if kernelcache.enabled():
        key = _driver_key(device)
        measured = _load().get(key)
    if not isinstance(measured, dict) or set(measured) != set(forced):
        measured = _measure(device)
        if kernelcache.enabled():
            _store(key, measured)
    return {
        name: bool(measured[name] if value is None else value)
        for name, value in forced.items()
    }
