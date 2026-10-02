"""Run one float64 math kernel with the installed package.

Used to check a freshly built wheel: it must find the bundled libclc, and a
Vulkan device (lavapipe, Mesa's CPU implementation, is enough) must run it.
"""

import math

import numpy as np

import numba_vulkan as nv
from numba_vulkan import libclc

path = libclc.find_bitcode()
assert path is not None and "site-packages" in path, f"libclc not bundled: {path}"
print("libclc:", libclc.version(), "at", path)
assert libclc.version() is not None, "no libclc version recorded"
print("devices:", nv.list_devices())


@nv.jit
def kernel(x, out):
    i = nv.global_id(0)
    if i < x.shape[0]:
        out[i] = math.sin(x[i]) + math.lgamma(x[i])


x = np.linspace(0.5, 3, 64)
out = np.zeros_like(x)
kernel.forall(64)(x, out)
want = np.sin(x) + np.vectorize(math.lgamma)(x)
error = np.abs(out - want).max()
assert error < 1e-12, error
print("ok, max error", error)
