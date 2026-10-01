"""Run one kernel on every Vulkan device and compare against NumPy."""

import math

import numpy as np

import numba_vulkan as nv


@nv.jit
def kernel(a, x, y):
    i = nv.global_id(0)
    if i < x.shape[0]:
        acc = np.float32(0.0)
        for k in range(4):
            acc += math.sin(x[i]) * a
        n = 0
        while n < 10 and acc > 0.5:
            acc = acc / 2
            n += 1
        y[i] = acc + y[i] + n


def reference(a, x, y):
    out = y.copy()
    for i in range(x.shape[0]):
        acc = 0.0
        for k in range(4):
            acc += np.sin(x[i]) * a
        n = 0
        while n < 10 and acc > 0.5:
            acc = acc / 2
            n += 1
        out[i] = acc + y[i] + n
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, 1000).astype(np.float32)
    y0 = rng.uniform(-1, 1, 1000).astype(np.float32)
    a = np.float32(2.5)
    expected = reference(a, x, y0)
    for info in nv.list_devices():
        y = y0.copy()
        try:
            kernel.forall(x.size, device=info.index)(a, x, y)
        except nv.VulkanSupportError as exc:
            print(f"{info.name:45s} SKIPPED: {exc}")
            continue
        err = np.abs(y - expected).max()
        print(f"{info.name:45s} max abs error vs NumPy: {err:.2e}")
