"""numba.vectorize and numba.guvectorize on a Vulkan device.

uv run python examples/vectorize.py
"""

import math
import time

import numpy as np
from numba import guvectorize, vectorize

import numba_vulkan as nv


@vectorize(["float32(float32, float32)"], target="vulkan")
def gaussian(x, sigma):
    return math.exp(-0.5 * (x / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


@guvectorize(
    ["void(float32[:], float32[:], float32[:])"], "(n),(n)->()", target="vulkan"
)
def distance(a, b, out):
    total = np.float32(0)
    for i in range(a.shape[0]):
        d = a[i] - b[i]
        total += d * d
    out[0] = math.sqrt(total)


def main():
    x = np.linspace(-5, 5, 1 << 22, dtype=np.float32)
    sigma = np.float32(1.5)
    gaussian(x, sigma)  # compile, and fill the buffer pool
    start = time.perf_counter()
    y = gaussian(x, sigma)
    print(
        f"gaussian over {x.size:,} values: {(time.perf_counter() - start) * 1e3:.1f} ms"
    )
    want = np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    print(f"  max error {np.abs(y - want).max():.1e}")

    points = np.random.default_rng(0).random((100_000, 16), dtype=np.float32)
    centre = np.full(16, 0.5, dtype=np.float32)
    print("distances:", distance(points, centre)[:4])

    # Device arrays stay on the GPU between calls.
    dx = nv.to_device(x)
    dy = gaussian(dx, sigma)
    print("on the device:", dy, "->", dy.copy_to_host()[:3])


if __name__ == "__main__":
    main()
