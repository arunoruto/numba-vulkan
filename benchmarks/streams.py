"""Processing data in chunks: one chunk after another, or on streams.

Each chunk is copied to the device, transformed by a kernel and copied
back. Without streams every step waits for the one before; with three
streams the copies of one chunk overlap the kernels of others, on devices
with a separate copy engine.

    uv run python benchmarks/streams.py
"""

import argparse
import time
import warnings

import numpy as np

import numba_vulkan as nv

f32 = np.float32


@nv.jit
def work(x, out, rounds):
    i = nv.global_id(0)
    if i < x.shape[0]:
        v = x[i]
        for _ in range(rounds):
            v = v * f32(0.999) + f32(0.001)
        out[i] = v


def one_by_one(device, host, result, chunks, rounds):
    """Each chunk copied in, transformed and copied out before the next."""
    size = host.size // chunks
    for c in range(chunks):
        part = slice(c * size, (c + 1) * size)
        dx = nv.to_device(host[part], device)
        dout = nv.device_array_like(dx)
        work.forall(size, device=device)(dx, dout, rounds)
        dout.copy_to_host(result[part])


def streamed(device, host, result, chunks, rounds, count=3):
    """The chunks spread over `count` streams, everything enqueued at once."""
    size = host.size // chunks
    streams = [nv.stream(device) for _ in range(count)]
    buffers = [
        (nv.device_array(size, f32, device), nv.device_array(size, f32, device))
        for _ in streams
    ]
    for c in range(chunks):
        stream = streams[c % count]
        dx, dout = buffers[c % count]
        part = slice(c * size, (c + 1) * size)
        dx.copy_to_device(host[part], stream=stream)
        work.forall(size, stream=stream)(dx, dout, rounds)
        dout.copy_to_host(result[part], stream=stream)
    for stream in streams:
        stream.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=1 << 22)
    args = parser.parse_args()
    warnings.simplefilter("ignore", nv.VulkanPerformanceWarning)
    total = args.chunks * args.chunk_size
    for info in nv.list_devices():
        device = info.index
        host = nv.pinned_array(total, f32, device=device)
        host[:] = np.random.default_rng(0).random(total, dtype=f32)
        result = nv.pinned_array(total, f32, device=device)
        for rounds in (0, 50, 400):
            one_by_one(device, host[: args.chunk_size], result, 1, rounds)  # compile
            start = time.perf_counter()
            one_by_one(device, host, result, args.chunks, rounds)
            plain = time.perf_counter() - start
            want = result.copy()
            start = time.perf_counter()
            streamed(device, host, result, args.chunks, rounds)
            overlapped = time.perf_counter() - start
            same = "same results" if np.array_equal(result, want) else "DIFFERENT"
            print(
                f"{info.name[:30]:30s} {rounds:4d} rounds: one by one "
                f"{plain * 1e3:7.1f} ms, three streams {overlapped * 1e3:7.1f} ms "
                f"({plain / overlapped:.2f}x), {same}"
            )


if __name__ == "__main__":
    main()
