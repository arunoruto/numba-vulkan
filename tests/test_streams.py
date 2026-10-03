"""Streams: copies and launches enqueued without waiting, in order per stream."""

import gc

import numpy as np
import pytest

import numba_vulkan as nv

f32 = np.float32


@nv.jit
def affine(x, out, scale):
    i = nv.global_id(0)
    if i < x.shape[0]:
        out[i] = x[i] * scale + f32(1)


@nv.jit
def checked(x):
    i = nv.global_id(0)
    if i < x.shape[0] and x[i] < 0:
        raise ValueError("negative")


def test_a_pipeline_of_chunks(device):
    """Copy in, compute and copy out chunk by chunk on rotating streams."""
    chunks, size = 6, 5000
    host = nv.pinned_array(chunks * size, np.float32, device=device)
    host[:] = np.arange(host.size)
    out = nv.pinned_array(host.size, np.float32, device=device)
    plain = np.zeros(host.size, np.float32)  # not pinned: staged
    streams = [nv.stream(device) for _ in range(3)]
    buffers = [
        (nv.device_array(size, f32, device), nv.device_array(size, f32, device))
        for _ in streams
    ]
    for c in range(chunks):
        stream = streams[c % 3]
        dx, dy = buffers[c % 3]
        part = slice(c * size, (c + 1) * size)
        dx.copy_to_device(host[part], stream=stream)
        affine.forall(size, stream=stream)(dx, dy, f32(2))
        dy.copy_to_host(out[part], stream=stream)
        dy.copy_to_host(plain[part], stream=stream)
    for stream in streams:
        stream.synchronize()
        assert stream.query()
    np.testing.assert_array_equal(out, host * 2 + 1)
    np.testing.assert_array_equal(plain, host * 2 + 1)


def test_streams_wait_for_each_other(device):
    first, second = nv.stream(device), nv.stream(device)
    x = nv.to_device(np.arange(1 << 16, dtype=f32), device, stream=first)
    y = nv.device_array_like(x)
    z = nv.device_array_like(x)
    affine[(1 << 16) // 64, 64, first](x, y, f32(3))
    # z depends on y, written by the other stream.
    affine[(1 << 16) // 64, 64, second](y, z, f32(1))
    result = z.copy_to_host(stream=second)
    second.synchronize()
    np.testing.assert_array_equal(result, np.arange(1 << 16) * 3 + 2)


def test_the_default_path_waits_for_streams(device):
    stream = nv.stream(device)
    x = nv.to_device(np.ones(4096, f32), device, stream=stream)
    y = nv.device_array_like(x)
    affine.forall(4096, stream=stream)(x, y, f32(5))
    # Without a stream: waits for the stream's work on y.
    np.testing.assert_array_equal(y.copy_to_host(), np.full(4096, 6))
    affine.forall(4096, device=device)(y, x, f32(2))  # default path
    result = x.copy_to_host(stream=stream)  # stream after default work
    stream.synchronize()
    np.testing.assert_array_equal(result, np.full(4096, 13))
    nv.synchronize(device)


def test_errors_are_raised_by_synchronize(device):
    stream = nv.stream(device)
    x = nv.to_device(np.array([1, -1], dtype=f32), device, stream=stream)
    with pytest.raises(ValueError, match="negative"):
        checked.forall(2, stream=stream)(x)  # raises here with NUMBA_VULKAN_SYNC=1
        stream.synchronize()
    stream.synchronize()  # reported once


def test_auto_synchronize_and_dropped_arrays(device):
    stream = nv.stream(device)
    with stream.auto_synchronize():
        x = nv.to_device(np.arange(1000, dtype=f32), device, stream=stream)
        y = nv.device_array_like(x)
        affine.forall(1000, stream=stream)(x, y, f32(1))
        del x  # its buffer waits for the stream before it is reused
        gc.collect()
        nv.device_array(1000, f32, device).copy_to_device(np.zeros(1000))
        out = y.copy_to_host(stream=stream)
    np.testing.assert_array_equal(out, np.arange(1000) + 1)


def test_pinned_arrays_and_their_views(device):
    pinned = nv.pinned_array((10, 100), np.int32, device=device)
    pinned[:] = np.arange(1000).reshape(10, 100)
    row = pinned[3]  # a view keeps the memory
    del pinned
    gc.collect()
    stream = nv.stream(device)
    d = nv.to_device(row, device, stream=stream)
    stream.synchronize()
    np.testing.assert_array_equal(d.copy_to_host(), np.arange(300, 400))


def test_what_streams_reject(device):
    stream = nv.stream(device)
    with pytest.raises(TypeError, match="device arrays only"):
        affine.forall(4, stream=stream)(np.ones(4, f32), np.zeros(4, f32), f32(1))
    d = nv.to_device(np.ones((4, 4), f32), device)
    with pytest.raises(ValueError, match="contiguous"):
        d[:, 1].copy_to_host(stream=stream)
    other = next((i for i in range(len(nv.list_devices())) if i != device), None)
    if other is not None:
        with pytest.raises(ValueError):
            nv.to_device(np.ones(4, f32), other).copy_to_host(stream=stream)


def test_dropped_streams_finish_their_work(device):
    x = nv.to_device(np.zeros(1000, f32), device)
    out = nv.pinned_array(1000, np.float32, device=device)
    stream = nv.stream(device)
    affine.forall(1000, stream=stream)(x, x, f32(1))
    x.copy_to_host(out, stream=stream)
    del stream
    gc.collect()
    nv.synchronize(device)  # waits for the dropped stream
    np.testing.assert_array_equal(out, np.ones(1000))
