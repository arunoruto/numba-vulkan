"""Arrays that stay on the device, and the buffer pool behind them."""

import gc

import numpy as np
import pytest

import numba_vulkan as nv
from numba_vulkan import runtime

f32 = np.float32


@nv.jit
def add_one(a):
    i = nv.global_id(0)
    if i < a.shape[0]:
        a[i] += f32(1)


@nv.jit
def scale_rows(a, factors, out):
    i, j = nv.global_id(0), nv.global_id(1)
    if i < a.shape[0] and j < a.shape[1]:
        out[i, j] = a[i, j] * factors[j]


@nv.jit
def mark_positive(a, flags):
    i = nv.global_id(0)
    if i < a.shape[0]:
        flags[i] = a[i] > 0 and not flags[i]


@pytest.mark.parametrize(
    "values",
    [
        np.linspace(-1, 1, 37, dtype=f32),
        np.arange(24, dtype=np.int32).reshape(2, 3, 4),
        np.array([True, False, True, True]),
        np.zeros((0, 3), dtype=f32),
        np.array(2.5, dtype=f32),
    ],
    ids=["float32", "int32-3d", "bool", "empty", "0d"],
)
def test_round_trip(device, values):
    array = nv.to_device(values, device)
    assert (array.shape, array.dtype, array.ndim) == (
        values.shape,
        values.dtype,
        values.ndim,
    )
    assert array.size == values.size and array.nbytes == values.nbytes
    back = array.copy_to_host()
    assert back.dtype == values.dtype
    np.testing.assert_array_equal(back, values)
    np.testing.assert_array_equal(np.asarray(array), values)


def test_kernel_works_in_place_on_the_device(run, device):
    array = nv.to_device(np.zeros(100, dtype=f32), device)
    for _ in range(3):
        run(add_one, 100, array)
    # nothing was copied back in between; the data stayed on the device
    np.testing.assert_array_equal(array.copy_to_host(), np.full(100, 3, dtype=f32))


def test_host_and_device_arrays_mix(run, device):
    a = np.arange(12, dtype=f32).reshape(3, 4)
    factors = np.array([1, 2, 3, 4], dtype=f32)
    out = nv.device_array_like(a, device)
    run(scale_rows, (3, 4), nv.to_device(a, device), factors, out)
    np.testing.assert_array_equal(out.copy_to_host(), a * factors)
    host_out = np.zeros_like(a)
    run(scale_rows, (3, 4), a, nv.to_device(factors, device), host_out)
    np.testing.assert_array_equal(host_out, a * factors)


def test_boolean_device_array(run, device):
    a = np.array([1, -1, 2, 0, 3], dtype=f32)
    flags = nv.to_device(np.array([False, False, True, False, False]), device)
    run(mark_positive, 5, nv.to_device(a, device), flags)
    np.testing.assert_array_equal(
        flags.copy_to_host(), [True, False, False, False, True]
    )


def test_copies_check_shapes_and_types(device):
    array = nv.device_array((2, 3), f32, device)
    assert array.copy_to_device(np.ones((2, 3))) is array  # converted to float32
    out = np.zeros((2, 3), dtype=f32)
    assert array.copy_to_host(out) is out
    np.testing.assert_array_equal(out, 1)
    with pytest.raises(ValueError, match="cannot copy shape"):
        array.copy_to_device(np.ones(6, dtype=f32))
    with pytest.raises(ValueError, match="out must be"):
        array.copy_to_host(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="out must be"):
        array.copy_to_host(np.zeros((3, 2), dtype=f32))
    assert len(array) == 2 and "shape=(2, 3)" in repr(array)


def test_array_of_another_device_is_rejected():
    devices = nv.list_devices()
    if len(devices) < 2:
        pytest.skip("needs two Vulkan devices")
    array = nv.to_device(np.zeros(8, dtype=f32), devices[0].index)
    with pytest.raises(ValueError, match="the array is on"):
        add_one.forall(8, device=devices[1].index)(array)


def test_released_buffers_are_reused_and_can_be_trimmed(device):
    dev = nv.get_device(device)
    dev.trim()
    assert dev._pooled == 0
    array = nv.device_array(100_000, f32, device)
    handle, size = array._buffer.handle, array._buffer.nbytes
    assert size >= 400_000
    del array
    gc.collect()
    assert dev._pooled == size
    again = nv.device_array(99_990, f32, device)  # rounds to the same size
    assert again._buffer.handle is handle and dev._pooled == 0
    del again
    gc.collect()
    dev.trim()
    assert dev._pooled == 0 and not any(dev._free.values())


def test_pool_stays_below_its_limit(device, monkeypatch):
    dev = nv.get_device(device)
    dev.trim()
    monkeypatch.setattr(dev, "pool_limit", 1 << 20)
    arrays = [nv.device_array(1 << 18, np.uint8, device) for _ in range(8)]
    del arrays
    gc.collect()
    assert dev._pooled <= 1 << 20
    dev.trim()


def test_host_arrays_reuse_pooled_buffers(run, device):
    dev = nv.get_device(device)
    a = np.zeros(5000, dtype=f32)
    run(add_one, 5000, a)
    pooled = dev._pooled
    for _ in range(3):
        run(add_one, 5000, a)
    assert dev._pooled == pooled  # no new buffers were needed
    np.testing.assert_array_equal(a, 4)


def test_pool_sizes_round_up():
    assert runtime._pool_size(1) == 256
    assert runtime._pool_size(257) == 512
    assert runtime._pool_size(1 << 16) == 1 << 16
    assert runtime._pool_size((1 << 16) + 1) == 2 << 16
    assert runtime._pool_size(1_000_000) == 16 << 16


@nv.jit
def scaled(factor, a, out):
    i = nv.global_id(0)
    if i < a.shape[0]:
        out[i] = factor * a[i]


def test_repeated_launches_follow_changes_of_buffers_kernels_and_grids(run, device):
    # Launches reuse the descriptor set and the recorded commands while
    # nothing changes; every kind of change must be noticed.
    dev = nv.get_device(device)
    a = nv.to_device(np.arange(100, dtype=f32), device)
    b = nv.to_device(np.arange(100, dtype=f32) + 1000, device)
    out1, out2 = nv.device_array_like(a), nv.device_array_like(a)

    run(scaled, 100, f32(2), a, out1)
    kernel = next(k for k in scaled._kernels.values() if k.mode == dev.mode)
    state = dev._pipelines[id(kernel)]
    bound = state.bound
    run(scaled, 100, f32(3), a, out1)  # same buffers, other scalar value
    assert state.bound == bound
    np.testing.assert_array_equal(out1.copy_to_host(), 3 * np.arange(100))

    run(scaled, 100, f32(2), b, out2)  # other buffers
    assert state.bound != bound
    np.testing.assert_array_equal(out2.copy_to_host(), 2 * (np.arange(100) + 1000))
    np.testing.assert_array_equal(out1.copy_to_host(), 3 * np.arange(100))

    out2.copy_to_device(np.zeros(100, dtype=f32))
    run(scaled, 50, f32(5), a, out2)  # smaller grid: fewer workgroups
    got = out2.copy_to_host()
    np.testing.assert_array_equal(got[:64], 5 * np.arange(64))
    assert (got[64:] == 0).all()

    run(add_one, 100, out1)  # another kernel in between
    run(scaled, 100, f32(5), a, out2)
    np.testing.assert_array_equal(out1.copy_to_host(), 3 * np.arange(100) + 1)
    np.testing.assert_array_equal(out2.copy_to_host(), 5 * np.arange(100))
