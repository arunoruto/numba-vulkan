"""Atomics, workgroup-shared memory, barriers and launch configuration."""

import numpy as np
import pytest
from numba.core import errors

import numba_vulkan as nv

f32 = np.float32
N = 1000


@nv.jit
def histogram(values, bins):
    i = nv.global_id(0)
    if i < values.shape[0]:
        nv.atomic.add(bins, values[i] % bins.shape[0], 1)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
def test_histogram_with_atomic_add(run, dtype):
    values = np.arange(N, dtype=np.int32) * 7
    bins = np.zeros(13, dtype=dtype)
    run(histogram, N, values, bins)
    np.testing.assert_array_equal(bins, np.bincount(values % 13, minlength=13))


def test_integer_operations_return_the_old_value(run):
    @nv.jit
    def ops(values, cells, olds):
        i = nv.global_id(0)
        if i < values.shape[0]:
            v = values[i]
            nv.atomic.sub(cells, 0, v)
            nv.atomic.max(cells, 1, v)
            nv.atomic.min(cells, 2, v)
            nv.atomic.and_(cells, 3, v | 0x100)
            nv.atomic.or_(cells, 4, v)
            nv.atomic.xor(cells, 5, v)
            olds[i] = nv.atomic.exch(cells, 6, v)

    values = np.arange(-50, 50, dtype=np.int32)
    cells = np.array([0, -1000, 1000, -1, 0, 0, 7777], dtype=np.int32)
    olds = np.zeros_like(values)
    run(ops, values.size, values, cells, olds)
    assert cells[0] == -values.sum()
    assert (cells[1], cells[2]) == (49, -50)
    assert cells[3] == np.bitwise_and.reduce(values | 0x100)
    assert cells[4] == np.bitwise_or.reduce(values)
    assert cells[5] == np.bitwise_xor.reduce(values)
    # exch: every value was written once, and every old value seen once
    assert sorted(np.append(olds, cells[6])) == sorted(np.append(values, 7777))


def test_unsigned_max_and_min_compare_unsigned(run):
    @nv.jit
    def extremes(values, cells):
        i = nv.global_id(0)
        if i < values.shape[0]:
            nv.atomic.max(cells, 0, values[i])
            nv.atomic.min(cells, 1, values[i])

    values = np.array([1, 2, 0xFFFFFFF0, 5], dtype=np.uint32)
    cells = np.array([0, 0xFFFFFFFF], dtype=np.uint32)
    run(extremes, 4, values, cells)
    assert list(cells) == [0xFFFFFFF0, 1]


def test_float_atomics(run):
    @nv.jit
    def accumulate(values, cells):
        i = nv.global_id(0)
        if i < values.shape[0]:
            nv.atomic.add(cells, 0, values[i])
            nv.atomic.sub(cells, 1, values[i])
            nv.atomic.max(cells, 2, values[i])
            nv.atomic.min(cells, 3, values[i])

    values = np.linspace(-3, 5, N, dtype=f32)
    cells = np.array([0, 0, -np.inf, np.inf], dtype=f32)
    run(accumulate, N, values, cells)
    np.testing.assert_allclose(cells[:2], [values.sum(), -values.sum()], rtol=1e-4)
    assert (cells[2], cells[3]) == (values.max(), values.min())


def test_compare_and_swap(run):
    @nv.jit
    def first_writer(owner, winners):
        i = nv.global_id(0)
        if i < winners.shape[0]:
            # every invocation tries to claim slot i % 4
            winners[i] = nv.atomic.cas(owner, i % 4, -1, i) == -1

    owner = np.full(4, -1, dtype=np.int32)
    winners = np.zeros(256, dtype=np.bool_)
    run(first_writer, 256, owner, winners)
    assert winners.sum() == 4  # exactly one per slot
    for slot in range(4):
        assert owner[slot] % 4 == slot and winners[owner[slot]]


def test_atomics_on_multidimensional_arrays_and_views(run):
    @nv.jit
    def scatter(values, grid):
        i = nv.global_id(0)
        if i < values.shape[0]:
            nv.atomic.add(grid, (i % 3, i % 5), values[i])

    values = np.ones(150, dtype=np.int32)
    grid = np.zeros((3, 5), dtype=np.int32)
    run(scatter, 150, values, grid)
    np.testing.assert_array_equal(grid, np.full((3, 5), 10))


def test_atomic_errors():
    @nv.jit
    def on_float64(cells):
        nv.atomic.add(cells, 0, 1.0)

    with pytest.raises(errors.NumbaError, match="float64 needs 64-bit compare"):
        on_float64.forall(1)(np.zeros(1))

    @nv.jit
    def bitwise_on_float(cells):
        nv.atomic.and_(cells, 0, f32(1))

    with pytest.raises(errors.TypingError):
        bitwise_on_float.forall(1)(np.zeros(1, dtype=f32))


def test_float64_atomics_work_when_narrowed(run):
    @nv.jit(narrow=True)
    def accumulate(values, total):
        i = nv.global_id(0)
        if i < values.shape[0]:
            nv.atomic.add(total, 0, values[i])

    values = np.linspace(0, 1, N)
    total = np.zeros(1)
    run(accumulate, N, values, total)
    np.testing.assert_allclose(total, values.sum(), rtol=1e-5)


# -- shared memory and barriers ---------------------------------------------------


@nv.jit
def block_sums(values, out):
    tile = nv.shared.array(64, np.float32)
    local, i = nv.local_id(0), nv.global_id(0)
    tile[local] = values[i] if i < values.shape[0] else f32(0)
    nv.barrier()
    step = 32
    while step > 0:
        if local < step:
            tile[local] += tile[local + step]
        nv.barrier()
        step //= 2
    if local == 0:
        out[nv.group_id(0)] = tile[0]


def test_reduction_in_shared_memory(run):
    values = np.random.default_rng(0).random(N).astype(f32)
    out = np.zeros(16, dtype=f32)
    run(block_sums, N, values, out)
    padded = np.pad(values, (0, 16 * 64 - N))
    np.testing.assert_allclose(out, padded.reshape(16, 64).sum(1), rtol=1e-5)


@nv.jit
def transpose(a, out):
    tile = nv.shared.array((16, 17), np.float32)  # padded against bank conflicts
    x, y = nv.local_id(0), nv.local_id(1)
    gx = nv.group_id(0) * nv.local_size(0)
    gy = nv.group_id(1) * nv.local_size(1)
    if gy + y < a.shape[0] and gx + x < a.shape[1]:
        tile[y, x] = a[gy + y, gx + x]
    nv.barrier()
    if gx + y < out.shape[0] and gy + x < out.shape[1]:
        out[gx + y, gy + x] = tile[x, y]


def test_tiled_transpose_with_launch_configuration(run, device):
    a = np.arange(40 * 70, dtype=f32).reshape(40, 70)
    out = np.zeros((70, 40), dtype=f32)
    groups = (-(-70 // 16), -(-40 // 16))
    try:
        transpose[groups, (16, 16), device](a, out)
    except nv.VulkanSupportError as exc:
        pytest.skip(str(exc))
    np.testing.assert_array_equal(out, a.T)
    compiled = list(transpose._kernels.values())[-1]
    assert compiled.local_size == (16, 16, 1)
    assert compiled.shared_bytes == 16 * 17 * 4


def test_ids_and_sizes(device):
    @nv.jit
    def ids(out):
        i = nv.global_id(0)
        out[i, 0] = nv.local_id(0)
        out[i, 1] = nv.group_id(0)
        out[i, 2] = nv.local_size(0)
        out[i, 3] = nv.num_groups(0)

    out = np.zeros((96, 4), dtype=np.int32)
    ids[3, 32, device](out)
    want = np.stack([np.arange(96) % 32, np.arange(96) // 32,
                     np.full(96, 32), np.full(96, 3)], axis=1)  # fmt: skip
    np.testing.assert_array_equal(out, want)
    ids.forall(96, device=device, local_size=48)(out)
    assert out[:, 2].tolist() == [48] * 96 and out[:, 3].tolist() == [2] * 96


@nv.jit
def two_arrays_and_shared_atomics(values, out):
    counts = nv.shared.array(4, np.int32)
    other = nv.shared.array(4, np.int32)
    local = nv.local_id(0)
    if local < 4:
        counts[local] = 0
        other[local] = 100
    nv.barrier()
    i = nv.global_id(0)
    if i < values.shape[0]:
        nv.atomic.add(counts, values[i] % 4, 1)
    nv.barrier()
    if local < 4:
        nv.atomic.add(out, local, counts[local] + other[local] - 100)


def test_shared_atomics_and_several_shared_arrays(run):
    values = np.arange(N, dtype=np.int32)
    out = np.zeros(4, dtype=np.int32)
    run(two_arrays_and_shared_atomics, N, values, out)
    np.testing.assert_array_equal(out, np.bincount(values % 4))


def test_shared_array_must_have_a_constant_shape():
    @nv.jit
    def dynamic(n, out):
        tile = nv.shared.array(n, np.float32)
        out[0] = tile[0]

    with pytest.raises(errors.TypingError):
        dynamic.forall(1)(4, np.zeros(1, dtype=f32))


def test_launch_limits(device):
    @nv.jit
    def nothing(out):
        out[0] = 1

    out = np.zeros(1, dtype=np.int32)
    info = nv.list_devices()[device]
    with pytest.raises(nv.VulkanSupportError, match="has workgroups of"):
        nothing[1, info.max_local_invocations + 1, device](out)

    @nv.jit
    def huge(out):
        big = nv.shared.array(1 << 20, np.float32)
        big[0] = 1
        out[0] = big[0]

    with pytest.raises(nv.VulkanSupportError, match="bytes of shared memory"):
        huge.forall(1, device=device)(out)
    with pytest.raises(TypeError, match="kernel\\[groups, local_size\\]"):
        nothing[1](out)


def test_64_bit_integer_atomics(run):
    values = np.arange(N, dtype=np.int32)
    # integers are 32-bit in kernels by default, so int64 counters work
    bins = np.zeros(13, dtype=np.int64)
    run(nv.jit(narrow="ints")(histogram.py_func), N, values, bins)
    np.testing.assert_array_equal(bins, np.bincount(values % 13, minlength=13))
    # with 64-bit integers, LLVM's SPIR-V backend cannot emit them
    exact = nv.jit(narrow=False)(histogram.py_func)
    with pytest.raises(errors.NumbaError, match="does not offer 64-bit integer"):
        exact.forall(N)(values, np.zeros(13, dtype=np.int64))


def test_float_add_uses_the_native_instruction_where_available(device):
    import dataclasses

    from numba_vulkan import runtime

    @nv.jit
    def total(values, out):
        i = nv.global_id(0)
        if i < values.shape[0]:
            nv.atomic.add(out, 0, values[i])

    values = np.ones(N, dtype=f32)
    info = nv.list_devices()[device]
    for native in (False, True):
        if native and not info.float32_atomic_add:
            continue
        target = runtime.Device(dataclasses.replace(info, float32_atomic_add=native))
        out = np.zeros(1, dtype=f32)
        total.forall(N, device=target)(values, out)
        assert out[0] == N
        compiled = next(
            k for k in total._kernels.values() if k.mode.float_atomics == native
        )
        assert ("float32_atomic_add" in compiled.capabilities) == native


def test_narrow_ints_only(run):
    @nv.jit(narrow="ints")
    def scale(values, out):
        i = nv.global_id(0)
        if i < values.shape[0]:
            out[i] = values[i] * 0.1

    values = np.arange(10, dtype=np.int64)
    out = np.zeros(10)
    run(scale, 10, values, out)
    np.testing.assert_allclose(out, values * 0.1, rtol=1e-15)  # floats stay float64
    compiled = list(scale._kernels.values())[-1]
    assert compiled.mode.ints and not compiled.mode.floats
    with pytest.raises(ValueError, match="narrow must be"):
        nv.jit(narrow="yes")(scale.py_func)


def test_grids_beyond_the_device_limit_are_split(device, monkeypatch):
    @nv.jit
    def mark(out):
        i = nv.global_id(0)
        if i < out.shape[0]:
            out[i] = i + 1

    out = np.zeros(1000, dtype=np.int32)
    dev = nv.get_device(device)
    monkeypatch.setattr(dev.info, "max_groups", (3, 65535, 65535))
    mark.forall(1000, device=device)(out)  # 16 workgroups, dispatched 3 at a time
    np.testing.assert_array_equal(out, np.arange(1, 1001))
