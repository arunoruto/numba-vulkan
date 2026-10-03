"""How kernels receive scalars and array extents: push constants or buffers."""

import numpy as np
import pytest
from numba import types

import numba_vulkan as nv
from numba_vulkan import narrowing, runtime
from numba_vulkan.compiler import PUSH_CONSTANT_BYTES

f32 = np.float32


@nv.jit
def store_scalars(a, b, c, d, out):
    if nv.global_id(0) == 0:
        out[0] = a
        out[1] = b
        out[2] = c
        out[3] = d


@nv.jit
def add_scalar(value, a):
    i = nv.global_id(0)
    if i < a.shape[0]:
        a[i] += value


@nv.jit
def extents(a, b, out):
    if nv.global_id(0) == 0:
        out[0] = a.shape[0]
        out[1] = b.shape[0]
        out[2] = b.shape[1]


@nv.jit
def many_scalars(
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, out
):
    if nv.global_id(0) == 0:
        out[0] = (
            a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13
        ) + (a14 + a15 + a16)


@nv.jit
def ignores_scalar(unused, out):
    if nv.global_id(0) == 0:
        out[0] = 1  # a constant index does not need the extent


@nv.jit
def uses_all(a, n, flag, small, x, y):
    i = nv.global_id(0)
    if flag and i < x.shape[0] and i < y.shape[1]:
        y[0, i] = x[i] * a + n + small


def _compiled(kernel):
    """The kernel's most recent specialisation."""
    return list(kernel._launched.values())[-1]


@pytest.mark.parametrize(
    "dtype, values",
    [
        (np.int8, [-128, 127, -1, 5]),
        (np.uint8, [0, 255, 128, 7]),
        (np.int16, [-32768, 32767, -3, 9]),
        (np.int32, [-(2**31), 2**31 - 1, -7, 11]),
        (np.uint32, [0, 2**32 - 1, 2**31, 13]),
        (np.float32, [1.5, -0.0, 3.4e38, 1e-45]),
        (np.bool_, [True, False, True, True]),
    ],
)
def test_scalars_arrive_unchanged(device, dtype, values):
    out = np.zeros(4, dtype=dtype)
    args = [dtype(v) for v in values]
    store_scalars.forall(1, device=device)(*args, out)
    np.testing.assert_array_equal(out, np.array(values, dtype=dtype))
    assert _compiled(store_scalars).push_format


@pytest.mark.float64
def test_64_bit_scalars_arrive_unchanged(device):
    out = np.zeros(4)
    store_scalars.forall(1, device=device)(1e300, -2.5, 2.0**-1074, 7.0, out)
    np.testing.assert_array_equal(out, [1e300, -2.5, 2.0**-1074, 7.0])
    if not runtime.get_device(device).info.int64:
        return
    ints = np.zeros(4, dtype=np.int64)
    values = [-(2**40), 2**62, -1, 3]
    kernel = nv.jit(store_scalars.py_func, narrow=False)
    kernel.forall(1, device=device)(*[np.int64(v) for v in values], ints)
    np.testing.assert_array_equal(ints, values)
    assert _compiled(kernel).push_format.startswith("<qqqq")


def test_extents_arrive_unchanged(device):
    out = np.zeros(3, dtype=np.int32)
    extents.forall(1, device=device)(np.zeros(7, f32), np.zeros((5, 3), f32), out)
    np.testing.assert_array_equal(out, [7, 5, 3])
    extents.forall(1, device=device)(np.zeros(2, f32), np.zeros((9, 4), f32), out)
    np.testing.assert_array_equal(out, [2, 9, 4])


def test_changing_scalars_between_asynchronous_launches(device):
    a = nv.to_device(np.zeros(1000, f32), device=device)
    launch = add_scalar.forall(1000, device=device)
    for k in range(100):
        launch(f32(k), a)
    np.testing.assert_array_equal(a.copy_to_host(), np.full(1000, 4950, f32))
    # Repeated values reuse slots recorded for them.
    for k in [1, 2, 1, 2]:
        launch(f32(k), a)
    np.testing.assert_array_equal(a.copy_to_host(), np.full(1000, 4956, f32))


def test_changing_scalars_between_synchronous_launches(device):
    a = np.zeros(1000, f32)
    launch = add_scalar.forall(1000, device=device)
    for k in range(20):
        launch(f32(k), a)
    np.testing.assert_array_equal(a, np.full(1000, 190, f32))


def test_too_many_arguments_fall_back_to_buffers(device):
    values = [f32(k) for k in range(17)]
    out = np.zeros(1, f32)
    many_scalars.forall(1, device=device)(*[f32(v) for v in values], out)
    assert out[0] == sum(range(17))
    # 17 float32 values and an extent fit; 17 float64 values do not.
    assert _compiled(many_scalars).push_format
    out64 = np.zeros(1, np.float64)
    # A device without float64 computes the kernel in float32 instead.
    with pytest.warns((nv.VulkanPerformanceWarning, nv.VulkanPrecisionWarning)):
        many_scalars.forall(1, device=device)(*[float(v) for v in values], out64)
    kernel = _compiled(many_scalars)
    if kernel.mode.floats:
        assert kernel.push_format  # narrowed to 17 float32 values
    else:
        assert not kernel.push_format
        assert 8 * 17 > PUSH_CONSTANT_BYTES
    assert out64[0] == sum(range(17))
    # Asynchronously, with the scalars in buffers of their own.
    stored = nv.to_device(np.zeros(1), device=device)
    for k in range(3):
        values[0] = f32(k)
        many_scalars.forall(1, device=device)(*[float(v) for v in values], stored)
        assert stored.copy_to_host()[0] == sum(range(1, 17)) + k


def test_unused_scalar(device):
    out = np.zeros(1, dtype=np.int32)
    ignores_scalar.forall(1, device=device)(f32(3), out)
    assert out[0] == 1
    # Unused members leave no block behind.
    assert not _compiled(ignores_scalar).push_format


@pytest.mark.parametrize(
    "mode",
    [narrowing.Mode(), narrowing.Mode(floats=True, ints=True)],
    ids=["wide", "narrowed"],
)
@pytest.mark.filterwarnings("ignore::numba_vulkan.VulkanPrecisionWarning")
@pytest.mark.filterwarnings("ignore::numba_vulkan.VulkanPerformanceWarning")
def test_push_constant_layout_follows_narrowing(validate, mode):
    args = (
        types.float64,
        types.int64,
        types.boolean,
        types.int8,
        types.float32[::1],
        types.float32[:, ::1],
    )
    kernel = uses_all.compile(args, 1, mode)
    validate(kernel)
    wide = 0 if mode.ints else 1
    assert kernel.push_format == ("<dq" if wide else "<fi") + "iiiii"
    assert kernel.push_sources[:4] == (
        ("arg", 0),
        ("arg", 1),
        ("arg", 2),
        ("arg", 3),
    )
    assert kernel.unbound == {1, 2, 3, 4}
