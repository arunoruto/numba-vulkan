"""Indexing device arrays, and views of them in kernels."""

import gc

import numpy as np
import pytest
from numba import vectorize

import numba_vulkan as nv

f32 = np.float32

KEYS = [
    (1,),
    (-1, 2),
    (slice(1, 3),),
    (slice(None, None, -1),),
    (slice(None), slice(1, None, 2)),
    (Ellipsis, slice(None, None, -2)),
    (2, Ellipsis, 1),
    (None, slice(1, 3), None),
    (slice(3, 0, -2), 1, slice(None)),
    (slice(10, 20),),  # past the end: empty
]


@pytest.fixture
def pair(device):
    host = np.arange(4 * 5 * 6, dtype=f32).reshape(4, 5, 6)
    return host, nv.to_device(host, device)


@pytest.mark.parametrize("key", KEYS, ids=str)
def test_views_hold_what_numpy_selects(pair, key):
    host, array = pair
    view = array[key]
    assert view.shape == host[key].shape
    assert view.strides == host[key].strides
    np.testing.assert_array_equal(view.copy_to_host(), host[key])
    np.testing.assert_array_equal(np.asarray(view), host[key])


def test_elements(pair):
    host, array = pair
    assert array[1, 2, 3] == host[1, 2, 3]
    assert array[-1, -2, -3] == host[-1, -2, -3]
    assert array[1:3][1, 0, ::-1][2] == host[1:3][1, 0, ::-1][2]
    assert isinstance(array[0, 0, 0], np.float32)


@pytest.mark.parametrize("key", KEYS, ids=str)
def test_writes_go_where_numpy_puts_them(pair, key):
    host, array = pair
    want = host.copy()
    values = np.arange(want[key].size, dtype=f32).reshape(want[key].shape) + 1000
    want[key] = values
    array[key] = values
    np.testing.assert_array_equal(array.copy_to_host(), want)
    want[key] = -1  # broadcast scalars
    array[key] = -1
    np.testing.assert_array_equal(array.copy_to_host(), want)


def test_transpose_and_reshape(pair):
    host, array = pair
    np.testing.assert_array_equal(array.T.copy_to_host(), host.T)
    np.testing.assert_array_equal(
        array.transpose(1, 0, 2).copy_to_host(), host.transpose(1, 0, 2)
    )
    np.testing.assert_array_equal(
        array[1:3].reshape(2, 30).copy_to_host(), host[1:3].reshape(2, 30)
    )
    # Splitting an axis and adding unit axes never needs a copy.
    strided = array[:, ::2]
    np.testing.assert_array_equal(
        strided.reshape(4, 3, 1, 2, 3).copy_to_host(),
        host[:, ::2].reshape(4, 3, 1, 2, 3),
    )
    with pytest.raises(ValueError, match="without copying"):
        strided.reshape(-1)
    np.testing.assert_array_equal(strided.ravel().copy_to_host(), host[:, ::2].ravel())
    np.testing.assert_array_equal(strided.copy().copy_to_host(), host[:, ::2])


def test_bad_indices(pair):
    _, array = pair
    with pytest.raises(IndexError):
        array[4]
    with pytest.raises(IndexError):
        array[0, 0, 0, 0]
    with pytest.raises(TypeError, match="basic indexing"):
        array[[0, 1]]
    with pytest.raises(TypeError, match="basic indexing"):
        array[np.array([True, False, True, False])]


def test_views_keep_the_buffer_alive(device):
    view = nv.to_device(np.arange(10, dtype=f32), device)[2:5]
    gc.collect()
    nv.device_array(10, f32, device)  # may reuse a released buffer
    np.testing.assert_array_equal(view.copy_to_host(), [2, 3, 4])


def test_booleans_and_narrowed_types(device):
    flags = np.array([True, False, True, True, False])
    array = nv.to_device(flags, device)
    np.testing.assert_array_equal(array[::-2].copy_to_host(), flags[::-2])
    array[1::2] = True
    flags[1::2] = True
    np.testing.assert_array_equal(array.copy_to_host(), flags)
    wide = np.arange(12, dtype=np.int64).reshape(3, 4)
    array = nv.to_device(wide, device)  # 32-bit on the device by default
    np.testing.assert_array_equal(array[:, 1::2].copy_to_host(), wide[:, 1::2])
    assert array[2, 3] == 11


@nv.jit
def scale(a, factor):
    i, j = nv.global_id(0), nv.global_id(1)
    if i < a.shape[0] and j < a.shape[1]:
        a[i, j] *= factor


@nv.jit
def copy_into(src, dst):
    i, j = nv.global_id(0), nv.global_id(1)
    if i < dst.shape[0] and j < dst.shape[1]:
        dst[i, j] = src[i, j]


VIEWS = [
    (slice(None), slice(None)),
    (slice(2, 7), slice(None)),  # contiguous, starting later
    (slice(1, None, 2), slice(None, None, -3)),
    (slice(None, None, -1), slice(2, 6)),
]


@pytest.mark.parametrize("key", VIEWS, ids=str)
def test_kernels_write_through_views(device, key):
    host = np.arange(80, dtype=f32).reshape(8, 10)
    array = nv.to_device(host, device)
    view = array[key]
    scale.forall(view.shape, device=device)(view, f32(2))
    host[key] *= 2
    np.testing.assert_array_equal(array.copy_to_host(), host)


def test_kernels_read_transposed_views(device):
    host = np.arange(80, dtype=f32).reshape(8, 10)
    src = nv.to_device(host, device)
    dst = nv.device_array((10, 8), f32, device)
    copy_into.forall((10, 8), device=device)(src.T, dst)
    np.testing.assert_array_equal(dst.copy_to_host(), host.T)


def test_views_when_arguments_do_not_fit_push_constants(device):
    @nv.jit
    def weighted(a, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13):
        i = nv.global_id(0)
        if i < a.shape[0]:
            a[i] = a[i] * (w0 + w1 + w2 + w3 + w4 + w5 + w6 + w7) + (
                w8 + w9 + w10 + w11 + w12 + w13
            )

    host = np.arange(20, dtype=np.float64)
    array = nv.to_device(host, device)
    weights = [1.0] * 14  # float64 scalars, too many for push constants
    with pytest.warns((nv.VulkanPerformanceWarning, nv.VulkanPrecisionWarning)):
        weighted.forall(5, device=device)(array[15:4:-2], *weights)
    host[15:4:-2] = host[15:4:-2] * 8 + 6
    np.testing.assert_allclose(array.copy_to_host(), host)
    kernel = list(weighted._launched.values())[-1]
    assert not kernel.args_pushed or kernel.mode.floats


@vectorize(target="vulkan")
def mul_add(a, b):
    return a * b + 1


@vectorize(["float32(float32, float32)"], target="vulkan", identity=0)
def plus(a, b):
    return a + b


def test_ufuncs_on_views(device):
    host = np.arange(48, dtype=f32).reshape(6, 8)
    array = nv.to_device(host, device)
    out = nv.device_array((3, 4), f32, device)
    mul_add(array[::2, ::2], array[1::2, 1::2], out=out)
    np.testing.assert_array_equal(
        out.copy_to_host(), host[::2, ::2] * host[1::2, 1::2] + 1
    )
    target = nv.to_device(np.zeros((6, 8), f32), device)
    mul_add(array[::2, ::2], f32(2), out=target[1::2, ::2])
    want = np.zeros((6, 8), f32)
    want[1::2, ::2] = host[::2, ::2] * 2 + 1
    np.testing.assert_array_equal(target.copy_to_host(), want)
    assert plus.reduce(array[:, 3]) == host[:, 3].sum()
    assert plus.reduce(array[::-1, ::3], axis=None) == host[::-1, ::3].sum()


def test_arithmetic_stays_on_the_device(device):
    host = np.arange(1, 25, dtype=f32).reshape(4, 6)
    a = nv.to_device(host, device)
    b = a[:, ::-2]
    hb = host[:, ::-2]
    results = {
        "a+1": (b + 1, hb + 1),
        "2*a-a": (2 * b - b, 2 * hb - hb),
        "a/a": (b / a[:, :3], hb / host[:, :3]),
        "a//3": (b // 3, hb // 3),
        "a%5": (b % 5, hb % 5),
        "a**2": (b**2, hb**2),
        "-a": (-b, -hb),
        "abs": (abs(b - 10), abs(hb - 10)),
        "host+device": (host[:, :3] + b, host[:, :3] + hb),
        "maximum": (np.maximum(b, 7), np.maximum(hb, 7)),
    }
    for name, (got, want) in results.items():
        assert isinstance(got, nv.DeviceArray), name
        np.testing.assert_allclose(got.copy_to_host(), want, rtol=1e-6, err_msg=name)
    np.testing.assert_array_equal((b > 10).copy_to_host(), hb > 10)
    np.testing.assert_array_equal((b == a[:, ::-2]).copy_to_host(), True)


def test_reductions_on_the_device(device):
    host = np.arange(30, dtype=np.int32).reshape(5, 6) - 7
    a = nv.to_device(host, device)
    assert a.sum() == host.sum()
    assert a[1:, ::2].min() == host[1:, ::2].min()
    assert a.T[3].max() == host.T[3].max()
    assert np.add.reduce(a[:, 2]) == host[:, 2].sum()


def test_unsupported_ufunc_uses_raise(device):
    a = nv.to_device(np.arange(4, dtype=f32), device)
    with pytest.raises(TypeError):
        np.add.accumulate(a)
    with pytest.raises(TypeError):
        np.divmod(a, 2)  # two results
