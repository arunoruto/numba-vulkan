"""Advanced indexing of device arrays: index arrays and masks, as in NumPy."""

import numpy as np
import pytest

import numba_vulkan as nv

HOST = np.arange(48, dtype=np.float32).reshape(6, 8) - 20

KEYS = {
    "list": [1, 3],
    "array and reversed slice": (np.array([5, 0, 5]), slice(None, None, -2)),
    "slice and list": (slice(1, 4), [0, 7]),
    "mask": HOST > 3,
    "pairs": ([0, 2], [1, 3]),
    "ellipsis": (Ellipsis, [2]),
    "broadcast arrays": (np.array([[1], [2]]), np.array([0, 4])),
    "row mask": np.array([True, False, True, False, False, True]),
    "negative": [-1, -6],
}


@pytest.mark.parametrize("name", KEYS)
def test_reading_selects_as_numpy_does(device, name):
    d = nv.to_device(HOST, device)
    got = d[KEYS[name]]
    assert isinstance(got, nv.DeviceArray)
    np.testing.assert_array_equal(got.copy_to_host(), HOST[KEYS[name]])


@pytest.mark.parametrize("name", KEYS)
def test_writing_selects_as_numpy_does(device, name):
    d = nv.to_device(HOST, device)
    want = HOST.copy()
    values = -np.arange(want[KEYS[name]].size, dtype=np.float32)
    values = values.reshape(want[KEYS[name]].shape)
    if name not in ("pairs", "negative", "broadcast arrays"):
        values = np.float32(-99.0)  # repeated indices would make arrays ambiguous
    d[KEYS[name]] = values
    want[KEYS[name]] = values
    np.testing.assert_array_equal(d.copy_to_host(), want)


def test_masks_and_indices_on_the_device(device):
    d = nv.to_device(HOST, device)
    np.testing.assert_array_equal(d[d > 3].copy_to_host(), HOST[HOST > 3])
    index = nv.to_device(np.array([2, 0], dtype=np.int32), device)
    np.testing.assert_array_equal(d[index].copy_to_host(), HOST[[2, 0]])
    d[d < 0] = 0
    np.testing.assert_array_equal(d.copy_to_host(), np.maximum(HOST, 0))


def test_views_and_scalars(device):
    d = nv.to_device(HOST, device)
    view = d[1:5, ::-2]
    np.testing.assert_array_equal(view[[0, 3]].copy_to_host(), HOST[1:5, ::-2][[0, 3]])
    assert d[np.array(2), np.array(3)] == HOST[2, 3]  # 0-d indices: an element
    assert d[[0, 1], 2].shape == (2,)
    assert d[[], 2].shape == (0,)


@pytest.mark.parametrize(
    "dtype", [np.int64, np.float64, np.bool_, np.int16, np.int8, np.uint32]
)
def test_element_types(device, dtype):
    """Sizes that are not a multiple of 4 bytes go through the host."""
    host = (np.arange(12) % 5).astype(dtype)
    d = nv.to_device(host, device)
    np.testing.assert_array_equal(d[[11, 0, 3]].copy_to_host(), host[[11, 0, 3]])
    d[[1, 2]] = host[[4, 4]]
    host[[1, 2]] = host[[4, 4]]
    np.testing.assert_array_equal(d.copy_to_host(), host)


def test_bad_advanced_indices(device):
    d = nv.to_device(HOST, device)
    with pytest.raises(IndexError):
        d[[0, 6]]
    with pytest.raises(IndexError):
        d[np.ones(5, bool)]
    with pytest.raises(IndexError):
        d[[0, 1], [0, 1, 2]]  # cannot be broadcast
