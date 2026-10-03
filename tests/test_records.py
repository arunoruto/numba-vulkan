"""Structured (record) arrays in kernels."""

import numpy as np
import pytest
from numba import types
from numba.np import numpy_support

import numba_vulkan as nv
from numba_vulkan.errors import VulkanUnsupportedError

PACKED = np.dtype([("flag", "?"), ("n", "i2"), ("x", "f4"), ("big", "f8"), ("u", "u1")])
ALIGNED = np.dtype([("x", "f4"), ("n", "i4"), ("big", "f8")], align=True)


def _data(dtype, count=13):
    rng = np.random.default_rng(0)
    out = np.zeros(count, dtype)
    for name in dtype.names:
        kind = dtype.fields[name][0]
        if kind == np.bool_:
            out[name] = rng.random(count) < 0.5
        elif kind.kind in "iu":
            info = np.iinfo(kind)
            out[name] = rng.integers(max(info.min, -1000), min(info.max, 1000), count)
        else:
            out[name] = rng.random(count) * 100
    return out


@nv.jit
def update(src, dst):
    i = nv.global_id(0)
    if i < src.shape[0]:
        r = src[i]
        d = dst[i]
        d.x = r.x * 2
        d["n"] = r.n + 1
        d.big = r["big"] - 1


@pytest.mark.parametrize("dtype", [PACKED, ALIGNED], ids=["packed", "aligned"])
def test_fields_by_attribute_and_name(run, dtype):
    src = _data(dtype)
    dst = np.zeros_like(src)
    run(update, src.size, src, dst)
    np.testing.assert_allclose(dst["x"], src["x"] * 2, rtol=1e-6)
    np.testing.assert_array_equal(dst["n"], src["n"] + 1)
    np.testing.assert_allclose(dst["big"], src["big"] - 1, rtol=1e-6)
    for name in set(dtype.names) - {"x", "n", "big"}:
        assert not dst[name].any()  # untouched neighbours stay zero


@pytest.mark.float64
def test_float64_fields_keep_their_precision(run):
    @nv.jit
    def tiny(src, dst):
        i = nv.global_id(0)
        if i < src.shape[0]:
            dst[i].big = src[i].big + 1e-12

    src = _data(PACKED)
    src["big"] = 1.0
    dst = np.zeros_like(src)
    run(tiny, src.size, src, dst)
    np.testing.assert_array_equal(dst["big"], 1.0 + 1e-12)


def test_whole_records_are_copied(run):
    @nv.jit
    def reverse(src, dst):
        i = nv.global_id(0)
        n = src.shape[0]
        if i < n:
            dst[n - 1 - i] = src[i]

    src = _data(PACKED)
    dst = np.zeros_like(src)
    run(reverse, src.size, src, dst)
    np.testing.assert_array_equal(dst, src[::-1])


def test_neighbouring_bytes_written_by_other_invocations(run):
    # Each invocation writes one byte; four share a 32-bit word.
    quad = np.dtype([("a", "u1"), ("b", "u1"), ("c", "u1"), ("d", "u1"), ("e", "u1")])

    @nv.jit
    def fill(out):
        i = nv.global_id(0)
        if i < out.shape[0] * 5:
            r = out[i // 5]
            k = i % 5
            if k == 0:
                r.a = i
            elif k == 1:
                r.b = i
            elif k == 2:
                r.c = i
            elif k == 3:
                r.d = i
            else:
                r.e = i

    out = np.zeros(40, quad)
    run(fill, out.size * 5, out)
    flat = out.view(np.uint8).reshape(-1)
    np.testing.assert_array_equal(flat, np.arange(200, dtype=np.uint8))


def test_iteration_and_device_functions(run):
    @nv.jit
    def weight(r):
        return r.x * r.n

    @nv.jit
    def total(src, out):
        if nv.global_id(0) == 0:
            s = np.float32(0)
            for r in src:
                s += weight(r)
            out[0] = s

    src = _data(ALIGNED)
    out = np.zeros(1, np.float32)
    run(total, 1, src, out)
    np.testing.assert_allclose(out[0], (src["x"] * src["n"]).sum(), rtol=1e-5)


def test_device_arrays_and_views_of_records(device):
    src = _data(PACKED)
    dsrc = nv.to_device(src, device)
    ddst = nv.to_device(np.zeros_like(src), device)
    update.forall(6, device=device)(dsrc[:12:2], ddst[1::2])  # six each
    want = np.zeros_like(src)
    want["x"][1::2] = src["x"][:12:2] * 2
    want["n"][1::2] = src["n"][:12:2] + 1
    want["big"][1::2] = src["big"][:12:2] - 1
    got = ddst.copy_to_host()
    for name in PACKED.names:
        np.testing.assert_allclose(got[name], want[name], rtol=1e-6)
    assert dsrc[3]["n"] == src[3]["n"]


def test_narrowed_records(run):
    kernel = nv.jit(update.py_func, narrow=True)
    src = _data(PACKED)
    dst = np.zeros_like(src)
    run(kernel, src.size, src, dst)
    np.testing.assert_allclose(dst["big"], src["big"] - 1, rtol=1e-6)
    compiled = list(kernel._launched.values())[-1]
    assert "float64" not in compiled.capabilities


def test_unsupported_fields_are_rejected():
    nested = np.dtype([("v", "f4", 3)])
    with pytest.raises(VulkanUnsupportedError, match="field 'v'"):
        update.compile((types.Array(numpy_support.from_dtype(nested), 1, "C"),) * 2, 1)


def test_unknown_fields_are_named(run):
    @nv.jit
    def wrong(src, dst):
        dst[0].x = src[0].missing

    with pytest.raises(Exception, match="missing"):
        run(wrong, 1, _data(ALIGNED), np.zeros(2, ALIGNED))
