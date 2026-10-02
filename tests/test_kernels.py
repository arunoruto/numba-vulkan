import math

import numpy as np
import pytest
from numba import types
from numba.core import errors

import numba_vulkan as nv
from numba_vulkan import libclc


@nv.jit
def saxpy(a, x, y):
    i = nv.global_id(0)
    if i < x.shape[0]:
        y[i] = a * x[i] + y[i]


@nv.jit
def int_ops(x, y, out):
    i = nv.global_id(0)
    if i < len(x):
        a = x[i]
        b = y[i]
        out[i, 0] = a // b
        out[i, 1] = a % b
        out[i, 2] = (a << 2) ^ (b & 7)
        out[i, 3] = max(a, b) - min(a, b) + abs(a)


@nv.jit
def float_ops(x, out):
    i = nv.global_id(0)
    if i < x.size:
        v = x[i]
        out[i, 0] = math.sqrt(abs(v)) + math.floor(v)
        out[i, 1] = v % np.float32(1.5)
        out[i, 2] = math.exp(v) / (np.float32(1.0) + math.cos(v) ** np.float32(2.0))
        out[i, 3] = math.atan2(v, np.float32(2.0)) if v > 0 else -v


@nv.jit
def mandelbrot(xmin, ymin, dx, dy, maxiter, out):
    col = nv.global_id(0)
    row = nv.global_id(1)
    if row < out.shape[0] and col < out.shape[1]:
        cr = xmin + col * dx
        ci = ymin + row * dy
        zr = np.float32(0.0)
        zi = np.float32(0.0)
        n = 0
        while n < maxiter and zr * zr + zi * zi < 4:
            t = zr * zr - zi * zi + cr
            zi = 2 * zr * zi + ci
            zr = t
            n += 1
        out[row, col] = n


@nv.jit
def stencil(src, dst):
    j = nv.global_id(0)
    i = nv.global_id(1)
    if 0 < i < src.shape[0] - 1 and 0 < j < src.shape[1] - 1:
        dst[i, j] = (src[i - 1, j] + src[i + 1, j] + src[i, j - 1] + src[i, j + 1]) / 4


@nv.jit
def strided_sum(x, out, flag):
    i = nv.global_id(0)
    if i < out.shape[0]:
        total = 0.0
        for k in range(x.shape[0] - 1, -1, -3):
            total += x[k] * (i + 1)
        if not flag:
            total = -total
        out[i] = total


def test_saxpy(run):
    rng = np.random.default_rng(0)
    x = rng.random(1000, dtype=np.float32)
    y = rng.random(1000, dtype=np.float32)
    expected = np.float32(2.5) * x + y
    run(saxpy, x.size, np.float32(2.5), x, y)
    np.testing.assert_allclose(y, expected, rtol=1e-6)


def test_readonly_inputs_stay_untouched(run):
    x = np.arange(100, dtype=np.float32)
    x.flags.writeable = False
    y = np.ones(100, dtype=np.float32)
    run(saxpy, 100, np.float32(2.0), x, y)
    np.testing.assert_array_equal(y, 2 * np.arange(100, dtype=np.float32) + 1)


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_int_ops_follow_python_semantics(run, dtype):
    rng = np.random.default_rng(1)
    x = rng.integers(-50, 50, 300).astype(dtype)
    y = rng.integers(1, 9, 300).astype(dtype) * rng.choice([-1, 1], 300).astype(dtype)
    out = np.zeros((300, 4), dtype=dtype)
    run(int_ops, 300, x, y, out)
    np.testing.assert_array_equal(out[:, 0], x // y)
    np.testing.assert_array_equal(out[:, 1], x % y)
    np.testing.assert_array_equal(out[:, 2], (x << 2) ^ (y & 7))
    np.testing.assert_array_equal(
        out[:, 3], np.maximum(x, y) - np.minimum(x, y) + abs(x)
    )


def test_float_ops(run):
    x = np.linspace(-3, 3, 257, dtype=np.float32)
    out = np.zeros((257, 4), dtype=np.float32)
    run(float_ops, x.size, x, out)
    one, two = np.float32(1.0), np.float32(2.0)
    np.testing.assert_allclose(out[:, 0], np.sqrt(abs(x)) + np.floor(x), rtol=1e-5)
    np.testing.assert_allclose(out[:, 1], x % np.float32(1.5), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(
        out[:, 2], np.exp(x) / (one + np.cos(x) ** two), rtol=2e-3
    )
    np.testing.assert_allclose(
        out[:, 3], np.where(x > 0, np.arctan2(x, two), -x), rtol=2e-3, atol=1e-5
    )


def test_mandelbrot_2d_grid(run):
    h, w, maxiter = 48, 80, 50
    xmin, ymin = np.float32(-2.0), np.float32(-1.0)
    dx, dy = np.float32(3.0 / w), np.float32(2.0 / h)
    out = np.zeros((h, w), dtype=np.int32)
    run(mandelbrot, (w, h), xmin, ymin, dx, dy, maxiter, out)

    expected = np.zeros((h, w), dtype=np.int32)
    for row in range(h):
        for col in range(w):
            cr, ci = xmin + col * dx, ymin + row * dy
            zr = zi = np.float32(0.0)
            n = 0
            while n < maxiter and zr * zr + zi * zi < 4:
                zr, zi = zr * zr - zi * zi + cr, 2 * zr * zi + ci
                n += 1
            expected[row, col] = n
    # Rounding differs slightly between drivers right at the set's boundary.
    assert np.mean(out == expected) > 0.99
    assert np.abs(out - expected).max() <= maxiter


@pytest.mark.float64
def test_stencil_2d_arrays(run):
    rng = np.random.default_rng(2)
    src = rng.random((33, 70))
    dst = np.zeros_like(src)
    run(stencil, (src.shape[1], src.shape[0]), src, dst)
    expected = np.zeros_like(src)
    expected[1:-1, 1:-1] = (
        src[:-2, 1:-1] + src[2:, 1:-1] + src[1:-1, :-2] + src[1:-1, 2:]
    ) / 4
    np.testing.assert_allclose(dst, expected, rtol=1e-12)


@pytest.mark.parametrize("flag", [True, False])
def test_range_with_negative_step_and_bool_arg(run, flag):
    x = np.arange(20, dtype=np.float64)
    out = np.zeros(5)
    run(strided_sum, 5, x, out, flag)
    expected = x[::-1][::3].sum() * np.arange(1, 6)
    np.testing.assert_allclose(out, expected if flag else -expected)


def test_generated_spirv_is_valid(validate):
    f32 = types.float32
    validate(saxpy.compile((f32, f32[::1], f32[::1])))
    validate(
        mandelbrot.compile(
            (f32, f32, f32, f32, types.int64, types.int32[:, ::1]), ndim=2
        )
    )


def _sine_body(x, out):
    i = nv.global_id(0)
    if i < x.shape[0]:
        out[i] = math.sin(x[i])


@pytest.mark.float64
@pytest.mark.skipif(not libclc.available(), reason="libclc is not installed")
def test_float64_transcendentals_use_libclc(run):
    x = np.linspace(0, 3, 64)
    out = np.zeros(64)
    run(nv.jit(_sine_body), 64, x, out)
    np.testing.assert_allclose(out, np.sin(x), rtol=1e-14, atol=1e-15)


@pytest.mark.float64
def test_float64_transcendentals_without_libclc(run, monkeypatch):
    # Without libclc there is no float64 math library: the call is rejected
    # unless narrow_math asks for float32 precision.
    monkeypatch.setattr(libclc, "available", lambda: False)
    x = np.linspace(0, 3, 64)
    out = np.zeros(64)
    with pytest.raises(errors.NumbaError, match="needs libclc"):
        nv.jit(_sine_body).forall(64)(x, out)
    run(nv.jit(narrow_math=True)(_sine_body), 64, x, out)
    np.testing.assert_allclose(out, np.sin(x), rtol=2e-3, atol=1e-6)


def test_unsupported_constructs_raise_with_location():
    @nv.jit
    def bad(x):
        i = nv.global_id(0)
        x[i] = x[i, None][0]

    with pytest.raises(errors.NumbaError, match="only integers, slices and"):
        bad.forall(4)(np.zeros(4, dtype=np.float32))


def test_calling_without_grid_explains_forall():
    with pytest.raises(TypeError, match="forall"):
        saxpy(1.0, np.zeros(1), np.zeros(1))
