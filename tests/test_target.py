"""Features that come from plugging into Numba's target extension API."""

import math

import numpy as np
import pytest
from numba import njit
from numba.core import errors, types
from numba.core.target_extension import (
    dispatcher_registry,
    jit_registry,
    resolve_dispatcher_from_str,
    target_registry,
)
from numba.extending import overload

import numba_vulkan as nv


def test_target_is_registered():
    target = target_registry["vulkan"]
    assert issubclass(target, target_registry["gpu"])
    assert dispatcher_registry[target] is nv.VulkanDispatcher
    assert jit_registry[target] is nv.jit
    assert resolve_dispatcher_from_str("vulkan") is nv.VulkanDispatcher


@nv.jit
def polar(x, y):
    return math.sqrt(x * x + y * y), math.atan2(y, x)


@nv.jit
def helper_kernel(x, y, radius, angle):
    i = nv.global_id(0)
    if i < x.size:
        r, phi = polar(x[i], y[i])
        radius[i] = r
        angle[i] = phi


def test_device_function_returning_tuple(run):
    rng = np.random.default_rng(3)
    x = rng.normal(size=200).astype(np.float32)
    y = rng.normal(size=200).astype(np.float32)
    radius, angle = np.zeros_like(x), np.zeros_like(x)
    run(helper_kernel, 200, x, y, radius, angle)
    np.testing.assert_allclose(radius, np.hypot(x, y), rtol=1e-5)
    np.testing.assert_allclose(angle, np.arctan2(y, x), rtol=2e-3, atol=2e-3)


@nv.jit
def row_sum(a, row):
    total = a[row, 0] - a[row, 0]
    for col in range(a.shape[1]):
        total += a[row, col]
    return total


@nv.jit
def row_sum_kernel(a, out):
    i = nv.global_id(0)
    if i < out.shape[0]:
        out[i] = row_sum(a, i)


def test_arrays_can_be_passed_to_device_functions(run):
    a = np.arange(60, dtype=np.float32).reshape(12, 5)
    out = np.zeros(12, dtype=np.float32)
    run(row_sum_kernel, 12, a, out)
    np.testing.assert_allclose(out, a.sum(axis=1))


@njit
def cpu_clamp(v, lo, hi):
    return min(max(v, lo), hi)


@nv.jit
def clamp_kernel(x, out):
    i = nv.global_id(0)
    if i < x.shape[0]:
        out[i] = cpu_clamp(x[i], np.float32(-1.0), np.float32(1.0))


def test_cpu_jitted_function_is_recompiled_for_vulkan(run):
    x = np.linspace(-3, 3, 100, dtype=np.float32)
    out = np.zeros_like(x)
    run(clamp_kernel, 100, x, out)
    np.testing.assert_array_equal(out, np.clip(x, -1, 1))
    assert cpu_clamp(5.0, 0.0, 1.0) == 1.0


def smoothstep(edge0, edge1, x):
    raise NotImplementedError


@overload(smoothstep, target="vulkan")
def ol_smoothstep(edge0, edge1, x):
    def impl(edge0, edge1, x):
        t = min(max((x - edge0) / (edge1 - edge0), 0), 1)
        return t * t * (3 - 2 * t)

    return impl


@nv.jit
def smoothstep_kernel(x, out):
    i = nv.global_id(0)
    if i < x.shape[0]:
        out[i] = smoothstep(0.0, 1.0, x[i])


def test_overload_extension_for_vulkan_target(run):
    x = np.linspace(-1, 2, 100)
    out = np.zeros_like(x)
    run(smoothstep_kernel, 100, x, out)
    t = np.clip(x, 0, 1)
    np.testing.assert_allclose(out, t * t * (3 - 2 * t), rtol=1e-12, atol=1e-15)


def test_vulkan_only_overload_is_invisible_to_cpu():
    @njit
    def use(x):
        return smoothstep(0.0, 1.0, x)

    with pytest.raises(errors.NumbaError):
        use(0.5)


@nv.jit
def language_kernel(x, out):
    i = nv.global_id(0)
    if i < x.shape[0]:
        lo, hi = (x[i], x[-1]) if x[i] < x[-1] else (x[-1], x[i])
        steps = 0
        for k in range(10):
            if k % 3 == 0:
                continue
            if k > 7:
                break
            steps += k
        out[i, 0] = hi - lo
        out[i, 1] = steps
        out[i, 2] = x[i] ** 3 + int(x[i]) ** 2 + x[i] ** -2
        out[i, 3] = float(i) if x[i] > 0 else -1.0


def test_tuples_loops_and_negative_indices(run):
    x = np.linspace(-4, 4, 50)
    out = np.zeros((50, 4))
    run(language_kernel, 50, x, out)
    np.testing.assert_allclose(out[:, 0], np.abs(x - x[-1]))
    np.testing.assert_array_equal(out[:, 1], sum(k for k in range(8) if k % 3))
    np.testing.assert_allclose(out[:, 2], x**3 + x.astype(np.int64) ** 2 + x**-2.0)
    np.testing.assert_array_equal(out[:, 3], np.where(x > 0, np.arange(50), -1.0))


def test_inspecting_compiled_kernel():
    f32 = types.float32
    compiled = helper_kernel.compile((f32[::1],) * 4)
    assert "define void @main()" in compiled.llvm_ir
    assert compiled.spirv[:4] == b"\x03\x02\x23\x07"
    assert compiled.local_size == (64, 1, 1)


@nv.jit
def not_kernel(flags, out):
    i = nv.global_id(0)
    if i < flags.shape[0]:
        out[i] = not flags[i]


def test_bool_arrays_round_trip(run):
    flags = np.arange(40) % 3 == 0
    out = np.zeros(40, dtype=np.bool_)
    run(not_kernel, 40, flags, out)
    np.testing.assert_array_equal(out, ~flags)


@nv.jit
def small_int_kernel(x, out):
    i = nv.global_id(0)
    if i < x.shape[0]:
        out[i] = x[i] * 2


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.uint32])
def test_small_and_unsigned_integer_arrays(run, dtype):
    x = np.arange(30, dtype=dtype)
    out = np.zeros_like(x)
    run(small_int_kernel, 30, x, out)
    np.testing.assert_array_equal(out, x * 2)


@nv.jit
def int_power_kernel(x, exponent, out):
    i = nv.global_id(0)
    if i < x.shape[0]:
        out[i] = x[i] ** exponent


@pytest.mark.parametrize("exponent", [0, 3, -1, -2])
def test_integer_power_with_runtime_exponent(run, exponent):
    x = np.array([-3, -2, -1, 1, 2, 3], dtype=np.int64)
    out = np.zeros_like(x)
    run(int_power_kernel, x.size, x, exponent, out)
    if exponent >= 0:
        expected = x**exponent
    else:
        # Integer results truncate to 0 unless the base is 1 or -1.
        expected = np.where(np.abs(x) == 1, x ** (-exponent), 0)
    np.testing.assert_array_equal(out, expected)


@nv.jit
def fibonacci(n):
    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)


def test_recursion_is_rejected():
    @nv.jit
    def kernel(out):
        i = nv.global_id(0)
        if i < out.shape[0]:
            out[i] = fibonacci(5)

    with pytest.raises(nv.VulkanUnsupportedError, match="recursive"):
        kernel.forall(4)(np.zeros(4, dtype=np.int64))
