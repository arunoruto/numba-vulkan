"""Kernels on devices without float64 and int64 (see numba_vulkan.narrowing)."""

import dataclasses
import math
import warnings

import numpy as np
import pytest
from numba.core import errors

import numba_vulkan as nv
from numba_vulkan import libclc, narrowing, runtime
from numba_vulkan.narrowing import Mode, narrow_ir

f32 = np.float32
X = np.linspace(0.5, 3.0, 64)
TABLE = np.array([10, 20, 30], dtype=np.int64)


def typical(scale, x, counts, out):
    # Python float literals, a float64 scalar, int64 data, index arithmetic,
    # math functions, a loop, a slice and a global table.
    i = nv.global_id(0)
    if i < x.shape[0]:
        total = 0.0
        for k in range(counts[i] % 4):
            total += math.sin(x[i] * 0.1 + k) * scale
        out[i] = total + math.log(x[i]) + x[max(i - 2, 0) : i + 1].sum()
        out[i] += TABLE[i % 3] + math.atan2(x[i], 2.0) + x[i] ** 2.5


def reference(scale, x, counts):
    out = np.zeros_like(x)
    for i in range(x.shape[0]):
        total = sum(math.sin(x[i] * 0.1 + k) * scale for k in range(counts[i] % 4))
        out[i] = total + math.log(x[i]) + x[max(i - 2, 0) : i + 1].sum()
        out[i] += TABLE[i % 3] + math.atan2(x[i], 2.0) + x[i] ** 2.5
    return out


@pytest.fixture
def limited(device):
    """The device, opened without any of the optional 64-bit features."""
    info = nv.list_devices()[device]
    return runtime.Device(
        dataclasses.replace(info, float64=False, int64=False, int16=False, int8=False)
    )


@pytest.mark.skipif(not libclc.available(), reason="libclc is not installed")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_forced_narrow_kernel_needs_no_optional_features(run, dtype):
    kernel = nv.jit(narrow=True)(typical)
    x = X.astype(dtype)
    counts = np.arange(64, dtype=np.int64)
    out = np.zeros_like(x)
    run(kernel, 64, 0.5, x, counts, out)
    np.testing.assert_allclose(out, reference(0.5, x, counts), rtol=2e-6)
    compiled = list(kernel._kernels.values())[-1]
    assert compiled.capabilities == set()
    assert compiled.mode[:2] == (True, True)
    assert compiled.narrowed[:2] == (True, True)
    assert out.dtype == dtype  # converted back on the host


def test_exact_kernel_still_uses_64_bit_types(run):
    kernel = nv.jit(narrow=False)(typical)
    counts = np.arange(64, dtype=np.int64)
    out = np.zeros_like(X)
    run(kernel, 64, 0.5, X, counts, out)
    np.testing.assert_allclose(out, reference(0.5, X, counts), rtol=1e-12)
    compiled = list(kernel._kernels.values())[-1]
    assert {"float64", "int64"} <= compiled.capabilities


def test_float32_kernels_no_longer_need_int8(run):
    @nv.jit
    def kernel(x, out):
        i = nv.global_id(0)
        if i < x.shape[0]:
            out[i] = x[i] + f32(1)

    x = X.astype(f32)
    out = np.zeros_like(x)
    run(kernel, 64, x, out)
    compiled = list(kernel._kernels.values())[-1]
    assert "int8" not in compiled.capabilities


@pytest.mark.skipif(not libclc.available(), reason="libclc is not installed")
def test_device_without_64_bit_types_narrows_and_warns(limited):
    kernel = nv.jit(typical)
    counts = np.arange(64, dtype=np.int64)
    out = np.zeros_like(X)
    with pytest.warns(nv.VulkanPrecisionWarning, match="computed in float32"):
        kernel.forall(64, device=limited)(0.5, X, counts, out)
    np.testing.assert_allclose(out, reference(0.5, X, counts), rtol=2e-6)
    # compiled once, so no second warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kernel.forall(64, device=limited)(0.5, X, counts, out)


def test_float32_kernel_on_a_limited_device_does_not_warn(limited):
    @nv.jit
    def kernel(x, out):
        i = nv.global_id(0)
        if i < x.shape[0]:
            out[i] = x[i] * f32(2)

    x = X.astype(f32)
    out = np.zeros_like(x)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kernel.forall(64, device=limited)(x, out)
    np.testing.assert_array_equal(out, x * 2)


def test_narrow_false_fails_on_a_limited_device(limited):
    kernel = nv.jit(narrow=False)(typical)
    with pytest.raises(nv.VulkanSupportError, match="needs float64, int64"):
        kernel.forall(64, device=limited)(
            0.5, X, np.arange(64, dtype=np.int64), np.zeros_like(X)
        )


def test_device_arrays_store_32_bit_elements_on_a_limited_device(limited):
    values = np.arange(10, dtype=np.float64) / 3
    array = nv.to_device(values, limited)
    assert array.dtype == np.float64 and array._stored == np.float32
    back = array.copy_to_host()
    assert back.dtype == np.float64
    np.testing.assert_array_equal(back, values.astype(f32))
    counts = nv.to_device(np.arange(5, dtype=np.int64) - 2, limited)
    assert counts._stored == np.int32
    np.testing.assert_array_equal(counts.copy_to_host(), np.arange(5) - 2)

    @nv.jit
    def kernel(a):
        i = nv.global_id(0)
        if i < a.shape[0]:
            a[i] = a[i] * 3.0

    with pytest.warns(nv.VulkanPrecisionWarning):
        kernel.forall(10, device=limited)(array)
    np.testing.assert_allclose(array.copy_to_host(), np.arange(10), rtol=1e-6)


@pytest.mark.float64
def test_wide_device_array_does_not_fit_a_narrow_kernel(device):
    @nv.jit(narrow=True)
    def kernel(a):
        i = nv.global_id(0)
        if i < a.shape[0]:
            a[i] += 1

    array = nv.to_device(np.zeros(4), device)
    with pytest.raises(ValueError, match="holds float64 elements"):
        kernel.forall(4, device=device)(array)


@pytest.mark.parametrize(
    "expression, message",
    [
        ("a[i] << 40", "shifts 64-bit integers by 32 bits or more"),
        ("a[i] + 10_000_000_000", "does not fit in 32 bits"),
    ],
)
def test_code_that_needs_64_bits_is_rejected(expression, message):
    scope = {"nv": nv}
    exec(
        "def kernel(a, out):\n"
        "    i = nv.global_id(0)\n"
        "    if i < a.shape[0]:\n"
        f"        out[i] = {expression}\n",
        scope,
    )
    a = np.arange(4, dtype=np.int64)
    with pytest.raises(errors.NumbaError, match=message):
        nv.jit(narrow=True)(scope["kernel"]).forall(4)(a, np.zeros_like(a))


def test_narrow_ir_rewrites_types_casts_and_literals():
    text = """\
target datalayout = "e-i64:64-n8:16:32:64"
declare double @llvm.sqrt.f64(double)
declare float @llvm.sqrt.f32(float)
define void @main() {
  %a = sext i32 %x to i64
  %b = trunc i64 %a to i32
  %c = fpext float %f to double
  %d = fmul double %c, 1.000000e-01
  %e = fptrunc double %d to float
  %g = call double @llvm.sqrt.f64(double %d)
  %h = icmp eq i64 %a, 9223372036854775807
  %j = lshr i64 %a, 62
  %k = ashr i64 %a, 63
  %m = fadd double %d, 0x3FB999999999999A
}
"""
    out = narrow_ir(text, Mode(True, True)).splitlines()
    assert out[0] == 'target datalayout = "e-i64:64-n8:16:32:64"'
    assert out.count("declare float @llvm.sqrt.f32(float)") == 1
    assert "  %a = bitcast i32 %x to i32" in out
    assert "  %b = bitcast i32 %a to i32" in out
    assert "  %c = bitcast float %f to float" in out
    # 0.1 rounded to float32, in LLVM's spelling for float constants
    assert "  %d = fmul float %c, 0x3FB99999A0000000" in out
    assert "  %m = fadd float %d, 0x3FB99999A0000000" in out
    assert "  %e = bitcast float %d to float" in out
    assert "  %g = call float @llvm.sqrt.f32(float %d)" in out
    assert "  %h = icmp eq i32 %a, 2147483647" in out
    assert "  %j = lshr i32 %nv.sign" in "\n".join(out) and out[-4].endswith(", 30")
    assert "  %k = ashr i32 %a, 31" in out
    assert not any("i64" in line or "double" in line for line in out[1:])


def test_narrow_ir_leaves_other_modes_alone():
    text = "define void @main() {\n  %a = sext i32 %x to i64\n  %d = fadd double %c, %c\n}\n"
    assert narrow_ir(text, Mode()) is text
    ints_only = narrow_ir(text, Mode(ints=True))
    assert "double" in ints_only and "i64" not in ints_only
    floats_only = narrow_ir(text, Mode(floats=True))
    assert "double" not in floats_only and "i64" in floats_only


def test_narrow_ir_allows_sign_tests_but_no_other_bit_patterns():
    sign = """\
define void @main() {
  %b = bitcast double %x to i64
  %n = icmp slt i64 %b, 0
}
"""
    out = narrow_ir(sign, Mode(True, True))
    assert "%b = bitcast float %x to i32" in out and "icmp slt i32 %b, 0" in out
    exponent = sign.replace("icmp slt i64 %b, 0", "lshr i64 %b, 52")
    with pytest.raises(nv.VulkanUnsupportedError, match="bit pattern of float64"):
        narrow_ir(exponent, Mode(True, True))


def test_stored_dtype():
    wide, both = Mode(), Mode(True, True)
    assert narrowing.stored_dtype(np.bool_, wide) == np.int32
    assert narrowing.stored_dtype(np.float64, wide) == np.float64
    assert narrowing.stored_dtype(np.float64, both) == np.float32
    assert narrowing.stored_dtype(np.int64, both) == np.int32
    assert narrowing.stored_dtype(np.uint64, both) == np.uint32
    assert narrowing.stored_dtype(np.int64, Mode(floats=True)) == np.int64
    assert narrowing.stored_dtype(np.int16, both) == np.int16


# -- 32-bit integers by default, float64 kept ---------------------------------------


@pytest.mark.float64
@pytest.mark.skipif(not narrowing.NARROW_INTS, reason="NUMBA_VULKAN_INT64=1")
def test_integers_are_32_bit_by_default_and_floats_64_bit(run):
    @nv.jit
    def mix(ints, floats, out):
        i = nv.global_id(0)
        if i < ints.shape[0]:
            out[i] = floats[i] * 3.0 + ints[i]

    ints = np.arange(8, dtype=np.int64)
    floats = np.linspace(0, 1, 8) + 1e-12  # needs float64 to survive
    out = np.zeros(8)
    with pytest.warns(nv.VulkanPerformanceWarning, match="computes with float64"):
        run(mix, 8, ints, floats, out)
    np.testing.assert_array_equal(out, floats * 3.0 + ints)
    compiled = list(mix._kernels.values())[-1]
    assert compiled.mode.ints and not compiled.mode.floats
    assert compiled.narrowed.ints and not compiled.narrowed.floats


@pytest.mark.skipif(not narrowing.NARROW_INTS, reason="NUMBA_VULKAN_INT64=1")
def test_large_int64_inputs_raise_instead_of_wrapping(run):
    @nv.jit
    def copy(values, out):
        i = nv.global_id(0)
        if i < values.shape[0]:
            out[i] = values[i]

    big = np.array([1, 2, 3 << 40], dtype=np.int64)
    out = np.empty(3, dtype=np.int64)  # garbage, but only written: no check
    with pytest.raises(OverflowError, match="does not fit into int32"):
        run(copy, 3, big, out)
    run(nv.jit(narrow=False)(copy.py_func), 3, big, out)
    np.testing.assert_array_equal(out, big)
    run(copy, 2, big[:2].copy(), out[:2])
    with pytest.raises(OverflowError):
        nv.to_device(big)


@pytest.mark.float64
def test_float_literal_hint(run):
    @nv.jit
    def scale(x, out):
        i = nv.global_id(0)
        if i < x.shape[0]:
            out[i] = x[i] * 0.1  # 0.5 would be folded into float32

    x = np.ones(4, dtype=np.float32)
    with pytest.warns(nv.VulkanPerformanceWarning, match="np.float32"):
        run(scale, 4, x, np.zeros_like(x))

    @nv.jit
    def scale32(x, out):
        i = nv.global_id(0)
        if i < x.shape[0]:
            out[i] = x[i] * np.float32(0.1)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run(scale32, 4, x, np.zeros_like(x))


def test_warnings_can_be_silenced(run, monkeypatch):
    monkeypatch.setattr(narrowing, "WARNINGS", False)

    @nv.jit
    def double(x, out):
        i = nv.global_id(0)
        if i < x.shape[0]:
            out[i] = x[i] * 2.0

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run(double, 4, np.ones(4), np.zeros(4))
