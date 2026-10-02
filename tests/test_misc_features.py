"""Division-by-zero errors, print() and float16 arrays."""

import numpy as np
import pytest

import numba_vulkan as nv
from numba_vulkan import runtime

f16 = np.float16


def divide(a, b, out):
    i = nv.global_id(0)
    if i < a.shape[0]:
        out[i] = a[i] // b[i] + a[i] % b[i]


def test_division_by_zero_with_the_python_error_model(run):
    a = np.array([7, -7, 9], dtype=np.int32)
    b = np.array([2, 3, 0], dtype=np.int32)
    out = np.zeros(3, dtype=np.int32)
    run(nv.jit(divide), 3, a, b, out)  # NumPy's model: no error
    with pytest.raises(ZeroDivisionError, match="integer division by zero"):
        run(nv.jit(error_model="python")(divide), 3, a, b, out)
    b[2] = 4
    run(nv.jit(error_model="python")(divide), 3, a, b, out)
    np.testing.assert_array_equal(out, a // b + a % b)
    with pytest.raises(ValueError, match="error_model must be"):
        nv.jit(error_model="c")(divide)


@nv.jit
def chatty(values):
    i = nv.global_id(0)
    if i < values.shape[0] and i % 2 == 1:
        print("i =", i, "value", values[i], values[i] > 2, np.float32(0.5), -3)


def test_print(run, capsys):
    run(chatty, 5, np.arange(5.0))
    lines = sorted(capsys.readouterr().out.splitlines())
    assert lines == ["i = 1 value 1.0 False 0.5 -3", "i = 3 value 3.0 True 0.5 -3"]


def test_print_waits_even_with_device_arrays(device, capsys):
    values = nv.to_device(np.arange(4.0), device)
    chatty.forall(4, device=device)(values)
    assert len(capsys.readouterr().out.splitlines()) == 2


def test_print_buffer_overflow(device, capsys, monkeypatch):
    monkeypatch.setattr(runtime, "PRINT_BUFFER_WORDS", 64)

    @nv.jit
    def flood(out):
        i = nv.global_id(0)
        if i < out.shape[0]:
            print(i)

    with pytest.warns(UserWarning, match="print buffer of 64 words overflowed"):
        flood.forall(1000, device=device)(np.zeros(1000))
    lines = capsys.readouterr().out.splitlines()
    assert 0 < len(lines) <= 64 // 3


def test_print_rejects_arrays():
    @nv.jit
    def bad(values):
        print(values)

    with pytest.raises(nv.VulkanUnsupportedError, match="constant strings and numbers"):
        bad.forall(1)(np.zeros(2))


@nv.jit
def half_ops(a, out):
    i = nv.global_id(0)
    if i < a.shape[0]:
        out[i] = a[i] * 3 + a[i:].sum() * 0 + a[: i + 1].max()


def test_float16_arrays(run):
    a = np.linspace(-1, 2, 50).astype(f16)
    out = np.zeros(50, dtype=f16)
    run(half_ops, 50, a, out)
    wide = a.astype(np.float32)
    want = (wide * 3 + np.maximum.accumulate(wide)).astype(f16)
    np.testing.assert_array_equal(out, want)


def test_float16_device_arrays_and_storage_feature(device):
    a = np.linspace(0, 1, 16).astype(f16)
    da, dout = nv.to_device(a, device), nv.device_array(16, f16, device)
    assert da.copy_to_host().dtype == f16
    half_ops.forall(16, device=device)(da, dout)
    want = a.astype(np.float32) * 3 + np.maximum.accumulate(a.astype(np.float32))
    np.testing.assert_array_equal(dout.copy_to_host(), want.astype(f16))
    compiled = list(half_ops._kernels.values())[-1]
    assert (
        "storage16" in compiled.capabilities and "float16" not in compiled.capabilities
    )
