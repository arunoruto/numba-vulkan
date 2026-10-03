"""Exceptions raised inside kernels, and bounds checking."""

import numpy as np
import pytest

import numba_vulkan as nv

f32 = np.float32


@nv.jit
def checked_sqrt(x):
    if x < 0:
        raise ValueError("negative input")
    return x ** f32(0.5)


@nv.jit
def checked_inverse(x):
    if x == 0:
        raise ZeroDivisionError("zero input")
    return f32(1) / x


@nv.jit
def uses_both(a, out):
    i = nv.global_id(0)
    if i < a.shape[0]:
        out[i] = checked_sqrt(a[i]) + checked_inverse(a[i])


@nv.jit
def raises_itself(a, out):
    i = nv.global_id(0)
    if i < a.shape[0]:
        if a[i] > 100:
            raise OverflowError
        out[i] = a[i]


@nv.jit
def cannot_raise(a, out):
    i = nv.global_id(0)
    if i < a.shape[0]:
        out[i] = a[i] + f32(1)


@nv.jit(boundscheck=True)
def shifted(a, shift, out):
    i = nv.global_id(0)
    if i < out.shape[0]:
        out[i] = a[i + shift]


@nv.jit(boundscheck=True)
def transposed(a, out):
    i, j = nv.global_id(0), nv.global_id(1)
    if i < a.shape[0] and j < a.shape[1]:
        out[j, i] = a[i, j]


def test_exception_from_a_called_function_is_raised_by_the_launch(run):
    a = np.array([1, 4, 9, 16], dtype=f32)
    out = np.zeros_like(a)
    run(uses_both, 4, a, out)
    np.testing.assert_allclose(out, np.sqrt(a) + 1 / a, rtol=1e-6)

    # Each function keeps its own exception, although both are inlined.
    a[2] = -1
    with pytest.raises(ValueError, match="negative input") as info:
        run(uses_both, 4, a, out)
    (note,) = info.value.__notes__
    assert "kernel 'uses_both'" in note and "in checked_sqrt at" in note
    assert "test_errors.py" in note

    a[2] = 0
    with pytest.raises(ZeroDivisionError, match="zero input"):
        run(uses_both, 4, a, out)


def test_exception_without_arguments_from_the_kernel_itself(run):
    a = np.array([1, 200, 3], dtype=f32)
    out = np.zeros_like(a)
    with pytest.raises(OverflowError):
        run(raises_itself, 3, a, out)
    # the other invocations still ran
    assert out[0] == 1 and out[2] == 3


def test_exception_with_device_arrays(run, device):
    a = nv.to_device(np.array([1, -4, 9], dtype=f32), device)
    out = nv.device_array_like(a)
    with pytest.raises(ValueError, match="negative input"):
        run(uses_both, 3, a, out)
        nv.synchronize(device)  # reported here unless NUMBA_VULKAN_SYNC=1


def test_status_is_only_read_back_from_kernels_that_can_raise(run):
    a = np.zeros(4, dtype=f32)
    run(cannot_raise, 4, a, a.copy())
    run(raises_itself, 4, a, a.copy())
    plain = list(cannot_raise._kernels.values())[-1]
    raising = list(raises_itself._kernels.values())[-1]
    assert 0 not in plain.written_bindings
    assert 0 in raising.written_bindings


def test_bounds_check(run):
    a = np.arange(8, dtype=f32)
    out = np.zeros(4, dtype=f32)
    run(shifted, 4, a, 4, out)
    np.testing.assert_array_equal(out, a[4:])
    run(shifted, 4, a, -4, out)  # negative indices wrap around
    np.testing.assert_array_equal(out, [4, 5, 6, 7])
    with pytest.raises(IndexError, match="out of bounds for axis 0 of a 1-dim"):
        run(shifted, 4, a, 5, out)
    with pytest.raises(IndexError, match="out of bounds"):
        run(shifted, 4, a, -9, out)


def test_bounds_check_names_the_axis(run):
    a = np.arange(6, dtype=f32).reshape(2, 3)
    out = np.zeros((3, 2), dtype=f32)
    run(transposed, (2, 3), a, out)
    np.testing.assert_array_equal(out, a.T)
    with pytest.raises(IndexError, match="axis 0 of a 2-dimensional"):
        run(transposed, (2, 3), a, np.zeros((2, 2), dtype=f32))
    with pytest.raises(IndexError, match="axis 1 of a 2-dimensional"):
        run(transposed, (2, 3), a, np.zeros((3, 1), dtype=f32))


def test_equal_exceptions_share_a_status_code():
    from numba_vulkan.target import exception_table

    first = exception_table._add_exception(KeyError, ("k",), None)
    assert exception_table._add_exception(KeyError, ("k",), None) == first
    assert exception_table._add_exception(KeyError, ("other",), None) != first
    assert exception_table.get_exception(first) == (KeyError, ("k",), None)
