"""Buffers are named by values, so functions taking arrays compile once."""

import numpy as np
import pytest

import numba_vulkan as nv
from numba_vulkan.errors import SpirvCodegenError

f32 = np.float32


@nv.jit
def total(a):
    s = f32(0)
    for v in a:
        s += v
    return s


@nv.jit
def bump(a, i):
    a[i] += 1


@nv.jit
def three_totals(x, y, z, out):
    if nv.global_id(0) == 0:
        out[0] = total(x)
        out[1] = total(y)
        out[2] = total(z)
        bump(out, 0)
        bump(x, 0)


def test_a_function_is_compiled_once_for_all_buffers(run):
    x = np.ones(4, f32)
    out = np.zeros(3, f32)
    run(three_totals, 1, x, 2 * x, 3 * x, out)
    np.testing.assert_array_equal(out, [5, 8, 12])
    assert x[0] == 2  # bump wrote to the right buffer
    # One specialisation per narrowing mode (device), not per buffer.
    assert all(len(found) == 1 for found in total._overloads.values())
    assert all(len(found) == 1 for found in bump._overloads.values())


def test_views_of_arguments_keep_their_buffer(run):
    @nv.jit
    def halves(x, y, out):
        if nv.global_id(0) == 0:
            out[0] = total(x[: x.shape[0] // 2])
            out[1] = total(y.T[1])
            out[2] = total(x[::-1])

    x = np.arange(6, dtype=f32)
    y = np.arange(6, dtype=f32).reshape(3, 2)
    out = np.zeros(3, f32)
    run(halves, 1, x, y, out)
    np.testing.assert_array_equal(out, [x[:3].sum(), y.T[1].sum(), x.sum()])


def test_aliasing_checks_still_see_the_same_buffer(run):
    @nv.jit
    def shift(a, b):
        if nv.global_id(0) == 0:
            a[1:] = a[:-1] + b[:-1]  # reads a at other positions

    with pytest.raises(ValueError, match="other positions"):
        run(shift, 1, np.arange(4, dtype=f32), np.zeros(4, f32))

    @nv.jit
    def add_other(a, b):
        if nv.global_id(0) == 0:
            a[1:] = a[1:] + b[:-1]  # b is another buffer: fine

    a = np.arange(4, dtype=f32)
    run(add_other, 1, a, np.ones(4, f32))
    np.testing.assert_array_equal(a, [0, 2, 3, 4])


def test_choosing_an_array_at_run_time_is_explained():
    @nv.jit
    def pick(x, y, flag, out):
        a = x if flag else y
        i = nv.global_id(0)
        if i < out.shape[0]:
            out[i] = a[i]

    x = np.ones(3, f32)
    with pytest.raises(SpirvCodegenError, match="chooses between arrays"):
        pick.forall(3)(x, x, True, np.zeros(3, f32))
