"""NumPy arrays used as global constants inside kernels."""

import numpy as np
import pytest
from numba.core import errors

import numba_vulkan as nv

f32 = np.float32
TABLE = np.array([1.0, 2.0, 3.0, 4.0], dtype=f32)
MATRIX = np.arange(12, dtype=np.int32).reshape(3, 4)
FLAGS = np.array([True, False, True])
STRIDED = np.arange(20, dtype=f32)[::5]
LARGE = np.sqrt(np.arange(1 << 20, dtype=np.float64)).astype(f32)
X = np.arange(8, dtype=f32)
I = np.arange(8)


@nv.jit
def lookup(j):
    return TABLE[j % 4] * 10


@nv.jit
def uses_tables(x, out):
    i = nv.global_id(0)
    if i < x.shape[0]:
        out[i] = x[i] + lookup(i) + MATRIX[i % 3, 1:].sum() + TABLE.shape[0]
        if FLAGS[i % 3]:
            out[i] += STRIDED[i % 4]


@nv.jit
def uses_one_table(x, out):
    i = nv.global_id(0)
    if i < x.shape[0]:
        out[i] = TABLE[-1 - i % 4] - x[i]


def test_global_arrays_in_kernels_and_called_functions(run):
    out = np.zeros_like(X)
    run(uses_tables, 8, X, out)
    want = X + TABLE[I % 4] * 10 + MATRIX[I % 3, 1:].sum(1) + 4
    want += np.where(FLAGS[I % 3], STRIDED[I % 4], 0)
    np.testing.assert_array_equal(out, want)
    # the same table in another kernel, with another number of arguments
    run(uses_one_table, 8, X, out)
    np.testing.assert_array_equal(out, TABLE[-1 - I % 4] - X)


def test_constants_follow_the_arguments_in_the_binding_order(run):
    out = np.zeros_like(X)
    run(uses_tables, 8, X, out)
    (compiled,) = uses_tables._kernels.values()
    assert sorted(compiled.constants) == [3, 4, 5, 6]
    assert compiled.num_bindings == 7
    assert compiled.written_bindings == {2}
    shapes = sorted(c.shape for c in compiled.constants.values())
    assert shapes == [(3,), (3, 4), (4,), (4,)]


def test_closure_variable_and_device_arrays(run, device):
    weights = np.array([0.5, 0.25], dtype=f32)

    @nv.jit
    def kernel(x, out):
        i = nv.global_id(0)
        if i < x.shape[0]:
            out[i] = x[i] * weights[i % 2]

    out = nv.device_array_like(X, device)
    run(kernel, 8, nv.to_device(X, device), out)
    np.testing.assert_array_equal(out.copy_to_host(), X * weights[I % 2])


def test_large_constant_array(run):
    @nv.jit
    def kernel(positions, out):
        i = nv.global_id(0)
        if i < positions.shape[0]:
            out[i] = LARGE[positions[i]]

    positions = np.array([0, 1, 1000, (1 << 20) - 1], dtype=np.int32)
    out = np.zeros(4, dtype=f32)
    run(kernel, 4, positions, out)
    np.testing.assert_array_equal(out, LARGE[positions])


def test_values_are_frozen_when_first_compiled(run):
    table = np.array([1, 2, 3], dtype=f32)

    @nv.jit
    def kernel(out):
        i = nv.global_id(0)
        if i < out.shape[0]:
            out[i] = table[i]

    out = np.zeros(3, dtype=f32)
    run(kernel, 3, out)
    table[:] = 9
    run(kernel, 3, out)
    np.testing.assert_array_equal(out, [1, 2, 3])


def test_constant_arrays_are_read_only():
    @nv.jit
    def kernel(x):
        i = nv.global_id(0)
        if i < x.shape[0]:
            TABLE[0] = x[i]

    with pytest.raises(errors.TypingError, match="Cannot modify readonly array"):
        kernel.forall(8)(X)


def test_bounds_check_covers_constant_arrays(run):
    @nv.jit(boundscheck=True)
    def kernel(x, out):
        i = nv.global_id(0)
        if i < x.shape[0]:
            out[i] = TABLE[i]

    with pytest.raises(IndexError, match="out of bounds"):
        run(kernel, 8, X, np.zeros_like(X))
