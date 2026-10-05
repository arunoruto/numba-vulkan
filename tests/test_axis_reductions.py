"""Reductions along an axis in kernels: array expressions computed lazily."""

import numpy as np
import pytest

import numba_vulkan as nv

f32 = np.float32
RNG = np.random.default_rng(7)
X3 = RNG.integers(-9, 10, (3, 4, 5)).astype(f32)


@nv.jit
def sums(x, out0, out1, out2):
    if nv.global_id(0) == 0:
        out0[:, :] = x.sum(axis=0)
        out1[:, :] = np.sum(x, axis=1)
        out2[:, :] = x.sum(-1)


def test_sum_along_each_axis_of_a_3d_array(device):
    outs = [np.zeros(X3.sum(axis=k).shape, f32) for k in range(3)]
    sums.forall(1, device=device)(X3, *outs)
    for k, out in enumerate(outs):
        np.testing.assert_array_equal(out, X3.sum(axis=k))


@nv.jit
def every(x, out):
    """All reductions along axis 1 of a 2-d array, one per row of out."""
    if nv.global_id(0) == 0:
        out[0, :] = x.sum(axis=1)
        out[1, :] = x.prod(axis=1)
        out[2, :] = x.mean(axis=1)
        out[3, :] = x.min(axis=1)
        out[4, :] = x.max(axis=1)
        out[5, :] = x.argmin(axis=1)
        out[6, :] = x.argmax(axis=1)
        out[7, :] = np.any(x > 2, axis=1)
        out[8, :] = np.all(x > -5, axis=1)


def test_every_reduction(device):
    x = RNG.integers(-6, 7, (6, 4)).astype(f32)
    out = np.zeros((9, 6), f32)
    every.forall(1, device=device)(x, out)
    expected = [
        x.sum(1),
        x.prod(1),
        x.mean(1),
        x.min(1),
        x.max(1),
        x.argmin(1),
        x.argmax(1),
        (x > 2).any(1),
        (x > -5).all(1),
    ]
    np.testing.assert_allclose(out, np.array(expected, f32), rtol=1e-6)


@nv.jit
def variable_axis(x, axis, out):
    if nv.global_id(0) == 0:
        out[:] = x.max(axis)


def test_the_axis_may_be_a_variable(device):
    x = RNG.standard_normal((4, 4)).astype(f32)
    out = np.zeros(4, f32)
    for axis in (0, 1, -1, -2):
        variable_axis.forall(1, device=device)(x, axis, out)
        np.testing.assert_array_equal(out, x.max(axis))
    with pytest.raises(ValueError, match="out of bounds"):
        variable_axis.forall(1, device=device)(x, 2, out)


@nv.jit
def centred(x, out):
    """Expressions of reductions broadcast like NumPy's."""
    i = nv.global_id(0)
    if i == 0:
        out[:, :] = x - x.mean(axis=0)
        out[0, 0] = (x * f32(2)).sum(axis=1).sum()


def test_reductions_inside_expressions(device):
    x = RNG.standard_normal((5, 3)).astype(f32)
    out = np.zeros_like(x)
    centred.forall(1, device=device)(x, out)
    expected = x - x.mean(axis=0)
    expected[0, 0] = (x * 2).sum(axis=1).sum()
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


@nv.jit
def per_thread(x, out):
    """Each invocation reads one element of the reduction."""
    i = nv.global_id(0)
    if i < out.shape[0]:
        out[i] = x[:, ::2].sum(axis=0)[i] + x.min(axis=0)[i]


def test_views_and_reading_single_elements(device):
    x = RNG.standard_normal((7, 8)).astype(f32)
    out = np.zeros(4, f32)
    per_thread.forall(4, device=device)(x, out)
    expected = x[:, ::2].sum(axis=0) + x.min(axis=0)[:4]
    np.testing.assert_allclose(out, expected, rtol=1e-5)


@nv.jit
def int_sums(x, out):
    if nv.global_id(0) == 0:
        out[:] = x.sum(axis=0)


def test_integers_and_booleans_accumulate_as_integers(device):
    x = np.full((3, 4), 1 << 20, np.int32)
    out = np.zeros(4, np.int64)
    int_sums.forall(1, device=device)(x, out)
    np.testing.assert_array_equal(out, x.sum(axis=0))
    flags = np.array([[True, False], [True, True]])
    counts = np.zeros(2, np.int64)
    int_sums.forall(1, device=device)(flags, counts)
    np.testing.assert_array_equal(counts, [2, 1])


@nv.jit
def nan_extremes(x, out):
    if nv.global_id(0) == 0:
        out[0, :] = x.min(axis=0)
        out[1, :] = x.max(axis=0)


def test_min_and_max_keep_nan(device):
    x = np.array([[1, np.nan], [0, 2]], f32)
    out = np.zeros((2, 2), f32)
    nan_extremes.forall(1, device=device)(x, out)
    np.testing.assert_array_equal(out, [x.min(axis=0), x.max(axis=0)])


@nv.jit
def in_place(x):
    if nv.global_id(0) == 0:
        x[:, :] = x.sum(axis=0)


@nv.jit
def through_other_names(x, y):
    if nv.global_id(0) == 0:
        y[:, :] = x.sum(axis=0)


def test_assigning_to_the_reduced_array_is_refused(device):
    x = np.ones((3, 3), f32)
    with pytest.raises(Exception, match="positions"):
        in_place.forall(1, device=device)(x)
    d = nv.to_device(x, device)
    # The same array as both arguments is not caught: KI-35.
    other = nv.device_array_like(d)
    through_other_names.forall(1, device=device)(d, other)
    np.testing.assert_array_equal(other.copy_to_host(), np.full((3, 3), 3))


@nv.jit
def one_dimensional(x, axis, out):
    if nv.global_id(0) == 0:
        out[0] = x.sum(axis=axis)


def test_one_dimensional_arrays_give_scalars(device):
    x = np.arange(6, dtype=f32)
    out = np.zeros(1, f32)
    one_dimensional.forall(1, device=device)(x, -1, out)
    assert out[0] == 15
    with pytest.raises(ValueError, match="out of bounds"):
        one_dimensional.forall(1, device=device)(x, 1, out)
