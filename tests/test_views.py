"""Slices, views, iteration and reductions over arrays."""

import textwrap

import numpy as np
import pytest
from numba.core import errors

import numba_vulkan as nv

f32 = np.float32
A = ((np.arange(30, dtype=f32).reshape(6, 5) * 7) % 11) + 1
N = (np.arange(30).reshape(6, 5) % 4).astype(np.int32)


def rows(body, *arrays, columns=8, dtype=f32, run=None, **options):
    """Run ``body`` once per row index ``i`` of the first array.

    The body sees the arrays as ``a``, ``b``... and writes to ``out``,
    which has one row per row of ``a``.
    """
    names = "abcd"[: len(arrays)]
    source = (
        f"def kernel({', '.join(names)}, out):\n"
        "    i = nv.global_id(0)\n"
        "    if i < a.shape[0]:\n"
        + textwrap.indent(textwrap.dedent(body).strip("\n"), " " * 8)
    )
    scope = dict(globals())
    exec(source, scope)
    out = np.zeros((arrays[0].shape[0], columns), dtype=dtype)
    run(nv.jit(**options)(scope["kernel"]), arrays[0].shape[0], *arrays, out)
    return out


def per_row(function, array=A):
    return np.array([function(array, i) for i in range(array.shape[0])])


VIEWS = {
    "row": ("a[i][2]", lambda a, i: a[i][2]),
    "reversed": ("a[i, ::-1][1]", lambda a, i: a[i, ::-1][1]),
    "step": ("a[i, 1::2][1]", lambda a, i: a[i, 1::2][1]),
    "negative bounds": ("a[i, 1:-1][-1]", lambda a, i: a[i, 1:-1][-1]),
    "clipped": ("a[i, 2:100].shape[0]", lambda a, i: a[i, 2:100].shape[0]),
    "empty": ("a[i, 4:2].size", lambda a, i: a[i, 4:2].size),
    "block": ("a[1:4, 1:3][i % 3, 1]", lambda a, i: a[1:4, 1:3][i % 3, 1]),
    "view of a view": ("a[::2][i % 3][::-2][1]", lambda a, i: a[::2][i % 3][::-2][1]),
    "column": ("a[:, 2][i]", lambda a, i: a[:, 2][i]),
    "ellipsis": ("a[..., 1][i] + a[i, ...][3]", lambda a, i: a[..., 1][i] + a[i, ...][3]),
    "transpose": ("a.T[2, i] + a.T[1:3][1, i]", lambda a, i: a.T[2, i] + a.T[1:3][1, i]),
    "attributes": (
        "a[i, ::2].size + 10 * len(a[i:]) + 100 * a[i, 1:].ndim + a[i, ::2].strides[0]",
        lambda a, i: (
            a[i, ::2].size + 10 * len(a[i:]) + 100 * a[i, 1:].ndim + a[i, ::2].strides[0]
        ),
    ),
}  # fmt: skip


@pytest.mark.parametrize("name", VIEWS)
def test_view(run, name):
    expression, reference = VIEWS[name]
    out = rows(f"out[i, 0] = {expression}", A, run=run)
    np.testing.assert_array_equal(out[:, 0], per_row(reference))


REDUCTIONS = {
    "sum": ("a[i].sum()", lambda a, i: a[i].sum()),
    "np.sum of a strided view": ("np.sum(a[i, ::2])", lambda a, i: a[i, ::2].sum()),
    "sum of a block": ("a[i : i + 2, 1:3].sum()", lambda a, i: a[i : i + 2, 1:3].sum()),
    "prod": ("a[i, :3].prod()", lambda a, i: a[i, :3].prod()),
    "mean": ("a[i].mean() + np.mean(a[:, 1])", lambda a, i: a[i].mean() + a[:, 1].mean()),
    "min and max": ("a[i].min() - np.max(a[i]) + max(a[i]) + min(a[i, 1:])",
                    lambda a, i: a[i].min() + a[i, 1:].min()),
    "argmin and argmax": ("a[i].argmax() + 10 * a[i, ::-1].argmin() + 100 * np.argmax(a.T[1:3])",
                          lambda a, i: a[i].argmax() + 10 * a[i, ::-1].argmin()
                          + 100 * a.T[1:3].argmax()),
    "dot": ("np.dot(a[i], a[0, ::-1])", lambda a, i: a[i] @ a[0, ::-1]),
}  # fmt: skip


@pytest.mark.parametrize("name", REDUCTIONS)
def test_reduction(run, name):
    expression, reference = REDUCTIONS[name]
    out = rows(f"out[i, 0] = {expression}", A, run=run)
    np.testing.assert_allclose(out[:, 0], per_row(reference), rtol=1e-6)


def test_integer_and_boolean_reductions(run):
    flags = N > 1
    out = rows(
        """
        out[i, 0] = b[i].sum()
        out[i, 1] = b.prod()
        out[i, 2] = b[i].mean() * 4
        out[i, 3] = 1 if b[i, :2].any() else 0
        out[i, 4] = 1 if b[i].all() else 0
        out[i, 5] = c[i].sum()
        out[i, 6] = 1 if np.any(c[i]) else 0
        out[i, 7] = 1 if np.all(c[i, 2:4]) else 0
        """,
        A, N, flags, dtype=np.int64, run=run,
    )  # fmt: skip
    want = np.stack(
        [N.sum(1), np.full(6, N.prod()), (N.mean(1) * 4).astype(np.int64), N[:, :2].any(1), N.all(1),
         flags.sum(1), flags.any(1), flags[:, 2:4].all(1)],
        axis=1,
    )  # fmt: skip
    np.testing.assert_array_equal(out, want)


def test_min_and_max_propagate_nan(run):
    a = A.copy()
    a[2, 3] = np.nan
    out = rows("out[i, 0] = a[i].min()\nout[i, 1] = a[i].max()", a, run=run)
    np.testing.assert_array_equal(out[:, :2], np.stack([a.min(1), a.max(1)], axis=1))


def test_reduction_of_an_empty_array_raises(run):
    with pytest.raises(ValueError, match="zero-size array has no minimum"):
        rows("out[i, 0] = a[i, 3:3].min()", A, run=run)
    out = rows("out[i, 0] = a[i, 3:3].sum() + 5", A, run=run)
    np.testing.assert_array_equal(out[:, 0], 5)


def test_iteration(run):
    out = rows(
        """
        for v in a[i]:
            out[i, 0] += v
        for j, v in enumerate(a[i, ::-1]):
            out[i, 1] += j * v
        for v, w in zip(a[i], a[0]):
            out[i, 2] += v * w
        for row in a[:3]:
            out[i, 3] += row[i % 5]
        for row in a.T:
            for v in row:
                out[i, 4] += v
        """,
        A, run=run,
    )  # fmt: skip
    want = np.stack(
        [A.sum(1), (A[:, ::-1] * np.arange(5)).sum(1), A @ A[0],
         A[:3, np.arange(6) % 5].sum(0), np.full(6, A.sum())],
        axis=1,
    )  # fmt: skip
    np.testing.assert_allclose(out[:, :5], want, rtol=1e-6)


@nv.jit
def weighted(values, weights):
    total = f32(0)
    for k in range(values.shape[0]):
        total += values[k] * weights[k]
    return total


@nv.jit
def fill_ramp(target, start):
    for k in range(target.shape[0]):
        target[k] = start + k


def test_views_are_passed_to_functions_and_written_through(run):
    out = rows(
        """
        out[i, 0] = weighted(a[i, 1:4], a[0, ::2])
        fill_ramp(out[i, 1:5], a[i, 0])
        fill_ramp(out[i, :4:-1], f32(100))
        """,
        A, run=run,
    )  # fmt: skip
    np.testing.assert_allclose(out[:, 0], (A[:, 1:4] * A[0, ::2]).sum(1))
    np.testing.assert_array_equal(out[:, 1:5], A[:, :1] + np.arange(4))
    np.testing.assert_array_equal(out[:, 5:], np.tile([102, 101, 100], (6, 1)))


def test_slice_assignment(run):
    out = rows(
        """
        out[i] = f32(-1)
        out[i, 1:3] = a[i, 3:]
        out[i, 4:] = i
        out[i, 3:0:-2] = b[i, :2]
        """,
        A, N, run=run,
    )  # fmt: skip
    want = np.full((6, 8), -1, dtype=f32)
    want[:, 1:3] = A[:, 3:]
    want[:, 4:] = np.arange(6)[:, None]
    want[:, 3:0:-2] = N[:, :2]
    np.testing.assert_array_equal(out, want)


def test_slice_assignment_checks_sizes(run):
    with pytest.raises(ValueError, match="cannot assign slice from input of different"):
        rows("out[i, 1:3] = a[i, 2:]", A, run=run)


def test_overlapping_copy_is_rejected(run):
    with pytest.raises(errors.NumbaError, match="could overlap"):
        rows("out[i, 1:] = out[i, :-1]", A, run=run)


def test_zero_step_raises(run):
    with pytest.raises(ValueError, match="slice step cannot be zero"):
        rows("out[i, 0] = a[i, ::b[0, 0]][0]", A, N, run=run)


def test_bounds_check_applies_to_views(run):
    out = rows("out[i, 0] = a[i, 1:4][2]", A, run=run, boundscheck=True)
    np.testing.assert_array_equal(out[:, 0], A[:, 3])
    with pytest.raises(IndexError, match="axis 0 of a 1-dimensional"):
        rows("out[i, 0] = a[i, 1:4][3]", A, run=run, boundscheck=True)


def test_unsupported_array_operations_say_so(run):
    with pytest.raises(nv.VulkanUnsupportedError, match="direct access to memory"):
        rows("out[i, 0] = (a[i] * 2)[0]", A, run=run)
    with pytest.raises(errors.NumbaError, match="only integers, slices and"):
        rows("out[i, 0] = a[i, np.newaxis][0, 0]", A, run=run)


@nv.jit
def double_in_place(a, out):
    i, j = nv.global_id(0), nv.global_id(1)
    if i < a.shape[0] and j < a.shape[1]:
        out[i, j] = 2 * a[i, j]


@pytest.mark.parametrize(
    "make",
    [
        lambda base: base[::2, ::-1],
        lambda base: np.asfortranarray(base),
        lambda base: base.T,
    ],
    ids=["strided", "fortran", "transposed"],
)
def test_host_arrays_need_not_be_contiguous(run, make):
    a = make(np.arange(48, dtype=f32).reshape(6, 8))
    backing = np.zeros((6, 8), dtype=f32)
    out = make(backing)
    assert not a.flags.c_contiguous and out.shape == a.shape
    run(double_in_place, a.shape, a, out)
    np.testing.assert_array_equal(out, 2 * a)
