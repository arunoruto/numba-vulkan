"""Arrays created inside kernels, and array expressions."""

import numpy as np
import pytest
from numba.core import errors
from test_views import rows

import numba_vulkan as nv

f32 = np.float32
A = (np.arange(12, dtype=f32).reshape(3, 4) * 7) % 5 + 1
B = np.linspace(1, 2, 4, dtype=f32)


def test_local_arrays(run):
    out = rows(
        """
        tmp = np.zeros(4, dtype=np.float32)
        grid = nv.local.array((2, 3), np.int32)
        for j in range(4):
            tmp[j] = a[i, j] * j
        for p in range(2):
            for q in range(3):
                grid[p, q] = p * 10 + q
        out[i, 0] = tmp.sum()
        out[i, 1] = grid.sum() + grid[1, 2]
        out[i, 2] = np.ones(5).sum() + np.full(3, 2.5).sum() + np.full((2, 2), 3, np.int32).sum()
        out[i, 3] = np.empty(3, np.float32).size
        """,
        A, run=run,
    )  # fmt: skip
    np.testing.assert_allclose(out[:, 0], (A * np.arange(4)).sum(1))
    np.testing.assert_array_equal(
        out[:, 1:4], np.tile([36 + 12, 5 + 7.5 + 12, 3], (3, 1))
    )


def test_local_arrays_are_private_to_each_invocation(run):
    out = rows(
        """
        mine = nv.local.array(4, np.float32)
        for j in range(4):
            mine[j] = a[i, j]
        for j in range(4):
            out[i, j] = mine[3 - j]
        """,
        A, run=run, columns=4,
    )  # fmt: skip
    np.testing.assert_array_equal(out, A[:, ::-1])


def test_local_arrays_need_a_constant_shape():
    @nv.jit
    def dynamic(n, out):
        tmp = np.zeros(n)
        out[0] = tmp[0]

    with pytest.raises(errors.TypingError, match="needs a constant shape"):
        dynamic.forall(1)(4, np.zeros(1))


EXPRESSIONS = {
    "element": ("(a[i] * 2 + b)[1]", lambda a, b: (a * 2 + b)[:, 1]),
    "negative index": ("(a[i] - b)[-1]", lambda a, b: (a - b)[:, -1]),
    "ufunc and reduction": ("np.sqrt(a[i] * b).sum()", lambda a, b: np.sqrt(a * b).sum(1)),
    "nested": ("(np.abs(a[i] - 3) * (b + 1)).max()",
               lambda a, b: (np.abs(a - 3) * (b + 1)).max(1)),
    "comparison": ("(a[i] > b * 2).sum()", lambda a, b: (a > b * 2).sum(1)),
    "two dimensions": ("(a * b)[i, 2]", lambda a, b: (a * b)[:, 2]),
    "broadcast column": ("(a + a[:, 0:1])[i, 3]", lambda a, b: (a + a[:, 0:1])[:, 3]),
    "dot": ("np.dot(a[i] - 1, b * 2)", lambda a, b: (a - 1) @ (b * 2)),
    "attributes": ("(a + b).shape[1] + (a[i] + b).size + (a + b).ndim + len(a[i] * 2)",
                   lambda a, b: np.full(3, 4 + 4 + 2 + 4)),
}  # fmt: skip


@pytest.mark.parametrize("name", EXPRESSIONS)
def test_expression(run, name):
    expression, reference = EXPRESSIONS[name]
    out = rows(f"out[i, 0] = {expression}", A, B, run=run)
    np.testing.assert_allclose(out[:, 0], reference(A, B), rtol=1e-6)


def test_assigning_and_updating_with_expressions(run):
    out = rows(
        """
        out[i, :4] = a[i] * b - 1
        out[i, 4:] = np.minimum(a[i], 3)
        out[i, :4] += b
        out[i, 4:] *= 2
        """,
        A, B, run=run,
    )  # fmt: skip
    np.testing.assert_allclose(out[:, :4], A * B - 1 + B, rtol=1e-6)
    np.testing.assert_array_equal(out[:, 4:], np.minimum(A, 3) * 2)


def test_whole_array_update_in_one_invocation(run):
    out = A.copy()

    @nv.jit
    def scale(a, factors):
        if nv.global_id(0) == 0:
            a *= factors
            a[:, 0] = a[:, 0] + 1

    run(scale, 1, out, B)
    want = A * B
    want[:, 0] += 1
    np.testing.assert_allclose(out, want, rtol=1e-6)


def test_expression_errors(run):
    with pytest.raises(ValueError, match="could not be broadcast"):
        rows("out[i, 0] = (a[i] + b[:3]).sum()", A, B, run=run)
    with pytest.raises(ValueError, match="reads the array it is assigned to"):
        rows("out[i] = a[i]\nout[i, 1:] = out[i, :-1] * 2", A, run=run, columns=4)
    with pytest.raises(errors.TypingError, match="indexed with 1 integers"):
        rows("out[i, 0] = (a[i] * 2)[1:][0]", A, run=run)
