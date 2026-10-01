"""Executable list of known issues (see docs/source/known_issues.md).

Every case is expected to fail. The marks are strict, so fixing an issue
turns its test red until the case is moved to the regular test suite and
the entry is removed from the documentation.
"""

import math

import numpy as np
import pytest

import numba_vulkan as nv

f32 = np.float32
N = 16
XF = np.linspace(0.5, 3.0, N, dtype=np.float32)
XD = np.linspace(0.5, 3.0, N)
TABLE = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

CASES = {}


def known_issue(issue, x, expected):
    """Register a kernel ``(x, out)`` that should compute ``expected(x)``."""

    def register(kernel):
        CASES[issue] = (kernel, x, expected)
        return kernel

    return register


@known_issue("KI-01 float64 ** float", XD, lambda x: x**2.5)
def _(x, out):
    i = nv.global_id(0)
    if i < x.size:
        out[i] = x[i] ** 2.5


@known_issue("KI-01 float32 ** Python float literal", XF, lambda x: x**2.5)
def _(x, out):
    i = nv.global_id(0)
    if i < x.size:
        out[i] = x[i] ** 2.5


@known_issue("KI-01 float64 transcendental (math.sin)", XD, np.sin)
def _(x, out):
    i = nv.global_id(0)
    if i < x.size:
        out[i] = math.sin(x[i])


@known_issue("KI-02 NumPy ufunc on a scalar (np.sqrt)", XF, np.sqrt)
def _(x, out):
    i = nv.global_id(0)
    if i < x.size:
        out[i] = np.sqrt(x[i])


@known_issue("KI-03 math.hypot", XF, lambda x: np.hypot(x, 2))
def _(x, out):
    i = nv.global_id(0)
    if i < x.size:
        out[i] = math.hypot(x[i], f32(2.0))


@known_issue("KI-03 math.log1p", XF, np.log1p)
def _(x, out):
    i = nv.global_id(0)
    if i < x.size:
        out[i] = math.log1p(x[i])


@known_issue("KI-03 math.erf", XF, lambda x: np.array([math.erf(v) for v in x]))
def _(x, out):
    i = nv.global_id(0)
    if i < x.size:
        out[i] = math.erf(x[i])


@known_issue("KI-03 math.isfinite", XF, np.ones_like)
def _(x, out):
    i = nv.global_id(0)
    if i < x.size:
        out[i] = 1.0 if math.isfinite(x[i]) else 0.0


@known_issue("KI-04 slicing", XF, lambda x: np.append(x[1:], 0))
def _(x, out):
    i = nv.global_id(0)
    if i < x.size - 1:
        out[i] = x[1:][i]


@known_issue("KI-04 array method (x.sum())", XF, lambda x: np.full_like(x, x.sum()))
def _(x, out):
    i = nv.global_id(0)
    if i < x.size:
        out[i] = x.sum()


@known_issue("KI-04 iterating over an array", XF, lambda x: np.full_like(x, x.sum()))
def _(x, out):
    i = nv.global_id(0)
    if i < x.size:
        total = f32(0.0)
        for value in x:
            total += value
        out[i] = total


@known_issue("KI-05 allocating an array (np.zeros)", XF, lambda x: x)
def _(x, out):
    i = nv.global_id(0)
    if i < x.size:
        tmp = np.zeros(4, dtype=np.float32)
        tmp[0] = x[i]
        out[i] = tmp[0]


@known_issue("KI-06 global constant array", XF, lambda x: x + TABLE[np.arange(N) % 4])
def _(x, out):
    i = nv.global_id(0)
    if i < x.size:
        out[i] = x[i] + TABLE[i % 4]


@known_issue("KI-07 complex numbers", XF, lambda x: x * x)
def _(x, out):
    i = nv.global_id(0)
    if i < x.size:
        z = complex(x[i], x[i])
        out[i] = (z * z).imag / 2


@known_issue("KI-08 print()", XF, lambda x: x)
def _(x, out):
    i = nv.global_id(0)
    if i < x.size:
        print(i)
        out[i] = x[i]


@known_issue("KI-09 float16 array", XF.astype(np.float16), lambda x: x)
def _(x, out):
    i = nv.global_id(0)
    if i < x.size:
        out[i] = x[i]


@known_issue("KI-09 non-contiguous array", np.arange(2 * N, dtype=f32)[::2], lambda x: x)
def _(x, out):
    i = nv.global_id(0)
    if i < x.size:
        out[i] = x[i]


@known_issue(
    "KI-09 structured array",
    np.zeros(N, dtype=[("a", np.float32), ("b", np.float32)]),
    lambda x: x,
)
def _(x, out):
    i = nv.global_id(0)
    if i < x.size:
        out[i].a = x[i].b


@pytest.mark.parametrize("issue", CASES)
@pytest.mark.xfail(strict=True, reason="documented in docs/source/known_issues.md")
def test_known_issue(issue):
    kernel, x, expected = CASES[issue]
    out = np.zeros_like(x)
    nv.jit(kernel).forall(x.shape[0])(x, out)
    want = expected(x)
    if x.dtype.names is None:
        np.testing.assert_allclose(
            out.astype(np.float64), np.asarray(want, dtype=np.float64), rtol=2e-3, atol=1e-5
        )
