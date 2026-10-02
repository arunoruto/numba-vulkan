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

CASES = {}


def known_issue(issue, x, expected):
    """Register a kernel ``(x, out)`` that should compute ``expected(x)``."""

    def register(kernel):
        CASES[issue] = (kernel, x, expected)
        return kernel

    return register


@known_issue("KI-04 fancy indexing", XF, lambda x: x)
def _(x, out):
    i = nv.global_id(0)
    if i < x.size:
        out[i] = x[x > 0][0]


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
            out.astype(np.float64),
            np.asarray(want, dtype=np.float64),
            rtol=2e-3,
            atol=1e-5,
        )


@nv.jit
def _gamma(x, out):
    i = nv.global_id(0)
    if i < x.size:
        out[i] = math.gamma(x[i])


# KI-30: libclc 22's tgamma is exp(lgamma), whose error grows with the size of
# lgamma, and its reflection overflows for large negative arguments.
GAMMA = {
    "float64 large": (np.linspace(100.25, 170.75, 64), 1e-14),
    "float64 tiny negative": (np.array([-171.3, -171.5, -175.5]), 1e-14),
    "float32 tiny negative": (np.array([-34.2, -34.5, -35.5], dtype=f32), 1e-5),
}


@pytest.mark.parametrize("case", GAMMA)
@pytest.mark.xfail(strict=True, reason="KI-30 in docs/source/known_issues.md")
def test_ki30_gamma(case):
    x, rtol = GAMMA[case]
    out = np.zeros_like(x)
    _gamma.forall(x.size)(x, out)
    want = np.array([math.gamma(float(v)) for v in x])
    # Results below the normal range are only as precise as their spacing.
    atol = 2 * float(np.finfo(x.dtype).smallest_subnormal)
    np.testing.assert_allclose(out.astype(np.float64), want, rtol=rtol, atol=atol)
