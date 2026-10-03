"""Executable list of known issues (see docs/source/known_issues.md).

Every case is expected to fail (KI-32 only on macOS arm64, KI-33 only on
NVIDIA and Intel GPUs). The marks are strict, so fixing an issue
turns its test red until the case is moved to the regular test suite and
the entry is removed from the documentation.
"""

import platform
import sys

import numpy as np
import pytest

import numba_vulkan as nv
from numba_vulkan import codegen

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


# KI-32: llvmlite's own backend (not NUMBA_VULKAN_LLC), in a child process.
SELECT_AFTER_FCMP_SELECT = f"""
target triple = "{codegen.TRIPLE}"
define i32 @main(i1 %c, float %f, i32 %i) {{
  %k = fcmp ogt float %f, 0.0
  %a = select i1 %k, float 0.0, float 1.0
  %b = select i1 %c, i32 %i, i32 0
  ret i32 %b
}}
"""


@pytest.mark.xfail(
    sys.platform == "darwin" and platform.machine() == "arm64",
    strict=True,
    reason="KI-32, documented in docs/source/known_issues.md",
)
def test_llvmlite_backend_on_macos_arm64():
    emitter = codegen.Emitter()
    try:
        assert emitter.emit(SELECT_AFTER_FCMP_SELECT)[:4] == b"\x03\x02\x23\x07"
    finally:
        emitter.close()


# KI-33: the drivers' float32 sqrt is not correctly rounded everywhere.
_INEXACT_SQRT_VENDORS = {0x10DE, 0x8086}  # NVIDIA, Intel


def _vendor(info):
    import vulkan as vk

    return vk.vkGetPhysicalDeviceProperties(info.handle).vendorID


@pytest.mark.parametrize(
    "info",
    [
        pytest.param(
            info,
            marks=pytest.mark.xfail(
                _vendor(info) in _INEXACT_SQRT_VENDORS,
                strict=True,
                reason="KI-33, documented in docs/source/known_issues.md",
            ),
            id=info.name,
        )
        for info in nv.list_devices()
    ],
)
def test_float32_sqrt_is_correctly_rounded(info):
    import math

    @nv.jit
    def root(x, out):
        i = nv.global_id(0)
        if i < x.shape[0]:
            out[i] = math.sqrt(x[i])

    x = (np.random.default_rng(0).random(4096) * 1000).astype(f32)
    out = np.zeros_like(x)
    root.forall(x.size, device=info.index)(x, out)
    np.testing.assert_array_equal(out, np.sqrt(x))
