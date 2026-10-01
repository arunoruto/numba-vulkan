"""Control flow that SPIR-V cannot express directly: early exits,
short-circuit conditions and returns from inside loops."""

import math

import numpy as np
import pytest

import numba_vulkan as nv

f32 = np.float32
X = np.linspace(0, 5, 80, dtype=f32)


@nv.jit
def piecewise(x):
    if x < 1:
        return math.sin(x)
    if x < 2:
        return math.cos(x)
    if x < 3:
        return math.sqrt(x)
    return x


def piecewise_reference(x):
    return np.where(
        x < 1, np.sin(x), np.where(x < 2, np.cos(x), np.where(x < 3, np.sqrt(x), x))
    )


@nv.jit
def piecewise_kernel(a, out):
    i = nv.global_id(0)
    if i < a.shape[0]:
        out[i] = piecewise(a[i])


@nv.jit
def piecewise_in_loop_kernel(a, out):
    i = nv.global_id(0)
    if i < a.shape[0]:
        total = f32(0.0)
        for j in range(4):
            total += piecewise(a[i] + f32(j) * f32(0.5))
        out[i] = total


@nv.jit
def early_exit_kernel(a, out):
    i = nv.global_id(0)
    if i >= a.shape[0]:
        return
    if a[i] < 1:
        out[i] = 1
        return
    if a[i] < 2:
        out[i] = 2
        return
    n = 0
    while n < 10:
        if a[i] * n > 20:
            break
        if n == 7:
            out[i] = -1
            return
        n += 1
    out[i] = n


def early_exit_reference(a):
    out = np.zeros_like(a)
    for i, v in enumerate(a):
        if v < 1:
            out[i] = 1
        elif v < 2:
            out[i] = 2
        else:
            n = 0
            while n < 10 and v * n <= 20 and n != 7:
                n += 1
            out[i] = -1 if (n == 7 and v * n <= 20) else n
    return out


@nv.jit
def short_circuit_kernel(a, out):
    i = nv.global_id(0)
    if i < a.shape[0]:
        v = a[i]
        if v > 4 or math.sin(v) > 0:
            out[i, 0] = math.cos(v)
        else:
            out[i, 0] = math.sqrt(v)
        if v > 1 and math.sin(v) > 0 and math.cos(v) < f32(0.5):
            out[i, 1] = v * v
        elif v > 3:
            out[i, 1] = math.sqrt(v)
        else:
            out[i, 1] = -v


@nv.jit
def complex_kernel(a, real, imag, magnitude):
    i = nv.global_id(0)
    if i < a.shape[0]:
        z = complex(a[i], f32(0.5))
        w = z * z + z / complex(f32(1.0), a[i])
        real[i] = w.real
        imag[i] = w.imag
        magnitude[i] = abs(w)


def test_chain_of_early_returns(run):
    out = np.zeros_like(X)
    run(piecewise_kernel, X.size, X, out)
    np.testing.assert_allclose(out, piecewise_reference(X), rtol=2e-3, atol=1e-5)


def test_early_returns_inside_a_loop_body(run):
    out = np.zeros_like(X)
    run(piecewise_in_loop_kernel, X.size, X, out)
    want = sum(piecewise_reference(X + f32(j) * f32(0.5)) for j in range(4))
    np.testing.assert_allclose(out, want, rtol=2e-3, atol=1e-4)


def test_kernel_returning_early_and_from_inside_a_loop(run):
    out = np.zeros_like(X)
    run(early_exit_kernel, X.size, X, out)
    np.testing.assert_array_equal(out, early_exit_reference(X))


def test_short_circuit_conditions_with_else_branches(run):
    out = np.zeros((X.size, 2), dtype=f32)
    run(short_circuit_kernel, X.size, X, out)
    first = np.where((X > 4) | (np.sin(X) > 0), np.cos(X), np.sqrt(X))
    second = np.where(
        (X > 1) & (np.sin(X) > 0) & (np.cos(X) < 0.5),
        X * X,
        np.where(X > 3, np.sqrt(X), -X),
    )
    np.testing.assert_allclose(out[:, 0], first, rtol=2e-3, atol=1e-5)
    np.testing.assert_allclose(out[:, 1], second, rtol=2e-3, atol=1e-5)


def test_complex_arithmetic_on_scalars(run):
    real, imag, magnitude = np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)
    run(complex_kernel, X.size, X, real, imag, magnitude)
    z = X.astype(np.complex64) + np.complex64(0.5j)
    w = z * z + z / (1 + 1j * X.astype(np.complex64))
    np.testing.assert_allclose(real, w.real, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(imag, w.imag, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(magnitude, np.abs(w), rtol=1e-4)


# Seeds of tests/fuzz_control_flow.py. Most of these programs were rejected
# before control flow was restructured (see docs/source/how_it_works.md).
FUZZ_SEEDS = [1, 3, 4, 7, 8, 12, 13, 16, 24, 27]


@pytest.mark.parametrize("seed", FUZZ_SEEDS)
def test_random_control_flow_program(device, seed):
    from fuzz_control_flow import run as run_program

    failure, source = run_program(seed, [device])
    assert failure is None, f"{failure}\n{source}"
