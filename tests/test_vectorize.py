"""numba.vectorize and numba.guvectorize with target="vulkan"."""

import math

import numpy as np
import pytest
from numba import guvectorize, vectorize

import numba_vulkan as nv
from numba_vulkan.vectorizers import VulkanGUFunc, VulkanUFunc

f32 = np.float32
RNG = np.random.default_rng(1)


@vectorize(["float32(float32, float32)", "float64(float64, float64)"], target="vulkan")
def hypot(a, b):
    return math.sqrt(a * a + b * b)


@vectorize(target="vulkan")
def axpb(a, b):
    return a * b + 1


@vectorize(["int64(int64)"], target="vulkan")
def collatz_steps(n):
    steps = 0
    while n > 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


def test_decorators_build_vulkan_ufuncs():
    assert isinstance(hypot, VulkanUFunc) and isinstance(axpb, VulkanUFunc)
    assert hypot.nin == 2 and hypot.nout == 1
    assert hypot.types == ["ff->f", "dd->d"]
    assert hypot.__name__ == "hypot"


def test_explicit_signatures_select_by_argument_types(device):
    x = RNG.random((3, 4)).astype(f32)
    y = RNG.random((3, 4)).astype(f32)
    got = hypot(x, y, device=device)
    assert got.dtype == f32
    np.testing.assert_allclose(got, np.hypot(x, y), rtol=1e-6)
    # float64 inputs take the second signature, int32 is converted to it
    got = hypot(x.astype(np.float64), np.arange(4, dtype=np.int32), device=device)
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, np.hypot(x, np.arange(4)), rtol=1e-6)


def test_no_matching_signature(device):
    with pytest.raises(TypeError, match="no signature for arguments of type"):
        collatz_steps(np.arange(4.0), device=device)
    with pytest.raises(TypeError, match="takes 1 inputs, got 2"):
        collatz_steps(1, 2, device=device)


def test_loops_inside_the_scalar_function(device):
    n = np.arange(1, 50, dtype=np.int64)
    want = [collatz_steps.pyfunc(int(k)) for k in n]
    np.testing.assert_array_equal(collatz_steps(n, device=device), want)


@pytest.mark.parametrize(
    "shapes",
    [((5,), (5,)), ((3, 4), (4,)), ((3, 1), (1, 4)), ((2, 3, 4), (3, 1)), ((), (3,))],
    ids=["same", "row", "outer", "3d", "0d"],
)
def test_broadcasting(device, shapes):
    a = RNG.random(shapes[0]).astype(f32)
    b = RNG.random(shapes[1]).astype(f32)
    got = axpb(a, b, device=device)
    np.testing.assert_allclose(got, a * b + 1, rtol=1e-6)
    assert got.shape == np.broadcast_shapes(*shapes)


@vectorize(target="vulkan")
def times(a, b):
    return a * b


def test_types_follow_numpy_without_signatures(device):
    x = np.arange(6, dtype=f32)
    assert times(x, 2.0, device=device).dtype == f32  # Python scalars are weak
    assert times(x, np.float64(2), device=device).dtype == np.float64
    # Inside the function, Numba's rules apply: the literal 1 is an int64.
    assert axpb(x, 2.0, device=device).dtype == np.float64
    # Numba widens integer products to int64; the scalar is not the cause.
    i = times(np.arange(4, dtype=np.int32), 3, device=device)
    assert i.dtype == np.int64
    np.testing.assert_array_equal(i, np.arange(4) * 3)
    result = times(f32(3), 2.0, device=device)
    assert np.isscalar(result) and result == 6 and result.dtype == f32


def test_device_arrays_in_and_out(device):
    x = RNG.random(100).astype(f32)
    dx = nv.to_device(x, device)
    got = axpb(dx, x)
    assert isinstance(got, nv.DeviceArray) and got.device is dx.device
    np.testing.assert_allclose(got.copy_to_host(), x * x + 1, rtol=1e-6)
    # chained without copies
    again = axpb(got, dx)
    np.testing.assert_allclose(again.copy_to_host(), (x * x + 1) * x + 1, rtol=1e-6)


def test_out_argument(device):
    x = RNG.random((4, 5)).astype(f32)
    out = np.zeros((4, 5), dtype=f32)
    assert axpb(x, x, out=out, device=device) is out
    np.testing.assert_allclose(out, x * x + 1, rtol=1e-6)
    backing = np.zeros((4, 10), dtype=f32)
    strided = backing[:, ::2]
    axpb(x, 2.0, out=strided, device=device)
    np.testing.assert_allclose(backing[:, ::2], 2 * x + 1, rtol=1e-6)
    dout = nv.device_array((4, 5), f32, device)
    assert axpb(x, x, out=dout) is dout
    np.testing.assert_allclose(dout.copy_to_host(), x * x + 1, rtol=1e-6)
    with pytest.raises(ValueError, match="out has shape"):
        axpb(x, x, out=np.zeros(3, dtype=f32), device=device)


def test_jit_options_are_passed_on(device):
    @vectorize(["float32(float32)"], target="vulkan", fastmath=True)
    def fast_sin(x):
        return math.sin(x)

    x = np.linspace(-3, 3, 64, dtype=f32)
    np.testing.assert_allclose(fast_sin(x, device=device), np.sin(x), atol=2e-3)
    (kernel,) = fast_sin._kernels.values()
    (compiled,) = kernel._kernels.values()
    assert "@llvm.sin.f32" in compiled.llvm_ir


# -- guvectorize -----------------------------------------------------------------


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(n),(n)->()",
    target="vulkan",
)
def dot(a, b, out):
    out[0] = np.dot(a, b)


@guvectorize(
    ["void(float64[:, :], float64[:], float64[:])"], "(m,n),(n)->(m)", target="vulkan"
)
def matvec(matrix, vector, out):
    for i in range(matrix.shape[0]):
        total = 0.0
        for j in range(matrix.shape[1]):
            total += matrix[i, j] * vector[j]
        out[i] = total


@guvectorize(["void(float32[:], float32, float32[:], float32[:])"], "(n),()->(n),()",
             target="vulkan")  # fmt: skip
def scale_and_sum(a, factor, scaled, total):
    total[0] = 0
    for i in range(a.shape[0]):
        scaled[i] = a[i] * factor
        total[0] += scaled[i]


@guvectorize("(n),()->(n)", target="vulkan")
def shift(a, by, out):
    for i in range(a.shape[0]):
        out[i] = a[i] + by


def test_gufunc_reductions_over_the_core_dimension(device):
    assert isinstance(dot, VulkanGUFunc) and dot.nin == 2 and dot.nout == 1
    a = RNG.random((7, 3)).astype(f32)
    b = RNG.random(3).astype(f32)
    got = dot(a, b, device=device)
    assert got.shape == (7,) and got.dtype == f32
    np.testing.assert_allclose(got, a @ b, rtol=1e-6)
    got = dot(a.astype(np.float64), b.astype(np.float64), device=device)
    assert got.dtype == np.float64


def test_gufunc_with_matrices_and_loop_broadcasting(device):
    matrix = RNG.random((4, 1, 3, 5))
    vectors = RNG.random((2, 5))
    got = matvec(matrix, vectors, device=device)
    assert got.shape == (4, 2, 3)
    np.testing.assert_allclose(got, np.einsum("abij,bj->abi", matrix, vectors))


def test_gufunc_with_several_outputs_and_scalar_inputs(device):
    a = RNG.random((5, 4)).astype(f32)
    factor = np.array([1, 2, 3, 4, 5], dtype=f32)
    scaled, total = scale_and_sum(a, factor, device=device)
    np.testing.assert_allclose(scaled, a * factor[:, None], rtol=1e-6)
    np.testing.assert_allclose(total, (a * factor[:, None]).sum(1), rtol=1e-5)
    scaled, total = scale_and_sum(a, 2.0, device=device)
    np.testing.assert_allclose(total, 2 * a.sum(1), rtol=1e-5)


def test_gufunc_outputs_passed_in(device):
    a = RNG.random((3, 4)).astype(f32)
    out = np.zeros((3, 4), dtype=f32)
    assert shift(a, f32(1), out, device=device) is out
    np.testing.assert_allclose(out, a + 1, rtol=1e-6)
    dout = nv.device_array((3, 4), f32, device)
    assert shift(nv.to_device(a, device), f32(2), out=dout) is dout
    np.testing.assert_allclose(dout.copy_to_host(), a + 2, rtol=1e-6)


def test_gufunc_errors(device):
    with pytest.raises(TypeError, match="outputs must be passed"):
        shift(np.zeros((2, 3), dtype=f32), f32(1), device=device)
    with pytest.raises(ValueError, match="dimension n of 'dot' is 3 in one"):
        dot(np.zeros((2, 3), f32), np.zeros(4, f32), device=device)
    with pytest.raises(ValueError, match="needs at least 1 dimensions"):
        dot(np.float32(1), np.zeros(4, f32), device=device)
    with pytest.raises(ValueError, match="output 0 has shape"):
        dot(np.zeros((2, 3), f32), np.zeros(3, f32), np.zeros(3, f32), device=device)
    with pytest.raises(TypeError, match="takes 3 arguments, but the layout"):

        @guvectorize("(n)->(n)", target="vulkan")
        def wrong(a, b, c):
            pass


def test_gufunc_with_empty_loop(device):
    got = dot(np.zeros((0, 3), f32), np.zeros(3, f32), device=device)
    assert got.shape == (0,)


def test_device_array_reshape(device):
    array = nv.to_device(np.arange(12, dtype=f32), device)
    view = array.reshape(3, -1)
    assert view.shape == (3, 4) and view.ravel().shape == (12,)
    del array
    np.testing.assert_array_equal(view.copy_to_host(), np.arange(12).reshape(3, 4))
    with pytest.raises(ValueError):
        view.reshape(5, 2)
