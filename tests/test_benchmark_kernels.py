"""The kernels of the benchmark suite compute the right results."""

import pathlib
import sys

import numpy as np
import pytest

import numba_vulkan as nv

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "benchmarks"))
import kernels  # noqa: E402


@pytest.fixture(scope="module")
def built():
    return kernels.build("vulkan")


@pytest.mark.parametrize(
    "name, tile", [("matmul", 16), ("matmul_blocked", 64)], ids=["tiled", "blocked"]
)
def test_matrix_products_at_odd_sizes(built, device, name, tile):
    rng = np.random.default_rng(0)
    a = rng.random((100, 70), dtype=np.float32)
    b = rng.random((70, 90), dtype=np.float32)
    c = np.zeros((100, 90), dtype=np.float32)
    groups = (-(-90 // tile), -(-100 // tile))
    built[name][groups, (16, 16), device](a, b, c)
    np.testing.assert_allclose(c, a @ b, rtol=1e-5)
