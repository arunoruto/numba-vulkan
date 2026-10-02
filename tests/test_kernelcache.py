"""The on-disk cache of compiled kernels."""

import math
import os

import numpy as np
import pytest

import numba_vulkan as nv
from numba_vulkan import codegen, kernelcache

f32 = np.float32
X = np.linspace(0.1, 3, 16, dtype=f32)
TABLE_A = np.array([1, 2, 3, 4], dtype=f32)
TABLE_B = np.array([10, 20, 30, 40], dtype=f32)


def body(a, out):
    i = nv.global_id(0)
    if i < a.shape[0]:
        if a[i] < 0:
            raise ValueError("negative")
        out[i] = math.sin(a[i]) * 2 + TABLE_A[i % 4]


def with_table(table):
    """The same code for every table, as a later process would compile it."""

    def kernel(a, out):
        i = nv.global_id(0)
        if i < a.shape[0]:
            out[i] = math.sin(a[i]) * 2 + table[i % 4]

    return kernel


@pytest.fixture
def cache(tmp_path, monkeypatch, device):
    """Turn the cache on, in an empty directory; report backend runs."""
    # Opening a device for the first time compiles the driver probe
    # (numba_vulkan.probes), which is not a backend run of the test's.
    nv.get_device(device)
    monkeypatch.setenv(kernelcache.ENV_VAR, "1")
    monkeypatch.setenv("NUMBA_VULKAN_CACHE_DIR", str(tmp_path))
    emitted = []
    emit = codegen.emitter.emit
    monkeypatch.setattr(
        codegen.emitter, "emit", lambda ir: emitted.append(1) or emit(ir)
    )

    def entries():
        found = []
        for folder, _, files in os.walk(tmp_path / "kernels"):
            found += [os.path.join(folder, name) for name in files]
        return found

    return emitted, entries


def launch(function, device, **options):
    kernel = nv.jit(**options)(function)
    out = np.zeros_like(X)
    try:
        kernel.forall(16, device=device)(X, out)
    except nv.VulkanSupportError as exc:
        pytest.skip(str(exc))
    (compiled,) = kernel._kernels.values()
    return out, compiled


def test_second_compilation_comes_from_the_cache(cache, device):
    emitted, entries = cache
    want = np.sin(X) * 2 + TABLE_A[np.arange(16) % 4]
    first, compiled = launch(body, device)
    assert len(emitted) == 1 and len(entries()) == 1
    second, cached = launch(body, device)
    assert len(emitted) == 1  # the backend did not run again
    np.testing.assert_allclose(first, want, rtol=1e-6)
    np.testing.assert_array_equal(first, second)
    assert cached.spirv == compiled.spirv and cached.llvm_ir == compiled.llvm_ir
    assert cached.written_bindings == compiled.written_bindings == {0, 2}
    assert cached.capabilities == compiled.capabilities
    assert list(cached.constants) == list(compiled.constants) == [3]


def test_exceptions_work_from_the_cache(cache, device):
    launch(body, device)
    kernel = nv.jit(body)
    x = X.copy()
    x[5] = -1
    with pytest.raises(ValueError, match="negative"):
        kernel.forall(16, device=device)(x, np.zeros_like(x))


def test_table_contents_are_not_part_of_the_entry(cache, device):
    emitted, entries = cache
    first, _ = launch(with_table(TABLE_A), device)
    second, _ = launch(with_table(TABLE_B), device)
    # One entry serves both, and each run sees the values of its own table.
    assert len(emitted) == 1 and len(entries()) == 1
    base = np.sin(X) * 2
    np.testing.assert_allclose(first, base + TABLE_A[np.arange(16) % 4], rtol=1e-6)
    np.testing.assert_allclose(second, base + TABLE_B[np.arange(16) % 4], rtol=1e-6)


@pytest.mark.float64
def test_settings_are_part_of_the_key(cache, device):
    emitted, entries = cache
    launch(body, device)
    launch(body, device, narrow=True)
    launch(body, device, fastmath=True)
    launch(body, device, boundscheck=True)
    assert len(emitted) == 4 and len(entries()) == 4
    for options in ({}, {"narrow": True}, {"fastmath": True}, {"boundscheck": True}):
        launch(body, device, **options)
    assert len(emitted) == 4


def test_damaged_entry_is_ignored(cache, device):
    emitted, entries = cache
    first, _ = launch(body, device)
    (path,) = entries()
    with open(path, "r+b") as fh:
        fh.truncate(100)
    second, _ = launch(body, device)
    assert len(emitted) == 2
    np.testing.assert_array_equal(first, second)
    launch(body, device)  # rewritten, so usable again
    assert len(emitted) == 2


def test_cache_can_be_turned_off(cache, device, monkeypatch):
    emitted, entries = cache
    monkeypatch.setenv(kernelcache.ENV_VAR, "0")
    launch(body, device)
    launch(body, device)
    assert len(emitted) == 2 and entries() == []


def test_unwritable_cache_directory_is_harmless(cache, device, monkeypatch):
    emitted, _ = cache
    monkeypatch.setenv("NUMBA_VULKAN_CACHE_DIR", "/proc/numba-vulkan-no-such-place")
    out, _ = launch(body, device)
    np.testing.assert_allclose(
        out, np.sin(X) * 2 + TABLE_A[np.arange(16) % 4], rtol=1e-6
    )


def test_normalise_removes_what_differs_between_processes():
    one = '; ModuleID = "k$1"\ncall @"_ZN1kB2v7B46abc"() call @"_ZN1hB2v9B46abc"()\n'
    two = '; ModuleID = "k$5"\ncall @"_ZN1kB3v12B46abc"() call @"_ZN1hB3v31B46abc"()\n'
    assert kernelcache.normalise(one) == kernelcache.normalise(two)
    swapped = two.replace("v12", "v31", 1)
    assert kernelcache.normalise(one) != kernelcache.normalise(swapped)
    assert kernelcache.key("a", 1) != kernelcache.key("a", 2)
    assert kernelcache.key("a", 1) != kernelcache.key("b", 1)
