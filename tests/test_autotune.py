"""Choosing the fastest candidate launch with nv.autotune."""

import numpy as np
import pytest

import numba_vulkan as nv


f32 = np.float32


@nv.jit
def scale(x, out, rounds):
    i = nv.global_id(0)
    if i < x.shape[0]:
        v = x[i]
        for _ in range(rounds):
            v = v * f32(0.5) + f32(0.5)
        out[i] = v * f32(0) + x[i] * f32(2)


def _tuned(runs):
    """A tuned function whose 'slow' candidate does far more work."""

    def double(x, out):
        def launch(rounds, name):
            def run():
                runs.append(name)
                scale.forall(x.shape[0], device=x.device)(x, out, rounds)

            return run

        yield "slow", launch(20000, "slow")
        yield "fast", launch(1, "fast")

    return nv.autotune(double, repeat=2)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("NUMBA_VULKAN_CACHE", "1")
    monkeypatch.setenv("NUMBA_VULKAN_CACHE_DIR", str(tmp_path))
    return tmp_path


def test_the_fastest_candidate_is_chosen_and_kept(device, cache):
    runs = []
    tuned = _tuned(runs)
    x = nv.to_device(np.arange(4096, dtype=f32), device)
    out = nv.device_array_like(x)
    tuned(x, out)
    assert runs.count("slow") == 3 and runs.count("fast") == 4  # 1 + 2 + chosen
    np.testing.assert_array_equal(out.copy_to_host(), np.arange(4096) * 2)
    assert list(tuned.choices.values()) == ["fast"]

    runs.clear()
    tuned(x, out)
    assert runs == ["fast"]  # no timing on later calls

    # Another process (here: another tuner) reads the choice from disk.
    runs.clear()
    _tuned(runs)(x, out)
    assert runs == ["fast"]
    assert (cache / "autotune.json").exists()


def test_a_new_key_is_tuned_again(device, cache):
    runs = []
    tuned = _tuned(runs)
    for n in (256, 512):
        x = nv.to_device(np.ones(n, dtype=f32), device)
        tuned(x, nv.device_array_like(x))
    assert len(tuned.choices) == 2
    assert runs.count("slow") == 6


def test_without_a_cache_nothing_is_written(device, tmp_path, monkeypatch):
    monkeypatch.setenv("NUMBA_VULKAN_CACHE", "0")
    monkeypatch.setenv("NUMBA_VULKAN_CACHE_DIR", str(tmp_path))
    runs = []
    x = nv.to_device(np.ones(64, dtype=f32), device)
    _tuned(runs)(x, nv.device_array_like(x))
    assert not (tmp_path / "autotune.json").exists()
