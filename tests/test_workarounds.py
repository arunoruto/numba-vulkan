"""Workarounds for drivers' float64 behaviour (see numba_vulkan.probes)."""

import math

import numpy as np
import pytest

import numba_vulkan as nv
from numba_vulkan import legalize, libclc, probes

needs_libclc = pytest.mark.skipif(
    not libclc.available(), reason="libclc is not installed"
)


@nv.jit
def _trig(x, s, c, t):
    i = nv.global_id(0)
    if i < x.shape[0]:
        s[i] = math.sin(x[i])
        c[i] = math.cos(x[i])
        t[i] = math.tan(x[i])


@nv.jit
def _rounding(x, t, r, e):
    i = nv.global_id(0)
    if i < x.shape[0]:
        t[i] = np.trunc(x[i])
        r[i] = round(x[i], 0)
        e[i] = np.rint(x[i])


def _trig_arguments():
    """Arguments from 1e-3 to 1e300 in magnitude, of both signs."""
    rng = np.random.default_rng(5)
    x = np.exp(rng.uniform(math.log(1e-3), math.log(1e300), 3000))
    return x * rng.choice([-1.0, 1.0], x.size)


def _check_trig(run, x):
    s, c, t = (np.zeros_like(x) for _ in range(3))
    run(_trig, x.size, x, s, c, t)
    # libclc's results are within 1 to 3 ulp; llvmpipe's were off by up to
    # 1e19 ulp (float64) and 4e9 ulp (float32) before the workarounds.
    rtol = 1e-15 if x.dtype == np.float64 else 5e-7
    for got, fn in ((s, math.sin), (c, math.cos), (t, math.tan)):
        want = np.vectorize(fn)(x.astype(np.float64)).astype(x.dtype)
        np.testing.assert_allclose(got, want, rtol=rtol, atol=0)


# Doubles of at least 2**24, which llvmpipe's vectorised Trunc returned
# unchanged, up to 2**52, where every double is an integer; and halves and
# small negative values, which its RoundEven rounded wrongly.
LARGE = np.array(
    [
        2.0**24 + 1.25, -(2.0**24) - 1.25, 3e7 + 0.5, -3e7 - 0.5,
        1e9 + 0.75, 1e15 + 0.5, -(2.0**52) + 0.5, 2.0**52, -0.5, 0.75,
        0.5, 1.5, 2.5, -2.5, -0.2, 3.5, 2.0**24 + 0.5, 2.0**24 + 1.5,
    ]
)  # fmt: skip


def _check_rounding(run):
    t, r, e = (np.zeros_like(LARGE) for _ in range(3))
    run(_rounding, LARGE.size, LARGE, t, r, e)
    np.testing.assert_array_equal(t, np.trunc(LARGE))
    np.testing.assert_array_equal(np.signbit(t), np.signbit(np.trunc(LARGE)))
    np.testing.assert_array_equal(r, np.round(LARGE))
    np.testing.assert_array_equal(e, np.rint(LARGE))
    np.testing.assert_array_equal(np.signbit(e), np.signbit(np.rint(LARGE)))


@needs_libclc
def test_float64_trig_over_the_whole_range(run):
    _check_trig(run, _trig_arguments())


def test_float32_trig_over_the_whole_range(run):
    x = _trig_arguments().astype(np.float32)
    _check_trig(run, x[np.isfinite(x)])


def test_float64_trunc_and_round_of_large_values(run):
    _check_rounding(run)


@pytest.fixture
def forced(device, monkeypatch):
    """The device, made to apply both workarounds whether it needs them."""
    dev = nv.get_device(device)
    monkeypatch.setattr(
        dev, "mode", dev.mode._replace(soft_fma=True, soft_rounding=True)
    )
    return dev


def test_float32_trig_is_correct_with_the_workarounds_forced(run, forced):
    x = _trig_arguments()[:500].astype(np.float32)
    _check_trig(run, x[np.isfinite(x)])


@needs_libclc
def test_trig_is_correct_with_the_workarounds_forced(run, forced):
    if not forced.info.float64:
        pytest.skip("no float64")
    _check_trig(run, _trig_arguments()[:500])


def test_rounding_is_correct_with_the_workarounds_forced(run, forced):
    if not forced.info.float64:
        pytest.skip("no float64")
    _check_rounding(run)


def test_environment_overrides_the_probe(monkeypatch):
    monkeypatch.setenv(probes.FMA_ENV_VAR, "1")
    monkeypatch.setenv(probes.ROUNDING_ENV_VAR, "0")
    # With both forced, the probe kernel is not run at all.
    monkeypatch.setattr(probes, "_probe_kernel", None)
    assert probes.workarounds(nv.get_device()) == {
        "soft_fma": True,
        "soft_rounding": False,
    }


def test_fma_rewrite_replaces_every_call():
    text = (
        "define double @f(double %a, double %b, double %c) {\n"
        "  %r = tail call nnan double @llvm.fma.f64(double %a, double %b, double %c)\n"
        "  ret double %r\n}\n"
    )
    out = legalize.emulate_fma(text)
    assert "call" not in out.split("define")[1].split("ret")[0]
    assert "%r = select i1" in out
    # float works the same way, and can be left alone.
    single = text.replace("double", "float").replace(".f64", ".f32")
    assert "fmul float %a, 4097.0" in legalize.emulate_fma(single)
    assert legalize.emulate_fma(single, ("double",)) == single


def test_rounding_rewrite_declares_what_it_uses():
    text = (
        "define double @f(double %x) {\n"
        "  %r = call double @llvm.trunc.f64(double %x)\n"
        "  ret double %r\n}\n"
        "declare double @llvm.trunc.f64(double)\n"
    )
    out = legalize.emulate_rounding64(text)
    assert "@llvm.trunc.f64(double %x)" not in out
    for name in ("fabs", "floor", "copysign"):
        assert out.count(f"declare double @llvm.{name}.f64(") == 1


def test_probe_results_are_stored(tmp_path, monkeypatch):
    monkeypatch.setenv("NUMBA_VULKAN_CACHE", "1")
    monkeypatch.setenv("NUMBA_VULKAN_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv(probes.FMA_ENV_VAR, raising=False)
    monkeypatch.delenv(probes.ROUNDING_ENV_VAR, raising=False)
    device = nv.get_device()
    if not device.info.float64:
        pytest.skip("devices without float64 are not probed")
    calls = []
    measure = probes._measure
    monkeypatch.setattr(probes, "_measure", lambda d: calls.append(d) or measure(d))
    first = probes.workarounds(device)
    assert probes.workarounds(device) == first
    assert len(calls) == 1
    assert (tmp_path / "probes.json").exists()
    # The probe agrees with what the device was set up with.
    assert first == {
        "soft_fma": device.mode.soft_fma,
        "soft_rounding": device.mode.soft_rounding,
    }


def test_a_failing_probe_warns_and_applies_no_workarounds(monkeypatch):
    monkeypatch.setenv("NUMBA_VULKAN_CACHE", "0")
    monkeypatch.delenv(probes.FMA_ENV_VAR, raising=False)
    monkeypatch.delenv(probes.ROUNDING_ENV_VAR, raising=False)

    def broken(device):
        raise RuntimeError("driver crashed")

    monkeypatch.setattr(probes, "_measure", broken)
    with pytest.warns(UserWarning, match="could not probe"):
        result = probes.workarounds(nv.get_device())
    assert result == {"soft_fma": False, "soft_rounding": False}
