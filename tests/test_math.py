"""``math`` functions and NumPy ufuncs on scalars, checked against NumPy."""

import math

import numpy as np
import pytest
from numba import float32 as nb_float32
from numba.core import errors

import numba_vulkan as nv
from numba_vulkan import libclc

f32 = np.float32
_rng = np.random.default_rng(0)
POS = _rng.uniform(0.1, 3.0, 64).astype(f32)
UNIT = _rng.uniform(-0.95, 0.95, 64).astype(f32)
SYM = _rng.uniform(-3, 3, 64).astype(f32)
INTS = _rng.integers(-20, 20, 64).astype(np.int64)
NONZERO = np.where(INTS == 0, 3, INTS)
SPECIAL = np.array([0.0, -0.0, 1.5, -2.5, np.inf, -np.inf, np.nan, 2.0] * 8, dtype=f32)
REVERSED = SPECIAL[::-1].copy()
# Halves, which round to even, and ordinary values.
HALVES = np.concatenate([np.arange(-8, 8) + 0.5, SYM[:48]]).astype(f32)


def _elementwise(fn, nargs):
    """Kernel applying ``fn`` to corresponding elements of its inputs."""
    if nargs == 1:

        def kernel(a, out):
            i = nv.global_id(0)
            if i < a.shape[0]:
                out[i] = fn(a[i])

    else:

        def kernel(a, b, out):
            i = nv.global_id(0)
            if i < a.shape[0]:
                out[i] = fn(a[i], b[i])

    return nv.jit(kernel)


_KERNELS = {}


def _check(run, key, fn, args, want, rtol=3e-3, atol=1e-5):
    # Kernels are cached so that each one is compiled once for all devices.
    if key not in _KERNELS:
        _KERNELS[key] = _elementwise(fn, len(args))
    out = np.zeros(want.shape, dtype=want.dtype)
    run(_KERNELS[key], out.size, *args, out)
    if want.dtype == np.bool_:
        np.testing.assert_array_equal(out, want)
    else:
        np.testing.assert_allclose(out, want, rtol=rtol, atol=atol, equal_nan=True)


def _py(fn):
    """Reference for a ``math`` function, evaluated in double precision."""
    return lambda *arrays: np.vectorize(fn)(
        *(a.astype(np.float64) for a in arrays)
    ).astype(f32)


MATH = {
    "hypot": (math.hypot, np.hypot, (SYM, POS)),
    "log1p": (math.log1p, np.log1p, (UNIT,)),
    "expm1": (math.expm1, np.expm1, (UNIT,)),
    "exp2": (math.exp2, np.exp2, (SYM,)),
    "asinh": (math.asinh, np.arcsinh, (SYM,)),
    "acosh": (math.acosh, np.arccosh, (POS + 1,)),
    "atanh": (math.atanh, np.arctanh, (UNIT,)),
    "degrees": (math.degrees, np.degrees, (SYM,)),
    "radians": (math.radians, np.radians, (SYM,)),
    "copysign": (math.copysign, np.copysign, (POS, SYM)),
    "erf": (math.erf, _py(math.erf), (SYM,)),
    "erfc": (math.erfc, _py(math.erfc), (SYM,)),
    "lgamma": (math.lgamma, _py(math.lgamma), (POS * 4,)),
    "gamma": (math.gamma, _py(math.gamma), (POS * 3,)),
    "isnan": (math.isnan, np.isnan, (SPECIAL,)),
    "isinf": (math.isinf, np.isinf, (SPECIAL,)),
    "isfinite": (math.isfinite, np.isfinite, (SPECIAL,)),
}


@pytest.mark.parametrize("name", MATH)
def test_math_function(run, name):
    fn, reference, args = MATH[name]
    _check(run, ("math", name), fn, args, reference(*args))


def test_math_small_arguments_keep_their_precision(run):
    # log1p and expm1 exist for arguments where log(1 + x) would lose digits.
    tiny = np.array([1e-7, -1e-7, 3e-6, -2e-5], dtype=f32)
    _check(
        run, ("math", "log1p"), math.log1p, (tiny,), np.log1p(tiny), rtol=1e-5, atol=0
    )
    _check(
        run, ("math", "expm1"), math.expm1, (tiny,), np.expm1(tiny), rtol=1e-5, atol=0
    )


def test_float64_classification_and_sign_functions(run):
    x = np.array([0.0, -1.5, np.inf, -np.inf, np.nan, 3.0])
    _check(run, ("math", "isnan"), math.isnan, (x,), np.isnan(x))
    _check(run, ("math", "isfinite"), math.isfinite, (x,), np.isfinite(x))
    y = np.array([-1.0, 2.0, -3.0, 4.0, -5.0, -0.0])
    finite = np.array([0.5, -1.5, 2.5, -3.5, 4.5, 6.0])
    _check(
        run,
        ("math", "copysign"),
        math.copysign,
        (finite, y),
        np.copysign(finite, y),
        rtol=1e-12,
    )
    _check(
        run, ("math", "hypot"), math.hypot, (finite, y), np.hypot(finite, y), rtol=1e-12
    )


def test_float64_versions_need_libclc(monkeypatch):
    monkeypatch.setattr(libclc, "available", lambda: False)
    x = np.linspace(0.1, 1, 8)
    with pytest.raises(errors.NumbaError, match="needs libclc"):
        _elementwise(math.log1p, 1).forall(8)(x, np.zeros(8))


def test_libclc_version_from_the_version_file(tmp_path, monkeypatch):
    (tmp_path / "clspv--.bc").write_bytes(b"BC\xc0\xde")
    (tmp_path / "libclc-version.txt").write_text("22.1.8 (conda-forge x)\n")
    monkeypatch.setattr(libclc, "find_bitcode", lambda: str(tmp_path / "clspv--.bc"))
    assert libclc.version() == "22.1.8 (conda-forge x)"


@pytest.mark.parametrize("package", ["libclc", "libclc-clspv"])
def test_libclc_version_from_a_nix_store_path(tmp_path, monkeypatch, package):
    path = tmp_path / f"abc-{package}-22.1.8" / "share" / "clc" / "clspv--.bc"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"BC\xc0\xde")
    monkeypatch.setattr(libclc, "find_bitcode", lambda: str(path))
    assert libclc.version() == "22.1.8 (nix)"


def test_libclc_version_without_libclc(monkeypatch):
    monkeypatch.setattr(libclc, "find_bitcode", lambda: None)
    assert libclc.version() is None


DOUBLE = {
    "sin": (math.sin, np.sin, SYM),
    "cos": (math.cos, np.cos, SYM),
    "tan": (math.tan, np.tan, UNIT),
    "asin": (math.asin, np.arcsin, UNIT),
    "acos": (math.acos, np.arccos, UNIT),
    "atan": (math.atan, np.arctan, SYM),
    "sinh": (math.sinh, np.sinh, SYM),
    "cosh": (math.cosh, np.cosh, SYM),
    "tanh": (math.tanh, np.tanh, SYM),
    "asinh": (math.asinh, np.arcsinh, SYM),
    "acosh": (math.acosh, np.arccosh, POS + 1),
    "atanh": (math.atanh, np.arctanh, UNIT),
    "exp": (math.exp, np.exp, SYM),
    "expm1": (math.expm1, np.expm1, UNIT),
    "log": (math.log, np.log, POS),
    "log2": (math.log2, np.log2, POS),
    "log10": (math.log10, np.log10, POS),
    "log1p": (math.log1p, np.log1p, UNIT),
    "erf": (math.erf, np.vectorize(math.erf), SYM),
    "erfc": (math.erfc, np.vectorize(math.erfc), SYM),
    "gamma": (math.gamma, np.vectorize(math.gamma), POS * 3),
    "lgamma": (math.lgamma, np.vectorize(math.lgamma), POS * 4),
}


@pytest.mark.skipif(not libclc.available(), reason="libclc is not installed")
@pytest.mark.parametrize("name", DOUBLE)
def test_float64_math_function(run, name):
    fn, reference, x = DOUBLE[name]
    x = x.astype(np.float64)
    _check(run, ("double", name), fn, (x,), reference(x), rtol=1e-13, atol=1e-15)


@pytest.mark.skipif(not libclc.available(), reason="libclc is not installed")
def test_gamma_functions_of_negative_arguments(run):
    x = np.array([-0.5, -1.5, -2.25, -3.75, -7.1, 0.5, 1.0, 2.0])
    gamma, lgamma = np.vectorize(math.gamma)(x), np.vectorize(math.lgamma)(x)
    _check(run, ("double", "gamma"), math.gamma, (x,), gamma, rtol=1e-13)
    _check(run, ("double", "lgamma"), math.lgamma, (x,), lgamma, rtol=1e-13, atol=1e-15)
    x = x.astype(f32)
    gamma, lgamma = _py(math.gamma)(x), _py(math.lgamma)(x)
    _check(run, ("math", "gamma"), math.gamma, (x,), gamma, rtol=2e-6)
    _check(run, ("math", "lgamma"), math.lgamma, (x,), lgamma, rtol=2e-6, atol=1e-6)


def _gamma_arguments(lo, hi, dtype):
    """Arguments across gamma's range, away from the poles."""
    x = np.linspace(lo, hi, 2000).astype(dtype)
    return x[(x > 0) | (np.abs(x - np.round(x)) > 1e-3)]


# Tolerances: the worst error measured on the Titan X, the UHD 630 and
# llvmpipe is 7 ulp in float32 and 6 ulp in float64. Results below the
# normal range may be flushed to zero.
GAMMA_RANGES = {
    "float64": (-185.5, 171.5, np.float64, 3e-15),
    "float32": (-41.5, 35.0, f32, 1e-6),
}


@pytest.mark.skipif(not libclc.available(), reason="libclc is not installed")
@pytest.mark.parametrize("case", GAMMA_RANGES)
def test_gamma_over_its_range(run, case):
    lo, hi, dtype, rtol = GAMMA_RANGES[case]
    x = _gamma_arguments(lo, hi, dtype)
    want = np.vectorize(math.gamma)(x.astype(np.float64)).astype(dtype)
    atol = 2 * float(np.finfo(dtype).tiny)
    _check(run, ("gamma range", case), math.gamma, (x,), want, rtol=rtol, atol=atol)


def test_gamma_with_fastmath_keeps_its_accuracy(run):
    x = _gamma_arguments(-41.5, 35.0, f32)
    want = np.vectorize(math.gamma)(x.astype(np.float64)).astype(f32)
    kernel = nv.jit(fastmath=True)(_elementwise(math.gamma, 1).py_func)
    out = np.zeros_like(x)
    run(kernel, x.size, x, out)
    # NVIDIA reorders arithmetic in fastmath kernels, which costs a few
    # digits but must not overflow.
    np.testing.assert_allclose(out, want, rtol=1e-5, atol=2 * float(np.finfo(f32).tiny))


# libclc's float64 exp, expm1, sinh, cosh and erfc lose up to 670 ulp for
# large arguments on llvmpipe and 200 on the UHD 630; their replacements in
# mathimpl measure 1 to 2 ulp. The references are accurate to about 1 ulp.
EXP_FAMILY = {
    "exp": (math.exp, np.exp, (-745.0, 709.7)),
    "expm1": (math.expm1, np.expm1, (-50.0, 709.7)),
    "sinh": (math.sinh, np.sinh, (-710.4, 710.4)),
    "cosh": (math.cosh, np.cosh, (-710.4, 710.4)),
    "erfc": (math.erfc, np.vectorize(math.erfc), (-7.0, 27.2)),
}


@pytest.mark.skipif(not libclc.available(), reason="libclc is not installed")
@pytest.mark.parametrize("name", EXP_FAMILY)
def test_float64_exp_family_over_its_range(run, name):
    fn, reference, (lo, hi) = EXP_FAMILY[name]
    x = np.concatenate([np.linspace(lo, hi, 3000), np.linspace(-1.5, 1.5, 301)])
    want = reference(x)
    tiny = float(np.finfo(np.float64).tiny)
    _check(run, ("exp family", name), fn, (x,), want, rtol=1e-15, atol=2 * tiny)


@pytest.mark.skipif(not libclc.available(), reason="libclc is not installed")
@pytest.mark.parametrize("name", EXP_FAMILY)
def test_float64_exp_family_special_values(run, name):
    fn, reference, _ = EXP_FAMILY[name]
    x = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 800.0, -800.0, 710.6])
    with np.errstate(over="ignore"):
        want = reference(x)
    out = np.zeros_like(x)
    run(_elementwise(fn, 1), x.size, x, out)
    np.testing.assert_array_equal(out, want)
    np.testing.assert_array_equal(np.signbit(out), np.signbit(want))


@pytest.mark.parametrize("dtype", [np.float64, f32])
def test_gamma_special_values(run, dtype):
    if dtype == np.float64 and not libclc.available():
        pytest.skip("libclc is not installed")
    x = np.array([0.0, -0.0, -3.0, np.inf, -np.inf, np.nan, 200.0, -200.5], dtype=dtype)
    out = np.zeros_like(x)
    run(_elementwise(math.gamma, 1), x.size, x, out)
    want = np.array([np.inf, -np.inf, np.nan, np.inf, np.nan, np.nan, np.inf, -0.0])
    np.testing.assert_array_equal(out, want.astype(dtype))
    assert np.signbit(out[[1, 7]]).all()


@nv.jit
def _power(a, b):
    return a**b


@pytest.mark.skipif(not libclc.available(), reason="libclc is not installed")
def test_float64_binary_math_and_power(run):
    x, y = POS.astype(np.float64), SYM.astype(np.float64)
    _check(run, ("double", "atan2"), math.atan2, (y, x), np.arctan2(y, x), rtol=1e-13)
    _check(run, ("double", "hypot"), math.hypot, (y, x), np.hypot(y, x), rtol=1e-13)
    _check(run, ("double", "pow"), math.pow, (x, y), np.power(x, y), rtol=1e-13)
    _check(run, ("double", "operator"), _power, (x, y), x**y, rtol=1e-13)
    _check(run, ("double", "np.power"), np.power, (x, y), np.power(x, y), rtol=1e-13)


@pytest.mark.skipif(not libclc.available(), reason="libclc is not installed")
def test_float32_math_is_accurate_and_identical_across_devices(run):
    # libclc replaces the drivers' own sin, which is only good to 1e-4 on some.
    _check(run, ("exact", "sin"), math.sin, (SYM,), np.sin(SYM), rtol=3e-7, atol=1e-7)
    _check(run, ("exact", "exp"), math.exp, (SYM,), np.exp(SYM), rtol=3e-7)


def test_fastmath_uses_the_device_functions(run):
    @nv.jit(fastmath=True)
    def kernel(a, out):
        i = nv.global_id(0)
        if i < a.shape[0]:
            out[i] = math.sin(a[i]) * math.exp(a[i])

    out = np.zeros_like(SYM)
    run(kernel, SYM.size, SYM, out)
    np.testing.assert_allclose(out, np.sin(SYM) * np.exp(SYM), rtol=3e-3, atol=1e-4)
    compiled = kernel.compile((nb_float32[::1], nb_float32[::1]))
    assert "@llvm.sin.f32" in compiled.llvm_ir


UNARY_UFUNCS = {
    "sin": SYM, "cos": SYM, "tan": UNIT, "arcsin": UNIT, "arccos": UNIT, "arctan": SYM,
    "sinh": SYM, "cosh": SYM, "tanh": SYM, "arcsinh": SYM, "arccosh": POS + 1, "arctanh": UNIT,
    "exp": SYM, "exp2": SYM, "expm1": UNIT, "log": POS, "log2": POS, "log10": POS, "log1p": UNIT,
    "sqrt": POS, "fabs": SYM, "floor": SYM, "ceil": SYM, "trunc": SYM, "rint": HALVES,
    "degrees": SYM, "radians": SYM, "rad2deg": SYM, "deg2rad": SYM,
    "negative": SYM, "absolute": SYM, "sign": SPECIAL, "square": SYM, "reciprocal": POS,
    "isnan": SPECIAL, "isinf": SPECIAL, "isfinite": SPECIAL, "logical_not": SYM,
}  # fmt: skip
BINARY_UFUNCS = {
    "add": (SYM, POS), "subtract": (SYM, POS), "multiply": (SYM, POS), "divide": (SYM, POS),
    "floor_divide": (SYM, POS), "remainder": (SYM, POS), "arctan2": (SYM, POS),
    "hypot": (SYM, POS), "copysign": (POS, SYM), "fmod": (SYM, POS), "power": (POS, SYM),
    "maximum": (SPECIAL, REVERSED), "minimum": (SPECIAL, REVERSED),
    "fmax": (SPECIAL, REVERSED), "fmin": (SPECIAL, REVERSED),
    "less": (SYM, POS), "greater_equal": (SYM, POS), "not_equal": (SPECIAL, SPECIAL),
    "logical_and": (SYM, SPECIAL),
}  # fmt: skip
INTEGER_UFUNCS = {
    "add": (INTS, NONZERO), "floor_divide": (INTS, NONZERO), "remainder": (INTS, NONZERO),
    "power": (np.abs(INTS) % 6, np.abs(NONZERO) % 4), "bitwise_and": (INTS, NONZERO),
    "bitwise_xor": (INTS, NONZERO), "left_shift": (np.abs(INTS), np.abs(NONZERO) % 5),
    "maximum": (INTS, NONZERO), "less": (INTS, NONZERO),
}  # fmt: skip


def _ufunc_case(run, kind, name, args):
    ufunc = getattr(np, name)
    with np.errstate(all="ignore"):
        want = ufunc(*args)
    _check(run, (kind, name), ufunc, args, want)


@pytest.mark.parametrize("name", UNARY_UFUNCS)
def test_unary_ufunc_on_scalars(run, name):
    _ufunc_case(run, "unary", name, (UNARY_UFUNCS[name],))


@pytest.mark.parametrize("name", BINARY_UFUNCS)
def test_binary_ufunc_on_scalars(run, name):
    _ufunc_case(run, "binary", name, BINARY_UFUNCS[name])


@pytest.mark.parametrize("name", INTEGER_UFUNCS)
def test_integer_ufunc_on_scalars(run, name):
    _ufunc_case(run, "integer", name, INTEGER_UFUNCS[name])
