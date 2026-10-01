"""``math`` functions and NumPy ufuncs on scalars, checked against NumPy."""

import math

import numpy as np
import pytest
from numba.core import errors

import numba_vulkan as nv

f32 = np.float32
_rng = np.random.default_rng(0)
POS = _rng.uniform(0.1, 3.0, 64).astype(f32)
UNIT = _rng.uniform(-0.95, 0.95, 64).astype(f32)
SYM = _rng.uniform(-3, 3, 64).astype(f32)
INTS = _rng.integers(-20, 20, 64).astype(np.int64)
NONZERO = np.where(INTS == 0, 3, INTS)
SPECIAL = np.array([0.0, -0.0, 1.5, -2.5, np.inf, -np.inf, np.nan, 2.0] * 8, dtype=f32)
REVERSED = SPECIAL[::-1].copy()


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


def test_float64_versions_inherit_the_32_bit_restriction():
    x = np.linspace(0.1, 1, 8)
    with pytest.raises(errors.NumbaError, match="float64"):
        _elementwise(math.log1p, 1).forall(8)(x, np.zeros(8))


UNARY_UFUNCS = {
    "sin": SYM, "cos": SYM, "tan": UNIT, "arcsin": UNIT, "arccos": UNIT, "arctan": SYM,
    "sinh": SYM, "cosh": SYM, "tanh": SYM, "arcsinh": SYM, "arccosh": POS + 1, "arctanh": UNIT,
    "exp": SYM, "exp2": SYM, "expm1": UNIT, "log": POS, "log2": POS, "log10": POS, "log1p": UNIT,
    "sqrt": POS, "fabs": SYM, "floor": SYM, "ceil": SYM, "trunc": SYM,
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
