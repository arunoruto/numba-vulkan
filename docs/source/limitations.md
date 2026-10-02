# Limitations

This page describes what the target supports and the constraints that follow
from how Vulkan shaders work. Bugs and missing pieces that could be fixed
are tracked in {doc}`known_issues`.

## What works

- `int32`, `int64`, `float32`, `float64` and `bool` scalars
- C-contiguous N-d arrays of `bool`, 8 to 64-bit integers, `float32` and
  `float64`, with integer indexing
  (including negative indices), `.shape`, `.size`, `.ndim` and `len()`
- arithmetic with Python semantics, comparisons, bit operations, casts
- tuples, `if`/`while`/`for ... in range(...)`
- `min`, `max`, `abs`, `**` with integer exponents
- the `math` module, including `hypot`, `log1p`, `expm1`, the inverse
  hyperbolic functions, `erf`, `erfc`, `gamma`, `lgamma`, `copysign`,
  `isnan`, `isinf` and `isfinite` (`math.fmod` is not typed by Numba; use
  `np.fmod` or `%`)
- NumPy functions applied to scalars (`np.sqrt(x[i])`, `np.maximum(a, b)`...):
  arithmetic, comparison, logical, bitwise and math ufuncs
- complex numbers as local values (not as array elements)
- early `return`s, `break`/`continue`, and conditions combined with
  `and`/`or`
- calling `@nv.jit` and `@njit` functions; `@overload(target="vulkan")`
- dispatch grids with one to three dimensions

## What does not work

`float64` math without libclc
: Vulkan's math library (GLSL.std.450) defines `sin`, `cos`, `exp`, `log`,
  `pow` and friends for 32-bit floats only. numba-vulkan takes the
  double-precision versions from libclc (see {doc}`math_library`); if that
  is not installed, calling them with `float64` raises an error, unless
  `@nv.jit(narrow_math=True)` asks for `float32` precision.

Slices, views and NumPy functions
: Only full integer indexing is supported. Array slicing, array-valued
  expressions, NumPy functions and array allocation are not.

Exceptions and recursion
: An exception raised inside a kernel is raised by the launch once the
  kernel has finished; it does not stop the other invocations, and
  `try`/`except` is not available. Out-of-bounds accesses are checked only
  with `boundscheck=True`, and arithmetic errors such as integer division
  by zero are not reported. Recursive functions cannot be compiled, because
  everything is inlined.

Devices without `float64`, `int64` or `int8`
: These are optional Vulkan features. Numba types Python literals as
  `float64`/`int64`, and the SPIR-V backend currently forces `int8`, so most
  kernels need all three. Desktop GPUs provide them; many mobile GPUs and
  Apple devices do not. Launching a kernel on a device that lacks a feature
  raises {py:class}`~numba_vulkan.errors.VulkanSupportError`.

Data transfer
: NumPy arguments are copied to the device on every call, and written ones
  back. Use device arrays to keep data on the device (see {doc}`usage`);
  they support transfers only, no indexing or arithmetic from Python.

Workgroup size
: The workgroup size is fixed per grid dimensionality (64, 8×8 or 4×4×4).

## Differences between drivers

Float arithmetic follows IEEE rules (shaders are compiled without
reassociation or fused multiply-add), and math functions come from libclc,
so results agree across devices to within the last digit. That no longer
holds with `fastmath=True` or without libclc, when the drivers' own
functions are used. In that case `float32` `sin` on Intel's Mesa driver is only
accurate to about 1e-4, against about 1e-6 on NVIDIA and llvmpipe. `round()` of halfway values is wrong on
llvmpipe.

## Tested hardware

The test suite has only been run on an NVIDIA TITAN X (Pascal), an Intel UHD
Graphics 630 and llvmpipe, all on Linux. AMD, Apple (MoltenVK), Windows and
mobile GPUs are untested.
