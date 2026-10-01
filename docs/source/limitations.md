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
- tuples, `if`/`while`/`for ... in range(...)`, `break`/`continue`
- `min`, `max`, `abs`, the `math` module, `**` with integer exponents
- calling `@nv.jit` and `@njit` functions; `@overload(target="vulkan")`
- dispatch grids with one to three dimensions

## What does not work

64-bit transcendental functions
: Vulkan's math library (GLSL.std.450) defines `sin`, `cos`, `exp`, `log`,
  `pow` and friends for 32-bit floats only. Calling them with `float64`
  raises an error. `@nv.jit(narrow_math=True)` evaluates them in `float32`
  instead. `sqrt`, `fabs`, `floor` and `ceil` work in both precisions.

Slices, views and NumPy functions
: Only full integer indexing is supported. Array slicing, array-valued
  expressions, NumPy functions and array allocation are not.

Exceptions and recursion
: Errors raised inside a kernel are silently dropped, and out-of-bounds
  accesses are not checked. Recursive functions cannot be compiled, because
  everything is inlined.

Devices without `float64`, `int64` or `int8`
: These are optional Vulkan features. Numba types Python literals as
  `float64`/`int64`, and the SPIR-V backend currently forces `int8`, so most
  kernels need all three. Desktop GPUs provide them; many mobile GPUs and
  Apple devices do not. Launching a kernel on a device that lacks a feature
  raises {py:class}`~numba_vulkan.errors.VulkanSupportError`.

Data transfer
: Every call copies all arguments to the device and all written arrays back.
  There are no device arrays yet, which dominates the run time of
  memory-bound kernels (see {doc}`benchmarks`).

Workgroup size
: The workgroup size is fixed per grid dimensionality (64, 8×8 or 4×4×4).

## Differences between drivers

Results are not bit-identical across devices. In particular, `float32`
`sin` on Intel's Mesa driver is only accurate to about 1e-4, against about
1e-6 on NVIDIA and llvmpipe. `round()` of halfway values is wrong on
llvmpipe.

## Tested hardware

The test suite has only been run on an NVIDIA TITAN X (Pascal), an Intel UHD
Graphics 630 and llvmpipe, all on Linux. AMD, Apple (MoltenVK), Windows and
mobile GPUs are untested.
