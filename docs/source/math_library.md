# The math library

Vulkan shaders come with a small math library, GLSL.std.450. It has two
gaps that matter for Numba code:

- its transcendental functions (`sin`, `exp`, `log`, `pow`...) exist for
  32-bit floats only, while Numba types Python floats as `float64`;
- it lacks `erf`, `gamma`, `log1p`, `expm1`, `hypot` and the inverse
  hyperbolic functions altogether.

CUDA has the same problem and solves it with libdevice, a math library that
NVIDIA ships as LLVM bitcode and that numba-cuda links into every kernel.
numba-vulkan does the same with **libclc**.

## libclc

[libclc](https://libclc.llvm.org/) is LLVM's implementation of the OpenCL C
library, distributed as LLVM bitcode. One of its builds, `clspv--.bc`, is
made for [clspv](https://github.com/google/clspv), Google's compiler from
OpenCL C to Vulkan compute shaders, and is therefore written for exactly
the constraints of this target. clspv bundles the same file.

When a kernel calls a math function, numba-vulkan emits a call to the
libclc symbol (for example `_Z3sind` for `sin(double)`), links the bitcode
into the kernel, removes everything unused and inlines the rest.

What this buys, measured on the three test devices:

| | built-in (GLSL.std.450) | libclc |
| --- | --- | --- |
| `float32` accuracy | driver-dependent; `sin` is off by 1e-4 on Intel/Mesa | about 1e-7, i.e. the last digit |
| `float64` | not available | about 1e-16 |
| same result on every device | no | very nearly: differences stay within the last digit |
| speed | fastest | see {doc}`benchmarks` |
| compile time per kernel | 0.4 s | 0.8 s |

### Which functions come from libclc

`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`,
`tanh`, `asinh`, `acosh`, `atanh`, `exp`, `exp2`, `expm1`, `log`, `log2`,
`log10`, `log1p`, `pow`, `hypot`, `erf` and `erfc`, in both precisions, as
`math` functions, as the matching NumPy ufuncs and through the `**`
operator.

`sqrt`, `fabs`, `floor`, `ceil` and `trunc` use the built-in versions.
`fabs`, `floor`, `ceil` and `trunc` are exact; `sqrt` is exact for
`float64`, but for `float32` NVIDIA's and Intel's drivers return the
neighbour of the correctly rounded result for some arguments (KI-33).

`lgamma` comes from libclc as well. `gamma` does not: libclc 22 computes it
as `exp(lgamma(x))`, which loses precision as the argument grows (1800 ulp
near 170 in `float64`) and returns infinity instead of tiny values for
large negative arguments. This package uses a port of the newer upstream
`tgamma` (from AMD's OCML) instead, in both precisions and also with
`fastmath`. It is within 7 ulp in both precisions on all three tested
devices; numba-cuda and Numba's CPU target are within 3 ulp.

In `float64`, `exp`, `expm1`, `sinh`, `cosh` and `erfc` are not libclc's
either, except for arguments where libclc's versions are accurate: on
Intel's and Mesa's drivers, they lose up to a few hundred ulp for large
arguments. This package reduces the argument of `exp` to `k ln 2 + r`
exactly and lets libclc compute only `exp(r)`, which gives 1 ulp on every
tested device; the other four are built on that (`erfc` with fdlibm's
formula, as in libclc) and measure 1 to 2 ulp.

### Choosing speed over accuracy

```python
@nv.jit(fastmath=True)
def kernel(x, out):
    ...
```

With `fastmath`, `float32` math uses the device's built-in functions, and
the driver is allowed to reassociate arithmetic and to fuse multiply-adds.
Results then differ between devices. `float64` math still comes from
libclc, since there is no alternative.

### Installing libclc

The file numba-vulkan needs is `clspv--.bc`. It is looked for in this
order:

1. the path in the `NUMBA_VULKAN_LIBCLC` environment variable (the file, or
   the directory containing it);
2. `numba_vulkan/data/` inside the installed package, where wheels carry a
   copy (with its licence, Apache-2.0 with LLVM exceptions);
3. `/usr/share/clc`, `/usr/lib/clc`, `/usr/lib64/clc`, `/usr/local/share/clc`.

The devenv shell sets the variable. In a source checkout elsewhere,
`make libclc` puts the file into `src/numba_vulkan/data/` (from Nix if it
is installed, from conda-forge otherwise); a distribution's libclc package
or a clspv or LLVM build works too. `numba_vulkan.libclc.version()` tells
which libclc is in use, for example
`'22.1.8 (nixpkgs 774debe7a0d1b496e35677ad955a1011c6ff74f3)'` for the copy
in a wheel.

The first kernel that needs libclc rewrites it into a form that links
quickly (a few seconds, once) and stores the result in
`~/.cache/numba-vulkan`, or in the directory named by
`NUMBA_VULKAN_CACHE_DIR`. The file is named after a hash of the original,
so a new libclc gets a new one; old ones can be deleted at any time.

:::{important}
LLVM can only read bitcode written by the same or an older LLVM. llvmlite
0.50 contains LLVM 22, so libclc must come from LLVM 22 or older.
:::

Without libclc, `float32` math falls back to the built-in functions and to
this package's Python implementations, and `float64` transcendental
functions raise an error (or run in `float32` precision with
`narrow_math=True`).

### What had to be adapted

libclc's clspv build expects the clspv compiler to finish the job. A few
things that clspv would handle are done by {py:mod}`numba_vulkan.legalize`
and {py:mod}`numba_vulkan.codegen` instead:

| libclc contains | what numba-vulkan does |
| --- | --- |
| functions marked `noinline`, with the `spir_func` calling convention | drops `noinline`; calls with the same convention |
| lookup tables in OpenCL's constant address space | moves them to the `Private` storage class |
| integers read from byte tables at arbitrary offsets | assembles them from single bytes |
| `llvm.ctlz`, `llvm.fshl`, `llvm.fmuladd`, `llvm.minimumnum` | expands or renames them |
| `__clc_mul_hi`, which clspv supplies | implements it with a 64-bit multiplication |

Functions that need other helpers clspv supplies (`rsqrt`, `copysign`,
`sqrt`) are not taken from libclc; a kernel that would need one fails with
an error naming the helper.

### Drivers that compute differently

Two kinds of `float64` behaviour broke libclc and kernels on some drivers.
{py:mod}`numba_vulkan.probes` runs a small kernel on each device the first
time it is used, and stores the result per device and driver version in
`probes.json` in the cache directory:

| Behaviour | Seen on | Workaround |
| --- | --- | --- |
| `Fma` rounds the product before adding (Vulkan allows this), in both precisions | llvmpipe | `llvm.fma` computed in software, exactly (`legalize.emulate_fma`) |
| `Trunc` returns values of at least 2**24 unchanged; `RoundEven` rounds halves towards zero (vectorised code only) | llvmpipe (Mesa 26.1) | both computed from `Floor` (`legalize.emulate_rounding64`) |

libclc relies on a fused `Fma` and on `Trunc` for the argument reduction
of `sin`, `cos` and `tan`, which on llvmpipe were off by up to 1e19 ulp
(`float64`) and 4e9 ulp (`float32`) beyond small arguments; with the
workarounds they are within 1 to 3 ulp there, as on the other devices. The rounding bug also affected `math.trunc`,
`np.trunc`, `round` and `np.rint` in kernels. Kernels for devices that do
not need the workarounds are compiled as before.
`NUMBA_VULKAN_SOFT_FMA` and `NUMBA_VULKAN_SOFT_ROUNDING` (`1` or `0`)
override the probe for every device.

Intel's driver returns `+0.0` for `trunc` and `ceil` of values between -1
and 0; rounding functions now take the sign of their argument on every
device, which is always correct for them.

## Alternatives that were considered

**Taichi.** Its Vulkan backend exposes only what GLSL.std.450 has, limited
to 32 bits, and has no `erf`, `gamma`, `log1p` or `hypot`. `isnan` and
`isinf` are bit tests written in Python, as here.

**Python Vulkan packages** ([vulkan](https://pypi.org/project/vulkan/),
[Kompute](https://github.com/KomputeProject/kompute),
[wgpu-py](https://github.com/pygfx/wgpu-py)). These are bindings and
runtimes: the user supplies shaders written in GLSL or WGSL, with those
languages' built-in functions. None of them brings a math library.

**Compiling a C math library to bitcode** (musl, openlibm, LLVM-libc,
CORE-MATH). Possible in principle, and it is how libdevice and libclc came
to be. General-purpose libms lean on pointers, unions and `errno`, all of
which a shader cannot express, so each would need the kind of porting that
libclc's clspv build has already received. LLVM-libc has GPU builds, but
for NVIDIA and AMD targets, not for SPIR-V.

**Letting another compiler do the work** (clspv itself, or a GLSL/WGSL
compiler such as glslang or naga). That would mean generating OpenCL C or
GLSL source instead of LLVM IR, which gives up Numba's lowering. GLSL and
WGSL compilers would also bring only the same built-in functions.

**Driver-side emulation.** Mesa can emulate `float64` arithmetic on GPUs
without it, but that is about the basic operations, is driver-specific,
and does not add library functions.

**Double-float arithmetic** (representing a `float64` as two `float32`
values). Not a library to reuse here, but the likely way to offer real
`float64` precision on devices without the `shaderFloat64` feature, where
kernels are currently narrowed to `float32` (KI-17).

## Open points

- **Where libclc comes from.** Wheels bundle `clspv--.bc` (2.7 MB), built
  by `nix/libclc.nix` from the LLVM 22 sources in nixpkgs. nixpkgs-unstable
  removed its own `llvmPackages.libclc` in August 2026, because nothing
  used it any more: Mesa moved to its own fork, `mesa-libclc`, which pins
  libclc since its interface to OpenCL runtimes is not stable, and which no
  longer builds the `clspv` target. The derivation here builds only that
  target; for LLVM 22.1.8 its output is byte for byte the file nixpkgs
  26.05 still ships. Without Nix, conda-forge's package is used. LLVM 23 also
  renames the `clspv` target to the `spirv-unknown-vulkan` triple, built as
  part of LLVM's runtimes build, which will change the file name once
  llvmlite moves to LLVM 23. Once the bundled libclc has the new `tgamma`,
  this package's port can be dropped, provided the libclc version is as
  accurate without fused multiply-add (the port avoids it; Mesa's fork
  fuses).
- **Compile time.** The whole library is parsed and linked for every
  kernel that uses it. Caching a reduced copy would remove most of the
  0.4 s this costs.
