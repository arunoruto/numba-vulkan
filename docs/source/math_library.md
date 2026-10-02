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

`sqrt`, `fabs`, `floor`, `ceil` and `trunc` use the built-in versions, which
are exact in both precisions.

`gamma` and `lgamma` use this package's own Python implementation (a
Lanczos approximation, accurate to about 1e-10 in `float64`), because
libclc's versions have control flow that cannot be restructured yet
(KI-24 in {doc}`known_issues`).

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
2. `numba_vulkan/data/` inside the installed package;
3. `/usr/share/clc`, `/usr/lib/clc`, `/usr/lib64/clc`, `/usr/local/share/clc`.

The devenv shell sets the variable. Elsewhere, install your distribution's
libclc package, or copy the file from a clspv or LLVM build.

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
values). Not a library to reuse here, but the likely way to offer
`float64` on devices without the `shaderFloat64` feature (KI-17).

## Open points

- **Packaging.** libclc is not on PyPI, and distributions are dropping the
  package (current nixpkgs has). Bundling `clspv--.bc` (2.7 MB, Apache-2.0
  with LLVM exception, as clspv does) in `numba_vulkan/data/` would make
  `pip install` self-contained. Not done yet.
- **Compile time.** The whole library is parsed and linked for every
  kernel that uses it. Caching a reduced copy would remove most of the
  0.4 s this costs.
- **llvmpipe.** For arguments beyond about 1e3, `sin` and `cos` lose
  accuracy on llvmpipe (to 1e-3 in `float32`), probably because its fused
  multiply-add is not fused. Hardware drivers are unaffected.
