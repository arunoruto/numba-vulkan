# Usage

## Kernels

A kernel is a Python function that returns nothing and writes its results
into arrays. It runs once per point of a dispatch grid.
{py:func}`~numba_vulkan.stubs.global_id` returns the position of the current
invocation along one axis, like `cuda.grid`.

```python
import numpy as np
import numba_vulkan as nv

@nv.jit
def saxpy(a, x, y):
    i = nv.global_id(0)
    if i < x.shape[0]:
        y[i] = a * x[i] + y[i]

x = np.arange(1000, dtype=np.float32)
y = np.ones(1000, dtype=np.float32)
saxpy.forall(x.size)(np.float32(2.0), x, y)
```

`forall(extent)` binds the grid and returns a callable that takes the kernel
arguments. The grid is rounded up to whole workgroups, so a kernel must check
its index against the array bounds, as above.

Grids can have up to three dimensions. Axis 0 is the first entry of `extent`:

```python
@nv.jit
def stencil(src, dst):
    j = nv.global_id(0)
    i = nv.global_id(1)
    if 0 < i < src.shape[0] - 1 and 0 < j < src.shape[1] - 1:
        dst[i, j] = (src[i - 1, j] + src[i + 1, j] + src[i, j - 1] + src[i, j + 1]) / 4

stencil.forall((src.shape[1], src.shape[0]))(src, dst)
```

Arguments are NumPy arrays (C-contiguous) and scalars. Arrays are copied to
the device before the call; arrays the kernel writes to are copied back.

## Functions called from kernels

A function decorated with `@nv.jit` can also be called from a kernel. It may
return a value or a tuple and may take arrays:

```python
import math

@nv.jit
def polar(x, y):
    return math.sqrt(x * x + y * y), math.atan2(y, x)

@nv.jit
def to_polar(x, y, radius, angle):
    i = nv.global_id(0)
    if i < x.size:
        radius[i], angle[i] = polar(x[i], y[i])
```

Functions compiled with `numba.njit` are recompiled for Vulkan when a kernel
calls them, so scalar helpers can be shared between CPU and GPU code:

```python
from numba import njit

@njit
def clamp(v, lo, hi):
    return min(max(v, lo), hi)

@nv.jit
def clamp_all(x, out):
    i = nv.global_id(0)
    if i < x.shape[0]:
        out[i] = clamp(x[i], np.float32(-1.0), np.float32(1.0))
```

## Extending the target

numba-vulkan is registered with Numba's target extension API under the name
`vulkan`, so Numba's extension decorators accept it:

```python
from numba.extending import overload

def smoothstep(edge0, edge1, x):
    raise NotImplementedError

@overload(smoothstep, target="vulkan")
def ol_smoothstep(edge0, edge1, x):
    def impl(edge0, edge1, x):
        t = min(max((x - edge0) / (edge1 - edge0), 0), 1)
        return t * t * (3 - 2 * t)
    return impl
```

## Math functions

The `math` module and NumPy's element-wise functions work on scalars inside
kernels and give the same results:

```python
@nv.jit
def kernel(x, out):
    i = nv.global_id(0)
    if i < x.shape[0]:
        out[i] = np.sqrt(x[i]) + math.erf(x[i]) + np.maximum(x[i], np.float32(0.0))
```

NumPy functions cannot be applied to whole arrays inside a kernel; a kernel
computes one element per invocation.

## Precision

Numba types Python literals as `float64` and `int64`, and mixing them into
`float32` arithmetic promotes the whole expression to `float64`. That works
(see {doc}`math_library`), but `float64` is slower on GPUs and unavailable
on some devices. Use typed constants to stay in single precision:

```python
HALF = np.float32(0.5)

@nv.jit
def kernel(x, out):
    i = nv.global_id(0)
    if i < x.shape[0]:
        out[i] = HALF * math.sin(x[i])      # float32 throughout
        # out[i] = 0.5 * math.sin(x[i])     # promotes to float64
```

By default, math functions come from libclc and float arithmetic is kept
exactly as written, so results match the CPU closely and agree between
devices. `@nv.jit(fastmath=True)` trades that for speed: `float32` math
uses the device's built-in functions and the driver may reorder
arithmetic.

## Choosing a device

```python
nv.list_devices()
nv.select_device("intel")                    # by name substring or index
saxpy.forall(n, device="llvmpipe")(a, x, y)  # for one call
```

Without a selection the first discrete GPU is used, then integrated GPUs,
then CPU implementations. The `NUMBA_VULKAN_DEVICE` environment variable sets
the default from outside the program.

Compiling for a device that lacks a capability a kernel needs (for example
`float64`) raises {py:class}`~numba_vulkan.errors.VulkanSupportError`.

## Inspecting generated code

```python
from numba import float32

compiled = saxpy.compile((float32, float32[::1], float32[::1]))
compiled.llvm_ir           # optimised LLVM IR handed to the SPIR-V backend
compiled.spirv             # the shader binary; view it with spirv-dis
compiled.capabilities      # optional device features it needs
compiled.written_bindings  # buffers that are copied back after a call
```

Setting `NUMBA_VULKAN_VALIDATE=1` runs every generated shader through
`spirv-val` before it reaches a driver.
