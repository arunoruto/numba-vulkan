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

Arguments are NumPy arrays, device arrays and scalars. NumPy arrays are
copied to the device before the call, and those the kernel writes to are
copied back.

## Arrays inside kernels

Indexing follows NumPy: integers select, slices give views of the same
data. Views cost nothing to create and can be iterated over, reduced,
passed to other functions and written through:

```python
@nv.jit
def normalise_rows(a, out):
    i = nv.global_id(0)
    if i < a.shape[0]:
        row = a[i]                    # a view, not a copy
        total = row.sum()
        for j, value in enumerate(row):
            out[i, j] = value / total
        out[i, -1] = row[1:-1].max()  # slices of slices
        out[i, :2] = 0                # fill a slice

normalise_rows.forall(a.shape[0])(a, out)
```

Available on arrays and views: `.shape`, `.size`, `.ndim`, `.T`, `len()`,
iteration (also with `enumerate` and `zip`), `sum`, `prod`, `mean`, `min`,
`max`, `argmin`, `argmax`, `any`, `all` over all elements, and `np.dot` of
two 1-d arrays.

### Arrays inside kernels and array expressions

Arrays of a constant shape can be created inside kernels. They are private
to each invocation and live as long as it does:

```python
tmp = np.zeros(8, dtype=np.float32)       # also np.empty, np.ones, np.full
acc = nv.local.array((4, 4), np.int32)    # like cuda.local.array
```

Arithmetic and ufuncs on arrays give *array expressions*. Kernels cannot
allocate the memory a result of run-time size would need, so an
expression stores only its operands; an element is computed when it is
read:

```python
d = a[i] - b                      # nothing is computed yet
out[i] = np.sqrt((d * d).sum())   # elements computed inside the reduction
out[i, :] = a[i] * w + c          # computed element by element into out
out[i, :] *= 2                    # in-place operators work the same way
x = (a[i] * 2)[3]                 # computes one element
```

Expressions support element indexing with integers, `shape`, `size`,
`ndim`, `len()`, the reductions above, `np.dot`, assignment to slices
and in-place operators; they broadcast like NumPy. Each read computes the
element again, so an expression read many times is better written into a
local array once. Indexing with masks or index arrays, `copy()` and
anything else that needs an array of run-time size is not available.

NumPy arrays defined outside a kernel can be used inside it, and inside
functions it calls, as read-only lookup tables:

```python
WEIGHTS = np.array([0.25, 0.5, 0.25], dtype=np.float32)

@nv.jit
def smooth(x, out):
    i = nv.global_id(0)
    if 0 < i < x.shape[0] - 1:
        out[i] = np.dot(x[i - 1 : i + 2], WEIGHTS)
```

As in Numba on the CPU, the values are frozen when the kernel is first
compiled; later changes to the array are not seen. The data is uploaded to
the device once, not on every call.

A reduction is a loop inside one invocation. `a[i].sum()` in a kernel over
rows is the intended use; `a.sum()` in every invocation repeats the whole
sum each time.

## Keeping data on the device

Copying is often slower than the kernel itself. A
{py:class}`~numba_vulkan.runtime.DeviceArray` lives on the device and is
used in place, so a chain of kernels transfers data only at its ends, as
with `numba.cuda`:

```python
x = nv.to_device(np.arange(1000, dtype=np.float32))   # host -> device
y = nv.device_array_like(x)                           # uninitialised
tmp = nv.device_array(1000, np.float32)

first.forall(1000)(x, tmp)        # no copies
second.forall(1000)(tmp, y)       # no copies
result = y.copy_to_host()         # device -> host
x.copy_to_device(new_values)      # overwrite in place
```

Host and device arrays can be mixed in one call. A device array belongs to
the device it was created on (`device=` selects it, as for `forall`) and
cannot be passed to a kernel that runs on another one. There is no indexing
or arithmetic on device arrays from Python; copy to the host for that.

The buffers behind device arrays and behind the temporary copies of NumPy
arguments are recycled: when an array is dropped, its buffer goes to a pool
and serves the next request of a similar size. The pool holds up to
`NUMBA_VULKAN_POOL_MB` megabytes (default 1024) per device;
`nv.get_device().trim()` empties it.

A launch returns when the kernel has finished. Devices are not thread-safe.

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

### Devices without 64-bit types

`float64` and `int64` are optional in Vulkan; many mobile GPUs and Apple
devices lack one or both. Since Numba uses `int64` for every index, almost
no kernel would run there as written. On such a device a kernel is
therefore *narrowed*: every `float64` in it becomes a `float32` and every
`int64` an `int32`, including the elements of arrays, which are converted
on the host.

```python
@nv.jit                 # narrows where the device requires it (default)
@nv.jit(narrow=True)    # always narrow: no optional features needed
@nv.jit(narrow=False)   # never: fail on devices without 64-bit types
```

When narrowing changes float arithmetic, a
{py:class}`~numba_vulkan.errors.VulkanPrecisionWarning` says so once per
kernel; `narrow=True` accepts it silently. Results then agree with the
64-bit ones to `float32` rounding, and integers wrap at 32 bits. Code that
cannot work in 32 bits is rejected at compile time: integer constants
beyond 32 bits, shifts by 32 or more, tricks on the bit pattern of a
`float64`. `narrow=True` is also worth trying on desktop GPUs, where
`float64` is slow. The environment variable `NUMBA_VULKAN_NARROW=1` makes
it the default.

### Exactness

By default, math functions come from libclc and float arithmetic is kept
exactly as written, so results match the CPU closely and agree between
devices. `@nv.jit(fastmath=True)` trades that for speed: `float32` math
uses the device's built-in functions and the driver may reorder
arithmetic.

## Workgroups, shared memory and atomics

Kernels can be written exactly like `numba.cuda` kernels that cooperate
within a block. The names map as follows:

| numba.cuda | numba-vulkan |
| --- | --- |
| `cuda.grid(1)` | `nv.global_id(0)` |
| `cuda.threadIdx.x`, `cuda.blockIdx.x` | `nv.local_id(0)`, `nv.group_id(0)` |
| `cuda.blockDim.x`, `cuda.gridDim.x` | `nv.local_size(0)`, `nv.num_groups(0)` |
| `cuda.shared.array(shape, dtype)` | `nv.shared.array(shape, dtype)` |
| `cuda.syncthreads()` | `nv.barrier()` (or `nv.syncthreads()`) |
| `cuda.atomic.add(a, i, v)` ... | `nv.atomic.add(a, i, v)` ... |
| `kernel[blocks, threads](...)` | `kernel[groups, local_size](...)` |

```python
@nv.jit
def block_sums(x, out):
    partial = nv.shared.array(256, np.float32)
    t = nv.local_id(0)
    i = nv.global_id(0)
    partial[t] = x[i] if i < x.shape[0] else np.float32(0)
    nv.barrier()
    step = 128
    while step > 0:
        if t < step:
            partial[t] += partial[t + step]
        nv.barrier()
        step //= 2
    if t == 0:
        nv.atomic.add(out, 0, partial[0])

block_sums[n_groups, 256](x, out)           # 256 invocations per workgroup
block_sums.forall(n, local_size=256)(x, out)  # or: n rounded up to workgroups
```

`kernel[groups, local_size]` launches exactly `groups × local_size`
invocations, like CUDA; `forall(n, local_size=...)` rounds `n` up. Without
a `local_size`, workgroups have 64 invocations. Sizes beyond the device's
limits, and shared arrays larger than its shared memory, raise
{py:class}`~numba_vulkan.errors.VulkanSupportError`.

Shared arrays need a constant shape and type; each workgroup has its own,
uninitialised copy. `nv.barrier()` must be reached by all invocations of a
workgroup, so it must not sit under a condition that differs between them.

The atomics `add`, `sub`, `max`, `min`, `exch`, `and_`, `or_`, `xor` and
`cas` work on elements of array arguments and shared arrays, return the
previous value, and use relaxed ordering, as in CUDA. Integer atomics need
`int32` or `uint32` elements: LLVM's SPIR-V backend does not offer 64-bit
integer atomics for Vulkan, so `int64` arrays are rejected unless the
kernel is narrowed. `float32` additions use the device's native
instruction where it has one (`VK_EXT_shader_atomic_float`) and a
compare-and-swap loop otherwise; float `max` and `min` always use the loop.
`float64` atomics are not available.

### 32-bit integers for speed

Numba computes with `int64` wherever it can, including every
`local_id(0) * 16 + j`. GPUs execute 64-bit integer arithmetic as several
32-bit instructions, so index-heavy kernels such as tiled matrix products
run up to 1.5× slower than they need to. `@nv.jit(narrow="ints")` compiles
all integers as 32-bit, which is what CUDA C programmers would write:

```python
@nv.jit(narrow="ints")
def matmul(a, b, c):
    ...
```

Integers then wrap at 2³¹, and `int64` arrays are converted on the host.

## Ufuncs: `vectorize` and `guvectorize`

Numba's own decorators accept `target="vulkan"` once `numba_vulkan` is
imported. The scalar function is compiled for Vulkan and applied to every
element on the device, with NumPy's broadcasting:

```python
import math
import numba
import numba_vulkan  # registers the target

@numba.vectorize(["float32(float32, float32)", "float64(float64, float64)"],
                 target="vulkan")
def gaussian(x, sigma):
    return math.exp(-0.5 * (x / sigma) ** 2)

y = gaussian(x, 1.5)             # x: a NumPy or device array
gaussian(x, sigmas[:, None], out=grid)
```

Without signatures (`@numba.vectorize(target="vulkan")`), each call
compiles for the types of its arguments. Python scalars then adapt to the
arrays as in NumPy 2, so `f(x32, 2.0)` stays in `float32`; inside the
function, Numba's own typing rules apply as usual.

`guvectorize` works the same way with a layout. The core function gets
views of the core dimensions and writes its results into the outputs:

```python
@numba.guvectorize(["void(float32[:], float32[:], float32[:])"],
                   "(n),(n)->()", target="vulkan")
def distance(a, b, out):
    total = np.float32(0)
    for i in range(a.shape[0]):
        total += (a[i] - b[i]) ** 2
    out[0] = math.sqrt(total)

d = distance(points, centre)      # points: (N, 16), centre: (16,)
```

As on the CPU, an argument with layout `()` is a scalar if its signature
type is a scalar and a one-element array otherwise, outputs with layout
`()` are written as `out[0]`, and a `guvectorize` without signatures needs
its outputs passed in. Both kinds of function take `out=` and `device=`,
and return device arrays when any input is one, so chains of calls stay
on the device.

The results are not real `numpy.ufunc` objects: `reduce`, `accumulate`,
`outer` and NumPy's dispatch (`np.add(a, b)` on device arrays) are not
available, and they cannot be called from inside a kernel; call the
`@nv.jit` function there instead.

## Errors

An exception raised in a kernel, or in a function it calls, is raised by
the launch after the kernel has finished:

```python
@nv.jit
def checked_log(x):
    if x <= 0:
        raise ValueError("log of a non-positive number")
    return math.log(x)

kernel.forall(n)(x, out)
# ValueError: log of a non-positive number
# raised in Vulkan kernel 'kernel', in checked_log at example.py:4
```

Each invocation that raises stops itself only; the others run to completion,
so the output arrays are partly written. If several invocations raise, one
of their exceptions is reported. Kernels that cannot raise pay nothing for
this.

Array indices are not checked by default. An out-of-bounds access reads
garbage or writes to memory it should not, depending on the driver. While
debugging, turn the check on for a function, or for everything with the
environment variable `NUMBA_BOUNDSCHECK=1`:

```python
@nv.jit(boundscheck=True)
def kernel(a, out):
    i = nv.global_id(0)
    out[i] = a[i]        # IndexError if the grid is larger than the arrays
```

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
