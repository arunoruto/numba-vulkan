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
cannot be passed to a kernel that runs on another one.

Device arrays are indexed like NumPy arrays, without advanced indexing
(arrays or lists as indices). An integer for every axis reads or writes one
element; slices, `...`, `None`, `.T`, `transpose` and `reshape` give
*views* that share the memory and that kernels take like any device array:

```python
a = nv.to_device(np.arange(48, dtype=np.float32).reshape(6, 8))
a[2, 3]                       # one element, copied to the host
a[0] = 0                      # writes a row
rows = a[1::2, ::-1]          # a view: every other row, reversed
scale.forall(rows.shape)(rows, 2.0)   # the kernel works on a's memory
b = (rows + 1) * a[::2]       # computed on the device; b is a device array
b.sum(), np.sqrt(b).max()     # reductions return scalars
```

A view passes the position of its first element and its steps to the
kernel as push constants, so kernels are compiled once for all views of a
dimensionality; arrays that are contiguous from their first element need
neither. `reshape` gives a view where the steps allow it and raises
otherwise; `copy()` and `ravel()` copy. NumPy's ufuncs (`np.add`,
`np.sqrt`, ...) and the operators run as kernels when an argument is a
device array, through the same machinery as `vectorize` (see below), and so
do their `reduce` along all axes, `sum()`, `min()` and `max()`; other ufunc
methods raise `TypeError`. Elements that a view skips are read and written
back when the view is copied to, so other kernels must not write them at
the same time.

The buffers behind device arrays and behind the temporary copies of NumPy
arguments are recycled: when an array is dropped, its buffer goes to a pool
and serves the next request of a similar size. The pool holds up to
`NUMBA_VULKAN_POOL_MB` megabytes (default 1024) per device;
`nv.get_device().trim()` empties it.

A launch whose arrays are all device arrays returns at once, before the
kernel has run, as in `numba.cuda`; later launches and copies wait for it
as needed, and a copy waits only for the launches that use its array.
`nv.synchronize()` waits for everything, which is what to call before
measuring time. Launches with NumPy arrays wait for the kernel, since
their results have to be copied back. `NUMBA_VULKAN_SYNC=1` makes every
launch wait. Devices are not thread-safe.

### Streams: overlapping copies and kernels

Data that does not fit on the device at once, or that arrives in pieces,
is best processed in chunks, and a chunk need not wait for the previous
one to be copied back: while one is being computed, the next can be
copied in and the last one out. Streams make this possible, as in
`numba.cuda`:

```python
streams = [nv.stream() for _ in range(3)]
host = nv.pinned_array(n, np.float32)       # host memory the device copies directly
result = nv.pinned_array(n, np.float32)
buffers = [(nv.device_array(size, np.float32), nv.device_array(size, np.float32))
           for _ in streams]

for c in range(n // size):
    stream = streams[c % 3]
    dx, dout = buffers[c % 3]
    part = slice(c * size, (c + 1) * size)
    dx.copy_to_device(host[part], stream=stream)     # returns at once
    kernel.forall(size, stream=stream)(dx, dout)     # or kernel[groups, local, stream]
    dout.copy_to_host(result[part], stream=stream)
for stream in streams:
    stream.synchronize()                             # result is complete now
```

Work on one stream runs in order; work on different streams can run at the
same time. Copies run on a separate copy engine where the device has one
(discrete GPUs), kernels on the compute units. `benchmarks/streams.py`
processes 256 MB in 16 chunks this way: on a TITAN X it took 47 ms instead
of 103 ms one chunk after another, close to what PCIe moves in and out.
Integrated GPUs and llvmpipe, whose device memory is the host's, gain
little or nothing.

Copies to and from arrays created with `nv.pinned_array` go directly
between that memory and the device. Other host arrays go through a staging
buffer: data copied to the device is taken when the copy is enqueued, and
data copied to the host is in the array after `synchronize()`.
`stream.query()` tells whether a stream has finished, and
`with stream.auto_synchronize():` waits for it when the block ends.
Exceptions raised by kernels on a stream are raised by its `synchronize()`.

Launches on a stream take device arrays only and cannot `print`, and only
contiguous views can be copied on a stream. Work without a stream that uses
an array that a stream still uses waits for that stream on the host, and the
other way round, so mixing the two is correct but serialises them.

To measure the time the device spends, rather than the time Python waits,
record events around the work, as with `numba.cuda`:

```python
start, end = nv.event(), nv.event()
start.record()
kernel.forall(n)(x, out)          # device arrays: does not wait
end.record()
start.elapsed_time(end)           # milliseconds, from the device's clock
```

The device writes a timestamp when it passes each event. Time in which it
waits for the host between the two events counts as well.

Launches that do not wait are collected into one command buffer, which is
submitted at once while the device is idle, and otherwise after 16 launches
or when something waits. A program that launches work and then computes on
the CPU for a while should call `nv.synchronize()` only when it needs the
results; the launches are submitted by then at the latest.

## Caching compiled kernels

Compiled SPIR-V is kept on disk (`~/.cache/numba-vulkan`) and reused by
later processes; `NUMBA_VULKAN_CACHE=0` turns that off. Numba's own
compilation still runs in every process, about 0.1 s per kernel, unless
the kernel asks for more:

```python
@nv.jit(cache=True)
def kernel(x, out):
    ...
```

Then the whole compiled kernel is stored, and a later process uses it
without compiling anything. It is compiled again when its source file, or
the source file of an `@nv.jit` function it calls, changes. As with
Numba's `cache=True`, changes to the values of global arrays are not
noticed, and functions defined interactively or with `exec` are not
cached (with a warning).

## Autotuning

Which kernel variant, tile size or workgroup size is fastest depends on the
device and the size of the problem. `nv.autotune` chooses by measurement:
the decorated generator yields named candidate launches, all of which must
leave the same results.

```python
@nv.autotune(key=lambda a, b, c: (a.shape, b.shape))
def product(a, b, c):
    m, n = c.shape
    yield "tiled", lambda: matmul[(-(-n // 16), -(-m // 16)), (16, 16)](a, b, c)
    yield "blocked", lambda: matmul_blocked[(-(-n // 64), -(-m // 64)), (16, 16)](a, b, c)

product(a, b, c)    # times both candidates, then runs the faster one
product.choices     # {...: "blocked"}; product.timings has the times
```

On the first call for a key (by default the shapes and types of the array
arguments), every candidate runs once untimed and three times timed, and
the one with the lowest median is kept: in memory, and per device and
driver in `~/.cache/numba-vulkan/autotune.json`, so later processes skip the
timing. Arrays that the candidates write are written several times while
tuning, so candidates that update an array in place should be tuned on a
copy. For the two matrix products of the benchmark suite, both GPUs here
chose the blocked one (1.2 against 2.1 ms on a TITAN X, 35 against 75 ms on
a UHD 630, for 1024×1024).

## Structured arrays

Arrays of a structured dtype work as in Numba on the CPU:

```python
particle = np.dtype([("x", "f4"), ("v", "f4"), ("alive", "?")])

@nv.jit
def step(p, dt):
    i = nv.global_id(0)
    if i < p.shape[0] and p[i].alive:
        p[i].x += p[i].v * dt
        if p[i].x > 100:
            p[i]["alive"] = False
```

Fields may be booleans, integers, `float32` and `float64`, in packed or
aligned dtypes. A record (`p[i]`) refers to the element in the buffer;
reading a field reads the buffer, and assigning to one writes it. Fields
narrower than 32 bits, and fields that do not start at a multiple of four
bytes, are written with atomic operations on the 32-bit words they share
with their neighbours, so invocations can write different fields of the
same record at once; whole aligned words are written directly. On devices
without `float64` or `int64`, those fields are converted on the host, so
the kernel sees them as `float32` and `int32`. Nested records and array
fields are not supported.

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

### Integer and float width

**Integers are 32-bit inside kernels.** Numba computes with `int64`
wherever it can, including every index and loop counter, but GPUs execute
64-bit integer arithmetic as several 32-bit instructions; index-heavy
kernels run up to 1.5× slower with it. Kernels therefore compute with
32-bit integers, as CUDA C code using `int` does. `int64` arrays and
scalars are converted on the host, and a value that does not fit raises
`OverflowError` rather than wrapping around. Arithmetic inside the kernel
wraps at 2³¹. For exact 64-bit integers, use `@nv.jit(narrow=False)` or set
`NUMBA_VULKAN_INT64=1`.

**Floats keep their width** where the device supports `float64`. Most GPUs
run `float64` at a small fraction of the `float32` speed (1/32 on the
TITAN X), so a {py:class}`~numba_vulkan.errors.VulkanPerformanceWarning`
points out kernels that compute in `float64`, once per kernel. Often the
cause is a Python float literal: `x[i] * 0.1` is a `float64` product even
for a `float32` array; `x[i] * np.float32(0.1)` is not.

**Devices without `float64`** (many mobile GPUs, Apple devices) get kernels
computed in `float32`, with a
{py:class}`~numba_vulkan.errors.VulkanPrecisionWarning`. Results then agree
with the 64-bit ones to `float32` rounding. Code that cannot work in 32
bits is rejected at compile time: integer constants beyond 32 bits, shifts
by 32 or more, tricks on the bit pattern of a `float64`.

```python
@nv.jit                    # 32-bit integers; float64 where available (default)
@nv.jit(narrow=True)       # 32-bit floats as well: no optional features needed
@nv.jit(narrow="floats")   # 32-bit floats, 64-bit integers
@nv.jit(narrow=False)      # exact: 64-bit integers and floats; fails without them
```

Both warnings are Python warnings: `NUMBA_VULKAN_WARNINGS=0` turns them off,
as does `warnings.filterwarnings("ignore", category=nv.VulkanPerformanceWarning)`.

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
previous value, and use relaxed ordering, as in CUDA. Integer atomics work
on 32-bit elements; `int64` arrays work as well, because kernels compute
with 32-bit integers. With `narrow=False` they are rejected: LLVM's SPIR-V
backend does not offer 64-bit integer atomics for Vulkan. `float32` additions use the device's native
instruction where it has one (`VK_EXT_shader_atomic_float`) and a
compare-and-swap loop otherwise; float `max` and `min` always use the loop.
`float64` atomics are not available.

## Subgroups

A workgroup runs as *subgroups* (warps, in CUDA's terms) whose invocations
execute together and can combine or exchange values without shared memory
or barriers. {py:mod}`nv.subgroup <numba_vulkan.stubs.subgroup>` offers:

| function | result |
| --- | --- |
| `size()`, `lane()`, `id()`, `count()` | subgroup size, index in the subgroup (`cuda.laneid`), index of the subgroup in the workgroup, subgroups per workgroup |
| `sum(v)`, `prod(v)`, `min(v)`, `max(v)` | over the active invocations of the subgroup |
| `inclusive_sum(v)`, `exclusive_sum(v)`, ... | prefix scans, also for `prod`, `min` and `max` |
| `any(p)`, `all(p)`, `elect()` | votes; `elect()` is true for the lowest active lane |
| `ballot(p)`, `ballot_count(p)` | the lanes for which `p` holds, as four `uint32` words, or their number |
| `broadcast(v, lane)`, `broadcast_first(v)` | `v` of one lane (`lane` the same for the whole subgroup) |
| `shuffle(v, lane)`, `shuffle_xor(v, mask)`, `shuffle_up(v, d)`, `shuffle_down(v, d)` | `v` of another lane, like `cuda.shfl_*_sync` |

Values are 32- or 64-bit integers or floats. Only the invocations that
reach the call take part, as with CUDA's `*_sync` functions under a mask of
the active threads.

```python
@nv.jit
def total(x, out):
    partial = nv.shared.array(32, np.float32)    # one value per subgroup
    i = nv.global_id(0)
    v = nv.subgroup.sum(x[i] if i < x.shape[0] else np.float32(0))
    if nv.subgroup.elect():
        partial[nv.subgroup.id()] = v
    nv.barrier()
    if nv.subgroup.id() == 0:
        lane = nv.subgroup.lane()
        v = partial[lane] if lane < nv.subgroup.count() else np.float32(0)
        v = nv.subgroup.sum(v)
        if nv.subgroup.elect():
            nv.atomic.add(out, 0, v)                 # one atomic per workgroup
```

A grid-stride sum of 16 M floats written like this took 3.2 ms on an Intel
UHD 630, against 4.6 ms with the shared-memory tree above; on an NVIDIA
TITAN X, which is limited by memory bandwidth here, both took 0.32 ms. One
atomic per *subgroup* instead was 17 times slower on the UHD 630, whose
float additions go through compare-and-swap loops.

Subgroups need not be full. `size()` is what the device reports, but
drivers may run fewer invocations per subgroup: Intel's runs many kernels
16 wide while reporting 32, and `count()` then gives the actual number of
subgroups. Devices that do not support a class of operations (see the
`subgroup_*` attributes of {py:class}`~numba_vulkan.runtime.DeviceInfo`)
raise {py:class}`~numba_vulkan.errors.VulkanSupportError` for kernels that
use it.

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

A `vectorize` function of two arguments can also reduce an array, as
`numba.cuda` offers:

```python
@numba.vectorize(["float32(float32, float32)"], target="vulkan", identity=0)
def add(a, b):
    return a + b

total = add.reduce(x)              # 1-d x; axis=None for any shape
```

The reduction runs on the device as a tree, so the function should be
associative; it needs no identity, except for empty arrays.

The results are not real `numpy.ufunc` objects: `accumulate`, `outer` and
`reduce` along other axes are not available, and they cannot be called from
inside a kernel; call the `@nv.jit` function there instead. NumPy's own
ufuncs applied to device arrays run on the device (see above).

## Errors

An exception raised in a kernel, or in a function it calls, is raised by
the launch after the kernel has finished, or, for a launch that does not
wait (all arrays on the device), by the next synchronisation: a copy of an
array the kernel used, `nv.synchronize()`, or a launch that waits. As in
CUDA, it is then not certain which launch raised it, and the launches in
between ran anyway; `NUMBA_VULKAN_SYNC=1` makes every launch wait and report
at once:

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

Integer division by zero gives NumPy's result by default (0), as in
`numba.cuda`. With `@nv.jit(error_model="python")` it raises
`ZeroDivisionError` instead, as Numba does on the CPU; each division then
costs a comparison.

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

## Printing

`print()` works inside kernels for constant strings and numbers:

```python
@nv.jit
def kernel(x):
    i = nv.global_id(0)
    if i < x.shape[0] and x[i] < 0:
        print("negative value at", i, ":", x[i])
```

As with CUDA's `printf`, each call appends a record to a buffer, and the
host prints the records after the kernel, in no particular order. A kernel
that prints therefore always waits for completion. The buffer holds
`NUMBA_VULKAN_PRINT_WORDS` 32-bit words (default 262144, 1 MiB); output
beyond that is dropped with a warning.

## float16 arrays

Arrays of `np.float16` can be passed in and created as device arrays. They
are stored as halves and computed with as `float32`: reading an element
gives a `float32`, writing one rounds to `float16`. This needs only the
`storageBuffer16BitAccess` device feature, which most GPUs have, not
`shaderFloat16`. Atomics on them are not supported.

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
