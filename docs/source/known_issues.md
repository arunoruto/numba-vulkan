# Known issues

This page lists what is currently broken or missing, as opposed to the
fundamental constraints described in {doc}`limitations`. It is meant as a
starting point for anyone, human or AI, who continues the work.

The language-coverage entries are reproduced by `tests/test_known_issues.py`.
Those tests are expected to fail and are marked strict: when an issue gets fixed,
its test turns red until the case is moved into the regular test suite and
the entry is removed from this page.

```sh
uv run pytest tests/test_known_issues.py -rxX
```

Numbers are not reused: KI-02, KI-03, KI-07, KI-13 and KI-24 (NumPy
functions on scalars, missing `math` functions, complex numbers, all data
copied on every call, nested loop exits that failed to compile) have been
fixed, and KI-01 no longer applies when libclc is installed.

## Language and library coverage

### KI-01: `float64` math needs libclc

```python
out[i] = math.sin(x[i])     # x is float64
out[i] = x[i] ** 2.5        # x is float64, or float32 with a Python float
```

These work, with full double accuracy, when libclc is installed (see
{doc}`math_library`). Without it:

**Symptom:** `VulkanUnsupportedError: math.pow on float64 needs libclc`.

**Cause:** Vulkan's own math library defines `sin`, `exp`, `log`, `pow` and
friends for 32-bit floats only.

**Workaround:** install libclc, keep the computation in `float32`, or use
`@nv.jit(narrow_math=True)`.

**Fix:** bundle libclc's `clspv--.bc` with the package, so that it is
always available.

### KI-04: slices, array methods and iteration

```python
x[1:][i]          # VulkanUnsupportedError: indexing ... with slice<a:b>
x.sum()           # TypingError: Unknown attribute 'sum'
for v in x: ...   # KeyError: VulkanArrayModel does not have a field named 'data'
```

**Cause:** {py:class}`~numba_vulkan.vktypes.VulkanArray` has no data pointer,
so every Numba array operation that needs one has to be reimplemented. Only
full integer indexing, `.shape`, `.size`, `.ndim` and `len()` exist.

**Fix:** views need an element offset and strides in the data model
(`models.py`) and in `_linear_index` (`vkimpl.py`). The strides member
already exists but is ignored. Iteration and reductions can then be written
as `@overload`s for the `vulkan` target on top of indexing. The `KeyError`
for iteration should become a `VulkanUnsupportedError`.

### KI-05: allocating arrays inside a kernel

**Symptom:** `np.zeros(...)` fails typing, because the target has no memory
allocator (Numba's NRT is disabled).

**Workaround:** tuples work as small fixed-size arrays; larger scratch
space has to be passed in as an argument.

**Fix:** fixed-size local arrays could map to SPIR-V function-local
variables. Nothing exists yet.

### KI-06: global constant arrays

```python
TABLE = np.array([...], dtype=np.float32)

@nv.jit
def kernel(x, out):
    ...
    out[i] = TABLE[i % 4]
```

**Symptom:** `VulkanUnsupportedError: global NumPy arrays cannot be used
inside Vulkan kernels yet`.

**Cause:** Numba emits the data as an LLVM global and indexes it through a
pointer, which the SPIR-V backend cannot translate.

**Workaround:** pass the array as an argument.

**Fix:** `VulkanTargetContext.make_constant_array` could upload the data as
an additional read-only buffer and return a `VulkanArray` bound to it.

### KI-08: `print`

**Symptom:** `No definition for lowering <built-in function print>`.

**Fix:** the `debugPrintf` extension of the validation layers could back
it, as Taichi does. Low priority.

### KI-09: unsupported array kinds

| Kind | Symptom |
| --- | --- |
| `float16` arrays | `NotImplementedError: float16` |
| Fortran-ordered and non-contiguous arrays | `ValueError: only C-contiguous arrays are supported` |
| structured (record) arrays | `VulkanUnsupportedError: arrays of Record(...)` |

Supported element types are `bool`, signed and unsigned integers of 8 to 64
bits, `float32` and `float64`.

## Behaviour that differs from Numba on the CPU

### KI-10: errors inside kernels are dropped

`raise` compiles but does nothing visible: the error status of the kernel is
discarded, as `numba.cuda` does without `debug=True`. There is no bounds
checking, and integer division by zero yields an unspecified value instead
of raising.

**Fix:** an error-code buffer written by the entry point wrapper in
`compiler.compile_kernel` and checked after the dispatch in
`dispatcher._launch`. The call helper needed to map codes back to
exceptions is already kept in the compile result.

### KI-11: `round()` on llvmpipe

`round(1.5)` gives `1.0` on llvmpipe; NVIDIA's driver gives the correct
`2.0`. Halfway cases should not be relied on.

### KI-12: recursion

Rejected with a clear error, because all functions are inlined into the
entry point. SPIR-V forbids recursion, so this will not change.

### KI-23: less accurate math without libclc or with `fastmath`

With libclc, all math functions are accurate to the last digit or two in
both precisions. Without it, and for `float32` with `fastmath=True`, the
drivers' and this package's own versions are used: `gamma` and `lgamma`
are then accurate to about 1e-5, `erf` and `erfc` have an absolute error
of about 1.5e-7, and the trigonometric functions are only as good as the
driver's.

### KI-27: very large functions are not restructured

SPIR-V needs structured control flow, and `structurize.py` rearranges the
graph to provide it (see {doc}`how_it_works`). Kernels with more than 4000
basic blocks after inlining are passed on unchanged, because restructuring
them would take minutes; LLVM's own structurizer then usually fails with
`SpirvCodegenError`. Random programs with five levels of nested loops and
conditions stay well below a thousand blocks.

Without `spirv-val` (`NUMBA_VULKAN_VALIDATE=1`), a structurally invalid
module would reach the driver unchecked. A built-in structural check of the
generated module would be worth having.

## Performance

### KI-28: launches are synchronous and cost about 0.4 ms

Every launch waits for the kernel to finish, and takes about 0.4 ms even
for a tiny kernel on device arrays (numba-cuda: about 0.2 ms). Most of it is spent in the Python Vulkan bindings: scalars and
array shapes are uploaded as small buffers and the descriptor set is
rewritten on each call.

**Fix:** pass scalars and shapes as push constants, keep descriptor sets
per argument combination, and offer an asynchronous launch that returns a
handle to wait on.

### KI-29: device arrays are bare buffers

A `DeviceArray` can be created, passed to kernels and copied; it has no
indexing, slicing, views or arithmetic, and no `__cuda_array_interface__`
equivalent for sharing memory with other libraries.

### KI-14: compilation is slow and not cached

A kernel takes about half a second to compile. A large part is starting a
Python child process for LLVM's SPIR-V backend (`codegen.emit_spirv`).
Nothing is cached on disk, and pipelines are cached only in memory.

**Fix:** cache SPIR-V on disk keyed by the optimised LLVM IR; keep one
long-lived worker process instead of starting one per kernel.

### KI-25: libclc is linked in full for every kernel

Each kernel that calls a math function parses and links the whole of libclc
(13,000 functions) before discarding what it does not need. That roughly
doubles the compile time, from 0.4 s to 0.8 s.

**Fix:** cache a reduced copy of the library holding only the functions
this target uses.

### KI-15: one specialisation per buffer binding

The binding is part of the array type, so a function that takes arrays is
compiled again for every combination of bindings it is called with. This is
invisible for kernels but multiplies work for shared helper functions.

### KI-16: fixed workgroup sizes

The workgroup size is fixed at 64, 8×8 or 4×4×4 depending on the grid
dimensionality (`compiler.LOCAL_SIZES`). There is no shared memory, no
atomics and no barriers.

## Portability

### KI-17: kernels need optional device features

Numba types Python literals and loop counters as `float64`/`int64`, so
almost every kernel requires the `shaderInt64` feature and many require
`shaderFloat64`. The SPIR-V backend additionally forces `shaderInt8`,
because it needs a named string per buffer. Devices without these features
(many mobile GPUs, Apple via MoltenVK) cannot run such kernels.

**Fix:** a typing mode that defaults to 32-bit types (a custom typing
context could type literals as `int32`/`float32`); stripping the unused
`Int8` capability from the generated module.

### KI-18: tested on three devices only

NVIDIA TITAN X (Pascal), Intel UHD Graphics 630 and llvmpipe, all on Linux.
AMD, Apple, Windows and mobile GPUs are untested. Two driver problems were
found on this small sample alone (see {doc}`how_it_works`), so more should
be expected.

## Implementation debt

### KI-19: LLVM IR is rewritten as text

Several steps patch textual LLVM IR with regular expressions: the shader
attributes on the entry point (`compiler.compile_kernel`), about ten
rewrites of constructs the backend cannot handle (`legalize.py`), the
control-flow restructuring (`structurize.py`) and the buffer access
expansion (`buffers.expand_buffer_access`). They depend on how LLVM 22
prints IR and may break with other LLVM versions. A small IR library that
can parse and edit modules would replace all of them.

### KI-20: the dispatcher is not a Numba `Dispatcher`

{py:class}`~numba_vulkan.dispatcher.VulkanDispatcher` implements only the
methods Numba's typing and lowering call. On-disk caching, `inspect_types`,
pickling, explicit signatures and `numba.jit(..., _target="vulkan")` are
untested or missing.

### KI-21: workarounds tied to specific versions

The `srem` rewrite and the pointer-select check exist because of behaviour
observed with NVIDIA driver 580.x and llvmlite 0.50 (LLVM 22). They should
be re-evaluated when either changes. The benchmark dependency group pins
`numpy<2.5`, because numba-cuda 0.30.4 does not import with NumPy 2.5.

### KI-26: libclc is not packaged

The math library depends on a file, `clspv--.bc`, that `pip` cannot
install and that distributions are dropping (current nixpkgs has removed
libclc; the devenv shell pins an older revision for it). It must also come
from an LLVM no newer than the one inside llvmlite.

**Fix:** bundle the file in the package, as clspv does.

### KI-22: lint warnings

Ruff reports a handful of style warnings (docstring mood, broad `except` in
the benchmark, `subprocess.run` without `check`). None affects behaviour.
