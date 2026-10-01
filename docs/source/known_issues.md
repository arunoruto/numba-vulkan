# Known issues

This page lists what is currently broken or missing, as opposed to the
fundamental constraints described in {doc}`limitations`. It is meant as a
starting point for anyone, human or AI, who continues the work.

Entries KI-01 to KI-09 are reproduced by `tests/test_known_issues.py`. Those
tests are expected to fail and are marked strict: when an issue gets fixed,
its test turns red until the case is moved into the regular test suite and
the entry is removed from this page.

```sh
uv run pytest tests/test_known_issues.py -rxX
```

## Language and library coverage

### KI-01: `float64` transcendental functions and float powers

```python
out[i] = math.sin(x[i])     # x is float64
out[i] = x[i] ** 2.5        # x is float64, or float32 with a Python float
```

**Symptom:** `VulkanUnsupportedError: math.pow on float64 is not available in
Vulkan shaders`.

**Cause:** GLSL.std.450, the math library of Vulkan shaders, defines `sin`,
`cos`, `tan`, their inverses, `exp`, `log` and `pow` for 32-bit floats only.
A Python float literal is typed `float64`, so `x ** 2.5` is a `float64` power
even for a `float32` array.

**Workaround:** keep the computation in `float32` (`x[i] ** np.float32(2.5)`),
or use `@nv.jit(narrow_math=True)`. Powers with integer exponents and
`math.sqrt` work in both precisions.

**Fix:** a software `float64` math library, in `mathimpl.py`.

### KI-02: NumPy functions on scalars

```python
out[i] = np.sqrt(x[i])
```

**Symptom:** `No definition for lowering <ufunc 'sqrt'>(float32,) -> float32`.

**Cause:** the target has no ufunc support. Typing succeeds, because Numba's
NumPy declarations are installed, but nothing lowers the call.

**Workaround:** use the `math` module.

**Fix:** implement `get_ufunc_info` on `VulkanTargetContext`, mapping ufunc
loops to the implementations in `mathimpl.py`. Numba's CUDA target does this
in `numba/cuda/ufuncs.py`.

### KI-03: missing `math` functions

**Symptom:** `No definition for lowering <built-in function hypot>...`.

**Affected:** confirmed for `math.hypot`, `math.log1p`, `math.erf` and
`math.isfinite`. Anything not listed in the tables at the top of
`mathimpl.py` is missing, which includes `expm1`, `erfc`, `gamma`,
`lgamma`, `isnan`, `isinf` and `copysign`.

**Fix:** `isnan`/`isinf`/`isfinite` are float comparisons, and `hypot`,
`log1p` and `expm1` can be composed from existing functions. `erf` and
`gamma` need polynomial approximations. See
[Adding a `math` function](development.md#adding-a-math-function).

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

### KI-07: complex numbers

**Symptom:** `No definition for lowering <built-in function isnan>`.

**Cause:** Numba's complex arithmetic calls `math.isnan` (KI-03). Complex
arrays are also not accepted as buffers.

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

## Performance

### KI-13: all data is copied on every call

Each call creates new buffers, uploads every argument and reads back every
array the kernel writes to. This dominates memory-bound kernels (see
{doc}`benchmarks`).

**Fix:** a device array type that owns a buffer and can be passed to
kernels, plus buffer reuse in `runtime.Device.run`. Buffers currently use
host-visible memory; device-local memory with staging copies would be
faster on discrete GPUs.

### KI-14: compilation is slow and not cached

A kernel takes about half a second to compile. A large part is starting a
Python child process for LLVM's SPIR-V backend (`codegen.emit_spirv`).
Nothing is cached on disk, and pipelines are cached only in memory.

**Fix:** cache SPIR-V on disk keyed by the optimised LLVM IR; keep one
long-lived worker process instead of starting one per kernel.

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

Three steps patch textual LLVM IR with regular expressions: the shader
attributes on the entry point (`compiler.compile_kernel`), the `srem`
rewrite (`codegen._expand_srem`) and the buffer access expansion
(`buffers.expand_buffer_access`). They depend on how LLVM 22 prints IR and
may break with other LLVM versions.

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

### KI-22: lint warnings

Ruff reports a handful of style warnings (docstring mood, broad `except` in
the benchmark, `subprocess.run` without `check`). None affects behaviour.
