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

Numbers are not reused: KI-02, KI-03, KI-06, KI-07, KI-13, KI-24 and KI-25
(NumPy functions on scalars, missing `math` functions, global constant
arrays, complex numbers, all data copied on every call, nested loop exits
that failed to compile, libclc linked in full for every kernel) have been
fixed.

## Language and library coverage

### KI-01: `float64` math needs libclc in a source checkout

```python
out[i] = math.sin(x[i])     # x is float64
out[i] = x[i] ** 2.5        # x is float64, or float32 with a Python float
```

Wheels contain libclc, and the devenv shell provides it, so these work
with full double accuracy (see {doc}`math_library`). A plain source
checkout or `pip install git+https://...` does not have it:

**Symptom:** `VulkanUnsupportedError: math.pow on float64 needs libclc`.

**Cause:** Vulkan's own math library defines `sin`, `exp`, `log`, `pow` and
friends for 32-bit floats only, and libclc's bitcode is a 2.7 MB binary
that is not kept in the repository.

**Workaround:** install from a wheel, point `NUMBA_VULKAN_LIBCLC` at
`clspv--.bc`, keep the computation in `float32`, or use
`@nv.jit(narrow_math=True)`.

### KI-04: array expressions and fancy indexing

```python
(x * 2)[i]        # VulkanUnsupportedError: ... needs direct access to memory
x[x > 0]          # the same
x.sum(axis=0)     # TypingError
x[1:] = x[:-1]    # VulkanUnsupportedError: the slices could overlap
```

Indexing with integers and slices, views, iteration and whole-array
reductions work (see {doc}`limitations`). What does not is everything that
has to create a new array, because there is nowhere to put it (KI-05), and
Numba's own implementations of other array functions, which walk the data
pointer that arrays on this target do not have.

**Fix:** further functions can be added as `@overload`s for the `vulkan`
target on top of indexing, as `arrayfuncs.py` does for the reductions.
Array expressions need local arrays first.

### KI-05: allocating arrays inside a kernel

**Symptom:** `np.zeros(...)` fails typing, because the target has no memory
allocator (Numba's NRT is disabled).

**Workaround:** tuples work as small fixed-size arrays; larger scratch
space has to be passed in as an argument.

**Fix:** fixed-size local arrays could map to SPIR-V function-local
variables. Nothing exists yet.

### KI-08: `print`

**Symptom:** `No definition for lowering <built-in function print>`.

**Fix:** the `debugPrintf` extension of the validation layers could back
it, as Taichi does. Low priority.

### KI-09: unsupported array kinds

| Kind | Symptom |
| --- | --- |
| `float16` arrays | `NotImplementedError: float16` |
| structured (record) arrays | `VulkanUnsupportedError: arrays of Record(...)` |

Supported element types are `bool`, signed and unsigned integers of 8 to 64
bits, `float32` and `float64`. NumPy arrays that are not C-contiguous
(strided, Fortran-ordered) work, but are copied to a contiguous array on the
host for every call; device arrays are always contiguous.

## Behaviour that differs from Numba on the CPU

### KI-10: only explicit errors are reported

An exception raised with `raise` is reported by the launch (see
{doc}`usage`), and `boundscheck=True` adds index checks. Everything else
that raises on the CPU does not: integer division by zero yields an
unspecified value, and array indices are unchecked by default, as in
`numba.cuda`. There is no `try`/`except`, and an exception stops only the
invocation that raised it.

**Fix:** arithmetic errors would need Numba's Python error model, which
adds a test to every division; it could become an option like
`boundscheck`.

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

### KI-14: compiled kernels are not cached on disk

A kernel compiles in 0.05 to 0.25 s, but every process compiles its kernels
again: SPIR-V and pipelines are cached in memory only. Kernels that call
math functions spend about 0.1 s of that parsing libclc.

**Fix:** cache SPIR-V on disk keyed by the LLVM IR, as Numba's
`cache=True` does for the CPU target, and pass a `VkPipelineCache` to
pipeline creation.

### KI-15: one specialisation per buffer binding

The binding is part of the array type, so a function that takes arrays is
compiled again for every combination of bindings it is called with. This is
invisible for kernels but multiplies work for shared helper functions.

### KI-16: fixed workgroup sizes

The workgroup size is fixed at 64, 8×8 or 4×4×4 depending on the grid
dimensionality (`compiler.LOCAL_SIZES`). There is no shared memory, no
atomics and no barriers.

## Portability

### KI-17: narrowing to 32-bit types is untested on real hardware

Devices without `shaderFloat64` or `shaderInt64` run kernels narrowed to
32-bit types (`narrowing.py`; see {doc}`how_it_works`). The tests exercise
this on the three desktop devices, opened without those features, where
the kernels also declare no optional capability at all. No device that
really lacks the features has been tried, and such drivers tend to have
restrictions of their own.

Narrowing is all or nothing per type: one `float64` value that is really
needed, for example a sum that must not lose precision, cannot be kept.
Kernels that use `int8` or `int16` arrays still need those features.

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

### KI-26: the bundled libclc is tied to an old nixpkgs revision

Wheels bundle `clspv--.bc`, which `build-dist` copies from the Nix store.
Current nixpkgs has removed libclc, so the devenv shell pins an older
revision for it (input `nixpkgs-libclc`). The file must come from an LLVM
no newer than the one inside llvmlite, so the bundled copy (LLVM 22) relies
on the `llvmlite>=0.50` requirement, and a newer libclc cannot be bundled
until llvmlite moves on.

**Fix:** build libclc's clspv target from the LLVM sources in a small Nix
derivation of this project's own, pinned to llvmlite's LLVM version.

### KI-22: lint warnings

Ruff reports a handful of style warnings (docstring mood, broad `except` in
the benchmark, `subprocess.run` without `check`). None affects behaviour.
