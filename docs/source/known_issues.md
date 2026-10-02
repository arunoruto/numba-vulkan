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

Numbers are not reused: KI-02, KI-03, KI-05, KI-06, KI-07, KI-08, KI-13,
KI-24 and KI-25 (NumPy functions on scalars, missing `math` functions,
allocating arrays in kernels, global constant arrays, complex numbers,
`print`, all data copied on every call, nested loop exits that failed to
compile, libclc linked in full for every kernel) have been fixed.

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

### KI-04: arrays of run-time size

```python
x[x > 0]            # TypingError: indexing with arrays ... is not supported
x.copy()            # VulkanUnsupportedError: ... size is only known at run time
np.zeros(n)         # TypingError: ... needs a constant shape
x.sum(axis=0)       # TypingError
x[1:] = x[:-1] * 2  # ValueError: the expression reads the array ... at other positions
```

Shaders cannot allocate memory, so nothing can create an array whose size
is only known when the kernel runs. Arrays of a constant shape
(`np.zeros(4)`, `nv.local.array`) and array expressions, which are computed
element by element where they are used, cover the rest (see {doc}`usage`).

Because an expression is computed while it is being assigned, assigning it
to an array it reads is only allowed where it reads each element at the
position it writes, as in `a[:] = a * 2`; anything else is detected at run
time and raises `ValueError`, where NumPy would compute a temporary first.

**Fix:** reductions along an axis can be added as `@overload`s on top of
expressions, as `arrayfuncs.py` does for whole-array reductions.

### KI-09: structured (record) arrays

**Symptom:** `VulkanUnsupportedError: arrays of Record(...)`.

Supported element types are `bool`, signed and unsigned integers of 8 to 64
bits, `float16` (stored as halves, computed as `float32`), `float32` and
`float64`. NumPy arrays that are not C-contiguous (strided,
Fortran-ordered) work, but are copied to a contiguous array on the host
for every call; device arrays are always contiguous.

**Fix:** a record would map to a SPIR-V struct in the buffer, with field
access as member access chains; Numba's record model assumes a data
pointer, so this needs its own type, like `VulkanArray`.

## Behaviour that differs from Numba on the CPU

### KI-10: errors are reported after the kernel

An exception raised with `raise`, an index out of bounds with
`boundscheck=True`, and an integer division by zero with
`error_model="python"` are raised by the launch once the kernel has
finished (see {doc}`usage`). The invocation that raised stops; the others
run to completion. There is no `try`/`except`. By default, indices are not
checked and divisions by zero give NumPy's results, as in `numba.cuda`.

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

### KI-28: launch overhead and synchronous launches with NumPy arrays

Launches whose arrays are all device arrays return at once, and a repeated
launch costs about 50 µs of Python time. Launches with NumPy arrays, and
of kernels that can raise, still wait for the kernel, because results and
the error status have to be read back. Scalars and array shapes travel as
small buffers rather than push constants. If a grid needs more workgroups
than the device allows, it is dispatched in parts, and `num_groups` then
reports the workgroups of the part, which breaks grid-stride loops over
`num_groups`.

**Fix:** push constants for scalars and shapes; reporting errors of
asynchronous launches at the next synchronisation, as CUDA does.

### KI-29: device arrays are bare buffers

A `DeviceArray` can be created, passed to kernels and copied; it has no
indexing, slicing, views or arithmetic, and no `__cuda_array_interface__`
equivalent for sharing memory with other libraries.

### KI-14: Numba's part of compilation is repeated in every process

The SPIR-V of a kernel is cached on disk (see {doc}`how_it_works`), but
type inference and lowering run again in every process, because the cache
is keyed by their output. That leaves about 0.05 s per kernel, and the
first kernel of a process additionally pays some 0.2 s for Numba's own
start-up. Driver pipelines are not cached by this package either; the
drivers keep shader caches of their own.

**Fix:** a second cache level keyed by source, like Numba's `cache=True`,
which has to decide when a function, the functions it calls and the
globals it reads have changed.

### KI-15: one specialisation per buffer binding

The binding is part of the array type, so a function that takes arrays is
compiled again for every combination of bindings it is called with. This is
invisible for kernels but multiplies work for shared helper functions.

### KI-16: 64-bit integer atomics and float64 atomics

LLVM's SPIR-V backend does not offer the `Int64Atomics` capability for
Vulkan, so atomics on `int64` and `uint64` arrays are rejected unless the
kernel is narrowed (`narrow="ints"`). `float64` atomics would need a 64-bit
compare-and-swap, which the backend cannot emit either. Its 32-bit
compare-and-swap needs a repair of the generated SPIR-V
(`codegen.fix_compare_exchange`), as do barriers
(`codegen.fix_barrier_semantics`).

**Fix:** in LLVM's SPIR-V backend.

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
