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
KI-14, KI-24, KI-25, KI-26, KI-30 and KI-31 (NumPy functions on scalars,
missing `math` functions, allocating arrays in kernels, global constant
arrays, complex numbers, `print`, all data copied on every call, nested
loop exits that failed to compile, Numba's compilation repeated in every
process, libclc linked in full for every kernel, libclc depending on an old
NixOS release, `gamma` losing precision for large arguments, `float64`
functions losing precision on Intel's and Mesa's drivers) have been fixed.

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
finished, or by the next synchronisation for launches on device arrays,
which do not wait (see {doc}`usage`). The invocation that raised stops; the others
run to completion. There is no `try`/`except`. By default, indices are not
checked and divisions by zero give NumPy's results, as in `numba.cuda`.

### KI-11: `round()` on llvmpipe

`round(1.5)` gives `1.0` on llvmpipe; NVIDIA's driver gives the correct
`2.0`. Halfway cases should not be relied on.

### KI-12: recursion

Rejected with a clear error, because all functions are inlined into the
entry point. SPIR-V forbids recursion, so this will not change.

### KI-23: less accurate math without libclc or with `fastmath`

With libclc, math functions are accurate to the last digit or two in both
precisions. Without it, and
for `float32` with `fastmath=True`, the drivers' and this package's own
versions are used: `lgamma` is then
accurate to about 1e-5, as is `gamma` without libclc (with libclc,
`gamma` ignores `fastmath`), `erf` and `erfc` have an absolute error
of about 1.5e-7, and the trigonometric functions are only as good as the
driver's.

### KI-33: `float32` square roots are off by one ulp on some drivers

Vulkan lets `sqrt` be as inaccurate as `1 / inversesqrt`. For random
arguments, NVIDIA's driver (580.x, TITAN X) returns the neighbour of the
correctly rounded `float32` result for 17 % of them, Intel's (Mesa, UHD 630)
for 8 %; llvmpipe, and `float64` everywhere, are exact. Python and NumPy
round correctly, so results can differ in the last bit.

**Fix:** correct the driver's result by comparing the squares of it and its
neighbours with `x`, computed exactly (Dekker's product), for kernels
without `fastmath`.

### KI-27: very large functions are not restructured

SPIR-V needs structured control flow, and `structurize.py` rearranges the
graph to provide it (see {doc}`how_it_works`). Kernels with more than 20000
basic blocks after inlining are passed on unchanged; LLVM's own
structurizer then usually fails with `SpirvCodegenError`, and a module it
gets wrong is rejected by `codegen.check_structure`. A kernel of 5700
blocks (sixty random programs with nested loops, inlined one after the
other) is restructured in about 2 s, but takes some 90 s to compile in all,
mostly in LLVM.

## Performance

### KI-28: launch overhead and synchronous launches with NumPy arrays

Launches whose arrays are all device arrays return at once, but a launch
still costs some 45–50 µs of Python time (typing the arguments, packing push
constants, recording the commands), which limits small kernels to about
20 000 launches per second. Launches with NumPy arrays wait for the kernel,
because their results have to be copied back.

**Fix:** a launch plan cached per argument types, so that a repeated launch
only packs values and records; or recording in C.

### KI-29: device arrays cannot be shared with other libraries

A `DeviceArray` supports basic indexing, views and NumPy's ufuncs, but
there is no `__cuda_array_interface__` or DLPack equivalent for handing its
memory to other Vulkan libraries, and no advanced indexing (arrays or
lists as indices).

### KI-15: one specialisation per buffer binding

The binding is part of the array type, so a function that takes arrays is
compiled again for every combination of bindings it is called with. This is
invisible for kernels but multiplies work for shared helper functions.

### KI-16: 64-bit integer atomics and float64 atomics

LLVM's SPIR-V backend does not offer the `Int64Atomics` capability for
Vulkan, so atomics on `int64` and `uint64` arrays are rejected in kernels
compiled with 64-bit integers (`narrow=False`); by default kernels use
32-bit integers and they work. `float64` atomics would need a 64-bit
compare-and-swap, which the backend cannot emit either. Its 32-bit
compare-and-swap needs a repair of the generated SPIR-V
(`codegen.fix_compare_exchange`), as do barriers
(`codegen.fix_barrier_semantics`).

**Fix:** in LLVM's SPIR-V backend.

## Portability

### KI-17: narrowing to 32-bit types is untested on real hardware

Devices without `shaderFloat64` or `shaderInt64` run kernels narrowed to
32-bit types (`narrowing.py`; see {doc}`how_it_works`). The tests exercise
this on the Linux devices, opened without those features, where the
kernels also declare no optional capability at all. The Apple M3 Pro
(MoltenVK) really lacks `shaderFloat64`, and the test suite passes on it;
tests that need `float64` precision are marked `float64` and skipped there.
No device that really lacks `shaderInt64` has been tried, and such drivers
tend to have restrictions of their own.

Narrowing is all or nothing per type: one `float64` value that is really
needed, for example a sum that must not lose precision, cannot be kept.
Kernels that use `int8` or `int16` arrays still need those features.

### KI-18: tested on four devices only

NVIDIA TITAN X (Pascal), Intel UHD Graphics 630 and llvmpipe on Linux, and
an Apple M3 Pro through MoltenVK 1.4.2 on macOS. AMD, Windows and mobile
GPUs are untested. Two driver problems were found on the Linux sample alone
(see {doc}`how_it_works`), and macOS needs another LLVM (KI-32), so more
should be expected.

### KI-32: llvmlite's SPIR-V backend crashes on macOS arm64

The SPIR-V backend in llvmlite 0.50.0's macOS arm64 wheel (LLVM 22.1.0)
segfaults on common kernels: those of 33 tests in the suite, among them
clamps, `min`, `max`, `hypot`, complex arithmetic and float atomics.
`llvm-reduce` brings one down to a `select` on an `fcmp`, followed by any
other `select`:

```llvm
define i32 @main(i1 %c, float %f, i32 %i) {
  %k = fcmp ogt float %f, 0.0
  %a = select i1 %k, float 0.0, float 1.0
  %b = select i1 %c, i32 %i, i32 0
  ret i32 %b
}
```

The same modules translate with nixpkgs' `llc` of LLVM 22.1.8 on the same
machine. llvmlite's Linux x86-64 wheel contains the same LLVM 22.1.0 and
translates them, so the version alone does not explain the crash; the arm64
build is the more likely culprit. On Linux the test suite also passes with
`NUMBA_VULKAN_LLC` set to that `llc`, on all three devices. The crash is in
a child process (see {doc}`development`) and is reported as "the backend
was killed by SIGSEGV".

**Workaround:** set `NUMBA_VULKAN_LLC` to an `llc` of LLVM 22; every module
is then translated by it instead. The devenv shell does this on macOS.
`tests/test_known_issues.py::test_llvmlite_backend_on_macos_arm64` turns
red once llvmlite's own backend translates the case above; the setting and
this entry can then go.

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
be re-evaluated when either changes. For the rewrites that work around LLVM's
SPIR-V backend, `tests/test_backend_limits.py` hands the backend each
construct unrewritten and checks that it still fails; after an LLVM upgrade,
a failing test there names a rewrite that may have become unnecessary. With
LLVM 22.1, `llvm.fmuladd` no longer needs rewriting for the backend's sake;
it is still split, so that results do not depend on whether a driver fuses. The benchmark dependency group pins
`numpy<2.5`, because numba-cuda 0.30.4 does not import with NumPy 2.5.

### KI-22: lint warnings

Ruff reports a handful of style warnings (docstring mood, broad `except` in
the benchmark, `subprocess.run` without `check`). None affects behaviour.
