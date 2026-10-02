# Development guide

For anyone continuing this work. Read {doc}`how_it_works` first; this page
covers the practical side.

## Layout

```text
src/numba_vulkan/
    target.py       target registration, typing and target contexts
    vktypes.py      VulkanArray, VulkanDispatcherType
    models.py       LLVM data models of those types
    vkdecl.py       typing of global_id
    vkimpl.py       lowering of global_id and array indexing
    mathimpl.py     lowering of the math module and integer powers
    mathfuncs.py    math functions written in Python (hypot, erf, gamma...)
    ufuncs.py       which implementation each NumPy ufunc loop uses
    libclc.py       locating and calling libclc, the math library
    buffers.py      buffer access placeholders and their expansion
    compiler.py     Numba pipeline, kernel entry point
    codegen.py      code library: link, optimise, emit and check SPIR-V
    legalize.py     rewrites of IR the SPIR-V backend cannot translate
    structurize.py  control-flow restructuring on LLVM IR text
    _emit.py        child process running LLVM's SPIR-V backend
    dispatcher.py   @nv.jit, VulkanDispatcher, forall
    runtime.py      Vulkan devices, buffers, pipelines, dispatch
    stubs.py        functions that only exist inside kernels
    errors.py       exception types
tests/
    test_kernels.py        language features, per device
    test_target.py         target extension API features, per device
    test_math.py           math functions and NumPy ufuncs, per device
    test_control_flow.py   early exits, short-circuit conditions, per device
    test_structurize.py    unit tests of the control-flow restructuring
    fuzz_control_flow.py   random-program fuzzer (a tool, not collected)
    test_known_issues.py   expected failures, one per known issue
benchmarks/bench.py
docs/source/
```

## Commands

```sh
devenv shell                                   # or: uv sync
uv run pytest                                  # all tests, all devices
uv run pytest -k llvmpipe                      # one device
uv run pytest tests/test_known_issues.py -rxX  # list known issues
uv run python benchmarks/bench.py
cd docs && uv run sphinx-build -M html ./source ./build -W
build-dist                                     # sdist and wheel, with libclc
```

`build-dist` is a devenv script: it copies libclc's `clspv--.bc` from the
Nix store to `src/numba_vulkan/data/` (ignored by git) and runs `uv build`,
so the distributions in `dist/` carry the math library. A plain `uv build`
without that file produces a wheel without it.

On NixOS, commands must run inside `devenv shell`; the wheels in the
virtual environment do not find their shared libraries otherwise.

## Debugging a kernel

Compile without running, then look at each stage:

```python
from numba import float32
compiled = kernel.compile((float32[::1], float32[::1]))

print(compiled.llvm_ir)           # after inlining and optimisation
open("k.spv", "wb").write(compiled.spirv)
```

```sh
spirv-val --target-env vulkan1.2 k.spv
spirv-dis k.spv | less
```

The unoptimised IR of a single function is available from
`kernel.compile_device(argtypes).library.get_llvm_str()`.

Useful environment variables:

| Variable | Effect |
| --- | --- |
| `NUMBA_VULKAN_VALIDATE=1` | run every shader through `spirv-val`; the test suite sets this |
| `NUMBA_VULKAN_DEVICE=llvmpipe` | default device, by name substring or index |
| `NUMBA_VULKAN_LIBCLC=/path/clspv--.bc` | where to find libclc; the devenv shell sets it |
| `NUMBA_VULKAN_CACHE_DIR=/path` | where the prepared copy of libclc is kept (default `~/.cache/numba-vulkan`) |
| `NUMBA_VULKAN_POOL_MB=1024` | size of the per-device pool of released buffers |
| `NUMBA_BOUNDSCHECK=1` | check array indices in all kernels |
| `NUMBA_VULKAN_NARROW=1` | compile every kernel without 64-bit types, as for a device that lacks them; the test suite then fails only where it expects float64 accuracy or a narrowing warning |
| `NUMBA_DUMP_IR=1`, `NUMBA_DUMP_LLVM=1` | Numba's own dumps of its IR and of unoptimised LLVM IR |

Where a failure comes from tells you where to look:

| Error | Stage | Look at |
| --- | --- | --- |
| `TypingError` | Numba type inference | `vkdecl.py`, `target.VulkanTypingContext` |
| `No definition for lowering ...` | lowering lookup | `vkimpl.py`, `mathimpl.py` |
| `VulkanUnsupportedError` | lowering | the message names the construct |
| `SpirvCodegenError: LLVM's SPIR-V backend failed` | LLVM | `compiled.llvm_ir`; reduce the kernel |
| `SpirvCodegenError: generated SPIR-V is invalid` | LLVM or the passes | `spirv-dis` output around the reported line; for control flow, `structurize.py` |
| `VulkanSupportError` | device features | `runtime.DeviceInfo`, `compiled.capabilities` |
| `VkError...` or a crash in the driver | driver | run on llvmpipe; validate the shader |

## Adding a `math` function

Check libclc first: if it has the function, add it to the `_LIBCLC` table
in `mathimpl.py` (and register a lowering, as the functions next to it do).
It then works in both precisions. A function that makes LLVM's backend
fail usually needs one more rewrite in `legalize.py`; see
{doc}`math_library`.

Functions that map to one LLVM intrinsic and are exact (like `sqrt`) are a
table entry in `mathimpl.py`: `_F32_ONLY` if GLSL.std.450 lacks a `float64`
version, `_ANY_FLOAT` otherwise.

Fallbacks written in Python go into `mathfuncs.py`. A factory receives the
float type and returns the implementation; all constants are created in
that type, because a Python literal would promote the computation to
`float64`:

```python
def _acosh(ty):
    one = ty(1)

    def acosh(x):
        return math.log(x + math.sqrt(x * x - one))

    return acosh
```

To make the matching NumPy ufunc work as well, add it to the tables in
`ufuncs.py`. Then add a case to `tests/test_math.py`.

## Fuzzing control flow

`tests/fuzz_control_flow.py` generates random functions made of nested
`if`/`elif`/`else`, `and`/`or`, `for` and `while` loops, `break`,
`continue` and early `return`s, runs them on Vulkan and compares with plain Python:

```sh
uv run python tests/fuzz_control_flow.py 0 100          # seeds 0..99, all devices
uv run python tests/fuzz_control_flow.py 0 100 --cpu    # llvmpipe only
uv run python tests/fuzz_control_flow.py 0 100 --rich   # also while loops, deeper
uv run python tests/fuzz_control_flow.py --show 42      # print one program
```

Use it after any change to `structurize.py` or to the pass list.

## Rules that were learned the hard way

Never hand unchecked SPIR-V to a driver
: NVIDIA's shader compiler segfaults on some invalid modules, taking the
  Python process with it. Everything goes through `codegen.check_spirv`;
  keep `NUMBA_VULKAN_VALIDATE=1` on while developing.

Never run LLVM's SPIR-V backend in-process, or twice in one process
: It aborts on input it cannot handle, and the second module translated by
  a process comes out invalid ("Id 1 is defined more than once").
  `codegen.Emitter` therefore keeps a helper process that forks once per
  module.

Do not create pointers into buffers before optimisation
: LLVM merges loads from different elements into a load through a selected
  pointer, which logical-addressing SPIR-V cannot express. Use
  `buffers.load_element` and `buffers.store_element`, which stay opaque
  until after optimisation.

Do not trust LLVM's structurizer
: It produces invalid SPIR-V for early returns, short-circuit conditions and
  returns inside loops. `structurize.py` exists to hand it a graph that
  needs no repair. A wrong result here would be silent, so every change
  needs the fuzzer and a validator.

instcombine re-creates what you avoided
: Spelling out `copysign` as bit operations, or a NaN test as `x != x`, does
  not help: instcombine recognises the pattern and emits the intrinsic or
  comparison the backend cannot handle. Such things are rewritten on the IR
  text *after* the passes have run, in `legalize.py`.

Never emit `select(a * b < 0, x, -x)` as is
: LLVM's SPIR-V backend rewrites that shape into GLSL's `faceforward` and
  crashes for scalars. `legalize.avoid_faceforward` hides the comparison;
  do not remove it because it "looks redundant".

Call libclc with its calling convention
: libclc functions are `spir_func`. A call with the default convention is
  undefined behaviour, which LLVM silently turns into unreachable code.
  Always go through `libclc.call`.

Helpers compiled by `compile_internal` need the NumPy error model
: Numba's default adds a zero-division check, and an extra exit, to every
  division. `VulkanTargetContext._compile_subroutine_no_cache` selects the
  NumPy error model for that reason.

Do not add LLVM passes casually
: `instcombine` is required (the backend miscompiles the aggregates Numba
  builds), NewGVN was tried and breaks the backend, and the standard
  optimisation pipelines introduce vectors and lookup tables the backend
  rejects. Any change to the pass list in
  `VulkanCodeLibrary._link_and_optimize` needs the full test suite on all
  devices.

A lowering registry cannot override Numba's concrete registrations
: Numba registers some implementations per concrete type (integer powers,
  for example). Those win over class-level entries in the target's
  registries. Intercept them in `VulkanTargetContext.get_function`, as
  `mathimpl.power_override` does.

Shader compilers do not keep float arithmetic as written
: Without the `NoContraction` decoration that `codegen.mark_exact` adds, all
  three tested drivers folded `(1 + x) - 1` to `x`. Algorithms that depend
  on rounding behaviour need it, and they still cannot rely on accurate
  `log`/`exp` near their critical points.

Check results on more than one device
: The same valid shader gave different integer results on NVIDIA than on
  Intel and llvmpipe. The test fixtures run every kernel on every device
  for this reason; llvmpipe is the most trustworthy reference.

Avoid `float64` in tests unless it is the point
: Python literals are `float64`. Use `np.float32(...)` constants, or the
  kernel will need `float64` support and fail on 32-bit-only math.

## Reference code

- Numba's in-tree CUDA target (`numba/cuda/` in the installed package) is
  the template for `target.py`, `compiler.py` and `dispatcher.py`.
- [numba-cuda](https://github.com/NVIDIA/numba-cuda) is its maintained
  successor and the place to look for device arrays, ufuncs and caching.
- LLVM's tests under `llvm/test/CodeGen/SPIRV/hlsl-resources/` show the
  resource intrinsics that `buffers.py` generates.
- [Taichi](https://github.com/taichi-dev/taichi) (`taichi/codegen/spirv/`,
  `taichi/rhi/vulkan/`) has a mature SPIR-V emitter and Vulkan runtime.

## Contributions by AI tools

See {doc}`authorship`. Contributions written with AI tools are welcome and
must say so, and a human has to review them before they are merged.
