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
    buffers.py      buffer access placeholders and their expansion
    compiler.py     Numba pipeline, kernel entry point
    codegen.py      code library: link, optimise, emit and check SPIR-V
    _emit.py        child process running LLVM's SPIR-V backend
    dispatcher.py   @nv.jit, VulkanDispatcher, forall
    runtime.py      Vulkan devices, buffers, pipelines, dispatch
    stubs.py        functions that only exist inside kernels
    errors.py       exception types
tests/
    test_kernels.py        language features, per device
    test_target.py         target extension API features, per device
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
```

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
| `NUMBA_DUMP_IR=1`, `NUMBA_DUMP_LLVM=1` | Numba's own dumps of its IR and of unoptimised LLVM IR |

Where a failure comes from tells you where to look:

| Error | Stage | Look at |
| --- | --- | --- |
| `TypingError` | Numba type inference | `vkdecl.py`, `target.VulkanTypingContext` |
| `No definition for lowering ...` | lowering lookup | `vkimpl.py`, `mathimpl.py` |
| `VulkanUnsupportedError` | lowering | the message names the construct |
| `SpirvCodegenError: LLVM's SPIR-V backend failed` | LLVM | `compiled.llvm_ir`; reduce the kernel |
| `SpirvCodegenError: generated SPIR-V is invalid` | LLVM or the passes | `spirv-dis` output around the reported line |
| `VulkanSupportError` | device features | `runtime.DeviceInfo`, `compiled.capabilities` |
| `VkError...` or a crash in the driver | driver | run on llvmpipe; validate the shader |

## Adding a `math` function

Functions that map to one LLVM intrinsic are a table entry in `mathimpl.py`:
add them to `_F32_ONLY` if GLSL.std.450 lacks a `float64` version, to
`_ANY_FLOAT` otherwise.

Anything else is a lowering function:

```python
@lower(math.isnan, types.Float)
def lower_isnan(context, builder, sig, args):
    """Lower ``math.isnan``."""
    return builder.fcmp_unordered("uno", args[0], args[0])
```

Functions that can be written in Python are easier as overloads, since they
are compiled by Numba like user code:

```python
@overload(math.hypot, target="vulkan")
def ol_hypot(x, y):
    return lambda x, y: math.sqrt(x * x + y * y)
```

Then move the case from `tests/test_known_issues.py` to a regular test and
update {doc}`known_issues`.

## Rules that were learned the hard way

Never hand unchecked SPIR-V to a driver
: NVIDIA's shader compiler segfaults on some invalid modules, taking the
  Python process with it. Everything goes through `codegen.check_spirv`;
  keep `NUMBA_VULKAN_VALIDATE=1` on while developing.

Never run LLVM's SPIR-V backend in-process
: It aborts on input it cannot handle. `codegen.emit_spirv` runs it in a
  child process for that reason.

Do not create pointers into buffers before optimisation
: LLVM merges loads from different elements into a load through a selected
  pointer, which logical-addressing SPIR-V cannot express. Use
  `buffers.load_element` and `buffers.store_element`, which stay opaque
  until after optimisation.

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
