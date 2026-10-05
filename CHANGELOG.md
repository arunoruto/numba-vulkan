# Changelog

## 0.1.0 (unreleased)

The first release of numba-vulkan, a proof-of-concept Vulkan compute target
for Numba: Python kernels go through Numba's pipeline to LLVM IR, through
LLVM's SPIR-V backend, and run as Vulkan compute shaders. It was written by
an AI under human direction and review (see the authorship page of the
documentation).

Tested on an NVIDIA TITAN X (Pascal), Intel UHD Graphics 630, Mesa's
llvmpipe on Linux, and an Apple M3 Pro through MoltenVK.

### Kernels

- `@nv.jit` kernels launched with `kernel.forall(n)(...)` or
  `kernel[groups, local_size](...)`, on NumPy arrays, device arrays and
  scalars; device functions called from kernels; `numba.vectorize` and
  `numba.guvectorize` with `target="vulkan"`, including `reduce`.
- `math` functions and NumPy ufuncs, `float64` math from a bundled libclc,
  with workarounds for drivers whose built-in functions are inaccurate.
- Control flow of any shape, restructured into the form shaders need.
- Workgroup-shared memory, barriers, atomics, subgroup operations
  (`nv.subgroup`), local arrays, `print`, `float16` arrays, structured
  (record) arrays, NumPy arrays as global constants.
- Array views, slicing and slice assignment, iteration, reductions over
  whole arrays and along an axis, and lazy array expressions.
- Exceptions raised in kernels, reported by the launch; `boundscheck=True`.
- 32-bit integers by default where Numba uses 64-bit ones; devices without
  `float64`, `int64` or `int8` are supported.

### Runtime

- Device arrays: views, NumPy indexing (basic and advanced), NumPy ufuncs
  and operators computed on the device.
- Asynchronous launches on device arrays, batched into few submissions;
  scalars and shapes as push constants; launch plans that skip typing for
  repeated launches.
- Streams (`nv.stream`, `nv.pinned_array`) that overlap copies and kernels,
  events (`nv.event`) for timing on the device, and `nv.autotune` for
  choosing between candidate kernels.
- Compiled kernels cached on disk; `cache=True` keeps whole kernels across
  processes.
- `NUMBA_VULKAN_DEBUG` runs Khronos' validation layer, including its
  synchronization checks.

### Known issues

See the known-issues page of the documentation: among them, recursion,
arrays whose size is only known at run time, 64-bit atomics, and LLVM IR
rewritten as text.
