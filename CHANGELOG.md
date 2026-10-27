# Changelog

The entry of each release is written by python-semantic-release from the
messages of the commits it contains; see the development guide.

<!-- version list -->

## v0.1.1 (2026-10-27)

### Bench

- Label drtrigon's power-saving runs with a valid machine name
  ([`1eb82b8`](https://github.com/arunoruto/numba-vulkan/commit/1eb82b89b5c2bc27ea3730eded3ce4ed67909e7b))

### Bug Fixes

- Run spirv-val by the path shutil.which finds
  ([#52](https://github.com/arunoruto/numba-vulkan/pull/52),
  [`9dbb2fb`](https://github.com/arunoruto/numba-vulkan/commit/9dbb2fb1666a1c0ea2e69c955e0df53614262a03))

### Continuous Integration

- Publish an existing release tag; attach files with gh
  ([`a490093`](https://github.com/arunoruto/numba-vulkan/commit/a4900937b93f74aa55f87f80052759e2598582a2))

### Documentation

- Tests and benchmarks run in a copy of the repository
  ([`15eebe3`](https://github.com/arunoruto/numba-vulkan/commit/15eebe3705fb27c9f4722effdc0bfb210598187a))

### Testing

- Spirv-val behind a wrapper script on PATH
  ([#52](https://github.com/arunoruto/numba-vulkan/pull/52),
  [`782ea13`](https://github.com/arunoruto/numba-vulkan/commit/782ea132d15466b7def280d3ceeba0dcea901a16))

- The suite on GPUs without float64 and without libclc
  ([`1d271eb`](https://github.com/arunoruto/numba-vulkan/commit/1d271eb502b2120fe1e351f3049e9a3c8a6804bf))


## v0.1.0 (2026-10-16)

- Initial Release

## 0.1.0 at a glance

The first release of numba-vulkan, a Vulkan compute target for Numba, as
alpha software. Python kernels go through Numba's pipeline to LLVM IR,
through LLVM's SPIR-V backend, and run as Vulkan compute shaders. It was written by
an AI under human direction and review (see the authorship page of the
documentation).

Tested on an NVIDIA TITAN X (Pascal), Intel UHD Graphics 630 and Mesa's
llvmpipe on Linux, an Apple M3 Pro through MoltenVK, and Mesa's lavapipe on
Windows; the benchmarks also ran on an Intel Arc GPU under Windows.

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
