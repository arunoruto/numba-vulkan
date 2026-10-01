# How it works

```text
Python function
   │  Numba: bytecode analysis, type inference, lowering
   ▼
LLVM IR (one module per function)
   │  link, inline everything into the entry point, simplify
   ▼
LLVM IR (one function, buffer access still abstract)
   │  expand buffer access, LLVM SPIR-V backend (child process)
   ▼
SPIR-V compute shader
   │  Vulkan: upload arguments, dispatch, read results back
   ▼
NumPy arrays
```

## A Numba target

The package registers a `vulkan` target in Numba's target registry, below
the generic `gpu` target, together with a dispatcher and a `jit` decorator.
It follows the structure of Numba's CUDA target:

| Piece | Module | Role |
| --- | --- | --- |
| Target and contexts | {py:mod}`numba_vulkan.target` | typing context, target context, calling convention |
| Types and data models | {py:mod}`numba_vulkan.vktypes`, {py:mod}`numba_vulkan.models` | the array type and its LLVM representation |
| Typing and lowering | {py:mod}`numba_vulkan.vkdecl`, {py:mod}`numba_vulkan.vkimpl`, {py:mod}`numba_vulkan.mathimpl` | `global_id`, array indexing, `math` |
| Pipeline | {py:mod}`numba_vulkan.compiler` | Numba compiler pipeline and the kernel entry point |
| Code generation | {py:mod}`numba_vulkan.codegen`, {py:mod}`numba_vulkan.buffers` | linking, optimisation, SPIR-V emission |
| Dispatcher | {py:mod}`numba_vulkan.dispatcher` | `@nv.jit`, `forall`, specialisation cache |
| Runtime | {py:mod}`numba_vulkan.runtime` | devices, buffers, pipelines, dispatch |

Kernels go through Numba's stock compiler passes and its stock lowering, so
scalar code, tuples, `range` loops, casts and Numba's generic overloads work
without Vulkan-specific code.

## Arrays without pointers

Numba represents an array as a structure holding a data pointer. Vulkan
shaders use *logical addressing*: there are no general pointers, and memory
is reached only through buffers bound at fixed descriptor bindings.

numba-vulkan therefore uses its own array type,
{py:class}`~numba_vulkan.vktypes.VulkanArray`, which records the binding of
its buffer **in the type**. Its runtime value holds only metadata (item
count, shape, strides), under the same member names as Numba's array model,
so Numba's implementations of `.shape`, `.size` and `len()` work unchanged.
Indexing is implemented by the target.

One consequence: a function taking arrays is compiled once per combination
of bindings it is called with.

The bindings of a kernel are laid out as follows:

| Binding | Contents |
| --- | --- |
| 0 | shapes of all array arguments, as `int32` |
| `1 + k` | argument `k`: the array data, or a one-element buffer for a scalar |

## Deferring buffer access

LLVM's optimiser freely merges two loads from different array elements into
one load through a selected pointer. SPIR-V cannot express that, and at least
one driver crashes on it.

Element access is therefore emitted as a call to an opaque placeholder
function that takes a binding and an index and involves no pointer at all.
Only after optimisation are the placeholders expanded into LLVM's Vulkan
resource intrinsics.

## Inlining and optimisation

Numba's calling convention returns values through pointers and reports
errors through status codes. Neither survives in a shader, so all functions
are inlined into a single entry point and cleaned up with a small, fixed set
of LLVM passes (inlining, scalar replacement of aggregates, instruction
combining, CFG simplification, dead code elimination).

LLVM's full optimisation pipeline is deliberately not used: it produces
constructs, such as vector operations and lookup tables, that the SPIR-V
backend does not handle.

## SPIR-V emission

LLVM's SPIR-V backend, shipped in llvmlite since LLVM 20, has a Vulkan mode
that produces structured control flow and storage-buffer access. It runs in
a child process, because it aborts the whole process on input it cannot
handle; a failure then surfaces as
{py:class}`~numba_vulkan.errors.SpirvCodegenError`.

Before a module is handed to a driver it is checked for constructs known to
crash drivers.

## Runtime

{py:mod}`numba_vulkan.runtime` is a small Vulkan compute runtime on top of
the [`vulkan`](https://pypi.org/project/vulkan/) bindings. For each call it
creates one host-visible storage buffer per binding, uploads the arguments,
dispatches the shader and reads back the buffers the shader writes to.
Pipelines are cached per device and kernel specialisation.

## Workarounds for toolchain and driver behaviour

| Observation | Workaround |
| --- | --- |
| NVIDIA's driver returns wrong results for signed 64-bit remainders with negative operands | remainders are computed as `a - (a / b) * b` |
| NVIDIA's shader compiler segfaults on some invalid SPIR-V | modules are checked before use; buffer access is deferred |
| The SPIR-V backend miscompiles nested aggregate inserts | aggregates are folded away before emission |
| The SPIR-V backend aborts the process on unsupported input | emission runs in a child process |
| Numba's integer power falls back to a `float64` `pow` | integer exponents are implemented with multiplications |
