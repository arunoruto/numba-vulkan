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
| Typing and lowering | {py:mod}`numba_vulkan.vkdecl`, {py:mod}`numba_vulkan.vkimpl`, {py:mod}`numba_vulkan.mathimpl`, {py:mod}`numba_vulkan.mathfuncs`, {py:mod}`numba_vulkan.ufuncs` | `global_id`, array indexing, `math`, NumPy ufuncs |
| Pipeline | {py:mod}`numba_vulkan.compiler` | Numba compiler pipeline and the kernel entry point |
| Code generation | {py:mod}`numba_vulkan.codegen`, {py:mod}`numba_vulkan.structurize`, {py:mod}`numba_vulkan.buffers` | linking, optimisation, control-flow restructuring, SPIR-V emission |
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
combining, CFG simplification, dead code elimination, loop simplification
and demotion of phi nodes to memory).

LLVM's full optimisation pipeline is deliberately not used: it produces
constructs, such as vector operations and lookup tables, that the SPIR-V
backend does not handle.

## Structured control flow

SPIR-V requires structured control flow: every conditional branch names a
merge block, selections nest properly, and a loop has one continue block and
one merge block. Python code does not look like that after compilation:

- several early `return`s all jump to the same continuation;
- `if a or b: ... else: ...` enters its first branch from two places;
- a `return` inside a loop leaves the loop without passing its exit.

LLVM's SPIR-V backend contains a structurizer for such graphs, but in LLVM
22 it produces invalid modules for many of them. {py:mod}`numba_vulkan.structurize`
therefore brings the graph into properly nested form before code
generation, in three steps:

1. **Loop exits.** Every loop gets a single exit block. Blocks that leave
   the loop record which target they wanted in a stack slot, and a chain of
   tests after the exit block dispatches to it.
2. **Unstructured joins.** A block entered from several places that is not
   the merge block of a selection is copied, once per entry.
3. **Shared merge blocks.** A selection that shares its merge block with an
   enclosing one gets a merge block of its own that forwards to the shared
   one.

To make these rewrites simple, the IR is first brought into a form without
phi nodes (LLVM's `reg2mem`) and with canonical loops (`loop-simplify`).

This step is the least mature part of the compiler; see KI-24 in
{doc}`known_issues`.

## SPIR-V emission

LLVM's SPIR-V backend, shipped in llvmlite since LLVM 20, has a Vulkan mode
that produces structured control flow and storage-buffer access. It runs in
a child process, because it aborts the whole process on input it cannot
handle; a failure then surfaces as
{py:class}`~numba_vulkan.errors.SpirvCodegenError`.

Two things happen to the binary afterwards. Every float operation is
decorated `NoContraction`, because shader compilers otherwise reassociate
arithmetic freely, which Numba code does not expect. And the module is
checked for constructs known to crash drivers before it is handed to one.

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
| LLVM's structurizer emits invalid SPIR-V for early exits and short-circuit conditions | control flow is restructured before code generation |
| The SPIR-V backend emits `OpUnordered`/`OpOrdered`, which shaders may not use | NaN comparisons are rewritten as tests on the bit pattern |
| The SPIR-V backend cannot select `llvm.copysign`, and instcombine creates it from bit operations | calls are expanded after optimisation |
| All tested drivers reassociate float arithmetic, e.g. `(1 + x) - 1` becomes `x` | every float operation is decorated `NoContraction` |
| The `log` of GPU drivers is imprecise close to 1 | `log1p` and `expm1` use series for small arguments |
| Vulkan has no math library beyond GLSL.std.450 | `hypot`, `log1p`, `erf`, `gamma`... are implemented in Python |
