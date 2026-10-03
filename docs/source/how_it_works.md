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
| Typing and lowering | {py:mod}`numba_vulkan.vkdecl`, {py:mod}`numba_vulkan.vkimpl`, {py:mod}`numba_vulkan.mathimpl`, {py:mod}`numba_vulkan.mathfuncs`, {py:mod}`numba_vulkan.ufuncs`, {py:mod}`numba_vulkan.libclc` | `global_id`, array indexing, `math`, NumPy ufuncs, the math library |
| Pipeline | {py:mod}`numba_vulkan.compiler` | Numba compiler pipeline and the kernel entry point |
| Code generation | {py:mod}`numba_vulkan.codegen`, {py:mod}`numba_vulkan.legalize`, {py:mod}`numba_vulkan.structurize`, {py:mod}`numba_vulkan.buffers` | linking, optimisation, IR rewrites, control-flow restructuring, SPIR-V emission |
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
| 0 | error status of the kernel (`int32`) |
| `1 + k` | argument `k`, if it is an array |
| after the arguments | one buffer per NumPy array the kernel uses as a global constant |

Scalar arguments and the extents of array arguments are **push constants**,
one block per kernel specialisation, as clspv passes plain-data arguments:
8-byte scalars first, then 4-byte scalars (booleans and integers narrower
than 32 bits are widened to 32 bits), then the extents as `int32`. The
block is read through placeholder loads like buffers (see below), which
become loads from an `addrspace(13)` global only after narrowing, so its
member types are final; the runtime packs the values with offsets read
back from the SPIR-V module. A block must fit into 128 bytes, the size every
Vulkan device guarantees, so that compiled kernels do not depend on the
device. Kernels whose arguments need more fall back to one-element buffers
for scalars (at binding `1 + k`) and the extents after the status in
binding 0.

The **workgroup size** is a specialization constant: the module declares
it as clspv does, with three `OpSpecConstant` (ids 0, 1 and 2, defaulting
to the size of the grid's dimensionality) combined into an
`OpSpecConstantComposite` decorated `BuiltIn WorkgroupSize`
(`codegen.specialize_local_size`). `nv.local_size()` reads a private array
initialised with the same constants. A kernel is therefore compiled once for
all workgroup sizes; each size gets its own pipeline, and drivers fold the
constants when they create it.

A global array is typed with a placeholder binding that is the same in every
function using it; each kernel renumbers the ones it reaches to follow its
arguments (`buffers.renumber_constants`). The runtime uploads their data
once per kernel and device.

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

1. **Loop exits.** A loop that is left from inside its body (`break`,
   `return`) is rewritten to leave through its latch only. A block that
   wants to leave records its target in a flag and jumps to the latch; the
   latch tests the flag, and a chain of tests after the loop dispatches to
   the recorded target. The loop body is then an acyclic region in which a
   `break` is just an early jump to its end.
2. **Unstructured joins.** A block entered from several places that is not
   the merge block of a selection is either copied, once per entry, when it
   is small; or guarded: the paths that bypassed it are sent through it
   with a flag set, and a test in front of the block skips it when the flag
   is set. Copying costs no run time, guarding costs no code size, and
   only guarding keeps functions with many early returns from growing
   exponentially. A join that is one tiny block, such as the block that
   stores the status of a kernel that raised, is always copied, so that
   each place that raises leaves on a path of its own.
3. **Shared merge blocks.** A selection that shares its merge block with an
   enclosing one gets a merge block of its own that forwards to the shared
   one.

To make these rewrites simple, the IR is first brought into a form without
phi nodes (LLVM's `reg2mem`), without `switch` (`lower-switch`) and with
canonical loops (`loop-simplify`).

Steps 1 and 2 need dominators, post-dominators and loops of the whole
graph, which are recomputed after the rewrites. A rewrite changes edges
only within a small *footprint*: for a loop, its body, the blocks in front
of it and the targets of its exits; for a join, the blocks on the paths
from its immediate dominator to it, the blocks it dominates if it is copied,
and their successors. Dominance outside a footprint stays as it was, so all
rewrites with disjoint footprints are made on one analysis. A kernel of
some 5700 blocks needs 11 analyses instead of several hundred, and about
2 s instead of 40 s.

Subgroup operations must stay where the program puts them: the
invocations that reach one instruction together take part in it. They are
calls of `convergent` functions, which LLVM neither moves across conditions
nor duplicates, and the structurizer never copies a block that contains one.

After code generation, `codegen.check_structure` checks every module for the
rules of structured control flow that a failure here or in LLVM's backend
would break (a conditional branch without a merge instruction, a branch
back to a block that is no loop header, two constructs sharing a merge
block), so that such a module never reaches a driver, with or without
`spirv-val`.

The flags are ordinary local variables, which the drivers' compilers
optimise like any other. `tests/fuzz_control_flow.py` checks this step with
random programs: all of the 900 it was last run on (600 of them with
`--rich`) compiled and gave the same results as Python on three devices
(see {doc}`development`). `tests/test_control_flow_corpus.py` runs the
shapes of rust-gpu's control-flow tests as part of the test suite.

## SPIR-V emission

LLVM's SPIR-V backend, shipped in llvmlite since LLVM 20, has a Vulkan mode
that produces structured control flow and storage-buffer access. It runs in
a child process, because it aborts the whole process on input it cannot
handle; a failure then surfaces as
{py:class}`~numba_vulkan.errors.SpirvCodegenError`. It also needs a fresh
process for every module, because it keeps state that corrupts the second
module it translates. To keep that cheap, one helper process is started
when the first function is compiled; it imports llvmlite once and forks
for each module ({py:class}`numba_vulkan.codegen.Emitter`).

Passes over the binary then finish the module (`codegen.emit_spirv`):

- calls of functions named `nv.sg.*`, which the backend emits as imported
  functions, become the `OpGroupNonUniform*` instructions of their name
  (`lower_group_operations`). The backend offers subgroup reductions but no
  products, scans or shuffles; declaring a function and replacing its calls
  afterwards gives access to any instruction it lacks;
- repairs of what the backend gets wrong (compare-and-swap results,
  barrier semantics, `OpConstantNull` indices), and capabilities it
  declares too broadly (`Float16`, `Int8`) or not at all
  (`StorageBuffer8BitAccess`);
- the workgroup size becomes a specialization constant
  (`specialize_local_size`);
- every float operation is decorated `NoContraction`, because shader
  compilers otherwise reassociate arithmetic freely, which Numba code does
  not expect;
- the module is checked for constructs known to crash drivers and for the
  rules of structured control flow (`check_spirv`).

At pipeline creation, the runtime adds `SignedZeroInfNanPreserve` for
kernels without `fastmath` (see the table below).

## Shared memory, atomics and barriers

A shared array is typed like an array argument whose "binding" lies above
`buffers.SHARED_BASE`; an untyped pass gives each `shared.array` call a
number derived from its function and position, so that the type, and with
it the kernel cache key, is the same in every process. Element accesses
use the same placeholders as buffers and become accesses to a Workgroup
variable when they are expanded. Atomics are placeholders as well, which
become `atomicrmw` with relaxed ordering and device or workgroup scope.

The backend crashes on LLVM's `cmpxchg`, so compare-and-swap uses its
`llvm.spv.cmpxchg` intrinsic, whose mistyped result is repaired in the
SPIR-V binary. Barriers are a `convergent` placeholder with unknown memory
effects, so that LLVM neither moves memory accesses across them nor
duplicates them; the control-flow restructuring guards joins instead of
copying them when they contain one, and the barrier's memory semantics are
changed to acquire-release in the binary, because Vulkan forbids the
sequentially consistent ones the backend emits.

## Ufuncs

Numba's `vectorize` and `guvectorize` look up a builder class per target;
{py:mod}`numba_vulkan.vectorizers` registers one for `"vulkan"`, as
numba-cuda does for `"cuda"`. A call resolves the signature from the
argument types, broadcasts on the host and launches a kernel that is
generated as Python source from a template: one invocation per element
(or per loop iteration of a gufunc), which unravels its position into
per-axis indices, picks index 0 along axes of extent 1, and calls the
scalar or core function with elements or views. Inputs that already have
the result's shape are passed flattened, which saves the index arithmetic.
The generated kernels go through the same pipeline and cache as any other.

## Caching

Everything after Numba's lowering (linking libclc, the LLVM passes, the
rewrites described above and the SPIR-V backend) is a function of the
unoptimised LLVM IR and of this package. {py:mod}`numba_vulkan.kernelcache`
stores the result under a hash of the IR, the options that affect code
generation, the package's own source, the llvmlite version and the libclc
file. A later process that arrives at the same IR reads the SPIR-V from
`~/.cache/numba-vulkan/kernels` instead of producing it again.

Because the key is derived from the code Numba generated, an edit to the
kernel, to a function it calls or to the shape or type of a global it reads
always leads to a new entry; there is nothing to invalidate by hand. Two
things that differ between processes are removed before hashing: the
counter Numba appends to function names, and the placeholder numbers of
constant arrays. The contents of constant arrays are not part of an entry;
they are taken from the running process.

## Devices without 64-bit types

`float64` and `int64` are optional device features, and Numba uses `int64`
for every index. For a device that lacks them, a kernel is compiled in a
narrow mode ({py:mod}`numba_vulkan.narrowing`), in three places:

1. **Lowering.** Math functions called with `float64` values call their
   `float32` versions, and code that works on bit patterns (`isnan`,
   `copysign`) does so at 32 bits. The conversions around them are opaque
   placeholder calls, so that LLVM cannot turn the `float32` code back into
   operations on the bits of the `float64`.
2. **LLVM IR.** After optimisation, every `double` becomes `float` and
   every `i64` becomes `i32` in the text: types, constants, casts and
   intrinsic names. Anything that depends on the width is rejected; shifts
   that only extract the sign and the extreme values used as "no limit"
   are translated.
3. **SPIR-V.** The backend indexes constant tables with 64-bit constants
   of its own; those are retyped, and unused definitions are stripped.
   Stripping happens for every kernel: it removes the 8-bit name strings
   that would otherwise make all shaders require the `Int8` feature.

Buffers then hold 32-bit elements for 64-bit array types; the host converts
NumPy arguments, and device arrays on such a device store the narrow type.

Integers are narrowed by default on every device, for speed. A kernel that
keeps `float64` cannot be narrowed as a whole, because libclc's `float64`
functions work on the bits of doubles with 64-bit integers. Its integers
are narrowed before libclc is linked, in the kernel's own code only, which
gets a first round of inlining and clean-up for that purpose. The
target's own bit manipulations of doubles (`isnan`, `copysign`, printing)
use two 32-bit words instead of a 64-bit integer, so that they survive.

## Runtime

{py:mod}`numba_vulkan.runtime` is a small Vulkan compute runtime on top of
the [`vulkan`](https://pypi.org/project/vulkan/) bindings. For each call it
binds one storage buffer per binding, dispatches the shader and waits for it
to finish. A NumPy argument gets a temporary buffer in mappable memory: it
is uploaded before the dispatch and read back afterwards if the shader
writes to it. A device array brings its own buffer, which on discrete GPUs
lives in device-local memory and is filled and read through a staging
buffer. Where device-local memory can be mapped and is cached, as on
integrated GPUs and llvmpipe, device arrays are mapped and copies need no
staging. MoltenVK declares such memory for Apple GPUs too
(`mvk_datatypes.h`), which has not been checked on one yet. If the shader left an error status in binding 0, the matching
exception is raised.

Buffers are recycled through a per-device pool, and pipelines, descriptor
sets and the command buffer are kept per device and kernel specialisation.
A launch that binds the same buffers as the previous one of that kernel,
which is the usual case in a loop, neither rewrites the descriptor set nor
records the commands again; it only submits. Push constants are part of the
recorded commands, so a launch with other scalar values or shapes records
them again, through the C functions directly, which takes a few
microseconds.

Launches on device arrays do not wait. They are recorded into a *batch*,
one command buffer with a fence, as clvk does: a memory barrier, the
pipeline and descriptor set (bound only when they change), the push
constants and the dispatch. A batch is submitted when the device has
nothing else to do (its last fence is signalled), after 16 launches, or at
any synchronisation; at most four are in flight. Each launch uses a *slot*
of its kernel, a descriptor set with mapped buffers for the status (and,
without push constants, shapes and scalars). Launches with the same buffers
and the same values in those share a slot, so a loop over the same arrays
uses one descriptor set and copies nothing.

A kernel that can raise leaves its status in the slot's mapped buffer.
When a batch has finished, the status of each slot it used last is read
and reset; an error is kept and raised by the next synchronisation, as
CUDA reports errors of asynchronous launches. Buffers that are released
while launches may use them join the pool once the batch recorded before
their release has finished. Each buffer records the last batch that uses
it, so that copying to or from a device array waits only for that one.
Grids beyond the device's workgroup limits are dispatched in parts with
`vkCmdDispatchBase`; `num_groups` comes from push constants holding the
size of the whole grid.

## Workarounds for toolchain and driver behaviour

| Observation | Workaround |
| --- | --- |
| NVIDIA's driver returns wrong results for signed 64-bit remainders with negative operands | remainders are computed as `a - (a / b) * b` |
| NVIDIA's shader compiler segfaults on some invalid SPIR-V | modules are checked before use; buffer access is deferred |
| The SPIR-V backend miscompiles nested aggregate inserts | aggregates are folded away before emission |
| The SPIR-V backend aborts the process on unsupported input | emission runs in a child process |
| The SPIR-V backend emits duplicate ids for the second module translated in one process | one forked process per module |
| Numba's integer power falls back to a `float64` `pow` | integer exponents are implemented with multiplications |
| LLVM's structurizer emits invalid SPIR-V for early exits and short-circuit conditions | control flow is restructured before code generation |
| The SPIR-V backend emits `OpUnordered`/`OpOrdered`, which shaders may not use | NaN comparisons are rewritten as tests on the bit pattern |
| The SPIR-V backend cannot select `llvm.copysign`, and instcombine creates it from bit operations | calls are expanded after optimisation |
| All tested drivers reassociate float arithmetic, e.g. `(1 + x) - 1` becomes `x` | every float operation is decorated `NoContraction` |
| Without `SignedZeroInfNanPreserve`, drivers may assume there are no NaNs; NVIDIA's turns `y < x ? y : x` (Python's `min`) into an instruction that ignores NaN | kernels without `fastmath` declare the execution mode, for the float widths they use, on devices that support it |
| The `log` of GPU drivers is imprecise close to 1 | the fallback `log1p` and `expm1` use series for small arguments |
| Vulkan has no math library beyond GLSL.std.450 | math functions are linked in from libclc; see {doc}`math_library` |
| The SPIR-V backend mistakes `x = a * b; (x < 0) ? -y : y` for GLSL's `faceforward` and crashes | strict comparisons with zero are emitted as negated complements |
| The SPIR-V backend cannot legalise `llvm.ctlz`, `llvm.fshl` and `llvm.minimumnum`, nor loads from byte tables | each is expanded on the IR text |
