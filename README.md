<a id="readme-top"></a>

<div align="center">
  <img src="https://raw.githubusercontent.com/arunoruto/numba-vulkan/main/docs/source/_static/logo.svg" alt="numba-vulkan logo" width="160" />
  <h1 align="center">numba-vulkan</h1>

  <p align="center">
    A proof-of-concept Vulkan compute target for Numba: write a kernel in Python, run it on any GPU with a Vulkan driver.
    <br />
    <a href="https://github.com/numba/numba/issues/10116"><strong>Background: numba/numba#10116 »</strong></a>
    <br />
    <br />
    <a href="https://arunoruto.github.io/numba-vulkan/">Documentation</a>
    &middot;
    <a href="#usage">Usage</a>
    &middot;
    <a href="#benchmarks">Benchmarks</a>
    &middot;
    <a href="#roadmap">Roadmap</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
        <li><a href="#how-it-works">How It Works</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#benchmarks">Benchmarks</a></li>
    <li><a href="#what-works-and-what-does-not">What Works and What Does Not</a></li>
    <li><a href="#known-issues">Known Issues</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#authorship-and-ai-disclosure">Authorship and AI Disclosure</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

> [!NOTE]
> **AI disclosure.** The code, tests and documentation in this repository
> were written by an AI (Anthropic's Claude) under the direction of a human,
> who came up with the idea, set the design direction and reviewed the
> result. See [Authorship](#authorship-and-ai-disclosure).

## About The Project

Numba's only maintained GPU target is CUDA, which ties GPU-accelerated Numba
code to NVIDIA hardware. Supporting every vendor separately is a lot of work;
Vulkan is implemented by nearly all of them.

numba-vulkan registers a `vulkan` target through Numba's target extension API
and compiles `numba.cuda`-style kernels to SPIR-V compute shaders. The same
kernel has been run on an NVIDIA GPU, an Intel integrated GPU and a CPU
rasteriser (llvmpipe), with results checked against NumPy.

This is a proof of concept. It shows that the approach works end to end and
where it hurts; it is not ready for real workloads.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

- [Numba](https://numba.pydata.org/): bytecode frontend, type inference,
  lowering and the target extension API
- [libclc](https://libclc.llvm.org/), LLVM's OpenCL math library, linked into
  kernels for accurate and double-precision math
- [llvmlite](https://llvmlite.readthedocs.io/) 0.50+ (LLVM 22), whose SPIR-V
  backend emits Vulkan-flavoured SPIR-V
- [vulkan](https://pypi.org/project/vulkan/): Python bindings for the Vulkan API
- [devenv](https://devenv.sh/) and [uv](https://docs.astral.sh/uv/) for the
  development environment

### How It Works

1. `@nv.jit` creates a dispatcher for the `vulkan` target. Numba's own pipeline
   (bytecode analysis, type inference, lowering) produces LLVM IR.
2. Arrays are a custom Numba type that carries the descriptor binding of its
   storage buffer, because shaders have no general pointers. Element access is
   emitted as a placeholder call taking a binding and an index.
3. Everything is inlined into one entry point and simplified with a few LLVM
   passes. Only then are the placeholders expanded into buffer accesses.
4. LLVM's SPIR-V backend turns the result into a compute shader. It runs in a
   child process, so a backend failure raises an exception instead of
   aborting Python.
5. The runtime uploads the arguments (binding 0: error status and array shapes, binding
   `1 + k`: argument `k`), dispatches, and copies written buffers back.
   Device arrays are used in place.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Prerequisites

- libclc's `clspv--.bc` from LLVM 22 or older, for float64 and accurate
  float32 math. Wheels bundle it and the devenv shell provides it; only a
  bare source checkout needs it installed (see the
  [math library docs](https://arunoruto.github.io/numba-vulkan/math_library.html)).
- A Vulkan 1.2 driver. [lavapipe](https://docs.mesa3d.org/drivers/llvmpipe.html)
  (Mesa's CPU implementation) is enough to try it without a GPU.
- Python 3.11 or newer.
- Optional: `spirv-val` from SPIRV-Tools to validate generated shaders.

### Installation

With devenv (NixOS or any system with Nix), everything including the Vulkan
loader and SPIRV-Tools is provided by the shell:

```sh
devenv shell
uv run pytest
```

Without Nix, install the Vulkan loader from your distribution, then:

```sh
uv sync
uv run pytest
```

On macOS, Vulkan runs on MoltenVK, and llvmlite's SPIR-V backend has to be
replaced by an `llc` of LLVM 22 (KI-32). The devenv shell sets up both
(`devenv shell -- uv run pytest`); the test suite passes on an Apple M3 Pro.
See [Getting started](https://arunoruto.github.io/numba-vulkan/getting_started.html#macos)
for the details and a Homebrew alternative.

The test suite runs every test on every Vulkan device it finds.

### Documentation

The Sphinx documentation is published at
<https://arunoruto.github.io/numba-vulkan/> and covers usage, the design, the
limitations, the known issues and the benchmarks. To build it locally:

```sh
uv sync --group docs
cd docs && uv run sphinx-build -M html ./source ./build
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

A kernel looks like a `numba.cuda` kernel. `global_id(axis)` plays the role of
`cuda.grid`, and `forall` launches it over a 1-d to 3-d grid:

```python
import numpy as np
import numba_vulkan as nv

@nv.jit
def saxpy(a, x, y):
    i = nv.global_id(0)
    if i < x.shape[0]:
        y[i] = a * x[i] + y[i]

x = np.arange(1000, dtype=np.float32)
y = np.ones(1000, dtype=np.float32)
saxpy.forall(x.size)(np.float32(2.0), x, y)
```

NumPy arrays are copied to the device and back on every call. Device arrays
stay there, as with `numba.cuda`:

```python
dx, dy = nv.to_device(x), nv.to_device(y)
saxpy.forall(x.size)(np.float32(2.0), dx, dy)   # no copies
saxpy.forall(x.size)(np.float32(0.5), dx, dy)
y = dy.copy_to_host()
```

Functions compiled with `@nv.jit` can be called from kernels, may return
values or tuples, and can take arrays. Functions compiled with `numba.njit`
are recompiled for Vulkan when a kernel calls them:

```python
import math
from numba import njit

@njit
def norm(x, y):
    return math.sqrt(x * x + y * y)

@nv.jit
def radius(x, y, out):
    i = nv.global_id(0)
    if i < out.size:
        out[i] = norm(x[i], y[i])
```

Numba's ufunc decorators work too, without writing a kernel:

```python
from numba import vectorize, guvectorize

@vectorize(["float32(float32, float32)"], target="vulkan")
def hypot(a, b):
    return math.sqrt(a * a + b * b)

@guvectorize(["void(float32[:], float32[:], float32[:])"], "(n),(n)->()",
             target="vulkan")
def dot(a, b, out):
    out[0] = np.dot(a, b)

hypot(x, y)               # element-wise, with broadcasting
dot(matrix, vector)       # one dot product per row
```

Cooperating kernels use the same building blocks as in `numba.cuda`:
workgroup-shared arrays, barriers, atomics and an explicit launch
configuration:

```python
@nv.jit
def block_sums(x, out):
    partial = nv.shared.array(256, np.float32)
    t, i = nv.local_id(0), nv.global_id(0)
    partial[t] = x[i] if i < x.shape[0] else np.float32(0)
    nv.barrier()
    step = 128
    while step > 0:
        if t < step:
            partial[t] += partial[t + step]
        nv.barrier()
        step //= 2
    if t == 0:
        nv.atomic.add(out, 0, partial[0])

block_sums[groups, 256](x, out)    # like kernel[blocks, threads] in CUDA
```

Because it is a real Numba target, `@overload` works with it:

```python
from numba.extending import overload

@overload(my_function, target="vulkan")
def ol_my_function(x):
    return lambda x: x * 2
```

Choosing a device:

```python
nv.list_devices()               # [<0: NVIDIA ... (discrete)>, <1: ...>, ...]
nv.select_device("intel")       # by name, index, or NUMBA_VULKAN_DEVICE
saxpy.forall(n, device="llvmpipe")(a, x, y)
```

Inspecting what was generated:

```python
from numba import float32
compiled = saxpy.compile((float32, float32[::1], float32[::1]))
compiled.llvm_ir, compiled.spirv, compiled.capabilities
```

Set `NUMBA_VULKAN_VALIDATE=1` to run every generated shader through
`spirv-val` before it reaches a driver.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Benchmarks

Two suites compare numba-vulkan with Numba's CPU target and numba-cuda: the
same scalar function compiled for every backend (`benchmarks/bench.py`), and
the same CUDA-style kernels with shared memory, barriers and atomics
(`benchmarks/kernels.py`). Results are collected in the repository, one file
per run, and charted in the
[documentation](https://arunoruto.github.io/numba-vulkan/benchmarks.html),
including every backend, device and variant and how results change over
time. The fastest result per backend, GPUs with data on the device (log
scale; further left is faster):

<p align="center">
  <img src="https://arunoruto.github.io/numba-vulkan/_images/machines.svg" alt="Best time per workload and backend" width="100%" />
</p>

In the first collected run (Intel i9-9900K, NVIDIA TITAN X, Intel UHD 630):

- With data on the device, numba-vulkan takes 25 to 55 % longer than
  numba-cuda on the same card on most workloads, and is about level on the
  tiled matrix product (1.85 against 1.75 ms). Repeated runs vary by 10 to
  20 %, so small differences are not meaningful.
- With NumPy arrays as arguments, the copies dominate: the memory-bound
  `saxpy` takes 7 to 10 ms on every GPU backend and 1.8 ms on one CPU thread.
- A kernel's first call, which compiles it, takes 0.04 to 0.25 s on Vulkan,
  comparable to numba-cuda.

On an Apple M3 Pro (MoltenVK 1.4.2, kernels translated by `llc`, KI-32),
with data on the device:

- The compute-bound Mandelbrot set takes 7 to 9 ms, about level with the
  TITAN X; within a full run the GPU shares the chip's power budget with
  the CPU benchmarks before it, which costs up to 20 %.
- Every launch costs about 0.2 ms, so the small workloads mostly measure
  that: `saxpy` takes 0.67 ms, about the same as the CPU.
- The tiled matrix product reaches only about 180 GFLOP/s (11.8 ms), the
  clearest candidate for tuning.

Please add a run from your machine; it takes a few minutes:

```sh
uv run python benchmarks/collect.py --user <your GitHub name>
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## What Works and What Does Not

Works:

- `int32`/`int64`/`float32`/`float64`/`bool` scalars and N-d arrays of
  `bool`, 8 to 64-bit integers, `float32` and `float64`
- indexing with integers and slices; slices are views, as in NumPy, and can
  be iterated over, passed to functions, assigned to and reduced
  (`a[i, 1:-1].sum()`, `for v in a[i]`, `out[i, :3] = 0`, `np.dot`, `.T`)
- arithmetic, comparisons, bit operations, casts, tuples, complex numbers,
  `if`/`while`/`for ... in range(...)`, early `return`s, `min`/`max`/`abs`
- the `math` module in float32 **and float64** (including `hypot`, `log1p`,
  `erf`, `gamma`, `isnan`...), accurate to the last digit and consistent
  across devices, and the `**` operator
- NumPy functions on scalars, such as `np.sqrt(x[i])` or `np.maximum(a, b)`
- calling other `@nv.jit` and `@njit` functions; `@overload(target="vulkan")`

Does not work:

- **float64 math without libclc.** Vulkan's math library is 32-bit only, so
  `sin`, `exp`, `pow` and friends in float64 come from libclc, LLVM's OpenCL
  math library. Wheels bundle it; in a source checkout without it such calls
  raise, unless you opt in to float32 precision with
  `@nv.jit(narrow_math=True)`.
- **Arrays of run-time size** (`a[mask]`, `a.copy()`, `np.zeros(n)`),
  `try`/`except`, recursion. Arrays of constant shape and array expressions
  such as `a * 2 + b` work.
- **float64 precision on devices without float64.** Many mobile GPUs and
  Apple devices lack float64 or int64. Kernels are narrowed to 32-bit types
  there, with a warning; float narrowing is tested on an Apple M3 Pro,
  integer narrowing only on desktop GPUs with the feature switched off.
- **Selecting between array elements by reference** in ways the optimiser
  turns into pointer selects is rejected.

Things found along the way:

- NVIDIA's driver returns wrong results for 64-bit signed remainders with
  negative operands; remainders are computed from the quotient instead.
- NVIDIA's shader compiler segfaults on some invalid SPIR-V, which is why
  modules are checked before being handed to a driver.
- float32 `sin` on Intel/Mesa is only accurate to about 1e-4.
- LLVM's SPIR-V backend miscompiles nested aggregate inserts and aborts on
  several unsupported inputs.
- LLVM's SPIR-V structurizer emits invalid shaders for early returns and
  short-circuit conditions; control flow is restructured beforehand.
- All three drivers reassociate float arithmetic unless told not to, so
  every float operation is marked exact.
- LLVM's SPIR-V backend mistakes `x = a * b; (x < 0) ? -y : y` for GLSL's
  `faceforward` and crashes; such comparisons are rewritten.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Known Issues

The full list, with causes, workarounds and pointers to where each would be
fixed, is in [known issues](https://arunoruto.github.io/numba-vulkan/known_issues.html). The
ones most likely to bite:

| | Issue | Workaround |
| --- | --- | --- |
| KI-01 | float64 `math.sin(x)`, `x ** 2.5`... need libclc, which a source checkout lacks | install a wheel, or set `NUMBA_VULKAN_LIBCLC` |
| KI-04 | no arrays of run-time size: `a[mask]`, `a.copy()`, `np.zeros(n)`, `axis=` reductions | constant-shape arrays, array expressions |
| KI-10 | indices and divisions by zero are only checked on request | `@nv.jit(boundscheck=True, error_model="python")` while debugging |
| KI-17 | running without int64 (mobile) is tested only with the feature switched off on desktop GPUs | report what you find |
| KI-32 | llvmlite's SPIR-V backend segfaults on macOS arm64 | set `NUMBA_VULKAN_LLC` to an LLVM 22 `llc`; the devenv shell does |

`tests/test_known_issues.py` reproduces the coverage gaps as expected
failures, so the list stays honest:

```sh
uv run pytest tests/test_known_issues.py -rxX
```

To continue the work, start with the
[development guide](https://arunoruto.github.io/numba-vulkan/development.html).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

- [x] Vulkan target registered with Numba's target extension API
- [x] Kernels, device functions, overloads
- [x] Test suite running on every available Vulkan device
- [x] Sphinx documentation and benchmark suite
- [x] Math functions, NumPy ufuncs on scalars, complex numbers
- [x] Early returns and short-circuit conditions (control-flow restructuring)
- [x] Loops with `break`/`return` anywhere, `while` loops (fuzz-tested)
- [x] Device arrays and buffer reuse, to avoid copying on every call
- [x] Asynchronous launches on device arrays (`nv.synchronize()`)
- [ ] Scalars as push constants
- [x] Slices, array views, iteration and reductions
- [x] Array expressions and local arrays
- [x] Kernels narrowed to 32-bit types on devices without float64/int64
- [x] float64 math, through libclc
- [x] Bundle libclc in wheels
- [ ] Publish to PyPI
- [x] `numba.vectorize` / `numba.guvectorize` with `target="vulkan"`
- [x] `reduce` of `vectorize` functions
- [x] `print()` in kernels, `float16` arrays, `error_model="python"`
- [ ] Other ufunc methods (`accumulate`, `outer`)
- [x] Shared memory, atomics, barriers and CUDA-style launch configuration
- [x] On-disk caching of compiled kernels
- [x] Testing on Apple (MoltenVK)
- [ ] Testing on AMD and mobile GPUs

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Authorship and AI Disclosure

| | Who |
| --- | --- |
| Idea | Mirza Arnaut, in [numba/numba#10116](https://github.com/numba/numba/issues/10116) |
| Direction and design decisions | Mirza Arnaut |
| Implementation: code, tests, benchmarks, documentation | Claude (Anthropic's Claude Fable 5.1, running in Claude Code) |
| Review | Mirza Arnaut |

Every line here was generated by a language model in October 2026; the human
author steered the work and reviewed it but did not write the code. Many
implementation-level choices were made by the model while solving problems it
ran into, and should be judged on their merits. Benchmark numbers and
hardware observations come from real runs on the author's machine.

This is an independent project and is not endorsed by the Numba maintainers.
Numba has an
[AI tools policy](https://numba.readthedocs.io/en/stable/reference/ai_tools_policy.html)
that requires attributing LLM-generated contributions; it applies if any part
of this is ever proposed upstream.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Issues and pull requests are welcome, in particular reports of how the test
suite behaves on other GPUs and drivers:

```sh
uv run pytest
uv run python benchmarks/bench.py
```

Contributions written with AI tools are welcome on the same terms as the
project itself: say which tool produced them, and have a human review them.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the BSD 2-Clause License, the same license as Numba. See
[LICENSE](https://github.com/arunoruto/numba-vulkan/blob/main/LICENSE).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Mirza Arnaut - [@arunoruto](https://github.com/arunoruto)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

- [Numba](https://github.com/numba/numba) and
  [numba-cuda](https://github.com/NVIDIA/numba-cuda), whose CUDA target is the
  template for this one
- [Taichi](https://github.com/taichi-dev/taichi), for showing that Python
  kernels on Vulkan work
- The LLVM SPIR-V backend developers
- [Best-README-Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
