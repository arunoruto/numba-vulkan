<a id="readme-top"></a>

<div align="center">
  <img src="docs/source/_static/logo.svg" alt="numba-vulkan logo" width="160" />
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
5. The runtime uploads the arguments (binding 0: array shapes, binding
   `1 + k`: argument `k`), dispatches, and copies written buffers back.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Prerequisites

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

`benchmarks/bench.py` compiles the same scalar function for Numba's CPU
target, numba-vulkan and numba-cuda and times a full call with NumPy arrays,
so GPU timings include data transfer.

```sh
uv run python benchmarks/bench.py            # --size, --maxiter, --repeat, --json
```

One run on an Intel i9-9900K (8 cores), NVIDIA TITAN X (Pascal) and Intel UHD
630, best of 5, in milliseconds (lower is better):

| Backend | Mandelbrot 2048², 200 iter. | Option pricing, 4.2M | saxpy, 4.2M |
| --- | ---: | ---: | ---: |
| Numba CPU, 1 thread | 460.4 | 106.8 | 2.4 |
| Numba CPU, parallel | 91.9 | 22.2 | 5.5 |
| **numba-vulkan**, NVIDIA TITAN X | 19.7 | 29.0 | 22.8 |
| **numba-vulkan**, Intel UHD 630 | 51.2 | 28.9 | 23.4 |
| **numba-vulkan**, llvmpipe (CPU) | 61.7 | 25.5 | 18.8 |
| numba-cuda, NVIDIA TITAN X | 12.6 | 18.2 | 11.1 |

Reading the numbers:

- On compute-heavy kernels (Mandelbrot) Vulkan beats the parallel CPU on both
  GPUs and is roughly 1.5x slower than CUDA on the same card.
- On memory-bound kernels (saxpy) every GPU backend loses to a single CPU
  thread, because the time goes into copying arrays. numba-vulkan copies all
  arguments on every call and has no device arrays yet.
- First-call (compile) time is about 0.5 s for Vulkan, against 0.05 to 0.3 s
  for CUDA.
- All backends agree with the CPU result to float32 rounding; saxpy is
  bit-identical on Vulkan.
- Repeated runs vary by around 25 %, so small differences are not meaningful.

The [documentation](docs/source/benchmarks.md) has the full tables, including
compile times and the software versions used.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## What Works and What Does Not

Works:

- `int32`/`int64`/`float32`/`float64`/`bool` scalars and C-contiguous N-d
  arrays of `bool`, 8 to 64-bit integers, `float32` and `float64`, with
  integer indexing (including negative indices), `.shape`, `.size` and `len()`
- arithmetic, comparisons, bit operations, casts, tuples, complex numbers,
  `if`/`while`/`for ... in range(...)`, early `return`s, `min`/`max`/`abs`
- the `math` module (including `hypot`, `log1p`, `erf`, `gamma`, `isnan`...)
  and `**` with integer exponents
- NumPy functions on scalars, such as `np.sqrt(x[i])` or `np.maximum(a, b)`
- calling other `@nv.jit` and `@njit` functions; `@overload(target="vulkan")`

Does not work:

- **float64 `sin`/`exp`/`pow`/...** Vulkan's math library is 32-bit only.
  Such calls raise unless you opt in with `@nv.jit(narrow_math=True)`, which
  computes them in float32.
- **Slices, array views, NumPy functions on whole arrays, array allocation,
  exceptions, recursion.** Errors raised inside a kernel are silently dropped.
- **Devices without float64, int64 or int8 support.** Numba types Python
  literals as float64/int64, and the SPIR-V backend currently forces int8, so
  most kernels need all three. Desktop GPUs have them; many mobile GPUs and
  Apple devices do not. Kernels fail with a clear error on such devices.
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

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Known Issues

The full list, with causes, workarounds and pointers to where each would be
fixed, is in [docs/source/known_issues.md](docs/source/known_issues.md). The
ones most likely to bite:

| | Issue | Workaround |
| --- | --- | --- |
| KI-01 | `x ** 2.5` and `math.sin(x)` fail for float64, and for float32 mixed with Python float literals | stay in float32 (`np.float32(2.5)`), or `narrow_math=True` |
| KI-24 | some deeply nested loops with `return`/`break` fail to compile | simplify the loop exits |
| KI-04 | no slices, array methods or iteration over arrays | index explicitly |
| KI-06 | global NumPy arrays cannot be used in kernels | pass them as arguments |
| KI-10 | errors raised in kernels are dropped; no bounds checks | check inputs on the host |
| KI-13 | every call copies all arrays to and from the device | none yet |
| KI-17 | most kernels need the optional float64/int64/int8 device features | none yet |

`tests/test_known_issues.py` reproduces the coverage gaps as expected
failures, so the list stays honest:

```sh
uv run pytest tests/test_known_issues.py -rxX
```

To continue the work, start with the
[development guide](docs/source/development.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

- [x] Vulkan target registered with Numba's target extension API
- [x] Kernels, device functions, overloads
- [x] Test suite running on every available Vulkan device
- [x] Sphinx documentation and benchmark suite
- [ ] Device arrays and buffer reuse, to avoid copying on every call
- [ ] Slices and array views
- [ ] A float32-by-default typing mode, so kernels run on devices without float64
- [ ] Software float64 math library
- [ ] `@vectorize`-style ufuncs
- [ ] Shared memory, atomics and barriers
- [ ] On-disk caching of compiled kernels
- [ ] Testing on AMD, Apple (MoltenVK) and mobile GPUs

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
[LICENSE](LICENSE).

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
