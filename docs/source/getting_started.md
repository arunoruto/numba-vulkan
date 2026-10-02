# Getting started

## Prerequisites

- A Vulkan 1.2 driver. [lavapipe](https://docs.mesa3d.org/drivers/llvmpipe.html),
  Mesa's CPU implementation, is enough to try it without a GPU.
- Python 3.11 or newer.
- libclc's `clspv--.bc`, from LLVM 22 or older, for accurate and
  double-precision math (see {doc}`math_library`). Without it, `float64`
  math functions are unavailable.
- Optional: `spirv-val` from SPIRV-Tools, to validate generated shaders.

## Installation

### With devenv

On NixOS, or any system with Nix, the [devenv](https://devenv.sh/) shell
provides Python, [uv](https://docs.astral.sh/uv/), the Vulkan loader,
libclc, SPIRV-Tools and `vulkaninfo`. GPU drivers come from the host.

```sh
devenv shell
uv run pytest
```

### Without Nix

Install the Vulkan loader (`libvulkan1` on Debian and Ubuntu, `vulkan-loader`
on Fedora and Arch) and a driver for your GPU. Wheels contain libclc. When
working from a source checkout instead, install your distribution's libclc
package or point `NUMBA_VULKAN_LIBCLC` at `clspv--.bc`.
Then:

```sh
uv sync
uv run pytest
```

### macOS

On macOS, Vulkan runs on top of Metal through
[MoltenVK](https://github.com/KhronosGroup/MoltenVK). numba-vulkan asks the
Vulkan loader for such "portability" drivers and enables what they need.
The test suite passes on an Apple M3 Pro with MoltenVK 1.4.2.

llvmlite's SPIR-V backend crashes on macOS arm64 (KI-32 in
{doc}`known_issues`), so numba-vulkan needs an `llc` of LLVM 22 there, named
by `NUMBA_VULKAN_LLC`. The devenv shell provides it, together with MoltenVK,
the Vulkan loader and libclc, and sets the environment variables that find
them; this is the tested setup:

```sh
devenv shell -- uv run pytest
```

Without Nix, Homebrew provides the same parts. This way has not been
tested:

```sh
brew install molten-vk vulkan-loader vulkan-tools
# Homebrew on Apple silicon installs into /opt/homebrew/lib, where the
# `vulkan` Python package does not look by itself:
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
# An llc of LLVM 22, the version inside llvmlite, from wherever it is installed:
export NUMBA_VULKAN_LLC=/path/to/llvm-22/bin/llc
vulkaninfo --summary       # should list the GPU with the MoltenVK driver
uv sync
make libclc                # optional: accurate float32 math (see below)
uv run python -c "import numba_vulkan as nv; print(nv.list_devices())"
```

If `vulkaninfo` lists no device, point the loader at MoltenVK's manifest
with `export VK_DRIVER_FILES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json`
(or the `share/vulkan/icd.d` directory, depending on the version). The
LunarG Vulkan SDK is an alternative to Homebrew.

Apple GPUs have no `float64` (Metal does not offer it), so kernels compute
`float64` values in `float32` and warn about it (see {doc}`limitations`).
`make libclc` puts libclc into the source checkout; it downloads
conda-forge's package unless Nix is installed, and the file is the same on
every platform. Without it, `float32` math uses the driver's functions.
CUDA does not exist on macOS, so the benchmarks leave out numba-cuda.

## Checking the setup

```python
import numba_vulkan as nv

nv.list_devices()
# [<0: NVIDIA TITAN X (Pascal) (discrete)>,
#  <1: Intel(R) UHD Graphics 630 (CFL GT2) (integrated)>,
#  <2: llvmpipe (LLVM 21.1.8, 256 bits) (cpu)>]
```

The test suite runs every test on every device in that list and validates
each generated shader with `spirv-val` when it is installed:

```sh
uv run pytest
uv run python examples/saxpy.py
```

## Building this documentation

```sh
uv sync --group docs
cd docs
uv run sphinx-build -M html ./source ./build
```
