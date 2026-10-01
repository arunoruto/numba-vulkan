# Getting started

## Prerequisites

- A Vulkan 1.2 driver. [lavapipe](https://docs.mesa3d.org/drivers/llvmpipe.html),
  Mesa's CPU implementation, is enough to try it without a GPU.
- Python 3.11 or newer.
- Optional: `spirv-val` from SPIRV-Tools, to validate generated shaders.

## Installation

### With devenv

On NixOS, or any system with Nix, the [devenv](https://devenv.sh/) shell
provides Python, [uv](https://docs.astral.sh/uv/), the Vulkan loader,
SPIRV-Tools and `vulkaninfo`. GPU drivers come from the host.

```sh
devenv shell
uv run pytest
```

### Without Nix

Install the Vulkan loader (`libvulkan1` on Debian and Ubuntu, `vulkan-loader`
on Fedora and Arch) and a driver for your GPU, then:

```sh
uv sync
uv run pytest
```

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
