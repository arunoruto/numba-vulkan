# Benchmarks

`benchmarks/bench.py` compares numba-vulkan with Numba's CPU target and with
numba-cuda. Each workload is one scalar Python function that is compiled
unchanged for every backend, plus a thin per-backend driver.

| Workload | Character |
| --- | --- |
| `mandelbrot` | compute-bound, data-dependent loop, little data |
| `option` | Black-Scholes call price: `log`, `exp`, `sqrt` per element |
| `saxpy` | memory-bound: one multiply-add per element |

Timings are wall-clock for a complete call with NumPy arrays as arguments,
so GPU numbers **include copying data to and from the device**. "First call"
includes compilation.

## Results

```{include} _generated/benchmark_results.md
```

## Reading the numbers

- On the compute-bound workload, Vulkan beats the parallel CPU on both GPUs
  and is roughly 1.5x slower than CUDA on the same card.
- On the memory-bound workload every GPU backend, CUDA included, loses to a
  single CPU thread: the time goes into copying arrays. numba-vulkan copies
  all arguments on every call and has no device arrays yet.
- Compiling a Vulkan kernel takes about half a second, several times longer
  than CUDA. Much of that is starting the child process that runs LLVM's
  SPIR-V backend.
- llvmpipe executes the shader on the CPU, so its numbers show the overhead
  of the Vulkan path rather than any hardware speed-up.
- All backends agree with the CPU result to `float32` rounding.

These are results of a single run on one machine. Repeated runs vary by
around 25 %, so small differences between rows are not meaningful.

## Reproducing

```sh
uv sync --group bench        # adds numba-cuda; optional
uv run python benchmarks/bench.py

# Regenerate the tables on this page:
uv run python benchmarks/bench.py --markdown docs/source/_generated/benchmark_results.md
```

Options: `--size` (grid edge; arrays have `size**2` elements), `--maxiter`,
`--repeat`, `--json PATH`, `--markdown PATH`. Backends that are unavailable
are skipped.
