# Benchmarks

`benchmarks/bench.py` compares numba-vulkan with Numba's CPU target and with
numba-cuda. Each workload is one scalar Python function that is compiled
unchanged for every backend, plus a thin per-backend driver.

| Workload | Character |
| --- | --- |
| `mandelbrot` | compute-bound, data-dependent loop, little data |
| `option` | Black-Scholes call price: `log`, `exp`, `sqrt` per element |
| `saxpy` | memory-bound: one multiply-add per element |

Timings are wall-clock for a complete call. Each GPU backend appears twice:
with NumPy arrays as arguments, which **includes copying data to and from
the device**, and with device arrays, which leaves the kernel and the launch
overhead. "First call" includes compilation, except in the device-array
rows, which reuse the kernels compiled for the rows above.

## Results

```{include} _generated/benchmark_results.md
```

## Reading the numbers

- On the compute-bound workload, Vulkan beats the parallel CPU on both GPUs
  and is within about 20 % of CUDA on the same card.
- With NumPy arguments, the memory-bound workload is slower on every GPU
  backend than on a single CPU thread: the time goes into copying arrays.
  Vulkan copies somewhat faster than CUDA here, because it maps buffers
  that are kept from call to call.
- With device arrays nothing is copied. `saxpy` over 4 million elements
  then takes about 0.5 ms on the discrete GPU, several times faster than
  the CPU. CUDA is two to three times faster still on such short kernels:
  a Vulkan launch costs about 0.4 ms and waits for the kernel to finish
  (KI-28 in {doc}`known_issues`).
- The integrated GPU and llvmpipe share memory with the CPU and gain less
  from device arrays; `saxpy` is limited by memory bandwidth there.
- Compiling a Vulkan kernel takes 0.05 to 0.25 s, about as long as for
  CUDA; the higher figure applies to kernels that call math functions,
  which link libclc. (The very first kernel of a workload also pays for
  Numba's type inference of the shared core function.)
- The `option` workload calls `log`, `exp` and `sqrt` from libclc. On the
  GPUs that costs little compared with the built-in functions; on llvmpipe
  it roughly doubles the run time.
- llvmpipe executes the shader on the CPU, so its numbers show the overhead
  of the Vulkan path rather than any hardware speed-up.
- All backends agree with the CPU result to `float32` rounding; `saxpy` is
  bit-identical on Vulkan, because shaders are compiled without fused
  multiply-add.

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
