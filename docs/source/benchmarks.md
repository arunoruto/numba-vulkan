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
  then takes about 0.3 ms on the discrete GPU, eight times faster than the
  CPU, and within a factor of 1.3 of CUDA, mostly in waiting for the
  kernel to finish.
- The integrated GPU and llvmpipe share memory with the CPU and gain less
  from device arrays; `saxpy` is limited by memory bandwidth there.
- Compiling a Vulkan kernel takes 0.05 to 0.25 s, about as long as for
  CUDA; the higher figure applies to kernels that call math functions,
  which link libclc. (The very first kernel of a workload also pays for
  Numba's type inference of the shared core function.) The tables were
  produced with the on-disk kernel cache turned off; with it, a kernel
  that was compiled in an earlier run takes about 0.05 s.
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

## The same kernels on Vulkan and CUDA

The workloads above let every backend compile a function its own way.
`benchmarks/kernels.py` instead writes kernels once, in the style of
numba.cuda, with shared memory, barriers and atomics, and substitutes the
API names to produce a numba-cuda kernel and a numba-vulkan kernel with the
same workgroup and grid sizes. Data stays on the device, so these numbers
compare the code generators and runtimes on equal terms.

```{include} _generated/kernel_benchmark_results.md
```

- On the same card, Vulkan is within 25 to 40 % of CUDA on the
  reduction and the histogram, which take about 0.25 ms; the difference
  is mostly the fixed cost of submitting and waiting for a dispatch.
- The tiled matrix product is about 1.5× slower with Numba's default 64-bit
  integers; with `narrow="ints"` it is within 6 % of CUDA. NVIDIA's CUDA
  compiler narrows such index arithmetic itself, the Vulkan driver does
  not.
- Launching is cheaper than in numba-cuda: 1000 launches of a small kernel
  on device arrays take about 55 µs each on Vulkan and 70 µs on CUDA, both
  dominated by Python.
- The reduction uses a float atomic addition per workgroup. The Titan X
  supports native float atomics; on the integrated GPU, which does not,
  numba-vulkan falls back to a compare-and-swap loop.

## Reproducing

```sh
uv sync --group bench        # adds numba-cuda; optional
uv run python benchmarks/bench.py

# Regenerate the tables on this page (without the kernel cache, so that
# "first call" shows a real compilation):
NUMBA_VULKAN_CACHE=0 uv run python benchmarks/bench.py \
    --markdown docs/source/_generated/benchmark_results.md
```

For the kernel comparison:

```sh
uv run python benchmarks/kernels.py \
    --markdown docs/source/_generated/kernel_benchmark_results.md
```

Options of `bench.py`: `--size` (grid edge; arrays have `size**2` elements), `--maxiter`,
`--repeat`, `--json PATH`, `--markdown PATH`. Backends that are unavailable
are skipped.
