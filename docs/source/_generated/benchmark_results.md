| | |
| --- | --- |
| Date | 2026-10-01 |
| CPU | Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz (16 threads) |
| Vulkan devices | NVIDIA TITAN X (Pascal) (discrete), Intel(R) UHD Graphics 630 (CFL GT2) (integrated), llvmpipe (LLVM 21.1.8, 256 bits) (cpu) |
| Python | 3.14.7 |
| numba | 0.68.0 |
| llvmlite | 0.50.0 |
| numpy | 2.4.6 |
| numba-cuda | 0.30.4 |
| Timing | best of 5 runs after the first call |

### mandelbrot (2048x2048, 200 iterations)

| Backend | First call (ms) | Best (ms) | Speed-up | Agreement with CPU |
| --- | ---: | ---: | ---: | --- |
| numba cpu (1 thread) | 853.9 | 458.56 | 1.00x | 100.00% equal |
| numba cpu (parallel) | 480.0 | 92.47 | 4.96x | 100.00% equal |
| vulkan: NVIDIA TITAN X (Pascal) | 459.7 | 18.39 | 24.94x | 100.00% equal |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 459.6 | 47.73 | 9.61x | 100.00% equal |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits) | 498.6 | 74.11 | 6.19x | 100.00% equal |
| numba-cuda: NVIDIA TITAN X (Pascal) | 259.3 | 12.35 | 37.13x | 100.00% equal |

### option (4,194,304 options)

| Backend | First call (ms) | Best (ms) | Speed-up | Agreement with CPU |
| --- | ---: | ---: | ---: | --- |
| numba cpu (1 thread) | 288.0 | 108.28 | 1.00x | max err 0.0e+00 |
| numba cpu (parallel) | 359.5 | 22.79 | 4.75x | max err 0.0e+00 |
| vulkan: NVIDIA TITAN X (Pascal) | 543.2 | 33.11 | 3.27x | max err 3.2e-07 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 549.8 | 41.01 | 2.64x | max err 3.4e-07 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits) | 512.3 | 24.43 | 4.43x | max err 3.4e-07 |
| numba-cuda: NVIDIA TITAN X (Pascal) | 94.4 | 15.58 | 6.95x | max err 2.9e-07 |

### saxpy (4,194,304 elements)

| Backend | First call (ms) | Best (ms) | Speed-up | Agreement with CPU |
| --- | ---: | ---: | ---: | --- |
| numba cpu (1 thread) | 124.9 | 2.63 | 1.00x | max err 0.0e+00 |
| numba cpu (parallel) | 315.8 | 7.97 | 0.33x | max err 0.0e+00 |
| vulkan: NVIDIA TITAN X (Pascal) | 386.1 | 18.81 | 0.14x | max err 6.8e-08 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 616.6 | 33.05 | 0.08x | max err 6.8e-08 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits) | 676.4 | 17.95 | 0.15x | max err 0.0e+00 |
| numba-cuda: NVIDIA TITAN X (Pascal) | 66.2 | 14.22 | 0.18x | max err 6.8e-08 |
