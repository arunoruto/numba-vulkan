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
| numba cpu (1 thread) | 787.6 | 462.46 | 1.00x | 100.00% equal |
| numba cpu (parallel) | 488.9 | 90.37 | 5.12x | 100.00% equal |
| vulkan: NVIDIA TITAN X (Pascal) | 506.7 | 22.89 | 20.20x | 100.00% equal |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 554.4 | 56.45 | 8.19x | 100.00% equal |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits) | 656.9 | 71.61 | 6.46x | 100.00% equal |
| numba-cuda: NVIDIA TITAN X (Pascal) | 301.0 | 14.16 | 32.66x | 100.00% equal |

### option (4,194,304 options)

| Backend | First call (ms) | Best (ms) | Speed-up | Agreement with CPU |
| --- | ---: | ---: | ---: | --- |
| numba cpu (1 thread) | 292.3 | 109.57 | 1.00x | max err 0.0e+00 |
| numba cpu (parallel) | 351.0 | 23.27 | 4.71x | max err 0.0e+00 |
| vulkan: NVIDIA TITAN X (Pascal) | 861.0 | 31.59 | 3.47x | max err 3.4e-07 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 881.1 | 30.49 | 3.59x | max err 3.4e-07 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits) | 1008.1 | 48.91 | 2.24x | max err 2.5e-07 |
| numba-cuda: NVIDIA TITAN X (Pascal) | 107.1 | 16.68 | 6.57x | max err 2.9e-07 |

### saxpy (4,194,304 elements)

| Backend | First call (ms) | Best (ms) | Speed-up | Agreement with CPU |
| --- | ---: | ---: | ---: | --- |
| numba cpu (1 thread) | 313.3 | 2.78 | 1.00x | max err 0.0e+00 |
| numba cpu (parallel) | 288.9 | 5.96 | 0.47x | max err 0.0e+00 |
| vulkan: NVIDIA TITAN X (Pascal) | 469.4 | 23.08 | 0.12x | max err 0.0e+00 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 466.7 | 22.19 | 0.13x | max err 0.0e+00 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits) | 414.5 | 17.64 | 0.16x | max err 0.0e+00 |
| numba-cuda: NVIDIA TITAN X (Pascal) | 55.8 | 11.35 | 0.25x | max err 6.8e-08 |
