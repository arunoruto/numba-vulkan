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
| numba cpu (1 thread) | 839.2 | 460.42 | 1.00x | 100.00% equal |
| numba cpu (parallel) | 487.0 | 91.86 | 5.01x | 100.00% equal |
| vulkan: NVIDIA TITAN X (Pascal) | 479.1 | 19.70 | 23.37x | 100.00% equal |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 484.9 | 51.24 | 8.99x | 100.00% equal |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits) | 512.6 | 61.72 | 7.46x | 100.00% equal |
| numba-cuda: NVIDIA TITAN X (Pascal) | 414.5 | 12.63 | 36.47x | 100.00% equal |

### option (4,194,304 options)

| Backend | First call (ms) | Best (ms) | Speed-up | Agreement with CPU |
| --- | ---: | ---: | ---: | --- |
| numba cpu (1 thread) | 326.6 | 106.84 | 1.00x | max err 0.0e+00 |
| numba cpu (parallel) | 336.8 | 22.23 | 4.81x | max err 0.0e+00 |
| vulkan: NVIDIA TITAN X (Pascal) | 459.8 | 29.01 | 3.68x | max err 3.4e-07 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 461.2 | 28.92 | 3.69x | max err 3.4e-07 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits) | 495.4 | 25.49 | 4.19x | max err 3.4e-07 |
| numba-cuda: NVIDIA TITAN X (Pascal) | 124.6 | 18.17 | 5.88x | max err 2.9e-07 |

### saxpy (4,194,304 elements)

| Backend | First call (ms) | Best (ms) | Speed-up | Agreement with CPU |
| --- | ---: | ---: | ---: | --- |
| numba cpu (1 thread) | 145.1 | 2.40 | 1.00x | max err 0.0e+00 |
| numba cpu (parallel) | 285.4 | 5.50 | 0.44x | max err 0.0e+00 |
| vulkan: NVIDIA TITAN X (Pascal) | 411.8 | 22.79 | 0.11x | max err 0.0e+00 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 419.7 | 23.40 | 0.10x | max err 0.0e+00 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits) | 443.6 | 18.81 | 0.13x | max err 0.0e+00 |
| numba-cuda: NVIDIA TITAN X (Pascal) | 63.8 | 11.13 | 0.22x | max err 6.8e-08 |
