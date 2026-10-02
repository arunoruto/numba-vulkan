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
| numba cpu (1 thread) | 836.8 | 464.29 | 1.00x | 100.00% equal |
| numba cpu (parallel) | 518.1 | 93.91 | 4.94x | 100.00% equal |
| vulkan: NVIDIA TITAN X (Pascal) | 516.5 | 12.87 | 36.07x | 100.00% equal |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 507.4 | 54.59 | 8.51x | 100.00% equal |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits) | 525.2 | 70.51 | 6.59x | 100.00% equal |
| numba-cuda: NVIDIA TITAN X (Pascal) | 279.3 | 10.50 | 44.24x | 100.00% equal |
| vulkan: NVIDIA TITAN X (Pascal), device arrays | 9.2 | 8.75 | 53.08x | 100.00% equal |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2), device arrays | 55.3 | 52.80 | 8.79x | 100.00% equal |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits), device arrays | 75.7 | 66.27 | 7.01x | 100.00% equal |
| numba-cuda: NVIDIA TITAN X (Pascal), device arrays | 7.9 | 7.55 | 61.47x | 100.00% equal |

### option (4,194,304 options)

| Backend | First call (ms) | Best (ms) | Speed-up | Agreement with CPU |
| --- | ---: | ---: | ---: | --- |
| numba cpu (1 thread) | 287.9 | 107.67 | 1.00x | max err 0.0e+00 |
| numba cpu (parallel) | 391.3 | 22.98 | 4.68x | max err 0.0e+00 |
| vulkan: NVIDIA TITAN X (Pascal) | 881.0 | 14.10 | 7.63x | max err 3.4e-07 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 881.7 | 15.59 | 6.91x | max err 3.4e-07 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits) | 910.5 | 44.32 | 2.43x | max err 2.5e-07 |
| numba-cuda: NVIDIA TITAN X (Pascal) | 106.8 | 16.25 | 6.63x | max err 2.9e-07 |
| vulkan: NVIDIA TITAN X (Pascal), device arrays | 0.9 | 0.69 | 155.96x | max err 3.4e-07 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2), device arrays | 7.2 | 6.03 | 17.86x | max err 3.4e-07 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits), device arrays | 34.4 | 35.12 | 3.07x | max err 2.5e-07 |
| numba-cuda: NVIDIA TITAN X (Pascal), device arrays | 0.9 | 0.28 | 382.21x | max err 2.9e-07 |

### saxpy (4,194,304 elements)

| Backend | First call (ms) | Best (ms) | Speed-up | Agreement with CPU |
| --- | ---: | ---: | ---: | --- |
| numba cpu (1 thread) | 119.2 | 2.87 | 1.00x | max err 0.0e+00 |
| numba cpu (parallel) | 313.7 | 5.97 | 0.48x | max err 0.0e+00 |
| vulkan: NVIDIA TITAN X (Pascal) | 410.0 | 8.22 | 0.35x | max err 0.0e+00 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 414.5 | 10.82 | 0.27x | max err 0.0e+00 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits) | 433.4 | 8.74 | 0.33x | max err 0.0e+00 |
| numba-cuda: NVIDIA TITAN X (Pascal) | 54.1 | 12.64 | 0.23x | max err 6.8e-08 |
| vulkan: NVIDIA TITAN X (Pascal), device arrays | 1.0 | 0.51 | 5.63x | max err 0.0e+00 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2), device arrays | 6.4 | 5.04 | 0.57x | max err 0.0e+00 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits), device arrays | 4.2 | 3.90 | 0.74x | max err 0.0e+00 |
| numba-cuda: NVIDIA TITAN X (Pascal), device arrays | 0.5 | 0.22 | 12.97x | max err 6.8e-08 |
