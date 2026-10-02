| | |
| --- | --- |
| Date | 2026-10-02 |
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
| numba cpu (1 thread) | 792.6 | 469.69 | 1.00x | 100.00% equal |
| numba cpu (parallel) | 527.1 | 93.02 | 5.05x | 100.00% equal |
| vulkan: NVIDIA TITAN X (Pascal) | 168.9 | 15.50 | 30.31x | 100.00% equal |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 144.1 | 47.66 | 9.86x | 100.00% equal |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits) | 174.7 | 48.05 | 9.78x | 100.00% equal |
| numba-cuda: NVIDIA TITAN X (Pascal) | 294.6 | 14.32 | 32.81x | 100.00% equal |
| vulkan: NVIDIA TITAN X (Pascal), device arrays | 12.2 | 9.61 | 48.86x | 100.00% equal |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2), device arrays | 48.2 | 45.11 | 10.41x | 100.00% equal |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits), device arrays | 50.7 | 52.30 | 8.98x | 100.00% equal |
| numba-cuda: NVIDIA TITAN X (Pascal), device arrays | 9.3 | 8.46 | 55.49x | 100.00% equal |

### option (4,194,304 options)

| Backend | First call (ms) | Best (ms) | Speed-up | Agreement with CPU |
| --- | ---: | ---: | ---: | --- |
| numba cpu (1 thread) | 274.7 | 114.75 | 1.00x | max err 0.0e+00 |
| numba cpu (parallel) | 397.0 | 24.31 | 4.72x | max err 0.0e+00 |
| vulkan: NVIDIA TITAN X (Pascal) | 249.7 | 11.19 | 10.26x | max err 3.4e-07 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 246.5 | 13.50 | 8.50x | max err 3.4e-07 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits) | 467.3 | 46.83 | 2.45x | max err 2.5e-07 |
| numba-cuda: NVIDIA TITAN X (Pascal) | 107.5 | 16.05 | 7.15x | max err 2.9e-07 |
| vulkan: NVIDIA TITAN X (Pascal), device arrays | 1.2 | 0.41 | 279.96x | max err 3.4e-07 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2), device arrays | 8.6 | 6.03 | 19.03x | max err 3.4e-07 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits), device arrays | 38.6 | 40.59 | 2.83x | max err 2.5e-07 |
| numba-cuda: NVIDIA TITAN X (Pascal), device arrays | 0.8 | 0.26 | 447.18x | max err 2.9e-07 |

### saxpy (4,194,304 elements)

| Backend | First call (ms) | Best (ms) | Speed-up | Agreement with CPU |
| --- | ---: | ---: | ---: | --- |
| numba cpu (1 thread) | 116.0 | 2.27 | 1.00x | max err 0.0e+00 |
| numba cpu (parallel) | 281.4 | 4.99 | 0.45x | max err 0.0e+00 |
| vulkan: NVIDIA TITAN X (Pascal) | 53.4 | 8.01 | 0.28x | max err 0.0e+00 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 52.3 | 11.05 | 0.21x | max err 0.0e+00 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits) | 61.8 | 9.21 | 0.25x | max err 0.0e+00 |
| numba-cuda: NVIDIA TITAN X (Pascal) | 58.4 | 14.12 | 0.16x | max err 6.8e-08 |
| vulkan: NVIDIA TITAN X (Pascal), device arrays | 1.0 | 0.33 | 6.87x | max err 0.0e+00 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2), device arrays | 7.2 | 4.62 | 0.49x | max err 0.0e+00 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits), device arrays | 4.3 | 3.82 | 0.59x | max err 0.0e+00 |
| numba-cuda: NVIDIA TITAN X (Pascal), device arrays | 0.4 | 0.20 | 11.46x | max err 6.8e-08 |
