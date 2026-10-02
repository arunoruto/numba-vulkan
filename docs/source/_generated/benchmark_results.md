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
| numba cpu (1 thread) | 855.5 | 489.02 | 1.00x | 100.00% equal |
| numba cpu (parallel) | 556.7 | 95.71 | 5.11x | 100.00% equal |
| vulkan: NVIDIA TITAN X (Pascal) | 161.2 | 14.27 | 34.28x | 100.00% equal |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 149.5 | 57.98 | 8.43x | 100.00% equal |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits) | 179.3 | 75.51 | 6.48x | 100.00% equal |
| numba-cuda: NVIDIA TITAN X (Pascal) | 270.0 | 12.89 | 37.93x | 100.00% equal |
| vulkan: NVIDIA TITAN X (Pascal), device arrays | 9.5 | 9.27 | 52.77x | 100.00% equal |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2), device arrays | 55.5 | 52.88 | 9.25x | 100.00% equal |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits), device arrays | 79.6 | 62.37 | 7.84x | 100.00% equal |
| numba-cuda: NVIDIA TITAN X (Pascal), device arrays | 9.0 | 8.24 | 59.32x | 100.00% equal |

### option (4,194,304 options)

| Backend | First call (ms) | Best (ms) | Speed-up | Agreement with CPU |
| --- | ---: | ---: | ---: | --- |
| numba cpu (1 thread) | 278.1 | 107.93 | 1.00x | max err 0.0e+00 |
| numba cpu (parallel) | 333.4 | 19.89 | 5.43x | max err 0.0e+00 |
| vulkan: NVIDIA TITAN X (Pascal) | 221.0 | 10.14 | 10.65x | max err 3.4e-07 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 218.8 | 12.99 | 8.31x | max err 3.4e-07 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits) | 272.9 | 42.64 | 2.53x | max err 2.5e-07 |
| numba-cuda: NVIDIA TITAN X (Pascal) | 93.3 | 15.33 | 7.04x | max err 2.9e-07 |
| vulkan: NVIDIA TITAN X (Pascal), device arrays | 0.9 | 0.37 | 288.06x | max err 3.4e-07 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2), device arrays | 8.3 | 5.92 | 18.22x | max err 3.4e-07 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits), device arrays | 34.1 | 35.05 | 3.08x | max err 2.5e-07 |
| numba-cuda: NVIDIA TITAN X (Pascal), device arrays | 1.2 | 0.28 | 383.80x | max err 2.9e-07 |

### saxpy (4,194,304 elements)

| Backend | First call (ms) | Best (ms) | Speed-up | Agreement with CPU |
| --- | ---: | ---: | ---: | --- |
| numba cpu (1 thread) | 114.7 | 2.64 | 1.00x | max err 0.0e+00 |
| numba cpu (parallel) | 288.9 | 5.97 | 0.44x | max err 0.0e+00 |
| vulkan: NVIDIA TITAN X (Pascal) | 46.4 | 8.40 | 0.31x | max err 0.0e+00 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 48.2 | 11.59 | 0.23x | max err 0.0e+00 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits) | 51.9 | 12.25 | 0.22x | max err 0.0e+00 |
| numba-cuda: NVIDIA TITAN X (Pascal) | 76.8 | 16.62 | 0.16x | max err 6.8e-08 |
| vulkan: NVIDIA TITAN X (Pascal), device arrays | 0.7 | 0.28 | 9.50x | max err 0.0e+00 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2), device arrays | 6.4 | 4.78 | 0.55x | max err 0.0e+00 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits), device arrays | 4.0 | 3.86 | 0.69x | max err 0.0e+00 |
| numba-cuda: NVIDIA TITAN X (Pascal), device arrays | 0.5 | 0.22 | 11.97x | max err 6.8e-08 |
