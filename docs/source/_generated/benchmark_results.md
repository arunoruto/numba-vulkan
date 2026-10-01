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
| numba cpu (1 thread) | 836.3 | 472.03 | 1.00x | 100.00% equal |
| numba cpu (parallel) | 527.8 | 92.50 | 5.10x | 100.00% equal |
| vulkan: NVIDIA TITAN X (Pascal) | 179.1 | 16.06 | 29.39x | 100.00% equal |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 157.2 | 57.92 | 8.15x | 100.00% equal |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits) | 187.6 | 86.24 | 5.47x | 100.00% equal |
| numba-cuda: NVIDIA TITAN X (Pascal) | 308.2 | 13.42 | 35.18x | 100.00% equal |
| vulkan: NVIDIA TITAN X (Pascal), device arrays | 10.1 | 9.02 | 52.32x | 100.00% equal |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2), device arrays | 58.2 | 55.94 | 8.44x | 100.00% equal |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits), device arrays | 83.8 | 78.72 | 6.00x | 100.00% equal |
| numba-cuda: NVIDIA TITAN X (Pascal), device arrays | 7.8 | 7.65 | 61.70x | 100.00% equal |

### option (4,194,304 options)

| Backend | First call (ms) | Best (ms) | Speed-up | Agreement with CPU |
| --- | ---: | ---: | ---: | --- |
| numba cpu (1 thread) | 294.6 | 110.78 | 1.00x | max err 0.0e+00 |
| numba cpu (parallel) | 363.2 | 20.99 | 5.28x | max err 0.0e+00 |
| vulkan: NVIDIA TITAN X (Pascal) | 234.6 | 11.39 | 9.73x | max err 3.4e-07 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 233.5 | 16.40 | 6.76x | max err 3.4e-07 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits) | 397.9 | 40.08 | 2.76x | max err 2.5e-07 |
| numba-cuda: NVIDIA TITAN X (Pascal) | 104.7 | 14.17 | 7.82x | max err 2.9e-07 |
| vulkan: NVIDIA TITAN X (Pascal), device arrays | 0.9 | 0.72 | 154.23x | max err 3.4e-07 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2), device arrays | 9.1 | 6.11 | 18.14x | max err 3.4e-07 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits), device arrays | 40.3 | 34.92 | 3.17x | max err 2.5e-07 |
| numba-cuda: NVIDIA TITAN X (Pascal), device arrays | 0.9 | 0.28 | 389.54x | max err 2.9e-07 |

### saxpy (4,194,304 elements)

| Backend | First call (ms) | Best (ms) | Speed-up | Agreement with CPU |
| --- | ---: | ---: | ---: | --- |
| numba cpu (1 thread) | 148.1 | 2.11 | 1.00x | max err 0.0e+00 |
| numba cpu (parallel) | 274.8 | 5.43 | 0.39x | max err 0.0e+00 |
| vulkan: NVIDIA TITAN X (Pascal) | 45.1 | 7.49 | 0.28x | max err 0.0e+00 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 49.8 | 8.88 | 0.24x | max err 0.0e+00 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits) | 51.0 | 8.81 | 0.24x | max err 0.0e+00 |
| numba-cuda: NVIDIA TITAN X (Pascal) | 52.5 | 10.45 | 0.20x | max err 6.8e-08 |
| vulkan: NVIDIA TITAN X (Pascal), device arrays | 0.8 | 0.51 | 4.13x | max err 0.0e+00 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2), device arrays | 6.2 | 5.05 | 0.42x | max err 0.0e+00 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits), device arrays | 3.9 | 3.56 | 0.59x | max err 0.0e+00 |
| numba-cuda: NVIDIA TITAN X (Pascal), device arrays | 0.4 | 0.22 | 9.55x | max err 6.8e-08 |
