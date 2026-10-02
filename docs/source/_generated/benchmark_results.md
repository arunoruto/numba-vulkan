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
| numba cpu (1 thread) | 857.6 | 474.36 | 1.00x | 100.00% equal |
| numba cpu (parallel) | 513.8 | 92.66 | 5.12x | 100.00% equal |
| vulkan: NVIDIA TITAN X (Pascal) | 152.8 | 14.36 | 33.02x | 100.00% equal |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 157.2 | 56.65 | 8.37x | 100.00% equal |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits) | 164.6 | 82.03 | 5.78x | 100.00% equal |
| numba-cuda: NVIDIA TITAN X (Pascal) | 280.6 | 13.36 | 35.51x | 100.00% equal |
| vulkan: NVIDIA TITAN X (Pascal), device arrays | 11.1 | 9.50 | 49.92x | 100.00% equal |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2), device arrays | 57.4 | 53.62 | 8.85x | 100.00% equal |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits), device arrays | 78.0 | 70.89 | 6.69x | 100.00% equal |
| numba-cuda: NVIDIA TITAN X (Pascal), device arrays | 8.8 | 8.25 | 57.47x | 100.00% equal |

### option (4,194,304 options)

| Backend | First call (ms) | Best (ms) | Speed-up | Agreement with CPU |
| --- | ---: | ---: | ---: | --- |
| numba cpu (1 thread) | 255.5 | 105.15 | 1.00x | max err 0.0e+00 |
| numba cpu (parallel) | 348.0 | 19.41 | 5.42x | max err 0.0e+00 |
| vulkan: NVIDIA TITAN X (Pascal) | 227.3 | 10.14 | 10.37x | max err 3.4e-07 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 215.2 | 12.98 | 8.10x | max err 3.4e-07 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits) | 256.5 | 37.90 | 2.77x | max err 2.5e-07 |
| numba-cuda: NVIDIA TITAN X (Pascal) | 92.6 | 13.42 | 7.84x | max err 2.9e-07 |
| vulkan: NVIDIA TITAN X (Pascal), device arrays | 1.1 | 0.40 | 260.24x | max err 3.4e-07 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2), device arrays | 9.4 | 6.16 | 17.08x | max err 3.4e-07 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits), device arrays | 35.2 | 31.84 | 3.30x | max err 2.5e-07 |
| numba-cuda: NVIDIA TITAN X (Pascal), device arrays | 1.3 | 0.29 | 358.69x | max err 2.9e-07 |

### saxpy (4,194,304 elements)

| Backend | First call (ms) | Best (ms) | Speed-up | Agreement with CPU |
| --- | ---: | ---: | ---: | --- |
| numba cpu (1 thread) | 116.2 | 2.39 | 1.00x | max err 0.0e+00 |
| numba cpu (parallel) | 292.7 | 4.35 | 0.55x | max err 0.0e+00 |
| vulkan: NVIDIA TITAN X (Pascal) | 48.4 | 8.89 | 0.27x | max err 0.0e+00 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 49.5 | 8.32 | 0.29x | max err 0.0e+00 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits) | 64.8 | 9.81 | 0.24x | max err 0.0e+00 |
| numba-cuda: NVIDIA TITAN X (Pascal) | 58.0 | 15.94 | 0.15x | max err 6.8e-08 |
| vulkan: NVIDIA TITAN X (Pascal), device arrays | 0.9 | 0.28 | 8.49x | max err 0.0e+00 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2), device arrays | 5.5 | 5.28 | 0.45x | max err 0.0e+00 |
| vulkan: llvmpipe (LLVM 21.1.8, 256 bits), device arrays | 4.4 | 3.71 | 0.65x | max err 0.0e+00 |
| numba-cuda: NVIDIA TITAN X (Pascal), device arrays | 0.4 | 0.22 | 10.91x | max err 6.8e-08 |
