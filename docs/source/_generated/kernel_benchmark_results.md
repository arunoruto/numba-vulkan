### Sum of 16,777,216 float32 values (shared memory, barriers, one float atomic per workgroup)

| Backend | Time (ms) | Check |
| --- | ---: | --- |
| numba cpu (parallel) | 8.956 | rel. err 9.9e-06 |
| vulkan: NVIDIA TITAN X (Pascal) | 0.261 | rel. err 2.3e-07 |
| vulkan: NVIDIA TITAN X (Pascal), 64-bit integers | 0.338 | rel. err 1.7e-07 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 3.945 | rel. err 1.7e-07 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2), 64-bit integers | 4.872 | rel. err 1.7e-07 |
| numba-cuda: NVIDIA TITAN X (Pascal) | 0.209 | rel. err 2.9e-07 |

### Histogram of 16,777,216 int32 values into 256 bins (shared-memory integer atomics)

| Backend | Time (ms) | Check |
| --- | ---: | --- |
| numba cpu (1 thread) | 7.627 | exact |
| vulkan: NVIDIA TITAN X (Pascal) | 0.355 | exact |
| vulkan: NVIDIA TITAN X (Pascal), 64-bit integers | 0.303 | exact |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 3.921 | exact |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2), 64-bit integers | 5.043 | exact |
| numba-cuda: NVIDIA TITAN X (Pascal) | 0.221 | exact |

### 1024x1024 float32 matrix product (16x16 shared-memory tiles)

| Backend | Time (ms) | Check |
| --- | ---: | --- |
| numba cpu (parallel) | 23.110 | max rel. err 1.9e-06 |
| vulkan: NVIDIA TITAN X (Pascal) | 2.070 | max rel. err 1.9e-06 |
| vulkan: NVIDIA TITAN X (Pascal), 64-bit integers | 3.143 | max rel. err 1.9e-06 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 81.840 | max rel. err 1.9e-06 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2), 64-bit integers | 102.264 | max rel. err 1.9e-06 |
| numba-cuda: NVIDIA TITAN X (Pascal) | 1.943 | max rel. err 1.9e-06 |

