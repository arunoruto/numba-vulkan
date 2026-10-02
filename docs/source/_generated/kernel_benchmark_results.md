### Sum of 16,777,216 float32 values (shared memory, barriers, one float atomic per workgroup)

| Backend | Time (ms) | Check |
| --- | ---: | --- |
| numba cpu (parallel) | 5.957 | rel. err 9.9e-06 |
| vulkan: NVIDIA TITAN X (Pascal) | 0.280 | rel. err 2.3e-07 |
| vulkan: NVIDIA TITAN X (Pascal), narrow="ints" | 0.290 | rel. err 1.7e-07 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 4.391 | rel. err 5.2e-08 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2), narrow="ints" | 3.296 | rel. err 1.7e-07 |
| numba-cuda: NVIDIA TITAN X (Pascal) | 0.239 | rel. err 4.1e-07 |

### Histogram of 16,777,216 int32 values into 256 bins (shared-memory integer atomics)

| Backend | Time (ms) | Check |
| --- | ---: | --- |
| numba cpu (1 thread) | 8.094 | exact |
| vulkan: NVIDIA TITAN X (Pascal) | 0.314 | exact |
| vulkan: NVIDIA TITAN X (Pascal), narrow="ints" | 0.332 | exact |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 4.257 | exact |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2), narrow="ints" | 3.579 | exact |
| numba-cuda: NVIDIA TITAN X (Pascal) | 0.248 | exact |

### 1024x1024 float32 matrix product (16x16 shared-memory tiles)

| Backend | Time (ms) | Check |
| --- | ---: | --- |
| numba cpu (parallel) | 22.013 | max rel. err 1.9e-06 |
| vulkan: NVIDIA TITAN X (Pascal) | 3.089 | max rel. err 1.9e-06 |
| vulkan: NVIDIA TITAN X (Pascal), narrow="ints" | 2.058 | max rel. err 1.9e-06 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 92.222 | max rel. err 1.9e-06 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2), narrow="ints" | 76.220 | max rel. err 1.9e-06 |
| numba-cuda: NVIDIA TITAN X (Pascal) | 1.977 | max rel. err 1.9e-06 |

