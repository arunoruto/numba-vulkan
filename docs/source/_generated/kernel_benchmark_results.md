### Sum of 16,777,216 float32 values (shared memory, barriers, one float atomic per workgroup)

| Backend | Time (ms) | Check |
| --- | ---: | --- |
| numba cpu (parallel) | 6.018 | rel. err 9.9e-06 |
| vulkan: NVIDIA TITAN X (Pascal) | 0.291 | rel. err 1.7e-07 |
| vulkan: NVIDIA TITAN X (Pascal), narrow="ints" | 0.296 | rel. err 5.2e-08 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 4.476 | rel. err 1.7e-07 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2), narrow="ints" | 3.551 | rel. err 5.2e-08 |
| numba-cuda: NVIDIA TITAN X (Pascal) | 0.237 | rel. err 1.1e-07 |

### Histogram of 16,777,216 int32 values into 256 bins (shared-memory integer atomics)

| Backend | Time (ms) | Check |
| --- | ---: | --- |
| numba cpu (1 thread) | 7.895 | exact |
| vulkan: NVIDIA TITAN X (Pascal) | 0.340 | exact |
| vulkan: NVIDIA TITAN X (Pascal), narrow="ints" | 0.298 | exact |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 4.351 | exact |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2), narrow="ints" | 3.726 | exact |
| numba-cuda: NVIDIA TITAN X (Pascal) | 0.242 | exact |

### 1024x1024 float32 matrix product (16x16 shared-memory tiles)

| Backend | Time (ms) | Check |
| --- | ---: | --- |
| numba cpu (parallel) | 22.580 | max rel. err 1.9e-06 |
| vulkan: NVIDIA TITAN X (Pascal) | 2.450 | max rel. err 1.9e-06 |
| vulkan: NVIDIA TITAN X (Pascal), narrow="ints" | 1.697 | max rel. err 1.9e-06 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2) | 96.497 | max rel. err 1.9e-06 |
| vulkan: Intel(R) UHD Graphics 630 (CFL GT2), narrow="ints" | 76.867 | max rel. err 1.9e-06 |
| numba-cuda: NVIDIA TITAN X (Pascal) | 1.605 | max rel. err 1.9e-06 |

