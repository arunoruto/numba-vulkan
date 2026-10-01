# numba-vulkan

```{image} _static/logo.svg
:alt: numba-vulkan logo
:width: 140px
:align: center
:class: dark-light
```

A proof-of-concept Vulkan compute target for [Numba](https://numba.pydata.org/):
write a kernel in Python and run it on any GPU with a Vulkan driver.

```python
import numpy as np
import numba_vulkan as nv

@nv.jit
def saxpy(a, x, y):
    i = nv.global_id(0)
    if i < x.shape[0]:
        y[i] = a * x[i] + y[i]

x = np.arange(1000, dtype=np.float32)
y = np.ones(1000, dtype=np.float32)
saxpy.forall(x.size)(np.float32(2.0), x, y)
```

Numba's only maintained GPU target is CUDA, which ties GPU-accelerated Numba
code to NVIDIA hardware. Vulkan is implemented by nearly every vendor, so one
target could cover all of them. The idea was raised in
[numba/numba#10116](https://github.com/numba/numba/issues/10116); this project
explores how far it can be taken.

:::{note}
The code and this documentation were written by an AI (Claude) under the
direction of, and reviewed by, a human author. See {doc}`authorship`.
:::

:::{warning}
This is a proof of concept. It shows that the approach works end to end and
where it hurts; it is not ready for real workloads. See {doc}`limitations`
and {doc}`known_issues`.
:::

```{toctree}
:maxdepth: 2
:caption: Contents

getting_started
usage
how_it_works
math_library
limitations
known_issues
benchmarks
development
authorship
```

## Indices and tables

- {ref}`genindex`
- {ref}`search`
