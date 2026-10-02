"""Compare numba-vulkan against Numba's CPU target and numba-cuda.

Every workload is one scalar "core" function that is compiled unchanged for
each backend, plus a thin per-backend driver. Timings are wall-clock for a
full call. GPU backends are timed with host (NumPy) arrays, which includes
data transfer, and with device arrays, which does not.

    uv run python benchmarks/bench.py
    uv run python benchmarks/bench.py --size 1024 --repeat 3 --json out.json
"""

import argparse
import datetime
import json
import math
import os
import platform
import time
import warnings
from importlib import metadata

import numpy as np
from numba import njit, prange

import numba_vulkan as nv

try:
    from numba import cuda

    HAVE_CUDA = cuda.is_available()
except Exception:
    HAVE_CUDA = False

f32 = np.float32
HALF, ONE, TWO, FOUR, LOGISTIC = f32(0.5), f32(1.0), f32(2.0), f32(4.0), f32(1.702)


# -- workload cores (pure Python, float32 throughout) -------------------------


def mandelbrot_core(cr, ci, maxiter):
    zr = f32(0.0)
    zi = f32(0.0)
    n = 0
    while n < maxiter and zr * zr + zi * zi < FOUR:
        t = zr * zr - zi * zi + cr
        zi = TWO * zr * zi + ci
        zr = t
        n += 1
    return n


def option_core(spot, strike, years, rate, vol):
    # Black-Scholes call price with a logistic approximation of the normal CDF.
    root = vol * math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate + HALF * vol * vol) * years) / root
    d2 = d1 - root
    n1 = ONE / (ONE + math.exp(-LOGISTIC * d1))
    n2 = ONE / (ONE + math.exp(-LOGISTIC * d2))
    return spot * n1 - strike * math.exp(-rate * years) * n2


def saxpy_core(a, x, y):
    return a * x + y


# -- backends -------------------------------------------------------------------


def cpu_backend(parallel):
    mandel = njit(mandelbrot_core)
    option = njit(option_core)
    axpy = njit(saxpy_core)
    loop = prange if parallel else range

    @njit(parallel=parallel)
    def run_mandelbrot(xmin, ymin, dx, dy, maxiter, out):
        for row in loop(out.shape[0]):
            for col in range(out.shape[1]):
                out[row, col] = mandel(xmin + col * dx, ymin + row * dy, maxiter)

    @njit(parallel=parallel)
    def run_option(spot, strike, years, rate, vol, out):
        for i in loop(out.shape[0]):
            out[i] = option(spot[i], strike[i], years[i], rate, vol)

    @njit(parallel=parallel)
    def run_saxpy(a, x, y, out):
        for i in loop(out.shape[0]):
            out[i] = axpy(a, x[i], y[i])

    return {"mandelbrot": run_mandelbrot, "option": run_option, "saxpy": run_saxpy}


def vulkan_backend(device):
    mandel = nv.jit(mandelbrot_core)
    option = nv.jit(option_core)
    axpy = nv.jit(saxpy_core)

    @nv.jit
    def mandelbrot_kernel(xmin, ymin, dx, dy, maxiter, out):
        col = nv.global_id(0)
        row = nv.global_id(1)
        if row < out.shape[0] and col < out.shape[1]:
            out[row, col] = mandel(xmin + col * dx, ymin + row * dy, maxiter)

    @nv.jit
    def option_kernel(spot, strike, years, rate, vol, out):
        i = nv.global_id(0)
        if i < out.shape[0]:
            out[i] = option(spot[i], strike[i], years[i], rate, vol)

    @nv.jit
    def saxpy_kernel(a, x, y, out):
        i = nv.global_id(0)
        if i < out.shape[0]:
            out[i] = axpy(a, x[i], y[i])

    def run_mandelbrot(*args):
        h, w = args[-1].shape
        mandelbrot_kernel.forall((w, h), device=device)(*args)

    def run_option(*args):
        option_kernel.forall(args[-1].size, device=device)(*args)

    def run_saxpy(*args):
        saxpy_kernel.forall(args[-1].size, device=device)(*args)

    return {
        "mandelbrot": run_mandelbrot,
        "option": run_option,
        "saxpy": run_saxpy,
        "to_device": lambda array: nv.to_device(array, device),
        "sync": lambda: nv.synchronize(device),
    }


def cuda_backend():
    mandel = cuda.jit(device=True)(mandelbrot_core)
    option = cuda.jit(device=True)(option_core)
    axpy = cuda.jit(device=True)(saxpy_core)

    @cuda.jit
    def mandelbrot_kernel(xmin, ymin, dx, dy, maxiter, out):
        col, row = cuda.grid(2)
        if row < out.shape[0] and col < out.shape[1]:
            out[row, col] = mandel(xmin + col * dx, ymin + row * dy, maxiter)

    @cuda.jit
    def option_kernel(spot, strike, years, rate, vol, out):
        i = cuda.grid(1)
        if i < out.shape[0]:
            out[i] = option(spot[i], strike[i], years[i], rate, vol)

    @cuda.jit
    def saxpy_kernel(a, x, y, out):
        i = cuda.grid(1)
        if i < out.shape[0]:
            out[i] = axpy(a, x[i], y[i])

    def run_mandelbrot(*args):
        h, w = args[-1].shape
        mandelbrot_kernel[(-(-w // 16), -(-h // 16)), (16, 16)](*args)

    def run_option(*args):
        option_kernel.forall(args[-1].size)(*args)

    def run_saxpy(*args):
        saxpy_kernel.forall(args[-1].size)(*args)

    return {
        "mandelbrot": run_mandelbrot,
        "option": run_option,
        "saxpy": run_saxpy,
        "to_device": cuda.to_device,
        "sync": cuda.synchronize,
    }


# -- workloads ------------------------------------------------------------------


def make_workloads(size, maxiter):
    rng = np.random.default_rng(0)
    n = size * size

    def mandelbrot():
        out = np.zeros((size, size), dtype=np.int32)
        return (f32(-2.0), f32(-1.5), f32(3.0 / size), f32(3.0 / size), maxiter, out)

    def option():
        spot = rng.uniform(10, 100, n).astype(f32)
        strike = rng.uniform(10, 100, n).astype(f32)
        years = rng.uniform(0.25, 5, n).astype(f32)
        return (spot, strike, years, f32(0.02), f32(0.3), np.zeros(n, dtype=f32))

    def saxpy():
        x = rng.random(n, dtype=f32)
        y = rng.random(n, dtype=f32)
        return (f32(2.5), x, y, np.zeros(n, dtype=f32))

    return {
        "mandelbrot": (f"{size}x{size}, {maxiter} iterations", mandelbrot()),
        "option": (f"{n:,} options", option()),
        "saxpy": (f"{n:,} elements", saxpy()),
    }


def measure(kernels, name, args, repeat, on_device=False):
    """Time one workload; with `on_device`, without the transfers."""
    fn, sync = kernels[name], kernels.get("sync", lambda: None)
    if on_device:
        args = [
            kernels["to_device"](a) if isinstance(a, np.ndarray) else a for a in args
        ]
    start = time.perf_counter()
    fn(*args)
    sync()
    first = time.perf_counter() - start
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn(*args)
        sync()
        best = min(best, time.perf_counter() - start)
    return first, best, args[-1].copy_to_host() if on_device else args[-1].copy()


def agreement(name, result, reference):
    if name == "mandelbrot":
        # Iteration counts differ at the fractal boundary due to FMA/rounding.
        return f"{100 * np.mean(result == reference):.2f}% equal"
    # Normalised by the largest value, since some results are close to zero.
    return (
        f"max err {np.max(np.abs(result - reference)) / np.max(np.abs(reference)):.1e}"
    )


def system_info():
    """Hardware and software the benchmark ran on, as ``(label, value)`` pairs."""
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as fh:
            cpu = next(
                ln.split(":", 1)[1].strip() for ln in fh if ln.startswith("model name")
            )
    except (OSError, StopIteration):
        pass
    info = [
        ("Date", datetime.date.today().isoformat()),
        ("CPU", f"{cpu} ({os.cpu_count()} threads)"),
        (
            "Vulkan devices",
            ", ".join(f"{d.name} ({d.kind})" for d in nv.list_devices()),
        ),
        ("Python", platform.python_version()),
    ]
    for package in ("numba", "llvmlite", "numpy", "numba-cuda"):
        try:
            info.append((package, metadata.version(package)))
        except metadata.PackageNotFoundError:
            pass
    return info


def to_markdown(records, opts):
    """Render benchmark records as Markdown, one table per workload."""
    lines = ["| | |", "| --- | --- |"]
    lines += [f"| {label} | {value} |" for label, value in system_info()]
    lines += [f"| Timing | best of {opts.repeat} runs after the first call |", ""]
    workloads = dict.fromkeys(r["workload"] for r in records)
    for workload in workloads:
        rows = [r for r in records if r["workload"] == workload]
        lines += [
            f"### {workload} ({rows[0]['description']})",
            "",
            "| Backend | First call (ms) | Best (ms) | Speed-up | Agreement with CPU |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
        lines += [
            f"| {r['backend']} | {r['first_s'] * 1e3:.1f} | {r['best_s'] * 1e3:.2f} "
            f"| {r['speedup']:.2f}x | {r['agreement']} |"
            for r in rows
        ]
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--size", type=int, default=2048, help="grid edge; arrays have size**2 elements"
    )
    parser.add_argument(
        "--maxiter", type=int, default=200, help="Mandelbrot iteration limit"
    )
    parser.add_argument(
        "--repeat", type=int, default=5, help="timed runs after the first"
    )
    parser.add_argument("--json", metavar="PATH", help="also write the results as JSON")
    parser.add_argument(
        "--markdown", metavar="PATH", help="also write the results as Markdown tables"
    )
    opts = parser.parse_args()
    warnings.filterwarnings("ignore", message=".*copy overhead.*")
    warnings.filterwarnings("ignore", message=".*Grid size.*")

    backends = {
        "numba cpu (1 thread)": cpu_backend(False),
        "numba cpu (parallel)": cpu_backend(True),
    }
    # GPU backends run twice: with NumPy arrays, which are copied to and from
    # the device on every call, and with arrays that stay on the device.
    on_device = set()
    for info in nv.list_devices():
        backends[f"vulkan: {info.name}"] = vulkan_backend(info.index)
    if HAVE_CUDA:
        name = cuda.get_current_device().name
        name = name.decode() if isinstance(name, bytes) else name
        backends[f"numba-cuda: {name}"] = cuda_backend()
    for label in [b for b in backends if "to_device" in backends[b]]:
        backends[f"{label}, device arrays"] = backends[label]
        on_device.add(f"{label}, device arrays")

    records = []
    for name, (description, args) in make_workloads(opts.size, opts.maxiter).items():
        print(f"\n{name} ({description})")
        print(
            f"  {'backend':<60}{'first call':>12}{'best':>12}{'speedup':>9}  agreement"
        )
        reference = baseline = None
        for label, kernels in backends.items():
            try:
                first, best, result = measure(
                    kernels, name, args, opts.repeat, label in on_device
                )
            except Exception as exc:
                print(
                    f"  {label:<60}  failed: {type(exc).__name__}: {str(exc).splitlines()[0][:60]}"
                )
                continue
            if reference is None:
                reference, baseline = result, best
            check = agreement(name, result, reference)
            print(
                f"  {label:<60}{first * 1e3:>10.1f}ms{best * 1e3:>10.2f}ms"
                f"{baseline / best:>8.2f}x  {check}"
            )
            records.append(
                dict(
                    workload=name,
                    description=description,
                    backend=label,
                    first_s=first,
                    best_s=best,
                    speedup=baseline / best,
                    agreement=check,
                )
            )
    if opts.markdown:
        with open(opts.markdown, "w") as fh:
            fh.write(to_markdown(records, opts))
    if opts.json:
        with open(opts.json, "w") as fh:
            json.dump(
                dict(size=opts.size, maxiter=opts.maxiter, results=records),
                fh,
                indent=2,
            )


if __name__ == "__main__":
    main()
