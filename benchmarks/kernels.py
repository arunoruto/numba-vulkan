"""Compare numba-vulkan and numba-cuda on the same CUDA-style kernels.

`bench.py` compares functions that each backend compiles in its own way.
This script instead runs kernels written once, in the style of numba.cuda,
with workgroup-shared memory, barriers and atomics, and turns the source
into a numba-cuda kernel and a numba-vulkan kernel by substituting the API
names. The kernels, workgroup sizes and grid sizes are identical, and data
stays on the device, so the timings compare the two code generators and
runtimes rather than different algorithms. Numba's CPU target (parallel)
gives a reference point.

    uv run python benchmarks/kernels.py
    uv run python benchmarks/kernels.py --markdown kernels.md
"""

import argparse
import time

import numpy as np
from numba import njit, prange

import numba_vulkan as nv

try:
    from numba import cuda

    HAVE_CUDA = cuda.is_available()
except Exception:  # noqa: BLE001
    HAVE_CUDA = False

# -- the kernels, written once ----------------------------------------------------

API = {
    "vulkan": {
        "LID": "nv.local_id(0)",
        "LID_Y": "nv.local_id(1)",
        "GROUP": "nv.group_id(0)",
        "GROUP_Y": "nv.group_id(1)",
        "BLOCK": "nv.local_size(0)",
        "NGROUPS": "nv.num_groups(0)",
        "SHARED": "nv.shared.array",
        "LOCAL": "nv.local.array",
        "SYNC": "nv.barrier()",
        "ATOMIC_ADD": "nv.atomic.add",
    },
    "cuda": {
        "LID": "cuda.threadIdx.x",
        "LID_Y": "cuda.threadIdx.y",
        "GROUP": "cuda.blockIdx.x",
        "GROUP_Y": "cuda.blockIdx.y",
        "BLOCK": "cuda.blockDim.x",
        "NGROUPS": "cuda.gridDim.x",
        "SHARED": "cuda.shared.array",
        "LOCAL": "cuda.local.array",
        "SYNC": "cuda.syncthreads()",
        "ATOMIC_ADD": "cuda.atomic.add",
    },
}

SOURCE = """
def reduce_sum(x, out):
    partial = {SHARED}(256, float32)
    t = {LID}
    i = {GROUP} * {BLOCK} + t
    stride = {BLOCK} * {NGROUPS}
    acc = float32(0)
    while i < x.shape[0]:
        acc += x[i]
        i += stride
    partial[t] = acc
    {SYNC}
    step = 128
    while step > 0:
        if t < step:
            partial[t] += partial[t + step]
        {SYNC}
        step //= 2
    if t == 0:
        {ATOMIC_ADD}(out, 0, partial[0])


def histogram(x, bins):
    counts = {SHARED}(256, int32)
    t = {LID}
    counts[t] = 0
    {SYNC}
    i = {GROUP} * {BLOCK} + t
    stride = {BLOCK} * {NGROUPS}
    while i < x.shape[0]:
        {ATOMIC_ADD}(counts, x[i] & 255, 1)
        i += stride
    {SYNC}
    {ATOMIC_ADD}(bins, t, counts[t])


def matmul(a, b, c):
    ta = {SHARED}((16, 16), float32)
    tb = {SHARED}((16, 16), float32)
    tx = {LID}
    ty = {LID_Y}
    row = {GROUP_Y} * 16 + ty
    col = {GROUP} * 16 + tx
    acc = float32(0)
    for tile in range((a.shape[1] + 15) // 16):
        k = tile * 16
        ta[ty, tx] = a[row, k + tx] if row < a.shape[0] and k + tx < a.shape[1] else float32(0)
        tb[ty, tx] = b[k + ty, col] if k + ty < b.shape[0] and col < b.shape[1] else float32(0)
        {SYNC}
        for j in range(16):
            acc += ta[ty, j] * tb[j, tx]
        {SYNC}
    if row < c.shape[0] and col < c.shape[1]:
        c[row, col] = acc


def matmul_blocked(a, b, c):
    # A 64x64 tile of c per workgroup of 16x16, a 4x4 block per invocation,
    # accumulated in registers; a and b pass through shared memory in slices
    # of 16 along k, each invocation loading four values of each.
    ta = {SHARED}((16, 64), float32)  # a's slice, transposed: [k, row]
    tb = {SHARED}((16, 64), float32)  # b's slice: [k, column]
    tx = {LID}
    ty = {LID_Y}
    t = ty * 16 + tx
    row0 = {GROUP_Y} * 64
    col0 = {GROUP} * 64
    acc = {LOCAL}((4, 4), float32)
    for i in range(4):
        for j in range(4):
            acc[i, j] = float32(0)
    n = a.shape[1]
    for k0 in range(0, n, 16):
        for q in range(4):
            e = q * 256 + t
            r = row0 + e // 16
            k = k0 + e % 16
            ta[e % 16, e // 16] = a[r, k] if r < a.shape[0] and k < n else float32(0)
            k = k0 + e // 64
            col = col0 + e % 64
            tb[e // 64, e % 64] = b[k, col] if k < n and col < b.shape[1] else float32(0)
        {SYNC}
        for k in range(16):
            ra = {LOCAL}(4, float32)
            rb = {LOCAL}(4, float32)
            for i in range(4):
                ra[i] = ta[k, ty * 4 + i]
                rb[i] = tb[k, tx * 4 + i]
            for i in range(4):
                for j in range(4):
                    acc[i, j] += ra[i] * rb[j]
        {SYNC}
    for i in range(4):
        for j in range(4):
            row = row0 + ty * 4 + i
            col = col0 + tx * 4 + j
            if row < c.shape[0] and col < c.shape[1]:
                c[row, col] = acc[i, j]
"""


def build(backend, **options):
    """Compile the kernels' source for one backend."""
    scope = {"nv": nv, "float32": np.float32, "int32": np.int32}
    if backend == "cuda":
        scope["cuda"] = cuda
    exec(SOURCE.format(**API[backend]), scope)  # noqa: S102
    decorate = cuda.jit if backend == "cuda" else nv.jit(**options)
    return {
        name: decorate(scope[name])
        for name in ("reduce_sum", "histogram", "matmul", "matmul_blocked")
    }


# -- CPU references -----------------------------------------------------------


@njit(parallel=True)
def cpu_sum(x):
    total = np.float32(0)
    for i in prange(x.shape[0]):
        total += x[i]
    return total


@njit
def cpu_histogram(x, bins):
    for i in range(x.shape[0]):
        bins[x[i] & 255] += 1


@njit(parallel=True)
def cpu_matmul(a, b, c):
    for i in prange(a.shape[0]):
        for k in range(a.shape[1]):
            aik = a[i, k]
            for j in range(b.shape[1]):
                c[i, j] += aik * b[k, j]


# -- running ------------------------------------------------------------------------


class Backend:
    """Transfers, launches and synchronisation of one GPU backend."""

    def __init__(self, name, **options):
        self.name = name
        self.kernels = build(name, **options)

    def to_device(self, array):
        return cuda.to_device(array) if self.name == "cuda" else nv.to_device(array)

    def zeros(self, shape, dtype):
        return self.to_device(np.zeros(shape, dtype=dtype))

    def launch(self, kernel, groups, local, *args):
        self.kernels[kernel][groups, local](*args)

    def sync(self):
        if self.name == "cuda":
            cuda.synchronize()
        else:
            nv.synchronize()

    def events(self):
        """Two events for timing on the device."""
        if self.name == "cuda":
            return cuda.event(timing=True), cuda.event(timing=True)
        return nv.event(), nv.event()

    def elapsed(self, start, end):
        """Seconds on the device between two recorded events."""
        if self.name == "cuda":
            return cuda.event_elapsed_time(start, end) / 1e3
        return start.elapsed_time(end) / 1e3


def measure(backend, launch, before=None):
    """Time one launch: until it returns, until it is done, and on the device.

    Returns
    -------
    tuple of float
        Enqueue, end-to-end and device time in seconds.
    """
    if before is not None:
        before()
    backend.sync()
    start, end = backend.events()
    start.record()
    begin = time.perf_counter()
    launch()
    queued = time.perf_counter()
    end.record()
    backend.sync()
    done = time.perf_counter()
    return queued - begin, done - begin, backend.elapsed(start, end)


def timings(function, repeat, before=None):
    """Wall-clock times of `repeat` calls of `function` in seconds.

    `before` runs untimed before each call.
    """
    samples = []
    for _ in range(repeat):
        if before is not None:
            before()
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return samples


DESCRIPTIONS = {
    "reduce": "{n:,} float32 values",
    "histogram": "{n:,} int32 values into 256 bins",
    "matmul": "{m}x{m} float32",
    "matmul_blocked": "{m}x{m} float32, 4x4 per invocation",
}


def run(size=1 << 24, matrix=1024, repeat=10, all_devices=False, verbose=True):
    """Run the three kernels on every backend.

    Parameters
    ----------
    size : int
        Elements for the reduction and the histogram.
    matrix : int
        Edge of the square matrices.
    repeat : int
        Timed calls; every backend is called once before, untimed.
    all_devices : bool
        Include CPU Vulkan devices (llvmpipe).
    verbose : bool
        Print a line per measurement.

    Returns
    -------
    list of dict
        One record per workload and backend; see ``benchmarks/collect.py``
        for the fields.
    """
    rng = np.random.default_rng(0)
    n = size
    x = rng.random(n, dtype=np.float32)
    keys = rng.integers(0, 1 << 20, n, dtype=np.int32)
    m = matrix
    a = rng.random((m, m), dtype=np.float32)
    b = rng.random((m, m), dtype=np.float32)
    groups_1d = 1024
    records = []

    def record(workload, label, info, samples, error):
        backend, device, variant = info
        records.append(
            dict(
                suite="kernels",
                workload=workload,
                description=DESCRIPTIONS[workload].format(n=n, m=m),
                backend=backend,
                device=device,
                variant=variant,
                label=label,
                first_s=None,
                samples_s=samples,
                check=error,
            )
        )
        if verbose:
            print(
                f"  {workload:<10} {label:<52} {min(samples) * 1e3:9.3f} ms   {error}"
            )

    if verbose:
        print(
            f"reduce_sum: {n:,} float32 | histogram: {n:,} int32 into 256 bins | "
            f"matmul: {m}x{m} float32"
        )

    # CPU
    total = cpu_sum(x)
    record(
        "reduce",
        "numba cpu (parallel)",
        ("cpu", None, "parallel"),
        timings(lambda: cpu_sum(x), repeat),
        f"rel. err {abs(total - x.sum(dtype=np.float64)) / x.sum(dtype=np.float64):.1e}",
    )
    bins = np.zeros(256, dtype=np.int64)
    cpu_histogram(keys, bins)
    reference_bins = np.bincount(keys & 255, minlength=256)
    record(
        "histogram",
        "numba cpu (1 thread)",
        ("cpu", None, "1 thread"),
        timings(lambda: cpu_histogram(keys, bins), repeat, lambda: bins.fill(0)),
        "exact" if (bins == reference_bins).all() else "WRONG",
    )
    c = np.zeros((m, m), dtype=np.float32)
    cpu_matmul(a, b, c)
    reference_c = a.astype(np.float64) @ b.astype(np.float64)
    record(
        "matmul",
        "numba cpu (parallel)",
        ("cpu", None, "parallel"),
        timings(lambda: cpu_matmul(a, b, c), repeat, lambda: c.fill(0)),
        f"max rel. err {np.abs(c - reference_c).max() / np.abs(reference_c).max():.1e}",
    )

    backends = []
    for info in nv.list_devices():
        if all_devices or info.kind != "cpu":
            nv.select_device(info.index)
            backends.append(
                (
                    f"vulkan: {info.name}",
                    Backend("vulkan"),
                    info.index,
                    ("vulkan", info.name, "32-bit integers"),
                )
            )
            backends.append(
                (
                    f"vulkan: {info.name}, 64-bit integers",
                    Backend("vulkan", narrow=False),
                    info.index,
                    ("vulkan", info.name, "64-bit integers"),
                )
            )
    if HAVE_CUDA:
        name = cuda.get_current_device().name
        name = name.decode() if isinstance(name, bytes) else name
        backends.append(
            (
                f"numba-cuda: {name}",
                Backend("cuda"),
                None,
                ("cuda", name, "device arrays"),
            )
        )

    zero1 = np.zeros(1, dtype=np.float32)
    zero256 = np.zeros(256, dtype=np.int32)
    exact = x.sum(dtype=np.float64)
    reference_scale = np.abs(reference_c).max()
    # (backend label, info, workload, backend, device index, launch, before,
    #  check, samples)
    runs = []
    for label, backend, index, info in backends:
        if index is not None:
            nv.select_device(index)
        dx, dkeys, da, db = (backend.to_device(v) for v in (x, keys, a, b))
        out = backend.zeros(1, np.float32)
        dbins = backend.zeros(256, np.int32)
        dc = backend.zeros((m, m), np.float32)
        tiles = (-(-m // 16), -(-m // 16))
        blocks = (-(-m // 64), -(-m // 64))
        workloads = [
            (
                "reduce",
                lambda b=backend, dx=dx, out=out: b.launch(
                    "reduce_sum", groups_1d, 256, dx, out
                ),
                lambda out=out: out.copy_to_device(zero1),
                lambda out=out: (
                    f"rel. err {abs(out.copy_to_host()[0] - exact) / exact:.1e}"
                ),
            ),
            (
                "histogram",
                lambda b=backend, dk=dkeys, db_=dbins: b.launch(
                    "histogram", groups_1d, 256, dk, db_
                ),
                lambda db_=dbins: db_.copy_to_device(zero256),
                lambda db_=dbins: (
                    "exact" if (db_.copy_to_host() == reference_bins).all() else "WRONG"
                ),
            ),
            (
                "matmul",
                lambda b=backend, da=da, db=db, dc=dc: b.launch(
                    "matmul", tiles, (16, 16), da, db, dc
                ),
                None,
                lambda dc=dc: (
                    "max rel. err "
                    f"{np.abs(dc.copy_to_host() - reference_c).max() / reference_scale:.1e}"
                ),
            ),
            (
                "matmul_blocked",
                lambda b=backend, da=da, db=db, dc=dc: b.launch(
                    "matmul_blocked", blocks, (16, 16), da, db, dc
                ),
                None,
                lambda dc=dc: (
                    "max rel. err "
                    f"{np.abs(dc.copy_to_host() - reference_c).max() / reference_scale:.1e}"
                ),
            ),
        ]
        for workload, launch, before, check in workloads:
            measure(backend, launch, before)  # compiles
            runs.append(
                (label, info, workload, backend, index, launch, before, check, [])
            )

    # Round by round, so that every backend sees the same conditions (clock
    # boosts, thermal state, other load) as often as the others.
    for _ in range(repeat):
        for _, _, _, backend, index, launch, before, _, samples in runs:
            if index is not None:
                nv.select_device(index)
            samples.append(measure(backend, launch, before))
    for label, info, workload, backend, index, launch, before, check, samples in runs:
        if index is not None:
            nv.select_device(index)
        # The results of the last round of this workload are checked; a later
        # workload on the same backend may have overwritten them, so run once more.
        if before is not None:
            before()
        launch()
        backend.sync()
        record(workload, label, info, [s[1] for s in samples], check())
        records[-1]["enqueue_s"] = [s[0] for s in samples]
        records[-1]["device_s"] = [s[2] for s in samples]
    return records


def to_markdown(records, opts):
    """Render the results as one Markdown table per workload."""
    titles = {
        "reduce": f"Sum of {opts.size:,} float32 values (shared memory, barriers, "
        "one float atomic per workgroup)",
        "histogram": f"Histogram of {opts.size:,} int32 values into 256 bins "
        "(shared-memory integer atomics)",
        "matmul": f"{opts.matrix}x{opts.matrix} float32 matrix product "
        "(16x16 shared-memory tiles)",
        "matmul_blocked": f"{opts.matrix}x{opts.matrix} float32 matrix product "
        "(64x64 tiles, 4x4 per invocation in registers)",
    }
    lines = []
    for workload, title in titles.items():
        rows = [r for r in records if r["workload"] == workload]
        lines += [
            f"### {title}",
            "",
            "| Backend | Time (ms) | On the device (ms) | Check |",
            "| --- | ---: | ---: | --- |",
        ]
        lines += [
            f"| {r['label']} | {min(r['samples_s']) * 1e3:.3f} | "
            + (f"{min(r['device_s']) * 1e3:.3f}" if r.get("device_s") else "")
            + f" | {r['check']} |"
            for r in rows
        ]
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--size", type=int, default=1 << 24, help="elements for reduce/histogram"
    )
    parser.add_argument("--matrix", type=int, default=1024, help="matrix edge")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument(
        "--all-devices", action="store_true", help="include CPU Vulkan devices"
    )
    parser.add_argument("--markdown", metavar="PATH")
    opts = parser.parse_args()
    records = run(opts.size, opts.matrix, opts.repeat, opts.all_devices)
    if opts.markdown:
        with open(opts.markdown, "w") as fh:
            fh.write(to_markdown(records, opts) + "\n")


if __name__ == "__main__":
    main()
