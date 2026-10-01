"""Fuzz the control-flow handling with random programs.

Each seed generates a function made of nested if/elif/else, and/or
conditions, for loops, break, continue and early returns; with ``--rich``
also while loops and one more level of nesting. It is compiled
for Vulkan, run on the selected devices and compared with plain Python.

    uv run python tests/fuzz_control_flow.py 0 100          # all devices
    uv run python tests/fuzz_control_flow.py 0 100 --cpu    # CPU devices only
    uv run python tests/fuzz_control_flow.py 0 100 --rich   # also while loops
    uv run python tests/fuzz_control_flow.py --show 42      # print one program

A failure is either a compile error (loud) or a wrong result (silent in
normal use, and therefore the important kind).
"""

import os
import random
import sys
import warnings

import numpy as np

os.environ["NUMBA_VULKAN_VALIDATE"] = "1"
warnings.simplefilter("ignore")
import numba_vulkan as nv

f32 = np.float32
C = [f32(v) for v in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)]


def gen_expr(rng, depth=0):
    r = rng.random()
    if depth > 1 or r < 0.35:
        return rng.choice(["a", "b", "x", "y", f"C{rng.randrange(len(C))}"])
    op = rng.choice(["+", "-", "*"])
    return f"({gen_expr(rng, depth + 1)} {op} {gen_expr(rng, depth + 1)})"


def gen_cond(rng, depth=0):
    r = rng.random()
    if depth > 1 or r < 0.5:
        return f"{gen_expr(rng, 1)} {rng.choice(['<', '>', '<=', '>='])} {gen_expr(rng, 1)}"
    if r < 0.6:
        return f"not ({gen_cond(rng, depth + 1)})"
    return f"({gen_cond(rng, depth + 1)}) {rng.choice(['and', 'or'])} ({gen_cond(rng, depth + 1)})"


def gen_block(rng, indent, depth, in_loop, rich=False):
    lines = []
    for _ in range(rng.randint(1, 3)):
        r = rng.random()
        pad = "    " * indent
        if rich and depth < 4 and r >= 0.6 and rng.random() < 0.5:
            # a while loop; the counter moves first so `continue` is safe
            lines.append(f"{pad}k{depth} = 0")
            lines.append(
                f"{pad}while k{depth} < {rng.randint(1, 4)} and {gen_cond(rng, 2)}:"
            )
            lines.append(f"{pad}    k{depth} += 1")
            lines += gen_block(rng, indent + 1, depth + 1, True, rich)
        elif depth >= (4 if rich else 3) or r < 0.3:
            lines.append(f"{pad}{rng.choice('ab')} = {gen_expr(rng)} * C0")
        elif r < 0.6:
            lines.append(f"{pad}if {gen_cond(rng)}:")
            lines += gen_block(rng, indent + 1, depth + 1, in_loop, rich)
            for _ in range(rng.randint(0, 2)):
                lines.append(f"{pad}elif {gen_cond(rng)}:")
                lines += gen_block(rng, indent + 1, depth + 1, in_loop, rich)
            if rng.random() < 0.6:
                lines.append(f"{pad}else:")
                lines += gen_block(rng, indent + 1, depth + 1, in_loop, rich)
        elif r < 0.75:
            lines.append(f"{pad}for j{depth} in range({rng.randint(1, 4)}):")
            lines += gen_block(rng, indent + 1, depth + 1, True, rich)
        elif r < 0.85:
            lines.append(f"{pad}if {gen_cond(rng)}:")
            lines.append(f"{pad}    return {gen_expr(rng)}")
        elif in_loop:
            lines.append(f"{pad}if {gen_cond(rng)}:")
            lines.append(f"{pad}    {rng.choice(['break', 'continue'])}")
        else:
            lines.append(f"{pad}a = a + b * C1")
    return lines


def make(seed, rich=False):
    rng = random.Random(seed)
    src = (
        ["def f(x, y):", "    a = x", "    b = y"]
        + gen_block(rng, 1, 0, False, rich)
        + ["    return a + b"]
    )
    return "\n".join(src)


def run(seed, devices, rich=False):
    src = make(seed, rich)
    ns = {f"C{i}": c for i, c in enumerate(C)}
    exec(src, ns)
    pyf = ns["f"]
    x = np.linspace(-2, 2, 48, dtype=f32)
    y = np.linspace(3, -1, 48, dtype=f32)
    with np.errstate(all="ignore"):
        want = np.array([pyf(a, b) for a, b in zip(x, y)], dtype=f32)
    helper = nv.jit(pyf)

    @nv.jit
    def kernel(x, y, out):
        i = nv.global_id(0)
        if i < x.shape[0]:
            out[i] = helper(x[i], y[i])

    for dev in devices:
        out = np.zeros_like(x)
        try:
            kernel.forall(x.size, device=dev)(x, y, out)
        except Exception as exc:
            msg = [l for l in str(exc).splitlines() if l.strip()]
            return (
                f"{type(exc).__name__}: {(msg[1] if len(msg) > 1 else msg[0])[:140]}",
                src,
            )
        with np.errstate(all="ignore"):
            ok = np.isclose(out, want, rtol=1e-4, atol=1e-4, equal_nan=True) | (
                ~np.isfinite(want)
            )
        if not ok.all():
            j = np.flatnonzero(~ok)[0]
            return f"WRONG on device {dev} at {j}: got {out[j]} want {want[j]}", src
    return None, src


def main(argv):
    if argv and argv[0] == "--show":
        print(make(int(argv[1]), "--rich" in argv))
        return 0
    lo, hi = int(argv[0]), int(argv[1])
    devices = [
        d.index for d in nv.list_devices() if "--cpu" not in argv or d.kind == "cpu"
    ]
    counts = {"ok": 0, "compile error": 0, "WRONG RESULT": 0}
    for seed in range(lo, hi):
        res, _ = run(seed, devices, "--rich" in argv)
        kind = (
            "ok"
            if res is None
            else ("WRONG RESULT" if res.startswith("WRONG") else "compile error")
        )
        counts[kind] += 1
        if res:
            print(f"seed {seed}: {res}", flush=True)
    print(f"{hi - lo} programs on devices {devices}: {counts}")
    return 1 if counts["WRONG RESULT"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
