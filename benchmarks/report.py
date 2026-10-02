"""Draw charts from the collected benchmark results.

Reads every ``benchmarks/results/<user>/*.json`` written by ``collect.py``
(runs with ``--quick`` are left out) and writes, into an output directory:

``runs.md``
    a table of the runs: who, which machine, when, which commit, hardware
``charts.md``
    the charts below as MyST images, one section per machine
``backends-<user>-<machine>.svg``
    the latest run of each machine: every backend, device and variant
``machines.svg``
    the latest run of each machine side by side, fastest result per backend
``history.svg``
    the best time of each workload over time, per machine and backend

    uv run --group docs python benchmarks/report.py docs/source/_generated

The documentation build runs it (``docs/source/conf.py``).
"""

import argparse
import datetime
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

RESULTS = pathlib.Path(__file__).parent / "results"
SCHEMA = 1
SUITES = {
    "apps": "Same function on every backend",
    "kernels": "Same CUDA-style kernels",
}
COLORS = {"cpu": "#7f7f7f", "vulkan": "#c41e3a", "cuda": "#76b900"}
# The variants that stand for a backend when only its fastest result is
# shown: on the GPUs, data that stays on the device and 32-bit integers.
MAIN_VARIANTS = {
    "cpu": ("parallel", "1 thread"),
    "vulkan": ("device arrays", "32-bit integers"),
    "cuda": ("device arrays",),
}
MARKERS = "osD^v<>p*"
LINESTYLES = ("-", "--", ":", "-.")
plt.rcParams.update(
    {
        "svg.hashsalt": "numba-vulkan",  # stable SVG ids, for small diffs
        "svg.fonttype": "none",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load(directory=RESULTS):
    """Every complete run, oldest first.

    Parameters
    ----------
    directory : pathlib.Path
        The ``results`` directory.

    Returns
    -------
    list of dict
        The parsed files, each with an added ``path``.
    """
    runs = []
    for path in sorted(directory.glob("*/*.json")):
        data = json.loads(path.read_text())
        if data.get("schema") != SCHEMA:
            print(
                f"skipping {path}: unknown schema {data.get('schema')}", file=sys.stderr
            )
            continue
        if data["settings"].get("quick"):
            continue
        data["path"] = path
        runs.append(data)
    return sorted(runs, key=lambda run: run["date"])


def machine_key(run):
    """``user/machine`` of a run."""
    return f"{run['user']}/{run['machine']}"


def latest(runs):
    """The newest run of every machine."""
    newest = {}
    for run in runs:
        newest[machine_key(run)] = run
    return list(newest.values())


def best(record):
    """Best time of a result in milliseconds."""
    return min(record["samples_s"]) * 1e3


def is_main(record):
    """Whether a result is a variant that stands for its backend."""
    return record["variant"] in MAIN_VARIANTS[record["backend"]]


def _plain_log(axis):
    """Label a log axis with plain numbers at 1, 2 and 5 times powers of ten."""
    axis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def _device(name):
    """A device name without the driver detail in parentheses."""
    if name is None:
        return "CPU"
    short = name.split(" (")[0] if not name.startswith("llvmpipe") else "llvmpipe"
    return short.replace("NVIDIA ", "").replace("Intel(R) ", "Intel ")


def _workloads(records):
    """Suites and workloads in order of appearance."""
    return list(dict.fromkeys((r["suite"], r["workload"]) for r in records))


def _save(fig, path):
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def backends_chart(run, path):
    """Every result of one run, a panel per workload."""
    workloads = _workloads(run["results"])
    fig, axes = plt.subplots(
        len(workloads), 1, figsize=(7, 1.0 + 0.26 * len(run["results"])), squeeze=False
    )
    for ax, (suite, workload) in zip(axes[:, 0], workloads):
        rows = [
            r
            for r in run["results"]
            if (r["suite"], r["workload"]) == (suite, workload)
        ]
        rows.sort(key=lambda r: (list(COLORS).index(r["backend"]), r["device"] or ""))
        labels = [f"{_device(r['device'])}, {r['variant']}" for r in rows]
        times = [best(r) for r in rows]
        # Dots rather than bars: on a log axis, a bar's length depends on
        # where the axis starts.
        ax.scatter(times, range(len(rows)), color=[COLORS[r["backend"]] for r in rows],
                   s=24, zorder=3)  # fmt: skip
        for y, t in enumerate(times):
            ax.annotate(f"{t:.3g}", (t, y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=7)  # fmt: skip
        ax.set_yticks(range(len(rows)), labels)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.grid(axis="y", color="#e5e5e5", zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(min(times) / 1.6, max(times) * 3)
        _plain_log(ax.xaxis)
        ax.set_title(f"{workload}: {rows[0]['description']} ({SUITES[suite].lower()})",
                     loc="left", fontsize=9)  # fmt: skip
    axes[-1, 0].set_xlabel("best time (ms, log scale; further left is faster)")
    axes[0, 0].legend(
        handles=[Patch(color=c, label=b) for b, c in COLORS.items()],
        loc="lower right",
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(f"{machine_key(run)}, {run['date'][:10]}", x=0.0, ha="left")
    fig.tight_layout()
    _save(fig, path)


def _fastest(run, suite, workload, backend):
    """Fastest main-variant result of a backend in a run, or ``None``."""
    rows = [
        r
        for r in run["results"]
        if (r["suite"], r["workload"], r["backend"]) == (suite, workload, backend)
        and is_main(r)
    ]
    return min(rows, key=best) if rows else None


def machines_chart(runs, path):
    """The latest run of every machine, fastest result per backend."""
    runs = latest(runs)
    workloads = list(
        dict.fromkeys(w for run in runs for w in _workloads(run["results"]))
    )
    columns = 3
    fig, axes = plt.subplots(
        -(-len(workloads) // columns),
        columns,
        figsize=(10, (0.9 + 0.3 * len(runs)) * -(-len(workloads) // columns)),
        squeeze=False,
    )
    for ax, (suite, workload) in zip(axes.flat, workloads):
        times = []
        for backend in COLORS:
            for i, run in enumerate(runs):
                record = _fastest(run, suite, workload, backend)
                if record is not None:
                    times.append(best(record))
                    ax.scatter(best(record), i, color=COLORS[backend], s=28, zorder=3,
                               label=backend if i == 0 else None)  # fmt: skip
        ax.set_yticks(range(len(runs)), [machine_key(r) for r in runs])
        ax.set_ylim(len(runs) - 0.5, -0.5)
        ax.grid(axis="y", color="#e5e5e5", zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(min(times) / 1.6, max(times) * 1.6)
        _plain_log(ax.xaxis)
        ax.set_title(workload, fontsize=9)
    for ax in list(axes.flat)[len(workloads) :]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("best time (ms, log scale)")
    fig.legend(
        handles=[Patch(color=c, label=b) for b, c in COLORS.items()],
        loc="upper right",
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, path)


def history_chart(runs, path):
    """Best time of each workload over time, per machine and backend."""
    workloads = list(
        dict.fromkeys(w for run in runs for w in _workloads(run["results"]))
    )
    fig, axes = plt.subplots(
        2, -(-len(workloads) // 2), figsize=(10, 5.5), squeeze=False
    )
    for ax, (suite, workload) in zip(axes.flat, workloads):
        series = {}
        for run in runs:
            date = datetime.datetime.fromisoformat(run["date"])
            for backend in COLORS:
                record = _fastest(run, suite, workload, backend)
                if record is not None:
                    key = (machine_key(run), backend, record["device"])
                    series.setdefault(key, []).append((date, best(record)))
        machines = list(dict.fromkeys(key[0] for key in series))
        for (machine, backend, device), points in series.items():
            points.sort()
            k = machines.index(machine)
            ax.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                marker=MARKERS[k % len(MARKERS)],
                linestyle=LINESTYLES[k % len(LINESTYLES)],
                color=COLORS[backend],
                label=f"{machine}: {backend}"
                + (f" ({_device(device)})" if device else ""),
            )
        ax.set_yscale("log")
        _plain_log(ax.yaxis)
        ax.set_title(workload, fontsize=9)
        dates = sorted({p[0] for points in series.values() for p in points})
        if dates[0] == dates[-1]:
            # A single date: show the day around it.
            ax.set_xlim(dates[0] - datetime.timedelta(days=1),
                        dates[0] + datetime.timedelta(days=1))  # fmt: skip
        locator = mdates.AutoDateLocator(minticks=1, maxticks=5)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    for ax in list(axes.flat)[len(workloads) :]:
        ax.axis("off")
    axes[0, 0].set_ylabel("best time (ms, log scale)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _save(fig, path)


def runs_table(runs):
    """Markdown table of the runs."""
    lines = [
        "| Who / machine | Date | Commit | CPU | GPUs (Vulkan) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for run in reversed(runs):
        commit = (run["git"].get("commit") or "")[:7] + (
            " (modified)" if run["git"].get("dirty") else ""
        )
        gpus = ", ".join(
            f"{d['name']} ({d['driver_info']})"
            for d in run["hardware"]["vulkan"]
            if d["kind"] != "cpu"
        )
        lines.append(
            f"| {machine_key(run)} | {run['date'][:10]} | {commit} "
            f"| {run['hardware']['cpu']['model']} | {gpus} |"
        )
    return "\n".join(lines) + "\n"


def report(output, directory=RESULTS):
    """Write the table and all charts.

    Parameters
    ----------
    output : pathlib.Path
        Directory for the files.
    directory : pathlib.Path
        The ``results`` directory.

    Returns
    -------
    list of pathlib.Path
        The files written.
    """
    output.mkdir(parents=True, exist_ok=True)
    runs = load(directory)
    written = [output / "runs.md"]
    written[0].write_text(runs_table(runs) if runs else "No results collected yet.\n")
    if not runs:
        (output / "charts.md").write_text("")
        return written + [output / "charts.md"]
    sections = [
        "### Machines compared\n",
        "The latest run of every machine, fastest result of each backend.\n",
        "![Machines compared](machines.svg)\n",
        "### Over time\n",
        "Best time per workload, run by run.\n",
        "![History](history.svg)\n",
    ]
    for run in latest(runs):
        name = f"backends-{run['user']}-{run['machine']}.svg"
        backends_chart(run, output / name)
        written.append(output / name)
        sections += [
            f"### {machine_key(run)}\n",
            f"Latest run, {run['date'][:10]}: every backend, device and variant.\n",
            f"![{machine_key(run)}]({name})\n",
        ]
    machines_chart(runs, output / "machines.svg")
    history_chart(runs, output / "history.svg")
    (output / "charts.md").write_text("\n".join(sections))
    written += [output / "machines.svg", output / "history.svg", output / "charts.md"]
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--results", type=pathlib.Path, default=RESULTS)
    opts = parser.parse_args()
    for path in report(opts.output, opts.results):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
