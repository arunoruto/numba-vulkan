"""``@nv.jit(cache=True)``: kernels reused by later processes."""

import json
import os
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

import numba_vulkan as nv

HELPER = """
import numba_vulkan as nv

@nv.jit
def twice(v):
    return v * {factor}
"""

KERNELS = """
import numpy as np
import numba_vulkan as nv
from helper import twice

TABLE = np.array([10, 20, 30], dtype=np.float32)

@nv.jit(cache=True)
def apply(x, out):
    i = nv.global_id(0)
    if i < x.shape[0]:
        out[i] = twice(x[i]) + TABLE[i % 3]

@nv.jit(cache=True)
def checked(x):
    i = nv.global_id(0)
    if i < x.shape[0] and x[i] < 0:
        raise ValueError("negative input")

@nv.jit(cache=True)
def shout(x):
    if nv.global_id(0) == 0:
        print("first is", x[0])
"""

RUN = """
import json, sys
import numpy as np
from numba_vulkan import dispatcher
keys = []
original_name = dispatcher.VulkanDispatcher._cache_name
def naming(self, key):
    name = original_name(self, key)
    keys.append([self.py_func.__name__, name, repr(key)])
    return name
dispatcher.VulkanDispatcher._cache_name = naming
compiled = []
original = dispatcher.VulkanDispatcher.compile_device
def counting(self, *args, **kwargs):
    compiled.append(self.py_func.__name__)
    return original(self, *args, **kwargs)
dispatcher.VulkanDispatcher.compile_device = counting
import kernels
x = np.arange(6, dtype=np.float32)
out = np.zeros_like(x)
kernels.apply.forall(6)(x, out)
error = None
try:
    kernels.checked.forall(2)(np.array([1, -1], dtype=np.float32))
except ValueError as exc:
    error = [str(exc), exc.__notes__]
kernels.shout.forall(1)(x)
print(json.dumps({"out": out.tolist(), "compiled": compiled, "error": error,
                  "keys": keys}))
"""


def _run(directory, cache):
    """Run the script in a fresh process; its report and printed lines."""
    env = dict(os.environ, NUMBA_VULKAN_CACHE="1", NUMBA_VULKAN_CACHE_DIR=str(cache))
    # Python would reuse stale bytecode of a file rewritten within a second.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(directory), env.get("PYTHONPATH", "")])
    proc = subprocess.run(
        [sys.executable, "-c", RUN],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    return json.loads(lines[-1]), lines[:-1]


@pytest.fixture
def project(tmp_path):
    (tmp_path / "helper.py").write_text(HELPER.format(factor=2))
    (tmp_path / "kernels.py").write_text(KERNELS)
    return tmp_path


def _want(factor):
    x = np.arange(6, dtype=np.float32)
    return (x * factor + np.array([10, 20, 30] * 2, dtype=np.float32)).tolist()


def test_later_processes_skip_compiling(project, tmp_path_factory):
    cache = tmp_path_factory.mktemp("cache")
    first, printed = _run(project, cache)
    assert first["out"] == _want(2)
    assert "apply" in first["compiled"]
    assert first["error"][0] == "negative input"
    assert printed == ["first is 0.0"]

    second, printed = _run(project, cache)
    stored = sorted(p.name for p in cache.glob("functions/*/*"))
    # nothing went through Numba's pipeline
    same = [(a[0], a[1] == b[1]) for a, b in zip(first["keys"], second["keys"])]
    assert second["compiled"] == [], (len(stored), same)
    assert second["out"] == _want(2)
    # Exceptions and print formats come back with the kernel.
    assert second["error"] == first["error"]
    assert printed == ["first is 0.0"]


def test_a_changed_device_function_is_noticed(project, tmp_path_factory):
    cache = tmp_path_factory.mktemp("cache")
    _run(project, cache)
    time.sleep(0.01)  # a different modification time on any file system
    (project / "helper.py").write_text(HELPER.format(factor=3))
    report, _ = _run(project, cache)
    assert "apply" in report["compiled"]
    assert report["out"] == _want(3)
    assert "checked" not in report["compiled"]  # does not call the helper


def test_functions_without_a_source_file_are_compiled_with_a_warning(device):
    scope = {"nv": nv}
    exec(  # noqa: S102
        "def kernel(x):\n    x[nv.global_id(0)] = 1\n",
        scope,
    )
    kernel = nv.jit(cache=True)(scope["kernel"])
    os.environ["NUMBA_VULKAN_CACHE"] = "1"
    try:
        with pytest.warns(Warning, match="cache=True has no effect"):
            kernel.forall(1, device=device)(np.zeros(1, np.float32))
    finally:
        os.environ["NUMBA_VULKAN_CACHE"] = "0"


def test_exception_codes_are_the_same_in_every_process():
    from numba_vulkan.target import exception_table

    code = exception_table._add_exception(ValueError, ("x",), ("f", "file.py", 3))
    script = (
        "from numba_vulkan.target import exception_table as t\n"
        "print(t._add_exception(ValueError, ('x',), ('f', 'file.py', 3)))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert int(out) == code
