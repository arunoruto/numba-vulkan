"""Choosing the fastest of several ways to launch a computation.

Which kernel variant or workgroup size is fastest depends on the device
and the size of the problem. `autotune` turns a generator of candidate
launches into a function that times every candidate on the first call for
a key, and runs only the fastest one afterwards. The choices are kept per
device and driver in ``autotune.json`` in the cache directory
(``~/.cache/numba-vulkan``), unless ``NUMBA_VULKAN_CACHE=0``, so later
processes skip the timing.
"""

import functools
import json
import os
import statistics
import tempfile
import time

from numba_vulkan import kernelcache, libclc, runtime

_FILE = "autotune.json"


def autotune(function=None, *, key=None, repeat=3):
    """Run the fastest of several candidate launches.

    Parameters
    ----------
    function : generator function
        Called with the arguments of a call, it yields ``(name, launch)``
        pairs: a name for each candidate, and a function of no arguments
        that runs it. Every candidate must leave the same results.
    key : callable, optional
        Called with the arguments, returns what the choice depends on, such
        as the shapes of the arrays (hashable, and the same in every
        process). By default the shapes and types of the array arguments.
    repeat : int
        Timed runs of each candidate, after one untimed run; the median
        counts.

    Returns
    -------
    callable
        The tuned function. Its ``choices`` attribute maps the keys seen so
        far to the names chosen, and ``timings`` holds the seconds each
        candidate took when it last tuned.

    Notes
    -----
    While tuning, every candidate runs ``repeat + 1`` times, so arrays the
    candidates write to are overwritten several times: candidates that
    update an array in place (``a += b``) would accumulate. Pass copies, or
    tune on a representative call first.

    Examples
    --------
    >>> @nv.autotune(key=lambda a, b, c: (a.shape, b.shape))
    ... def product(a, b, c):
    ...     m, n = c.shape
    ...     yield "tiled", lambda: matmul[(n // 16, m // 16), (16, 16)](a, b, c)
    ...     yield "blocked", lambda: blocked[(n // 64, m // 64), (16, 16)](a, b, c)
    >>> product(a, b, c)    # times both; later calls run the faster one
    """
    if function is None:
        return lambda f: autotune(f, key=key, repeat=repeat)
    return _Tuned(function, key or _default_key, repeat)


def _default_key(*args):
    """Shapes and element types of the array arguments, types of the others."""
    parts = []
    for arg in args:
        if hasattr(arg, "shape") and hasattr(arg, "dtype"):
            parts.append((tuple(arg.shape), str(arg.dtype)))
        else:
            parts.append(type(arg).__name__)
    return tuple(parts)


class _Tuned:
    """A function tuned by `autotune`."""

    def __init__(self, function, key, repeat):
        self.function = function
        self.key = key
        self.repeat = repeat
        self.choices = {}
        functools.update_wrapper(self, function)

    def __call__(self, *args):
        """Run the chosen candidate, choosing it first if needed."""
        candidates = dict(self.function(*args))
        device = _device_of(args)
        name = self._name(candidates, device, self.key(*args))
        chosen = self.choices.get(name)
        if chosen not in candidates:
            chosen = _stored().get(name)
        if chosen not in candidates:
            chosen = self._choose(candidates, device)
            _store(name, chosen)
        self.choices[name] = chosen
        return candidates[chosen]()

    def _name(self, candidates, device, key):
        """What identifies a choice: function, candidates, driver, key."""
        from numba_vulkan import probes

        return " | ".join(
            [
                f"{self.function.__module__}.{self.function.__qualname__}",
                ",".join(candidates),
                probes._driver_key(device),
                repr(key),
            ]
        )

    def _choose(self, candidates, device):
        """Time every candidate on a device; the name of the fastest."""
        times = {}
        for name, launch in candidates.items():
            launch()  # compiles, and warms the caches
            device.synchronize()
            samples = []
            for _ in range(self.repeat):
                start = time.perf_counter()
                launch()
                device.synchronize()
                samples.append(time.perf_counter() - start)
            times[name] = statistics.median(samples)
        self.timings = times
        return min(times, key=times.get)


def _device_of(args):
    """The device of the first device array argument, or the selected one."""
    for arg in args:
        if isinstance(arg, runtime.DeviceArray):
            return arg.device
    return runtime.get_device()


def _path():
    """The file that keeps the choices."""
    return os.path.join(libclc.cache_directory(), _FILE)


def _stored():
    """The choices kept on disk; empty without them or without a cache."""
    if not kernelcache.enabled():
        return {}
    try:
        with open(_path()) as fh:
            stored = json.load(fh)
    except (OSError, ValueError):
        return {}
    return stored if isinstance(stored, dict) else {}


def _store(name, chosen):
    """Keep a choice on disk; failures are ignored."""
    if not kernelcache.enabled():
        return
    directory = libclc.cache_directory()
    try:
        os.makedirs(directory, exist_ok=True)
        stored = _stored()
        stored[name] = chosen
        # Written under another name first: another process may be reading.
        with tempfile.NamedTemporaryFile("w", dir=directory, delete=False) as fh:
            json.dump(stored, fh, indent=1, sort_keys=True)
        os.replace(fh.name, _path())
    except OSError:
        pass
