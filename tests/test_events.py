"""Timing work on the device with events."""

import time

import numpy as np
import pytest

import numba_vulkan as nv

f32 = np.float32


@nv.jit
def spin(x, rounds):
    i = nv.global_id(0)
    if i < x.shape[0]:
        v = x[i]
        for _ in range(rounds):
            v = v * f32(0.999) + f32(0.001)
        x[i] = v


def _time(device, rounds, x):
    start, end = nv.event(device), nv.event(device)
    start.record()
    wall = time.perf_counter()
    spin.forall(x.shape[0], device=device)(x, rounds)
    end.record()
    end.synchronize()
    return start.elapsed_time(end), time.perf_counter() - wall


def test_events_measure_the_work_on_the_device(device):
    x = nv.to_device(np.ones(1 << 14, dtype=f32), device)
    _time(device, 1, x)  # compiles
    light, _ = _time(device, 1, x)
    heavy, waited = _time(device, 200000, x)
    assert 0 <= light < heavy
    # The device time cannot exceed the host's time for the launch by much.
    assert heavy <= (waited * 1e3) * 1.5 + 1
    assert nv.event_elapsed_time  # the numba.cuda spelling exists


def test_events_around_launches_that_wait(device):
    start, end = nv.event(device), nv.event(device)
    start.record()
    x = np.ones(4096, dtype=f32)
    spin.forall(4096, device=device)(x, 100)  # NumPy array: waits
    end.record()
    assert start.elapsed_time(end) >= 0


def test_unrecorded_events_cannot_be_waited_for(device):
    with pytest.raises(RuntimeError, match="not recorded"):
        nv.event(device).synchronize()
