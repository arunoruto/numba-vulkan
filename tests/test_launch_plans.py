"""Launch plans: repeated launches skip typing, but must not mix signatures."""

import numpy as np
import pytest

import numba_vulkan as nv

f32 = np.float32
# Python ints and floats make the kernel compute with float64.
pytestmark = pytest.mark.filterwarnings("ignore:kernel 'scale' computes with float64")


@nv.jit
def scale(x, out, a):
    i = nv.global_id(0)
    if i < x.shape[0]:
        out[i] = x[i] * a


def test_repeated_launches_reuse_the_plan(device):
    x = nv.to_device(np.arange(100, dtype=f32), device)
    out = nv.device_array_like(x)
    for a in range(3):
        scale.forall(100, device=device)(x, out, f32(a))
        np.testing.assert_array_equal(out.copy_to_host(), np.arange(100) * a)
    assert len(scale._plans) >= 1


def test_layouts_and_scalar_types_get_their_own_plans(device):
    host = np.arange(200, dtype=f32)
    x = nv.to_device(host, device)
    out = nv.device_array(100, f32, device)
    run = scale.forall(100, device=device)
    run(x[:100], out, f32(2))  # plain
    np.testing.assert_array_equal(out.copy_to_host(), host[:100] * 2)
    run(x[::2], out, f32(2))  # with gaps: same types, another layout
    np.testing.assert_array_equal(out.copy_to_host(), host[::2] * 2)
    run(x[1::2], out, 3)  # an int scalar: another kernel
    np.testing.assert_array_equal(out.copy_to_host(), host[1::2] * 3)
    run(x[1::2], out, 0.5)  # a float
    np.testing.assert_array_equal(out.copy_to_host(), host[1::2] * 0.5)
    run(x[:100], out, True)
    np.testing.assert_array_equal(out.copy_to_host(), host[:100])


def test_scalars_keep_their_checks(device):
    """Values that do not fit a narrowed int are still caught on the plan."""
    x = nv.to_device(np.ones(10, np.int64), device)
    out = nv.device_array_like(x)
    run = scale.forall(10, device=device)
    run(x, out, 5)
    run(x, out, 7)  # planned
    np.testing.assert_array_equal(out.copy_to_host(), np.full(10, 7))
    if out._stored == np.int32:
        with pytest.raises((ValueError, OverflowError)):
            run(x, out, 1 << 40)
        with pytest.raises(OverflowError):
            run(x, out, 1 << 63)  # uint64: the full path, not the plan


def test_local_size_as_a_list(device):
    x = nv.to_device(np.arange(64, dtype=f32), device)
    out = nv.device_array_like(x)
    for _ in range(2):
        scale.forall(64, device=device, local_size=[32])(x, out, f32(2))
    np.testing.assert_array_equal(out.copy_to_host(), np.arange(64) * 2)
