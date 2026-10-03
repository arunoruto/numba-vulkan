"""The workgroup size as a specialization constant of the shader."""

import numpy as np
import pytest
from numba import types

import numba_vulkan as nv
from numba_vulkan import codegen

f32 = np.float32


@nv.jit
def report_size(out):
    i = nv.global_id(0)
    if i < out.shape[0]:
        out[i] = nv.local_size(0) * 10000 + nv.local_size(1) * 100 + nv.local_size(2)


@nv.jit
def block_sums(values, sums):
    cache = nv.shared.array(256, np.float32)
    lid = nv.local_id(0)
    i = nv.global_id(0)
    cache[lid] = values[i] if i < values.shape[0] else f32(0)
    nv.barrier()
    step = nv.local_size(0) // 2
    while step > 0:
        if lid < step:
            cache[lid] += cache[lid + step]
        nv.barrier()
        step //= 2
    if lid == 0:
        sums[nv.group_id(0)] = cache[0]


@nv.jit
def nothing(out):
    out[0] = 1


@pytest.mark.parametrize(
    "groups, local",
    [((2,), (32,)), ((4,), (64,)), ((1,), (256,)), ((1, 2), (4, 8)), ((1,), (2, 3, 4))],
)
def test_kernels_read_the_launched_size(device, groups, local):
    out = np.zeros(min(4, groups[0] * local[0]), np.int32)  # x covers it
    report_size[groups, local, device](out)
    x, y, z = local + (1,) * (3 - len(local))
    np.testing.assert_array_equal(out, x * 10000 + y * 100 + z)


@pytest.mark.parametrize("local", [32, 64, 128, 256])
def test_reduction_over_the_workgroup(device, local):
    values = np.arange(1024, dtype=f32)
    sums = np.zeros(1024 // local, f32)
    block_sums[1024 // local, local, device](values, sums)
    np.testing.assert_array_equal(sums, values.reshape(-1, local).sum(axis=1))


def test_sizes_share_one_module():
    args = (types.int32[::1],)
    default = report_size.compile(args, 1)
    for local in [(32,), (128,), (16, 1, 1)]:
        variant = report_size.compile(args, 1, local_size=local)
        assert variant.spirv is default.spirv
        assert variant.local_size == local + (1,) * (3 - len(local))
    assert report_size.compile(args, 1, local_size=(64,)) is default


def test_module_declares_the_size(validate):
    # This kernel declares no 3-component vector of its own.
    kernel = nothing.compile((types.int32[::1],), 1)
    validate(kernel)
    _, instructions = codegen._instructions(kernel.spirv)
    spec = [i for i in instructions if i[0] & 0xFFFF == 50]
    assert [i[3] for i in spec] == [64, 1, 1]
    builtin = [
        i
        for i in instructions
        if i[0] & 0xFFFF == 71 and i[2:] == (11, 25)  # BuiltIn WorkgroupSize
    ]
    assert len(builtin) == 1
