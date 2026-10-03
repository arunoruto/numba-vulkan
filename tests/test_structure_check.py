"""The structural check that every module passes before it reaches a driver."""

import shutil
import subprocess

import numpy as np
import pytest
from numba import types

import numba_vulkan as nv
from numba_vulkan import codegen
from numba_vulkan.errors import SpirvCodegenError

LOOP_MERGE, SELECTION_MERGE, BRANCH_CONDITIONAL = 246, 247, 250


@nv.jit
def branches(a, out):
    i = nv.global_id(0)
    if i < a.shape[0]:
        if a[i] > 0:
            out[i] = a[i]
        else:
            out[i] = -a[i]


@nv.jit
def loops(a, out):
    i = nv.global_id(0)
    if i < a.shape[0]:
        s = np.float32(0)
        for k in range(i):
            s += a[k]
        out[i] = s


ARRAYS = (types.float32[::1], types.float32[::1])


def _mutated(kernel, change):
    """The kernel's module with one instruction changed, and spirv-val's view."""
    header, instructions = codegen._instructions(kernel.compile(ARRAYS, 1).spirv)
    spirv = codegen._assemble(header, change(list(instructions)))
    rejected = None
    if shutil.which("spirv-val"):
        proc = subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.2", "-"],
            input=spirv,
            capture_output=True,
            check=False,
        )
        rejected = proc.returncode != 0
    return spirv, rejected


def test_generated_modules_pass():
    for kernel in (branches, loops):
        codegen.check_structure(kernel.compile(ARRAYS, 1).spirv)


def test_branch_without_merge_is_caught():
    def drop_merge(instructions):
        k = next(
            k
            for k, inst in enumerate(instructions)
            if inst[0] & 0xFFFF == SELECTION_MERGE
            and instructions[k + 1][0] & 0xFFFF == BRANCH_CONDITIONAL
        )
        return instructions[:k] + instructions[k + 1 :]

    spirv, rejected = _mutated(branches, drop_merge)
    assert rejected in (None, True)
    with pytest.raises(SpirvCodegenError, match="without a merge block"):
        codegen.check_structure(spirv)


def test_shared_merge_block_is_caught():
    def share(instructions):
        found = [
            k for k, i in enumerate(instructions) if i[0] & 0xFFFF == SELECTION_MERGE
        ]
        first, second = found[0], found[1]
        inst = list(instructions[second])
        inst[1] = instructions[first][1]
        instructions[second] = tuple(inst)
        return instructions

    spirv, rejected = _mutated(branches, share)
    assert rejected in (None, True)
    with pytest.raises(SpirvCodegenError, match="merge block of two"):
        codegen.check_structure(spirv)


def test_back_edge_to_a_block_that_is_no_loop_header_is_caught():
    def unmark_loop(instructions):
        k = next(k for k, i in enumerate(instructions) if i[0] & 0xFFFF == LOOP_MERGE)
        # A selection merge in its place keeps the branch structured.
        inst = instructions[k]
        instructions[k] = ((3 << 16) | SELECTION_MERGE, inst[1], 0)
        return instructions

    spirv, rejected = _mutated(loops, unmark_loop)
    assert rejected in (None, True)
    with pytest.raises(SpirvCodegenError, match="no loop header"):
        codegen.check_structure(spirv)
