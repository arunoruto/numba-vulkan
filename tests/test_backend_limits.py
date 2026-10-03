"""What LLVM's SPIR-V backend still gets wrong without our rewrites.

Each test hands the backend a construct that one of the rewrites in
`numba_vulkan.legalize`, `numba_vulkan.buffers` or `numba_vulkan.codegen`
removes, and checks that the backend still fails on it. When a test fails
after an LLVM upgrade, the backend has learnt the construct, and the rewrite
named in the test may no longer be needed (KI-21).
"""

import shutil
import subprocess

import numpy as np
import pytest
from numba import types

import numba_vulkan as nv
from numba_vulkan import buffers, codegen, legalize
from numba_vulkan.errors import SpirvCodegenError


def _shader(body, declarations="", ty="float", mangled="f32"):
    """A compute shader that loads a, b and c, computes r, and stores it."""
    text = f"""
target triple = "{codegen.TRIPLE}"
define void @main() #0 {{
entry:
  %a = call {ty} @numba_vulkan.load.{mangled}(i32 1, i32 0)
  %b = call {ty} @numba_vulkan.load.{mangled}(i32 1, i32 1)
  %c = call {ty} @numba_vulkan.load.{mangled}(i32 1, i32 2)
{body}
  call void @numba_vulkan.store.{mangled}(i32 1, i32 3, {ty} %r)
  ret void
}}
declare {ty} @numba_vulkan.load.{mangled}(i32, i32)
declare void @numba_vulkan.store.{mangled}(i32, i32, {ty})
{declarations}
attributes #0 = {{ "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }}
"""
    return buffers.expand_buffer_access(text)[0]


def _validate(spirv):
    """spirv-val's first complaint, or ``None`` if it accepts the module."""
    if shutil.which("spirv-val") is None:
        pytest.skip("spirv-val not available")
    proc = subprocess.run(
        ["spirv-val", "--target-env", "vulkan1.2", "-"],
        input=spirv,
        capture_output=True,
        check=False,
    )
    return proc.stderr.decode() if proc.returncode else None


INT = {"ty": "i32", "mangled": "i32"}

# Constructs the backend cannot translate, and the rewrite that removes them.
FAILING = {
    "llvm.ctlz (legalize.expand_ctlz)": _shader(
        "  %r = call i32 @llvm.ctlz.i32(i32 %a, i1 false)",
        "declare i32 @llvm.ctlz.i32(i32, i1)",
        **INT,
    ),
    "llvm.usub.sat (legalize.expand_saturating)": _shader(
        "  %r = call i32 @llvm.usub.sat.i32(i32 %a, i32 %b)",
        "declare i32 @llvm.usub.sat.i32(i32, i32)",
        **INT,
    ),
    "llvm.minimumnum (legalize.rename_minimumnum)": _shader(
        "  %r = call float @llvm.minimumnum.f32(float %a, float %b)",
        "declare float @llvm.minimumnum.f32(float, float)",
    ),
    "llvm.copysign (legalize.expand_copysign)": _shader(
        "  %r = call float @llvm.copysign.f32(float %a, float %b)",
        "declare float @llvm.copysign.f32(float, float)",
    ),
}

# Constructs the backend translates into modules that Vulkan does not allow.
INVALID = {
    "llvm.fshl (legalize.expand_funnel_shift)": _shader(
        "  %r = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)",
        "declare i32 @llvm.fshl.i32(i32, i32, i32)",
        **INT,
    ),
    "fcmp uno (legalize.expand_fcmp_ordering)": _shader(
        "  %u = fcmp uno float %a, %b\n  %r = select i1 %u, float 1.0, float 0.0"
    ),
    "llvm.spv.cmpxchg (codegen.fix_compare_exchange)": _shader(
        "  %r = call i32 @numba_vulkan.cas.i32(i32 1, i32 4, i32 %a, i32 %b)",
        "declare i32 @numba_vulkan.cas.i32(i32, i32, i32, i32)",
        **INT,
    ),
    "barrier (codegen.fix_barrier_semantics)": _shader(
        "  call void @numba_vulkan.barrier()\n  %r = fadd float %a, %b",
        "declare void @numba_vulkan.barrier()",
    ),
}


def test_the_harness_produces_valid_modules():
    assert _validate(codegen.emitter.emit(_shader("  %r = fadd float %a, %b"))) is None


@pytest.mark.parametrize("name", sorted(FAILING))
def test_backend_still_fails(name):
    with pytest.raises(SpirvCodegenError, match="backend failed"):
        codegen.emitter.emit(FAILING[name])


@pytest.mark.parametrize("name", sorted(INVALID))
def test_backend_output_is_still_invalid(name):
    assert _validate(codegen.emitter.emit(INVALID[name])) is not None, (
        f"LLVM now translates {name} validly"
    )


def test_fmuladd_still_becomes_a_fused_operation():
    """`legalize.expand_fmuladd` keeps results the same on every device."""
    spirv = codegen.emitter.emit(
        _shader(
            "  %r = call float @llvm.fmuladd.f32(float %a, float %b, float %c)",
            "declare float @llvm.fmuladd.f32(float, float, float)",
        )
    )
    _, instructions = codegen._instructions(spirv)
    fma = 50  # GLSL.std.450 Fma
    assert any(i[0] & 0xFFFF == 12 and i[4] == fma for i in instructions)


def test_faceforward_combine_still_crashes(monkeypatch, tmp_path):
    """`legalize.avoid_faceforward`; the shape needs a whole kernel."""
    monkeypatch.setenv("NUMBA_VULKAN_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(legalize, "avoid_faceforward", lambda text: text)

    @nv.jit
    def sign_flip(a, out):
        i = nv.global_id(0)
        if i < a.shape[0]:
            x = a[i] * np.float32(0.75)
            out[i] = -x if x < 0 else x

    with pytest.raises(SpirvCodegenError, match="backend failed"):
        sign_flip.compile((types.float32[::1], types.float32[::1]), 1)
