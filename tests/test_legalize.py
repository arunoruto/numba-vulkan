"""The IR rewrites, checked by running original and rewritten code on the CPU."""

import ctypes
import math
import struct

import llvmlite.binding as llvm
import numpy as np
import pytest

from numba_vulkan import legalize

llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

_CTYPES = {
    "i1": ctypes.c_bool,
    "i32": ctypes.c_int32,
    "i64": ctypes.c_int64,
    "float": ctypes.c_float,
    "double": ctypes.c_double,
}
_engines = []


def compile_ir(text, restype, argtypes):
    """JIT-compile ``@f`` from LLVM IR for the host and return it as a callable."""
    module = llvm.parse_assembly(text)
    module.verify()
    machine = llvm.Target.from_default_triple().create_target_machine()
    engine = llvm.create_mcjit_compiler(module, machine)
    engine.finalize_object()
    _engines.append(engine)  # keeps the code alive
    proto = ctypes.CFUNCTYPE(_CTYPES[restype], *[_CTYPES[t] for t in argtypes])
    return proto(engine.get_function_address("f"))


def both(text, restype, argtypes, rewrite):
    """The function as written and after ``rewrite``."""
    rewritten = rewrite(text)
    assert rewritten != text, "the rewrite did not apply"
    return compile_ir(text, restype, argtypes), compile_ir(rewritten, restype, argtypes)


INTS32 = [
    0,
    1,
    2,
    3,
    255,
    256,
    65535,
    65536,
    0x7FFFFFFF,
    -1,
    -2,
    -(2**31),
    123456789,
    -987654321,
]
FLOATS = [0.0, -0.0, 1.5, -2.5, 1e-30, -1e30, math.inf, -math.inf, math.nan]


@pytest.mark.parametrize("width", [32, 64])
def test_ctlz(width):
    ty = f"i{width}"
    text = f"""
declare {ty} @llvm.ctlz.{ty}({ty}, i1)
define {ty} @f({ty} %x) {{
  %r = call range({ty} 0, {width + 1}) {ty} @llvm.ctlz.{ty}({ty} %x, i1 false)
  ret {ty} %r
}}
"""
    original, rewritten = both(text, ty, [ty], legalize.expand_ctlz)
    values = INTS32 + ([2**40, -(2**62), 2**63 - 1] if width == 64 else [])
    for x in values:
        assert rewritten(x) == original(x), x


@pytest.mark.parametrize("direction", ["l", "r"])
def test_funnel_shift(direction):
    text = f"""
declare i32 @llvm.fsh{direction}.i32(i32, i32, i32)
define i32 @f(i32 %a, i32 %b, i32 %c) {{
  %r = call i32 @llvm.fsh{direction}.i32(i32 %a, i32 %b, i32 %c)
  ret i32 %r
}}
"""
    original, rewritten = both(text, "i32", ["i32"] * 3, legalize.expand_funnel_shift)
    for a in INTS32[:8]:
        for b in INTS32[4:]:
            for amount in (0, 1, 7, 16, 31, 32, 33):
                assert rewritten(a, b, amount) == original(a, b, amount), (a, b, amount)


def test_srem():
    text = """
define i64 @f(i64 %a, i64 %b) {
  %r = srem i64 %a, %b
  ret i64 %r
}
"""
    original, rewritten = both(text, "i64", ["i64", "i64"], legalize.expand_srem)
    for a in (7, -7, 0, 2**40 + 3, -(2**40) - 3):
        for b in (2, -2, 5, -5, 2**33):
            assert rewritten(a, b) == original(a, b), (a, b)


@pytest.mark.parametrize("pred", ["uno", "ord"])
def test_nan_comparisons(pred):
    text = f"""
define i1 @f(float %a, float %b) {{
  %r = fcmp {pred} float %a, %b
  ret i1 %r
}}
"""
    original, rewritten = both(
        text, "i1", ["float", "float"], legalize.expand_fcmp_ordering
    )
    for a in FLOATS:
        for b in FLOATS:
            assert rewritten(a, b) == original(a, b), (a, b)


@pytest.mark.parametrize("pred", ["olt", "ult", "ogt", "ugt"])
@pytest.mark.parametrize("zero_first", [False, True])
def test_comparisons_with_zero_keep_their_meaning(pred, zero_first):
    operands = "0.000000e+00, %a" if zero_first else "%a, 0.000000e+00"
    text = f"""
define i1 @f(double %a) {{
  %r = fcmp {pred} double {operands}
  ret i1 %r
}}
"""
    original, rewritten = both(text, "i1", ["double"], legalize.avoid_faceforward)
    for a in FLOATS:
        assert rewritten(a) == original(a), a


def test_copysign():
    text = """
declare double @llvm.copysign.f64(double, double)
define double @f(double %a, double %b) {
  %r = call noundef double @llvm.copysign.f64(double noundef %a, double %b)
  ret double %r
}
"""
    original, rewritten = both(
        text, "double", ["double", "double"], legalize.expand_copysign
    )
    for a in FLOATS[:8]:
        for b in FLOATS[:8]:
            want, got = original(a, b), rewritten(a, b)
            assert struct.pack("d", got) == struct.pack("d", want), (a, b)


def test_fmuladd():
    text = """
declare float @llvm.fmuladd.f32(float, float, float)
define float @f(float %a, float %b, float %c) {
  %r = call noundef float @llvm.fmuladd.f32(float %a, float 2.500000e+00, float %c)
  ret float %r
}
"""
    _, rewritten = both(text, "float", ["float"] * 3, legalize.expand_fmuladd)
    assert rewritten(3.0, 0.0, 1.0) == 8.5


@pytest.mark.parametrize("narrow", [False, True], ids=["64-bit", "32-bit"])
@pytest.mark.parametrize("kind,extend", [("jj", "u"), ("ii", "s")])
def test_mul_hi(kind, extend, narrow):
    text = f"""
declare dso_local spir_func i32 @_Z12__clc_mul_hi{kind}(i32 noundef, i32 noundef)
define i32 @f(i32 %a, i32 %b) {{
  %r = call spir_func i32 @_Z12__clc_mul_hi{kind}(i32 noundef %a, i32 noundef %b) #6
  ret i32 %r
}}
"""
    expanded = legalize.expand_mul_hi(text, narrow=narrow)
    assert ("i64" in expanded) is not narrow
    rewritten = compile_ir(expanded, "i32", ["i32", "i32"])
    for a in INTS32:
        for b in INTS32:
            if extend == "u":
                want = ((a & 0xFFFFFFFF) * (b & 0xFFFFFFFF)) >> 32
                want = want - 2**32 if want >= 2**31 else want
            else:
                want = (a * b) >> 32
            assert rewritten(a, b) == want, (a, b)


@pytest.mark.parametrize("width", [8, 32, 64])
def test_integer_loads_from_a_byte_table(width):
    data = bytes(range(1, 41))
    literal = "".join(f"\\{b:02X}" for b in data)
    text = f"""
@TABLE = internal unnamed_addr constant [40 x i8] c"{literal}"
define i{64} @f(i64 %i) {{
  %p = getelementptr inbounds i8, ptr @TABLE, i64 %i
  %q = getelementptr i8, ptr %p, i64 8
  %v = load i{width}, ptr %q, align 1
  %r = zext i{width} %v to i64
  ret i64 %r
}}
""".replace("zext i64 %v to i64", "add i64 %v, 0")
    original, rewritten = both(text, "i64", ["i64"], legalize.expand_byte_table_loads)
    assert "getelementptr inbounds [40 x i8]" in legalize.expand_byte_table_loads(text)
    for offset in (0, 1, 5, 16, 23):
        want = int.from_bytes(
            data[offset + 8 : offset + 8 + width // 8], "little", signed=False
        )
        assert original(offset) & (2**64 - 1) == want
        assert rewritten(offset) & (2**64 - 1) == want, offset


def test_minimumnum_is_renamed():
    text = "declare float @llvm.minimumnum.f32(float, float)\n"
    assert (
        legalize.rename_minimumnum(text)
        == "declare float @llvm.minnum.f32(float, float)\n"
    )


def test_legalize_leaves_plain_code_alone():
    text = """
define float @f(float %a, float %b) {
  %r = fadd float %a, %b
  ret float %r
}
"""
    assert legalize.legalize(text) == text
    assert np.isclose(compile_ir(text, "float", ["float", "float"])(1.5, 2.0), 3.5)
