"""Running kernels on devices without 64-bit floats or integers.

Numba types Python literals as ``float64`` and ``int64`` and does all index
arithmetic in ``int64``, so almost every kernel uses 64-bit types, which
are optional features of a Vulkan device. For a device that lacks them, a
kernel is compiled in a narrow `Mode`:

- with ``floats``, math functions called with ``float64`` values are
  evaluated by their ``float32`` versions (see `numba_vulkan.libclc.call`),
  and all remaining ``double`` values in the optimised LLVM IR become
  ``float``;
- with ``ints``, all ``i64`` values in the LLVM IR become ``i32``.

The values a kernel computes with are then 32 bits wide whatever Numba
typed them as, and buffers of ``float64`` or ``int64`` elements hold 32-bit
elements, which the host converts to and from (see
`numba_vulkan.dispatcher`). Code that depends on the width, such as bit
manipulation of ``float64`` values or shifts by 32 and more, is detected
and rejected rather than miscompiled.
"""

import contextlib
import functools
import os
import re
import struct
from typing import NamedTuple

import numpy as np
from llvmlite import ir
from numba.core import cgutils

from numba_vulkan.errors import VulkanUnsupportedError

# Kernels compute with 32-bit integers unless NUMBA_VULKAN_INT64=1: 64-bit
# integer arithmetic is emulated on GPUs and slows index-heavy kernels down.
NARROW_INTS = os.environ.get("NUMBA_VULKAN_INT64", "0") == "0"
# Whether to warn about float64; NUMBA_VULKAN_WARNINGS=0 silences it.
WARNINGS = os.environ.get("NUMBA_VULKAN_WARNINGS", "1") != "0"


def convert(values, dtype, check=True):
    """Convert host data to the element type a buffer holds.

    Parameters
    ----------
    values : numpy.ndarray
        The data.
    dtype : numpy.dtype
        The type to convert to, from `stored_dtype`.
    check : bool
        Whether to check that integers fit. Arrays that a kernel only
        writes to may hold anything, as from ``np.empty``.

    Returns
    -------
    numpy.ndarray
        A C-contiguous array of `dtype`; `values` itself if it is one.

    Raises
    ------
    OverflowError
        If integers do not fit into the narrower type. Wrapping them around
        would silently change the data.
    """
    dtype = np.dtype(dtype)
    if values.dtype.names is not None and dtype.names is not None and check:
        for name, target in zip(values.dtype.names, dtype.names):
            convert(values[name], dtype.fields[target][0], check)
    if (
        values.dtype.kind in "iu"
        and dtype.kind in "iu"
        and dtype.itemsize < values.dtype.itemsize
        and values.size
        and check
    ):
        info = np.iinfo(dtype)
        low, high = values.min(), values.max()
        if low < info.min or high > info.max:
            bad = low if low < info.min else high
            raise OverflowError(
                f"the {values.dtype} value {bad} does not fit into {dtype}: kernels "
                "compute with 32-bit integers by default. Pass narrow=False to "
                "@nv.jit, or set NUMBA_VULKAN_INT64=1, for 64-bit integers"
            )
    return np.asarray(values, dtype=dtype, order="C")


class Mode(NamedTuple):
    """How a kernel is compiled for the features of a device.

    Attributes
    ----------
    floats : bool
        Whether ``float64`` is narrowed to ``float32``.
    ints : bool
        Whether ``int64`` is narrowed to ``int32``.
    float_atomics : bool
        Whether ``float32`` atomic additions may use the device's native
        instruction (``VK_EXT_shader_atomic_float``) instead of a
        compare-and-swap loop.
    soft_fma : bool
        Whether fused multiply-add is computed in software, because the
        device does not fuse it (see `numba_vulkan.probes`).
    soft_rounding : bool
        Whether ``float64`` truncation and rounding to even are computed
        from ``floor``, because the device gets them wrong (llvmpipe).
    """

    floats: bool = False
    ints: bool = False
    float_atomics: bool = False
    soft_fma: bool = False
    soft_rounding: bool = False


# The mode of the kernel being compiled. Compilation holds Numba's global
# compiler lock, so one module-level value is enough.
current = Mode()


@contextlib.contextmanager
def using(mode):
    """Compile in the given mode within the ``with`` block.

    Parameters
    ----------
    mode : Mode
        The mode to make current.
    """
    global current
    previous, current = current, mode
    try:
        yield
    finally:
        current = previous


_PREFIX = "numba_vulkan"


def _conversion(builder, value, name, target):
    """Call the placeholder function that converts between float widths."""
    fnty = ir.FunctionType(target, [value.type])
    fn = cgutils.get_or_insert_function(builder.module, fnty, f"{_PREFIX}.{name}")
    fn.attributes.add("readnone")
    fn.attributes.add("nounwind")
    return builder.call(fn, [value])


def to_single(builder, value):
    """Convert a ``double`` to ``float`` in a kernel compiled without float64.

    In such a kernel every ``double`` ends up as a ``float``, so the
    conversion does nothing in the end. It is emitted as a call to a
    placeholder that `narrow_ir` removes, and not as ``fptrunc``, because
    LLVM would otherwise rewrite code working on the ``float`` into code
    working on the bits of the ``double``, which narrowing cannot follow.

    Parameters
    ----------
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    value : llvmlite.ir.Value
        A float value.

    Returns
    -------
    llvmlite.ir.Value
        The value as ``float``; `value` itself if it is one already.
    """
    if not isinstance(value.type, ir.DoubleType):
        return value
    return _conversion(builder, value, "narrow", ir.FloatType())


def to_double(builder, value):
    """Convert a ``float`` back to ``double``; the inverse of `to_single`."""
    if not isinstance(value.type, ir.FloatType):
        return value
    return _conversion(builder, value, "widen", ir.DoubleType())


_CONVERSION = re.compile(
    rf"^(\s*%[\w.]+) = (?:tail )?call float @{_PREFIX}\.(?:narrow|widen)\(float (\S+)\).*$"
)
_CONVERSION_DECLARATION = re.compile(rf"^declare [^\n]*@{_PREFIX}\.(?:narrow|widen)\(")
_SIGN_BITS = re.compile(r"^\s*(%[\w.]+) = bitcast double (\S+) to i64$")
_DECIMAL = r"-?\d+\.\d+e[+-]\d+"
_FLOAT_LITERAL = re.compile(rf"(?<![\w.%@])({_DECIMAL}|0x[0-9A-Fa-f]{{16}})(?![\w.])")
_FLOAT_BITS = re.compile(r"bitcast (?:double|i64) [^\n]* to (?:i64|double)\b")
_WIDEN_FLOAT = re.compile(
    r"\b(?:fpext float|fptrunc double) (\S+) to (?:double|float)\b"
)
_WIDEN_INT = re.compile(
    r"\b(?:(?:sext|zext)(?: nneg)? i32|trunc(?: nuw| nsw)* i64) (\S+) to (?:i64|i32)\b"
)
_WIDE_SHIFT = re.compile(r"\b(?:shl|lshr|ashr)(?: \w+)* i64 [^,\n]+, (\d+)")
_TOP_BITS = re.compile(
    r"^(\s*)(%[\w.]+) = (lshr|ashr)(?: exact)? i64 (%[\w.]+), (\d+)$"
)
_INT_LITERAL = re.compile(r"(?<![\w.%@#!\"-])(-?\d{10,})(?![\w.])")
# Constants this close to the extremes of int64 are kept at the same distance
# from the extremes of int32; see `_narrow_extreme`.
_NEAR = 1 << 16
_DECLARATION = re.compile(r"^declare [^@\n]*(@[^\s(]+)\(")


def _narrow_extreme(literal):
    """The 32-bit counterpart of an integer constant near an int64 extreme.

    Numba uses the extremes of ``int64`` as sentinels ("no bound" in
    slices), and LLVM derives constants from them: ``INT64_MAX - 1`` as an
    unsigned bound, ``INT64_MIN | 63`` as the mask of the sign bit and the
    low bits that a signed ``x % 64`` tests. In a narrowed kernel the same
    constants relative to the extremes of ``int32`` mean the same for
    32-bit values. Others are returned unchanged.

    Parameters
    ----------
    literal : str
        A decimal integer constant.

    Returns
    -------
    str
    """
    value = int(literal)
    top, bottom = (1 << 63) - 1, -(1 << 63)
    if top - _NEAR <= value <= top:
        return str((1 << 31) - 1 - (top - value))
    if bottom <= value <= bottom + _NEAR:
        return str(-(1 << 31) + (value - bottom))
    return literal


def _narrow_literal(match):
    """Round a ``double`` literal to ``float`` and spell it as LLVM wants.

    LLVM writes a ``float`` constant as the hexadecimal bit pattern of the
    ``double`` with the same value.
    """
    token = match.group(1)
    if token.startswith("0x"):
        (value,) = struct.unpack(">d", bytes.fromhex(token[2:]))
    else:
        value = float(token)
    with np.errstate(over="ignore"):
        narrowed = float(np.float32(value))
    return "0x" + struct.pack(">d", narrowed).hex().upper()


def _dedupe_declarations(lines):
    """Drop declarations that narrowing has made duplicates of others."""
    declared, out = set(), []
    for line in lines:
        match = _DECLARATION.match(line)
        if match:
            if match.group(1) in declared:
                continue
            declared.add(match.group(1))
        out.append(line)
    return out


def _sign_tests(lines):
    """Find reinterpretations of ``double`` values that only test the sign.

    The sign is the top bit at either width, so these survive narrowing.
    LLVM produces them from comparisons with zero.

    Parameters
    ----------
    lines : list of str
        The lines of the module.

    Returns
    -------
    set of str
        The ``bitcast double ... to i64`` lines whose result is only
        compared with zero or minus one.
    """
    casts = {}
    for line in lines:
        match = _SIGN_BITS.match(line)
        if match:
            casts[match.group(1)] = line
    if not casts:
        return set()
    harmless = dict(casts)
    names = re.compile("|".join(re.escape(n) + r"(?![\w.])" for n in casts))
    for line in lines:
        for name in names.findall(line):
            test = rf"= icmp (?:slt|sgt) i64 {re.escape(name)}, (?:0|-1)$"
            if line is not casts[name] and not re.search(test, line):
                harmless.pop(name, None)
    return set(harmless.values())


def narrow_ir(text, mode):
    """Replace 64-bit types in LLVM IR by their 32-bit counterparts.

    Parameters
    ----------
    text : str
        Textual LLVM IR of an optimised kernel.
    mode : Mode
        Which types to narrow.

    Returns
    -------
    str
        The rewritten IR; `text` itself if the mode narrows nothing.

    Raises
    ------
    VulkanUnsupportedError
        If the code depends on the width of the types: it reinterprets
        ``float64`` values as integers, shifts 64-bit integers by 32 bits or
        more, or contains integer constants that do not fit in 32 bits.
    """
    if not (mode.floats or mode.ints):
        return text
    out = []
    lines = text.splitlines()
    sign_tests = _sign_tests(lines)
    for line in lines:
        if line.startswith(("target ", "attributes ", "!", ";", "source_filename")):
            out.append(line)
            continue
        if _CONVERSION_DECLARATION.match(line):
            continue
        if _FLOAT_BITS.search(line) and line not in sign_tests:
            raise VulkanUnsupportedError(
                "this kernel works on the bit pattern of float64 values, which "
                "cannot be done on a device without 64-bit types"
            )
        if mode.floats and ("double" in line or ".f64" in line):
            line = _WIDEN_FLOAT.sub(r"bitcast float \1 to float", line)
            if "double" in line:
                line = _FLOAT_LITERAL.sub(_narrow_literal, line)
            line = re.sub(r"\bdouble\b", "float", line).replace(".f64", ".f32")
            line = _CONVERSION.sub(r"\1 = bitcast float \2 to float", line)
        if mode.ints and "i64" in line:
            wide = _TOP_BITS.match(line)
            if wide and int(wide.group(5)) >= 32:
                # The upper half of a 64-bit integer consists of copies of
                # the sign bit of its 32-bit counterpart.
                indent, result, kind, value, amount = wide.groups()
                if kind == "ashr":
                    line = f"{indent}{result} = ashr i32 {value}, 31"
                else:
                    sign = f"%nv.sign{len(out)}"
                    out.append(f"{indent}{sign} = ashr i32 {value}, 31")
                    line = f"{indent}{result} = lshr i32 {sign}, {int(amount) - 32}"
            elif any(int(n) >= 32 for n in _WIDE_SHIFT.findall(line)):
                raise VulkanUnsupportedError(
                    "this kernel shifts 64-bit integers by 32 bits or more, which "
                    "cannot be done on a device without 64-bit integers"
                )
            # The extreme values stand for "no limit", as in slice bounds.
            line = _INT_LITERAL.sub(lambda m: _narrow_extreme(m.group(1)), line)
            for literal in _INT_LITERAL.findall(line):
                if not -(1 << 31) <= int(literal) < (1 << 32):
                    raise VulkanUnsupportedError(
                        f"the integer constant {literal} in this kernel does not fit "
                        "in 32 bits, the widest integers of this device"
                    )
            line = _WIDEN_INT.sub(r"bitcast i32 \1 to i32", line)
            line = re.sub(r"\bi64\b", "i32", line)
        out.append(line)
    return "\n".join(_dedupe_declarations(out)) + "\n"


def stored_dtype(dtype, mode):
    """Element type that a buffer holds for arrays of ``dtype``.

    Parameters
    ----------
    dtype : numpy.dtype
        Element type of the array on the host.
    mode : Mode
        Mode of the kernel, or of the device holding the buffer.

    Returns
    -------
    numpy.dtype
        ``int32`` for booleans (SPIR-V has no storable bool), the 32-bit
        counterpart of a 64-bit type that the mode narrows, and ``dtype``
        otherwise. For a structured dtype, the same with the type of each
        field narrowed (booleans stay single bytes in records).
    """
    return _stored_dtype(np.dtype(dtype), mode)


@functools.lru_cache(maxsize=4096)
def _stored_dtype(dtype, mode):
    """`stored_dtype` for a `numpy.dtype`; asked for by every launch."""
    if dtype.names is not None:
        return _stored_record(dtype, mode)
    if dtype == np.bool_:
        return np.dtype(np.int32)
    if mode.floats and dtype == np.float64:
        return np.dtype(np.float32)
    if mode.ints and dtype in (np.int64, np.uint64):
        return np.dtype(np.int32 if dtype == np.int64 else np.uint32)
    return dtype


def _stored_record(dtype, mode):
    """A structured dtype with its 64-bit fields narrowed as `mode` says.

    Returns
    -------
    numpy.dtype
        `dtype` itself if nothing changes; nested and array fields are left
        as they are, for the typing to reject.
    """
    fields, changed = [], False
    for name in dtype.names:
        field = dtype.fields[name][0]
        stored = field
        if field.names is None and field.subdtype is None and field != np.bool_:
            stored = stored_dtype(field, mode)
        changed |= stored != field
        fields.append((name, stored))
    if not changed:
        return dtype
    return np.dtype(fields, align=dtype.isalignedstruct)
