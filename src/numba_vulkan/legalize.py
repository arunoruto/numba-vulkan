"""Rewrites of LLVM IR that the SPIR-V backend cannot translate for shaders.

LLVM's SPIR-V backend aborts on, or miscompiles, a number of constructs
that ordinary optimised code contains. Each function here replaces one of
them by an equivalent the backend handles. They work on textual IR and run
after LLVM's passes, because the optimiser would otherwise recreate the
original forms (it recognises a hand-written ``copysign`` or NaN test and
emits the intrinsic or comparison again).

Most of these constructs come from libclc, whose ``clspv`` build expects
the clspv compiler to legalise them.
"""

import functools
import re

_NAME = r'%(?:"[^"]+"|[-a-zA-Z$._0-9]+)'
# Parameter and return attributes, such as ``noundef`` or ``range(i32 0, 33)``.
_ATTRS = r"(?:[a-z_]+(?:\([^)]*\))? )*"
_COUNTER = iter(range(1 << 62))

_SREM = re.compile(r"^(\s*)(%\S+) = srem (\S+) ([^,]+), (.+)$", re.MULTILINE)
_FCMP_ORDERING = re.compile(
    r"^(\s*)(%\S+) = fcmp (?:[a-z]+ )*?(uno|ord) (float|double) ([^,]+), (.+)$",
    re.MULTILINE,
)
_COPYSIGN = re.compile(
    rf"^(\s*)(%\S+) = (?:tail )?call {_ATTRS}(float|double) @llvm\.copysign\.f(?:32|64)"
    rf"\((?:float|double) {_ATTRS}([^,]+), (?:float|double) {_ATTRS}([^)]+)\).*$",
    re.MULTILINE,
)
_FMULADD = re.compile(
    rf"^(\s*)(%\S+) = (?:tail )?call {_ATTRS}(float|double) @llvm\.fmuladd\.f(?:32|64)"
    rf"\((?:float|double) {_ATTRS}([^,]+), (?:float|double) {_ATTRS}([^,]+), "
    rf"(?:float|double) {_ATTRS}([^)]+)\).*$",
    re.MULTILINE,
)
_FMA64 = re.compile(
    rf"^(\s*)(%\S+) = (?:tail )?call {_ATTRS}double @llvm\.fma\.f64"
    rf"\(double {_ATTRS}([^,]+), double {_ATTRS}([^,]+), double {_ATTRS}([^)]+)\).*$",
    re.MULTILINE,
)
_ROUNDING64 = re.compile(
    rf"^(\s*)(%\S+) = (?:tail )?call {_ATTRS}double "
    rf"@llvm\.(trunc|rint|nearbyint|roundeven)\.f64\(double {_ATTRS}([^)]+)\).*$",
    re.MULTILINE,
)
_CTLZ = re.compile(
    rf"^(\s*)(%\S+) = (?:tail )?call {_ATTRS}i(32|64) @llvm\.ctlz\.i(?:32|64)"
    rf"\(i(?:32|64) {_ATTRS}([^,]+), i1 (?:true|false)\).*$",
    re.MULTILINE,
)
_FSHL = re.compile(
    rf"^(\s*)(%\S+) = (?:tail )?call {_ATTRS}i(32|64) @llvm\.fsh(l|r)\.i(?:32|64)"
    rf"\(i(?:32|64) {_ATTRS}([^,]+), i(?:32|64) {_ATTRS}([^,]+), i(?:32|64) {_ATTRS}([^)]+)\).*$",
    re.MULTILINE,
)
_MUL_HI = re.compile(
    rf"^(\s*)(%\S+) = (?:tail )?call {_ATTRS}i32 @_Z12__clc_mul_hi(jj|ii)"
    rf"\(i32 {_ATTRS}([^,]+), i32 {_ATTRS}([^)]+)\).*$",
    re.MULTILINE,
)
_MUL_HI_DECLARATION = re.compile(
    r"^declare [^\n]*@_Z12__clc_mul_hi(?:jj|ii)\([^\n]*\n", re.MULTILINE
)
_ZERO = r"-?0\.0+e\+00"
_FCMP_ZERO = re.compile(
    rf"^(\s*)(%\S+) = fcmp (olt|ult|ogt|ugt) (float|double) "
    rf"(?:({_ZERO}), ([^,\n]+)|([^,\n]+), ({_ZERO}))$",
    re.MULTILINE,
)
_COMPLEMENT = {"olt": "uge", "ult": "oge", "ogt": "ule", "ugt": "ole"}
_BYTE_TABLE = re.compile(r"^@(\S+) = [^\n]*constant \[(\d+) x i8\]", re.MULTILINE)
_BYTE_GEP = re.compile(
    rf"^\s*({_NAME}) = getelementptr (?:inbounds )?(?:nuw )?i8, ptr (addrspace\(\d+\) )?"
    rf"(@\S+|{_NAME}), i64 (\S+)$"
)
_INT_LOAD = re.compile(
    rf"^(\s*)({_NAME}) = load i(8|16|32|64), ptr (addrspace\(\d+\) )?({_NAME})(?:,.*)?$"
)

_FLOAT_BITS = {
    "float": ("i32", 0xFF << 23, (1 << 31) - 1, 32),
    "double": ("i64", 0x7FF << 52, (1 << 63) - 1, 64),
}


def expand_srem(text):
    """Rewrite ``srem`` instructions as ``a - (a / b) * b``.

    OpSRem returns wrong results for negative 64-bit operands on NVIDIA's
    driver, so signed remainders are derived from the quotient instead.

    Parameters
    ----------
    text : str
        Textual LLVM IR.

    Returns
    -------
    str
        The rewritten LLVM IR.
    """

    def repl(match):
        """Replacement text for one match."""
        indent, res, ty, lhs, rhs = match.groups()
        n = next(_COUNTER)
        return (
            f"{indent}%srem.q{n} = sdiv {ty} {lhs}, {rhs}\n"
            f"{indent}%srem.m{n} = mul {ty} %srem.q{n}, {rhs}\n"
            f"{indent}{res} = sub {ty} {lhs}, %srem.m{n}"
        )

    return _SREM.sub(repl, text)


def expand_fcmp_ordering(text):
    """Rewrite ``fcmp uno`` and ``fcmp ord`` as tests on the bit pattern.

    The SPIR-V backend emits OpUnordered and OpOrdered for them, which only
    OpenCL-flavoured SPIR-V may use. Numba's own number and ufunc
    implementations produce these comparisons when they check for NaN.

    Parameters
    ----------
    text : str
        Textual LLVM IR.

    Returns
    -------
    str
        The rewritten LLVM IR.
    """

    def repl(match):
        """Replacement text for one match."""
        indent, res, pred, ty, lhs, rhs = match.groups()
        bits, exponent, mask, _ = _FLOAT_BITS[ty]
        code, flags = [], []
        for operand in (lhs, rhs):
            if not operand.startswith("%"):
                continue  # a finite constant is never NaN
            n = next(_COUNTER)
            code += [
                f"{indent}%nan.b{n} = bitcast {ty} {operand} to {bits}",
                f"{indent}%nan.m{n} = and {bits} %nan.b{n}, {mask}",
                f"{indent}%nan.f{n} = icmp ugt {bits} %nan.m{n}, {exponent}",
            ]
            flags.append(f"%nan.f{n}")
        if not flags:
            unordered = "false"
        elif len(flags) == 1 or flags[0] == flags[1]:
            unordered = flags[0]
        else:
            n = next(_COUNTER)
            code.append(f"{indent}%nan.o{n} = or i1 {flags[0]}, {flags[1]}")
            unordered = f"%nan.o{n}"
        if pred == "uno":
            code.append(f"{indent}{res} = or i1 {unordered}, false")
        else:
            code.append(f"{indent}{res} = xor i1 {unordered}, true")
        return "\n".join(code)

    return _FCMP_ORDERING.sub(repl, text)


def expand_copysign(text):
    """Rewrite calls to ``llvm.copysign`` as operations on the bit pattern.

    The SPIR-V backend cannot select the intrinsic, and instcombine
    introduces it even where the source spelled out the bit operations.

    Parameters
    ----------
    text : str
        Textual LLVM IR.

    Returns
    -------
    str
        The rewritten LLVM IR.
    """

    def repl(match):
        """Replacement text for one match."""
        indent, res, ty, magnitude, sign = match.groups()
        bits, _, _, width = _FLOAT_BITS[ty]
        n = next(_COUNTER)
        return "\n".join(
            [
                f"{indent}%cs.m{n} = bitcast {ty} {magnitude} to {bits}",
                f"{indent}%cs.s{n} = bitcast {ty} {sign} to {bits}",
                f"{indent}%cs.a{n} = and {bits} %cs.m{n}, {(1 << (width - 1)) - 1}",
                f"{indent}%cs.b{n} = and {bits} %cs.s{n}, {-(1 << (width - 1))}",
                f"{indent}%cs.o{n} = or {bits} %cs.a{n}, %cs.b{n}",
                f"{indent}{res} = bitcast {bits} %cs.o{n} to {ty}",
            ]
        )

    return _COPYSIGN.sub(repl, text)


def expand_fmuladd(text):
    """Rewrite ``llvm.fmuladd`` as a multiplication and an addition.

    The intrinsic means "multiply and add, fused or not", and the SPIR-V
    backend fails on it. libclc uses it throughout.

    Parameters
    ----------
    text : str
        Textual LLVM IR.

    Returns
    -------
    str
        The rewritten LLVM IR.
    """

    def repl(match):
        """Replacement text for one match."""
        indent, res, ty, a, b, c = match.groups()
        n = next(_COUNTER)
        return f"{indent}%fma.m{n} = fmul {ty} {a}, {b}\n{indent}{res} = fadd {ty} %fma.m{n}, {c}"

    return _FMULADD.sub(repl, text)


def emulate_fma64(text):
    """Rewrite ``llvm.fma.f64`` as a fused multiply-add in software.

    Vulkan allows ``Fma`` to round the product before adding, and llvmpipe
    does so for ``double``. libclc's argument reductions, ``sin`` and
    ``cos`` among them, rely on fusion and lose all accuracy without it.
    The product is computed exactly as the sum of two doubles (Dekker's
    method, with Veltkamp's splitting), added to the third operand with
    Knuth's two-sum, and rounded once more. That differs from a true fused
    operation only in rare double-rounding cases. The splitting overflows
    for operands near the largest doubles; there, ``a * b + c`` is used.

    Parameters
    ----------
    text : str
        Textual LLVM IR.

    Returns
    -------
    str
        The rewritten LLVM IR.

    Notes
    -----
    Correct only while every operation is rounded on its own, which the
    ``NoContraction`` decoration of all float arithmetic guarantees outside
    ``fastmath`` kernels.
    """

    def repl(match):
        """Replacement text for one match."""
        indent, res, a, b, c = match.groups()
        v = f"%sfma{next(_COUNTER)}"
        lines = []
        for name, x in (("a", a), ("b", b)):
            lines += [
                f"{v}.{name}c = fmul double {x}, 134217729.0",
                f"{v}.{name}t = fsub double {v}.{name}c, {x}",
                f"{v}.{name}h = fsub double {v}.{name}c, {v}.{name}t",
                f"{v}.{name}l = fsub double {x}, {v}.{name}h",
            ]
        lines += [
            f"{v}.p = fmul double {a}, {b}",
            f"{v}.e0 = fmul double {v}.ah, {v}.bh",
            f"{v}.e1 = fsub double {v}.e0, {v}.p",
            f"{v}.e2 = fmul double {v}.ah, {v}.bl",
            f"{v}.e3 = fadd double {v}.e1, {v}.e2",
            f"{v}.e4 = fmul double {v}.al, {v}.bh",
            f"{v}.e5 = fadd double {v}.e3, {v}.e4",
            f"{v}.e6 = fmul double {v}.al, {v}.bl",
            f"{v}.e = fadd double {v}.e5, {v}.e6",
            f"{v}.s = fadd double {v}.p, {c}",
            f"{v}.v = fsub double {v}.s, {v}.p",
            f"{v}.w = fsub double {v}.s, {v}.v",
            f"{v}.x = fsub double {v}.p, {v}.w",
            f"{v}.y = fsub double {c}, {v}.v",
            f"{v}.z = fadd double {v}.x, {v}.y",
            f"{v}.lo = fadd double {v}.z, {v}.e",
            f"{v}.r = fadd double {v}.s, {v}.lo",
            # r - r is 0 exactly when r is finite.
            f"{v}.d = fsub double {v}.r, {v}.r",
            f"{v}.ok = fcmp oeq double {v}.d, 0.0",
            f"{res} = select i1 {v}.ok, double {v}.r, double {v}.s",
        ]
        return "\n".join(indent + line for line in lines)

    return _FMA64.sub(repl, text)


def emulate_rounding64(text):
    """Compute ``float64`` truncation and rounding to even from ``floor``.

    llvmpipe (Mesa 26.1) gets both wrong in its vectorised code, that is
    for values that differ between invocations: ``Trunc`` returns values of
    at least 2**24 unchanged, fraction included, and ``RoundEven`` rounds
    halves towards zero and loses the sign of negative zeros. ``Floor`` is
    correct. This matters beyond ``math.trunc`` and ``round``: libclc's
    ``sin``, ``cos`` and ``tan`` truncate the quadrant number, and go wrong
    from about 2**24 * pi / 2 on.

    With ``a = |x|`` and ``f = floor(a)``, ``trunc(x)`` is ``copysign(f, x)``
    and ``roundeven(x)`` is ``copysign(f + 1, x)`` where ``a - f`` exceeds a
    half or equals it with ``f`` odd, else ``copysign(f, x)``. Every step is
    exact, and infinities and NaN come out unchanged.

    Parameters
    ----------
    text : str
        Textual LLVM IR.

    Returns
    -------
    str
        The rewritten LLVM IR, with the intrinsics it uses declared.
    """

    def repl(match):
        """Replacement text for one match."""
        indent, res, kind, x = match.groups()
        v = f"%sround{next(_COUNTER)}"
        lines = [
            f"{v}.a = call double @llvm.fabs.f64(double {x})",
            f"{v}.f = call double @llvm.floor.f64(double {v}.a)",
        ]
        if kind == "trunc":
            lines.append(
                f"{res} = call double @llvm.copysign.f64(double {v}.f, double {x})"
            )
        else:
            lines += [
                f"{v}.d = fsub double {v}.a, {v}.f",
                f"{v}.h = fmul double {v}.f, 0.5",
                f"{v}.hf = call double @llvm.floor.f64(double {v}.h)",
                f"{v}.e = fsub double {v}.h, {v}.hf",
                # f is odd exactly when f / 2 has a fraction.
                f"{v}.odd = fcmp one double {v}.e, 0.0",
                f"{v}.above = fcmp ogt double {v}.d, 0.5",
                f"{v}.half = fcmp oeq double {v}.d, 0.5",
                f"{v}.tie = and i1 {v}.half, {v}.odd",
                f"{v}.up = or i1 {v}.above, {v}.tie",
                f"{v}.f1 = fadd double {v}.f, 1.0",
                f"{v}.r = select i1 {v}.up, double {v}.f1, double {v}.f",
                f"{res} = call double @llvm.copysign.f64(double {v}.r, double {x})",
            ]
        return "\n".join(indent + line for line in lines)

    text, count = _ROUNDING64.subn(repl, text)
    if count:
        for name, args in (
            ("fabs", "double"),
            ("floor", "double"),
            ("copysign", "double, double"),
        ):
            if f"declare double @llvm.{name}.f64(" not in text:
                text += f"\ndeclare double @llvm.{name}.f64({args})\n"
    return text


def expand_ctlz(text):
    """Rewrite ``llvm.ctlz`` as a binary search for the highest set bit.

    The backend cannot legalise the intrinsic. The result for zero is the
    bit width, as LLVM defines it.

    Parameters
    ----------
    text : str
        Textual LLVM IR.

    Returns
    -------
    str
        The rewritten LLVM IR.
    """

    def repl(match):
        """Replacement text for one match."""
        indent, res, width, value = match.groups()
        n, ty, width = next(_COUNTER), f"i{match.group(3)}", int(match.group(3))
        code, current, position, shift, step = [], value, "0", width // 2, 0
        while shift:
            p = f"%clz{n}.{step}"
            code += [
                f"{indent}{p}.t = lshr {ty} {current}, {shift}",
                f"{indent}{p}.c = icmp ne {ty} {p}.t, 0",
                f"{indent}{p}.v = select i1 {p}.c, {ty} {p}.t, {ty} {current}",
                f"{indent}{p}.s = select i1 {p}.c, {ty} {shift}, {ty} 0",
                f"{indent}{p}.p = add {ty} {position}, {p}.s",
            ]
            current, position, shift, step = f"{p}.v", f"{p}.p", shift // 2, step + 1
        code += [
            f"{indent}%clz{n}.n = sub {ty} {width - 1}, {position}",
            f"{indent}%clz{n}.z = icmp eq {ty} {value}, 0",
            f"{indent}{res} = select i1 %clz{n}.z, {ty} {width}, {ty} %clz{n}.n",
        ]
        return "\n".join(code)

    return _CTLZ.sub(repl, text)


def expand_funnel_shift(text):
    """Rewrite ``llvm.fshl`` and ``llvm.fshr`` as shifts and an ``or``.

    Parameters
    ----------
    text : str
        Textual LLVM IR.

    Returns
    -------
    str
        The rewritten LLVM IR.
    """

    def repl(match):
        """Replacement text for one match."""
        indent, res, width, direction, high, low, amount = match.groups()
        n, ty, width = next(_COUNTER), f"i{width}", int(width)
        p = f"%fsh{n}"
        # fshl keeps the high bits of (high:low) << amount, fshr the low bits
        # of (high:low) >> amount; an amount of zero must not shift by width.
        first, second = ("shl", "lshr") if direction == "l" else ("lshr", "shl")
        a, b = (high, low) if direction == "l" else (low, high)
        return "\n".join(
            [
                f"{indent}{p}.a = and {ty} {amount}, {width - 1}",
                f"{indent}{p}.r = sub {ty} {width}, {p}.a",
                f"{indent}{p}.x = {first} {ty} {a}, {p}.a",
                f"{indent}{p}.y = {second} {ty} {b}, {p}.r",
                f"{indent}{p}.o = or {ty} {p}.x, {p}.y",
                f"{indent}{p}.z = icmp eq {ty} {p}.a, 0",
                f"{indent}{res} = select i1 {p}.z, {ty} {a}, {ty} {p}.o",
            ]
        )

    return _FSHL.sub(repl, text)


def expand_mul_hi(text, narrow=False):
    """Rewrite libclc's 32-bit ``mul_hi`` helper as a 64-bit multiplication.

    ``__clc_mul_hi`` returns the upper half of a product. libclc's clspv
    build leaves it undefined, because the clspv compiler supplies it.

    Parameters
    ----------
    text : str
        Textual LLVM IR.
    narrow : bool
        Use 32-bit arithmetic only, for devices without 64-bit integers.

    Returns
    -------
    str
        The rewritten LLVM IR.
    """

    def repl(match):
        """Replacement text for one match."""
        indent, res, kind, a, b = match.groups()
        n, extend = next(_COUNTER), "zext" if kind == "jj" else "sext"
        shift = "lshr" if kind == "jj" else "ashr"
        p = f"%mulhi{n}"
        if narrow:
            # Without 64-bit integers: multiply the 16-bit halves.
            lines = [
                f"{p}.al = and i32 {a}, 65535",
                f"{p}.ah = lshr i32 {a}, 16",
                f"{p}.bl = and i32 {b}, 65535",
                f"{p}.bh = lshr i32 {b}, 16",
                f"{p}.ll = mul i32 {p}.al, {p}.bl",
                f"{p}.lh = mul i32 {p}.al, {p}.bh",
                f"{p}.hl = mul i32 {p}.ah, {p}.bl",
                f"{p}.hh = mul i32 {p}.ah, {p}.bh",
                f"{p}.c0 = lshr i32 {p}.ll, 16",
                f"{p}.c1 = and i32 {p}.lh, 65535",
                f"{p}.c2 = and i32 {p}.hl, 65535",
                f"{p}.c3 = add i32 {p}.c0, {p}.c1",
                f"{p}.c4 = add i32 {p}.c3, {p}.c2",
                f"{p}.c5 = lshr i32 {p}.c4, 16",
                f"{p}.h0 = lshr i32 {p}.lh, 16",
                f"{p}.h1 = lshr i32 {p}.hl, 16",
                f"{p}.h2 = add i32 {p}.hh, {p}.h0",
                f"{p}.h3 = add i32 {p}.h2, {p}.h1",
            ]
            if kind == "jj":
                lines.append(f"{res} = add i32 {p}.h3, {p}.c5")
            else:
                # signed from unsigned: subtract b where a < 0, a where b < 0
                lines += [
                    f"{p}.u = add i32 {p}.h3, {p}.c5",
                    f"{p}.sa = ashr i32 {a}, 31",
                    f"{p}.sb = ashr i32 {b}, 31",
                    f"{p}.ma = and i32 {p}.sa, {b}",
                    f"{p}.mb = and i32 {p}.sb, {a}",
                    f"{p}.d = sub i32 {p}.u, {p}.ma",
                    f"{res} = sub i32 {p}.d, {p}.mb",
                ]
            return "\n".join(indent + line for line in lines)
        return "\n".join(
            [
                f"{indent}{p}.a = {extend} i32 {a} to i64",
                f"{indent}{p}.b = {extend} i32 {b} to i64",
                f"{indent}{p}.m = mul i64 {p}.a, {p}.b",
                f"{indent}{p}.h = {shift} i64 {p}.m, 32",
                f"{indent}{res} = trunc i64 {p}.h to i32",
            ]
        )

    return _MUL_HI_DECLARATION.sub("", _MUL_HI.sub(repl, text))


def avoid_faceforward(text):
    """Rewrite strict comparisons with zero as negated complements.

    The SPIR-V backend has a combine that turns
    ``select(fcmp(a * b, 0), n, -n)`` into the GLSL ``faceforward``
    function. For scalars it crashes or trips an internal assertion, and
    the pattern is common: ``x = a * b`` followed by ``-y if x < 0 else y``,
    which is also how ``copysign`` and rounding are usually written. The
    combine only matches the four strict predicates, so ``x < 0`` is
    emitted as ``not (x >= 0 or unordered)``, which is equivalent.

    Parameters
    ----------
    text : str
        Textual LLVM IR.

    Returns
    -------
    str
        The rewritten LLVM IR.
    """

    def repl(match):
        """Replacement text for one match."""
        indent, res, pred, ty = match.group(1, 2, 3, 4)
        lhs, rhs = (
            (match.group(5), match.group(6)) if match.group(5) else match.group(7, 8)
        )
        n = next(_COUNTER)
        return (
            f"{indent}%ff.c{n} = fcmp {_COMPLEMENT[pred]} {ty} {lhs}, {rhs}\n"
            f"{indent}{res} = xor i1 %ff.c{n}, true"
        )

    return _FCMP_ZERO.sub(repl, text)


def rename_minimumnum(text):
    """Replace ``llvm.minimumnum``/``maximumnum`` by ``minnum``/``maxnum``.

    The backend cannot legalise the newer intrinsics. They differ from the
    older ones only in how signalling NaNs and signed zeros are ordered.

    Parameters
    ----------
    text : str
        Textual LLVM IR.

    Returns
    -------
    str
        The rewritten LLVM IR.
    """
    return text.replace("@llvm.minimumnum.", "@llvm.minnum.").replace(
        "@llvm.maximumnum.", "@llvm.maxnum."
    )


def expand_byte_table_loads(text):
    """Rewrite integer loads from byte tables as loads of single bytes.

    libclc stores some constants as byte arrays and reads 32- or 64-bit
    integers from arbitrary byte offsets. Shaders address memory by element,
    not by byte, so such a load is assembled from the individual bytes
    (little-endian), each reached through a structured ``getelementptr``.

    Parameters
    ----------
    text : str
        Textual LLVM IR.

    Returns
    -------
    str
        The rewritten LLVM IR.
    """
    tables = {f"@{name}": int(size) for name, size in _BYTE_TABLE.findall(text)}
    if not tables:
        return text
    pointers, out = {}, []
    for line in text.splitlines():
        gep = _BYTE_GEP.match(line)
        if gep:
            name, space, base, offset = gep.groups()
            if base in tables:
                pointers[name] = (base, space or "", [offset])
                continue
            if base in pointers:
                table, space, offsets = pointers[base]
                pointers[name] = (table, space, offsets + [offset])
                continue
        load = _INT_LOAD.match(line)
        if load and load.group(5) in pointers:
            indent, res, width, _, pointer = load.groups()
            table, space, offsets = pointers[pointer]
            n, width, size = next(_COUNTER), int(width), tables[table]
            p = f"%bt{n}"
            position = offsets[0]
            for i, term in enumerate(offsets[1:]):
                out.append(f"{indent}{p}.o{i} = add i64 {position}, {term}")
                position = f"{p}.o{i}"
            value = None
            for k in range(width // 8):
                out += [
                    f"{indent}{p}.i{k} = add i64 {position}, {k}",
                    f"{indent}{p}.p{k} = getelementptr inbounds [{size} x i8], "
                    f"ptr {space}{table}, i64 0, i64 {p}.i{k}",
                    f"{indent}{p}.b{k} = load i8, ptr {space}{p}.p{k}",
                ]
                if width == 8:
                    value = f"{p}.b{k}"
                    break
                out += [
                    f"{indent}{p}.e{k} = zext i8 {p}.b{k} to i{width}",
                    f"{indent}{p}.s{k} = shl i{width} {p}.e{k}, {8 * k}",
                ]
                if value is None:
                    value = f"{p}.s{k}"
                else:
                    out.append(f"{indent}{p}.v{k} = or i{width} {value}, {p}.s{k}")
                    value = f"{p}.v{k}"
            out.append(f"{indent}{res} = or i{width} {value}, 0")
            continue
        out.append(line)
    return "\n".join(out) + "\n"


_SATURATING = re.compile(
    rf"^(\s*)(%\S+) = (?:tail )?call {_ATTRS}(i\d+) @llvm\.(u|s)(add|sub)\.sat\.i\d+"
    rf"\(i\d+ (?:noundef )?([^,]+), i\d+ (?:noundef )?([^)]+)\).*$",
    re.MULTILINE,
)
_SATURATING_DECLARATION = re.compile(
    r"^declare [^\n]*@llvm\.[us](?:add|sub)\.sat\.[^\n]*\n", re.MULTILINE
)


def expand_saturating(text):
    """Rewrite saturating additions and subtractions as plain arithmetic.

    LLVM's instcombine forms ``llvm.usub.sat`` and its relatives from
    clamped arithmetic, which the SPIR-V backend cannot select.

    Parameters
    ----------
    text : str
        Textual LLVM IR.

    Returns
    -------
    str
        The rewritten LLVM IR.
    """

    def repl(match):
        """Replacement text for one match."""
        indent, res, ty, kind, op, a, b = match.groups()
        n, bits = next(_COUNTER), int(ty[1:])
        p = f"%sat{n}"
        lines = [f"{p}.r = {op} {ty} {a}, {b}"]
        if kind == "u":
            if op == "add":
                # wrapped around if the result is below an operand
                lines += [f"{p}.o = icmp ult {ty} {p}.r, {a}"]
                lines += [f"{res} = select i1 {p}.o, {ty} -1, {ty} {p}.r"]
            else:
                lines += [f"{p}.o = icmp ult {ty} {a}, {b}"]
                lines += [f"{res} = select i1 {p}.o, {ty} 0, {ty} {p}.r"]
        else:
            top, bottom = (1 << (bits - 1)) - 1, -(1 << (bits - 1))
            # Overflow: for an addition, both operands' signs differ from the
            # result's; for a subtraction, the operands' signs differ and the
            # result's differs from the first operand's.
            other = f"{b}, {p}.r" if op == "add" else f"{a}, {b}"
            lines += [
                f"{p}.x = xor {ty} {a}, {p}.r",
                f"{p}.y = xor {ty} {other}",
                f"{p}.z = and {ty} {p}.x, {p}.y",
                f"{p}.o = icmp slt {ty} {p}.z, 0",
                f"{p}.n = icmp slt {ty} {a}, 0",
                f"{p}.c = select i1 {p}.n, {ty} {bottom}, {ty} {top}",
                f"{res} = select i1 {p}.o, {ty} {p}.c, {ty} {p}.r",
            ]
        return "\n".join(indent + line for line in lines)

    return _SATURATING_DECLARATION.sub("", _SATURATING.sub(repl, text))


def legalize(text, narrow_ints=False):
    """Apply all rewrites.

    Parameters
    ----------
    text : str
        Textual LLVM IR after optimisation.
    narrow_ints : bool
        Whether the kernel must not use 64-bit integers.

    Returns
    -------
    str
        IR that the SPIR-V backend can translate.
    """
    for rewrite in (
        expand_srem,
        expand_fcmp_ordering,
        expand_copysign,
        expand_fmuladd,
        expand_ctlz,
        expand_saturating,
        expand_funnel_shift,
        functools.partial(expand_mul_hi, narrow=narrow_ints),
        avoid_faceforward,
        rename_minimumnum,
        expand_byte_table_loads,
    ):
        text = rewrite(text)
    return text
