"""Access to Vulkan storage buffers from LLVM IR.

Shaders have no general pointers, and LLVM's optimiser readily creates
pointer phis and selects that SPIR-V cannot express. Buffer accesses are
therefore emitted as calls to opaque placeholder functions that take a
binding and an element index, and only :func:`expand_buffer_access` turns
them into real loads and stores, after all optimisation has happened.
"""

import re

import numpy as np
from llvmlite import ir

from numba_vulkan.errors import SpirvCodegenError, VulkanUnsupportedError

# SPIR-V storage class "StorageBuffer" and the address space LLVM maps it to.
_SC_STORAGE_BUFFER = 12
_AS_STORAGE_BUFFER = 11

i32 = ir.IntType(32)
META_BINDING = 0
# Element of the shape buffer that receives the kernel's error status.
STATUS_INDEX = 0

_PREFIX = "numba_vulkan"
_LLVM_TYPES = {
    "i8": "i8",
    "i16": "i16",
    "i32": "i32",
    "i64": "i64",
    "f16": "half",
    "f32": "float",
    "f64": "double",
}
_LOAD = re.compile(
    rf"^(\s*)(%\S+) = (?:tail )?call \S+ @{_PREFIX}\.load\.(\w+)\(i32 (\d+), i32 ([^)]+)\).*$",
    re.MULTILINE,
)
_STORE = re.compile(
    rf"^(\s*)(?:tail )?call void @{_PREFIX}\.store\.(\w+)\(i32 (\d+), i32 ([^,]+), (.+)\)[^)\n]*$",
    re.MULTILINE,
)
_ATOMIC = re.compile(
    rf'^(\s*)(%\S+) = (?:tail )?call \S+ @"?{_PREFIX}\.atomic\.(\w+)\.(\w+)"?'
    rf"\(i32 (\d+), i32 ([^,]+), (\S+ [^)]+)\).*$",
    re.MULTILINE,
)
_CAS = re.compile(
    rf'^(\s*)(%\S+) = (?:tail )?call \S+ @"?{_PREFIX}\.cas\.(\w+)"?'
    rf"\(i32 (\d+), i32 ([^,]+), \S+ ([^,]+), \S+ ([^)]+)\).*$",
    re.MULTILINE,
)
_BARRIER = re.compile(
    rf'^(\s*)(?:tail )?call void @"?{_PREFIX}\.barrier"?\(\).*$', re.MULTILINE
)
# Conversions between half and float; see numba_vulkan.vkimpl.half_conversion.
_HALF_CONVERSION = re.compile(
    rf'^(\s*%\S+) = (?:tail )?call (half|float) @"?{_PREFIX}\.(?:to|from)half"?'
    r"\((\S+ [^)]+)\).*$",
    re.MULTILINE,
)
_DECLARE = re.compile(rf'^declare [^\n]*@"?{_PREFIX}\.[^\n]*\n', re.MULTILINE)


# Placeholder binding of the buffer that print() writes to. Each kernel that
# prints gets it as its last binding (see renumber_print).
PRINT_BINDING = (1 << 20) - 1
# Placeholder binding of the push-constant block; the index of a load is the
# number of the member. See numba_vulkan.compiler._push_members.
PUSH_BINDING = (1 << 20) - 2
# The block itself, in the address space LLVM maps to PushConstant.
PUSH_BLOCK = "nv.args"
_AS_PUSH_CONSTANT = 13
# What each print() call prints, by the number its records start with.
print_formats = {}
_PRINT_ACCESS = re.compile(
    rf'(@"?{_PREFIX}\.(?:load|store|atomic\.\w+)\.\w+"?\(i32 ){PRINT_BINDING}(,)'
)


def renumber_print(text, binding):
    """Give the print buffer of a kernel its binding.

    Parameters
    ----------
    text : str
        Textual LLVM IR with placeholder accesses.
    binding : int
        The binding to use.

    Returns
    -------
    text : str
        The IR with the placeholder binding replaced.
    used : bool
        Whether the kernel prints.
    """
    text, count = _PRINT_ACCESS.subn(rf"\g<1>{binding}\g<2>", text)
    return text, count > 0


# Arrays used as global constants are typed with bindings from here upwards.
# Each kernel renumbers the ones it uses to follow its arguments.
CONSTANT_BASE = 1 << 20
# Workgroup-shared arrays are typed with "bindings" from here upwards. They
# are no buffers: their accesses become accesses to a Workgroup variable.
SHARED_BASE = 1 << 24
# Arrays local to an invocation (nv.local.array, np.zeros... in kernels) are
# typed with "bindings" from here upwards; they become Private variables.
LOCAL_BASE = 1 << 25
# Number of elements of each shared or local array, by its binding.
shared_sizes = {}
# Memory scopes of SPIR-V.
_SCOPE_DEVICE, _SCOPE_WORKGROUP = 1, 2
_AS_WORKGROUP = 3
_AS_PRIVATE = 10
_constants = {}
# llvmlite quotes the function name, LLVM's own printer does not.
_CONSTANT_ACCESS = re.compile(
    rf'(@"?{_PREFIX}\.(?:load|store|atomic\.\w+|cas)\.\w+"?\(i32 )(\d+)(,)'
)


def constant_binding(array):
    """Binding that stands for a NumPy array used as a global constant.

    The array is copied when it is first seen: like Numba on the CPU, the
    compiled code keeps the values the array had at that time.

    Parameters
    ----------
    array : numpy.ndarray
        The array.

    Returns
    -------
    int
        A binding at or above `CONSTANT_BASE`, the same for every use of
        the same array object.
    """
    known = _constants.get(id(array))
    if known is None or known[1] is not array:
        data = np.ascontiguousarray(array)
        if data.dtype == np.bool_:
            data = data.astype(np.int32)  # booleans are int32 on the device
        elif data is array:
            data = data.copy()
        # The original is kept alive so that its id() is not reused.
        known = _constants[id(array)] = (CONSTANT_BASE + len(_constants), array, data)
    return known[0]


def constant_order(text):
    """The constant arrays that LLVM IR refers to, in order of appearance.

    Parameters
    ----------
    text : str
        Textual LLVM IR with placeholder buffer accesses.

    Returns
    -------
    list of int
        The placeholder bindings (at or above `CONSTANT_BASE`), each once.
    """
    found = (int(b) for _, b, _ in _CONSTANT_ACCESS.findall(text))
    return list(dict.fromkeys(b for b in found if CONSTANT_BASE <= b < SHARED_BASE))


def constant_data(binding):
    """Contents of the constant array behind a placeholder binding.

    Parameters
    ----------
    binding : int
        A binding returned by `constant_binding`.

    Returns
    -------
    numpy.ndarray
        The values the array had when it was first seen.
    """
    for known, _, data in _constants.values():
        if known == binding:
            return data
    raise KeyError(binding)


def renumber_constants(text, first, order=None):
    """Give the constant arrays of a kernel bindings after its arguments.

    Parameters
    ----------
    text : str
        Textual LLVM IR with placeholder buffer accesses.
    first : int
        Binding for the first constant array.
    order : list of int, optional
        Placeholder bindings in the order in which they receive their
        bindings; those that `text` no longer uses are skipped. By default
        the order of appearance in `text`.

    Returns
    -------
    text : str
        The IR with the placeholder bindings replaced.
    placeholders : dict of int to int
        Placeholder binding of each constant array by its binding in this
        kernel.
    """
    used = set(constant_order(text))
    kept = [b for b in (constant_order(text) if order is None else order) if b in used]
    actual = {placeholder: first + k for k, placeholder in enumerate(kept)}

    def rename(match):
        """Replace the binding of one access if it is a constant array."""
        binding = int(match.group(2))
        return match.group(1) + str(actual.get(binding, binding)) + match.group(3)

    text = _CONSTANT_ACCESS.sub(rename, text)
    return text, {actual[placeholder]: placeholder for placeholder in kept}


def arg_binding(index):
    """Descriptor binding of a kernel argument.

    Parameters
    ----------
    index : int
        Position of the argument in the kernel signature.

    Returns
    -------
    int
        ``1 + index``; binding 0 holds the error status and the array shapes.
    """
    return 1 + index


def _mangle(llty):
    """Short name of a buffer element type, used in placeholder names.

    Parameters
    ----------
    llty : llvmlite.ir.Type
        An integer, float or double type.

    Returns
    -------
    str
        For example ``"i32"`` or ``"f64"``.

    Raises
    ------
    VulkanUnsupportedError
        If buffers of ``llty`` are not supported.
    """
    if isinstance(llty, ir.IntType) and f"i{llty.width}" in _LLVM_TYPES:
        return f"i{llty.width}"
    if isinstance(llty, ir.HalfType):
        return "f16"
    if isinstance(llty, ir.FloatType):
        return "f32"
    if isinstance(llty, ir.DoubleType):
        return "f64"
    raise VulkanUnsupportedError(f"buffers of {llty} are not supported on Vulkan")


def _binding(binding):
    """A binding as the ``i32`` operand of a placeholder call.

    Parameters
    ----------
    binding : int or llvmlite.ir.Value
        A number, or a value that is a constant after inlining.
    """
    return i32(binding) if isinstance(binding, int) else binding


def _index(builder, index):
    """Truncate an element index to the ``i32`` that buffer access takes.

    Parameters
    ----------
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    index : llvmlite.ir.Value
        An integer index.

    Returns
    -------
    llvmlite.ir.Value
    """
    return index if index.type == i32 else builder.trunc(index, i32)


def load_element(builder, binding, elem, index):
    """Read one element of a storage buffer.

    Emits a call to a placeholder function, which
    `expand_buffer_access` later turns into the actual load.

    Parameters
    ----------
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    binding : int or llvmlite.ir.Value
        Descriptor binding of the buffer: a number, or a value that is a
        constant once the kernel is inlined.
    elem : llvmlite.ir.Type
        Element type of the buffer.
    index : llvmlite.ir.Value
        Element index.

    Returns
    -------
    llvmlite.ir.Value
        The element, of type ``elem``.
    """
    name = f"{_PREFIX}.load.{_mangle(elem)}"
    fn = builder.module.globals.get(name)
    if fn is None:
        fn = ir.Function(builder.module, ir.FunctionType(elem, [i32, i32]), name=name)
    return builder.call(fn, [_binding(binding), _index(builder, index)])


def store_element(builder, binding, elem, index, value):
    """Write one element of a storage buffer.

    Emits a call to a placeholder function, which
    `expand_buffer_access` later turns into the actual store.

    Parameters
    ----------
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    binding : int or llvmlite.ir.Value
        Descriptor binding of the buffer: a number, or a value that is a
        constant once the kernel is inlined.
    elem : llvmlite.ir.Type
        Element type of the buffer.
    index : llvmlite.ir.Value
        Element index.
    value : llvmlite.ir.Value
        Value to store, of type ``elem``.
    """
    name = f"{_PREFIX}.store.{_mangle(elem)}"
    fn = builder.module.globals.get(name)
    if fn is None:
        fnty = ir.FunctionType(ir.VoidType(), [i32, i32, elem])
        fn = ir.Function(builder.module, fnty, name=name)
    builder.call(fn, [_binding(binding), _index(builder, index), value])


# atomicrmw operations by the name used in placeholders.
ATOMIC_OPS = (
    "add",
    "sub",
    "and",
    "or",
    "xor",
    "xchg",
    "max",
    "min",
    "umax",
    "umin",
    "fadd",
)


def atomic_element(builder, binding, elem, index, op, value):
    """Apply an atomic read-modify-write to one element of a buffer.

    Emits a call to a placeholder function, which `expand_buffer_access`
    later turns into an ``atomicrmw`` instruction with relaxed ordering.

    Parameters
    ----------
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    binding : int or llvmlite.ir.Value
        Descriptor binding of the buffer (see `load_element`), or the
        binding of a shared array.
    elem : llvmlite.ir.Type
        Element type: a 32-bit integer, or ``float`` for ``fadd``.
    index : llvmlite.ir.Value
        Element index.
    op : str
        One of `ATOMIC_OPS`.
    value : llvmlite.ir.Value
        The operand, of type ``elem``.

    Returns
    -------
    llvmlite.ir.Value
        The value the element had before.
    """
    name = f"{_PREFIX}.atomic.{op}.{_mangle(elem)}"
    fn = builder.module.globals.get(name)
    if fn is None:
        fn = ir.Function(builder.module, ir.FunctionType(elem, [i32, i32, elem]), name)
    return builder.call(fn, [_binding(binding), _index(builder, index), value])


def compare_and_swap(builder, binding, index, expected, value):
    """Atomically replace one 32-bit element if it has the expected value.

    Emits a call to a placeholder function, which `expand_buffer_access`
    later expands.

    Parameters
    ----------
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    binding : int or llvmlite.ir.Value
        Descriptor binding of the buffer (see `load_element`), or the
        binding of a shared array.
    index : llvmlite.ir.Value
        Element index.
    expected, value : llvmlite.ir.Value
        ``i32`` values: what the element must hold, and its replacement.

    Returns
    -------
    llvmlite.ir.Value
        The value the element had before; the swap happened if it equals
        `expected`.
    """
    name = f"{_PREFIX}.cas.i32"
    fn = builder.module.globals.get(name)
    if fn is None:
        fn = ir.Function(
            builder.module, ir.FunctionType(i32, [i32, i32, i32, i32]), name
        )
    return builder.call(
        fn, [_binding(binding), _index(builder, index), expected, value]
    )


def barrier(builder):
    """Wait until all invocations of the workgroup arrive here.

    Emits a call to a placeholder function. It has unknown memory effects,
    so LLVM moves no memory access across it, and it is ``convergent``, so
    LLVM does not duplicate it into different branches.

    Parameters
    ----------
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    """
    name = f"{_PREFIX}.barrier"
    fn = builder.module.globals.get(name)
    if fn is None:
        fn = ir.Function(builder.module, ir.FunctionType(ir.VoidType(), []), name)
        fn.attributes.add("convergent")
    builder.call(fn, [])


def expand_buffer_access(text, push_types=()):
    """Replace the placeholder calls in LLVM IR by real buffer accesses.

    Each placeholder becomes a ``llvm.spv.resource.handlefrombinding``
    call, a ``llvm.spv.resource.getpointer`` call and a load or store.
    Loads from `PUSH_BINDING` become loads of members of the push-constant
    block. The declarations and name strings these need are appended.

    Parameters
    ----------
    text : str
        Textual LLVM IR after optimisation.
    push_types : sequence of str, optional
        LLVM types of the members of the push-constant block, if the
        kernel has one.

    Returns
    -------
    text : str
        The rewritten LLVM IR.
    written : set of int
        Bindings the code stores to.
    read : set of int
        Bindings the code loads from.

    Raises
    ------
    SpirvCodegenError
        If a placeholder call with a non-constant binding remains.

    Notes
    -----
    This works on text because llvmlite cannot build IR that uses LLVM
    target extension types, and because it has to run after LLVM's
    optimisation passes.
    """
    used = set()
    written = set()
    read = set()
    shared = {}
    cas_spaces = set()
    pushed = []
    block = "{ " + ", ".join(push_types) + " }"
    counter = iter(range(1 << 30))
    text = _float_add_as_loops(text)
    access_as = _access_types(text)

    def pointer(indent, mangled, binding, index):
        """Emit the instructions that compute the address of one element.

        Returns
        -------
        n : int
            Number identifying the ``%nv.p<n>`` pointer that was defined.
        code : str
            The instructions, each ending in a newline.
        space : int
            Address space of the pointer.
        """
        n = next(counter)
        mangled = access_as.get((binding, mangled), mangled)
        if int(binding) >= SHARED_BASE:
            ty = _LLVM_TYPES[mangled]
            space = _AS_PRIVATE if int(binding) >= LOCAL_BASE else _AS_WORKGROUP
            shared[int(binding)] = (ty, space)
            array = f"[{shared_sizes[int(binding)]} x {ty}]"
            return (
                n,
                (
                    f"{indent}%nv.p{n} = getelementptr inbounds {array}, "
                    f"ptr addrspace({space}) @nv.shared.{binding}, i32 0, i32 {index}\n"
                ),
                space,
            )
        used.add((mangled, binding))
        ext = _ext_type(mangled)
        suffix = _suffix(mangled)
        return (
            n,
            (
                f"{indent}%nv.h{n} = call {ext} @llvm.spv.resource.handlefrombinding.{suffix}"
                f"(i32 0, i32 {binding}, i32 1, i32 0, ptr @.nv.binding{binding})\n"
                f"{indent}%nv.p{n} = call ptr addrspace({_AS_STORAGE_BUFFER}) "
                f"@llvm.spv.resource.getpointer.p{_AS_STORAGE_BUFFER}.{suffix}"
                f"({ext} %nv.h{n}, i32 {index})\n"
            ),
            _AS_STORAGE_BUFFER,
        )

    def scope(space):
        """The syncscope of atomics on memory in the given address space."""
        return "workgroup" if space == _AS_WORKGROUP else "device"

    def load(match):
        """Replacement text for one placeholder load."""
        indent, res, mangled, binding, index = match.groups()
        if int(binding) == PUSH_BINDING:
            pushed.append(index)
            return (
                f"{indent}%nv.pc{len(pushed)} = getelementptr inbounds {block}, "
                f"ptr addrspace({_AS_PUSH_CONSTANT}) @{PUSH_BLOCK}, i32 0, i32 {index}\n"
                f"{indent}{res} = load {_LLVM_TYPES[mangled]}, "
                f"ptr addrspace({_AS_PUSH_CONSTANT}) %nv.pc{len(pushed)}"
            )
        read.add(int(binding))
        n, code, space = pointer(indent, mangled, binding, index)
        ty, stored = (
            _LLVM_TYPES[mangled],
            _LLVM_TYPES[access_as.get((binding, mangled), mangled)],
        )
        if stored == ty:
            return f"{code}{indent}{res} = load {ty}, ptr addrspace({space}) %nv.p{n}"
        return (
            f"{code}{indent}%nv.v{n} = load {stored}, ptr addrspace({space}) %nv.p{n}\n"
            f"{indent}{res} = bitcast {stored} %nv.v{n} to {ty}"
        )

    def store(match):
        """Replacement text for one placeholder store."""
        indent, mangled, binding, index, value = match.groups()
        if int(binding) < SHARED_BASE:
            written.add(int(binding))
        n, code, space = pointer(indent, mangled, binding, index)
        stored = _LLVM_TYPES[access_as.get((binding, mangled), mangled)]
        if stored == _LLVM_TYPES[mangled]:
            return f"{code}{indent}store {value}, ptr addrspace({space}) %nv.p{n}"
        return (
            f"{code}{indent}%nv.v{n} = bitcast {value} to {stored}\n"
            f"{indent}store {stored} %nv.v{n}, ptr addrspace({space}) %nv.p{n}"
        )

    def atomic(match):
        """Replacement text for one placeholder read-modify-write."""
        indent, res, op, mangled, binding, index, value = match.groups()
        read.add(int(binding))
        if int(binding) < SHARED_BASE:
            written.add(int(binding))
        n, code, space = pointer(indent, mangled, binding, index)
        return (
            f"{code}{indent}{res} = atomicrmw {op} ptr addrspace({space}) %nv.p{n}, "
            f'{value} syncscope("{scope(space)}") monotonic'
        )

    def cas(match):
        """Replacement text for one placeholder compare-and-swap."""
        indent, res, mangled, binding, index, expected, value = match.groups()
        read.add(int(binding))
        if int(binding) < SHARED_BASE:
            written.add(int(binding))
        n, code, space = pointer(indent, mangled, binding, index)
        cas_spaces.add(space)
        # LLVM's cmpxchg instruction crashes the backend; its intrinsic works
        # but gives the result a wrong type (see codegen.fix_compare_exchange).
        number = _SCOPE_WORKGROUP if space == _AS_WORKGROUP else _SCOPE_DEVICE
        return (
            f"{code}{indent}{res} = call i32 (ptr addrspace({space}), ...) "
            f"@llvm.spv.cmpxchg.p{space}(ptr addrspace({space}) %nv.p{n}, "
            f"i32 {expected}, i32 {value}, i32 {number}, i32 0, i32 0)"
        )

    text = _DECLARE.sub("", text)
    text = _HALF_CONVERSION.sub(
        lambda m: (
            f"{m.group(1)} = {'fptrunc' if m.group(2) == 'half' else 'fpext'} "
            f"{m.group(3)} to {m.group(2)}"
        ),
        text,
    )
    text = _STORE.sub(store, _LOAD.sub(load, text))
    text = _CAS.sub(cas, _ATOMIC.sub(atomic, text))
    text = _BARRIER.sub(
        lambda m: (
            f"{m.group(1)}call void @llvm.spv.group.memory.barrier.with.group.sync()"
        ),
        text,
    )
    if re.search(rf'@"?{_PREFIX}\.', text):
        raise SpirvCodegenError(
            "the kernel chooses between arrays while it runs (as in "
            "`a = x if flag else y`), which Vulkan shaders cannot express: each "
            "access must name its buffer. Index the arrays separately instead "
            "(`x[i] if flag else y[i]`)"
        )

    extra = []
    if pushed:
        # Declared as clang declares the push constants of HLSL: without
        # "hidden", the backend asks for the Linkage capability, which Vulkan
        # lacks.
        extra.append(
            f"@{PUSH_BLOCK} = external hidden addrspace({_AS_PUSH_CONSTANT}) "
            f"externally_initialized global {block}, align 4"
        )
    for binding in sorted({b for _, b in used}, key=int):
        label = f"nv.binding{binding}"
        extra.append(
            # The backend insists on a named global string for every resource.
            f'@.{label} = private constant [{len(label) + 1} x i8] c"{label}\\00"'
        )
    for binding, (ty, space) in sorted(shared.items()):
        extra.append(
            f"@nv.shared.{binding} = internal addrspace({space}) "
            f"global [{shared_sizes[binding]} x {ty}] poison"
        )
    for mangled in sorted({m for m, _ in used}):
        ext, suffix = _ext_type(mangled), _suffix(mangled)
        extra.append(
            f"declare {ext} @llvm.spv.resource.handlefrombinding.{suffix}"
            "(i32, i32, i32, i32, ptr)"
        )
        extra.append(
            f"declare ptr addrspace({_AS_STORAGE_BUFFER}) "
            f"@llvm.spv.resource.getpointer.p{_AS_STORAGE_BUFFER}.{suffix}({ext}, i32)"
        )
    for space in sorted(cas_spaces):
        extra.append(
            f"declare i32 @llvm.spv.cmpxchg.p{space}(ptr addrspace({space}), ...)"
        )
    if "@llvm.spv.group.memory.barrier.with.group.sync" in text:
        extra.append(
            "declare void @llvm.spv.group.memory.barrier.with.group.sync() convergent"
        )
    return text + "\n" + "\n".join(extra) + "\n", written, read


_FLOAT_ADD = re.compile(
    rf'^(\s*)(%\S+) = (?:tail )?call float @"?{_PREFIX}\.atomic\.fadd\.f32"?'
    rf"\(i32 (\d+), i32 ([^,]+), float ([^)]+)\).*$"
)
_BLOCK_LABEL = re.compile(r'^("[^"]+"|[-\w$.]+):')


def _float_add_as_loops(text):
    """Turn native float additions into compare-and-swap loops where needed.

    A buffer that is accessed with compare-and-swap, as float ``max`` and
    ``min`` are, must be accessed as integers throughout, and the native
    float addition needs a float pointer. On such buffers, each addition
    becomes a loop of integer placeholder accesses, which are expanded
    like the others.

    Parameters
    ----------
    text : str
        Textual LLVM IR with placeholder accesses and without phi nodes.

    Returns
    -------
    str
    """
    integer = {m.group(4) for m in _CAS.finditer(text)}
    if not integer or "atomic.fadd" not in text:
        return text
    out, block, counter = [], None, 0
    for line in text.splitlines():
        label = _BLOCK_LABEL.match(line)
        if label:
            block = label.group(1)
        match = _FLOAT_ADD.match(line)
        if not match or match.group(3) not in integer:
            out.append(line)
            continue
        if block is None:
            raise SpirvCodegenError("a float atomic addition in an unnamed block")
        indent, res, binding, index, value = match.groups()
        counter += 1
        n = f"nv.fa{counter}"
        start = (
            block if block.startswith('"') or _PLAIN.fullmatch(block) else f'"{block}"'
        )
        out += [
            f"{indent}%{n}.o = call i32 @{_PREFIX}.load.i32(i32 {binding}, i32 {index})",
            f"{indent}br label %{n}.loop",
            f"{n}.loop:",
            f"{indent}%{n}.c = phi i32 [ %{n}.o, %{start} ], [ %{n}.g, %{n}.loop ]",
            f"{indent}%{n}.f = bitcast i32 %{n}.c to float",
            f"{indent}%{n}.s = fadd float %{n}.f, {value}",
            f"{indent}%{n}.n = bitcast float %{n}.s to i32",
            f"{indent}%{n}.g = call i32 @{_PREFIX}.cas.i32(i32 {binding}, i32 {index}, "
            f"i32 %{n}.c, i32 %{n}.n)",
            f"{indent}%{n}.k = icmp eq i32 %{n}.g, %{n}.c",
            f"{indent}br i1 %{n}.k, label %{n}.done, label %{n}.loop",
            f"{n}.done:",
            f"{indent}{res} = bitcast i32 %{n}.c to float",
        ]
        block = f"{n}.done"
    return "\n".join(out) + "\n"


_PLAIN = re.compile(r"[-a-zA-Z$._][-a-zA-Z$._0-9]*")


def _access_types(text):
    """Decide which element type each buffer is accessed with.

    A float buffer that is also accessed as integers of the same width, as
    float atomics do through compare-and-swap, must be accessed with one
    type throughout; the integer type is used, and float values are bit
    casts of the integers.

    Parameters
    ----------
    text : str
        Textual LLVM IR with placeholder accesses.

    Returns
    -------
    dict
        The element type to use instead, by (binding, element type), for
        the accesses that need one.
    """
    seen = {}
    for match in _LOAD.finditer(text):
        seen.setdefault(match.group(4), set()).add(match.group(3))
    for match in _STORE.finditer(text):
        seen.setdefault(match.group(3), set()).add(match.group(2))
    for match in _ATOMIC.finditer(text):
        seen.setdefault(match.group(5), set()).add(match.group(4))
    for match in _CAS.finditer(text):
        seen.setdefault(match.group(4), set()).add(match.group(3))
    out = {}
    for binding, kinds in seen.items():
        for floating, integer in (("f32", "i32"), ("f64", "i64")):
            if floating in kinds and integer in kinds:
                out[(binding, floating)] = integer
    return out


def _ext_type(mangled):
    """LLVM target extension type of a read-write storage buffer.

    Parameters
    ----------
    mangled : str
        Element type name as returned by `_mangle`.

    Returns
    -------
    str
    """
    ty = _LLVM_TYPES[mangled]
    return f'target("spirv.VulkanBuffer", [0 x {ty}], {_SC_STORAGE_BUFFER}, 1)'


def _suffix(mangled):
    """Type suffix LLVM expects on the resource intrinsics of a buffer.

    Parameters
    ----------
    mangled : str
        Element type name as returned by `_mangle`.

    Returns
    -------
    str
    """
    return f"tspirv.VulkanBuffer_a0{mangled}_{_SC_STORAGE_BUFFER}_1t"
