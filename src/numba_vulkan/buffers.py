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
_DECLARE = re.compile(rf"^declare [^\n]*@{_PREFIX}\.[^\n]*\n", re.MULTILINE)


# Arrays used as global constants are typed with bindings from here upwards.
# Each kernel renumbers the ones it uses to follow its arguments.
CONSTANT_BASE = 1 << 20
_constants = {}
_CONSTANT_ACCESS = re.compile(rf"(@{_PREFIX}\.(?:load|store)\.\w+\(i32 )(\d+)(,)")


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


def renumber_constants(text, first):
    """Give the constant arrays of a kernel bindings after its arguments.

    Parameters
    ----------
    text : str
        Textual LLVM IR with placeholder buffer accesses.
    first : int
        Binding for the first constant array.

    Returns
    -------
    text : str
        The IR with the placeholder bindings replaced.
    constants : dict of int to numpy.ndarray
        Contents of each constant array by its binding in this kernel.
    """
    used = sorted(
        {
            int(b)
            for _, b, _ in _CONSTANT_ACCESS.findall(text)
            if int(b) >= CONSTANT_BASE
        }
    )
    actual = {virtual: first + k for k, virtual in enumerate(used)}
    by_binding = {binding: data for binding, _, data in _constants.values()}

    def rename(match):
        """Replace the binding of one access if it is a constant array."""
        binding = int(match.group(2))
        return match.group(1) + str(actual.get(binding, binding)) + match.group(3)

    text = _CONSTANT_ACCESS.sub(rename, text)
    return text, {actual[virtual]: by_binding[virtual] for virtual in used}


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
    if isinstance(llty, ir.FloatType):
        return "f32"
    if isinstance(llty, ir.DoubleType):
        return "f64"
    raise VulkanUnsupportedError(f"buffers of {llty} are not supported on Vulkan")


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
    binding : int
        Descriptor binding of the buffer. It must be a compile-time
        constant.
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
    return builder.call(fn, [i32(binding), _index(builder, index)])


def store_element(builder, binding, elem, index, value):
    """Write one element of a storage buffer.

    Emits a call to a placeholder function, which
    `expand_buffer_access` later turns into the actual store.

    Parameters
    ----------
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    binding : int
        Descriptor binding of the buffer. It must be a compile-time
        constant.
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
    builder.call(fn, [i32(binding), _index(builder, index), value])


def expand_buffer_access(text):
    """Replace the placeholder calls in LLVM IR by real buffer accesses.

    Each placeholder becomes a ``llvm.spv.resource.handlefrombinding``
    call, a ``llvm.spv.resource.getpointer`` call and a load or store.
    The declarations and name strings these need are appended.

    Parameters
    ----------
    text : str
        Textual LLVM IR after optimisation.

    Returns
    -------
    text : str
        The rewritten LLVM IR.
    written : set of int
        Bindings the code stores to.

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
    counter = iter(range(1 << 30))

    def pointer(indent, mangled, binding, index):
        """Emit the handle and element pointer for one access.

        Returns
        -------
        n : int
            Number identifying the ``%nv.p<n>`` pointer that was defined.
        code : str
            The two instructions, each ending in a newline.
        """
        n = next(counter)
        used.add((mangled, binding))
        ext = _ext_type(mangled)
        suffix = _suffix(mangled)
        return n, (
            f"{indent}%nv.h{n} = call {ext} @llvm.spv.resource.handlefrombinding.{suffix}"
            f"(i32 0, i32 {binding}, i32 1, i32 0, ptr @.nv.binding{binding})\n"
            f"{indent}%nv.p{n} = call ptr addrspace({_AS_STORAGE_BUFFER}) "
            f"@llvm.spv.resource.getpointer.p{_AS_STORAGE_BUFFER}.{suffix}"
            f"({ext} %nv.h{n}, i32 {index})\n"
        )

    def load(match):
        """Replacement text for one placeholder load."""
        indent, res, mangled, binding, index = match.groups()
        n, code = pointer(indent, mangled, binding, index)
        ty = _LLVM_TYPES[mangled]
        return f"{code}{indent}{res} = load {ty}, ptr addrspace({_AS_STORAGE_BUFFER}) %nv.p{n}"

    def store(match):
        """Replacement text for one placeholder store."""
        indent, mangled, binding, index, value = match.groups()
        written.add(int(binding))
        n, code = pointer(indent, mangled, binding, index)
        return (
            f"{code}{indent}store {value}, ptr addrspace({_AS_STORAGE_BUFFER}) %nv.p{n}"
        )

    text = _DECLARE.sub("", text)
    text = _STORE.sub(store, _LOAD.sub(load, text))
    if f"@{_PREFIX}." in text:
        raise SpirvCodegenError("a buffer access with a non-constant binding survived")

    extra = []
    for binding in sorted({b for _, b in used}, key=int):
        label = f"nv.binding{binding}"
        extra.append(
            # The backend insists on a named global string for every resource.
            f'@.{label} = private constant [{len(label) + 1} x i8] c"{label}\\00"'
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
    return text + "\n" + "\n".join(extra) + "\n", written


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
