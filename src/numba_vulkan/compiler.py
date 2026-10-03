"""Compilation pipeline of the Vulkan target."""

import re
import zlib

import numpy as np
from llvmlite import ir
from numba.core import cgutils, compiler, types, typing
from numba.core import ir as numba_ir
from numba.core.compiler import (
    CompilerBase,
    CompileResult,
    DefaultPassBuilder,
    Flags,
    sanitize_compile_result_entries,
)
from numba.core.compiler_lock import global_compiler_lock
from numba.core.compiler_machinery import (
    FunctionPass,
    LoweringPass,
    PassManager,
    register_pass,
)
from numba.core.ir_utils import find_callname, guard, mk_unique_var
from numba.core.target_extension import target_override
from numba.core.typed_passes import AnnotateTypes, IRLegalization, NativeLowering
from numba.core.untyped_passes import IRProcessing

from numba_vulkan import narrowing
from numba_vulkan.buffers import (
    META_BINDING,
    PUSH_BINDING,
    STATUS_INDEX,
    arg_binding,
    i32,
    load_element,
    store_element,
)
from numba_vulkan.codegen import (
    ENTRY_POINT,
    CompiledKernel,
    emitter,
    push_constant_layout,
    spirv_capabilities,
)
from numba_vulkan.errors import SpirvCodegenError
from numba_vulkan.target import TARGET_NAME, vulkan_target
from numba_vulkan.vkimpl import buffer_element_type, storage_type
from numba_vulkan.vktypes import VulkanArray

LOCAL_SIZES = {1: (64, 1, 1), 2: (8, 8, 1), 3: (4, 4, 4)}


class VulkanCompileResult(CompileResult):
    # Dispatchers key their overloads by entry point; there is no native one.
    """Compile result whose entry point is a synthetic identifier.

    Numba's dispatchers and target contexts key compiled functions by
    their entry point, normally a native address. Nothing is compiled to
    native code here, so the identity of the result is used instead.
    """

    @property
    def entry_point(self):
        """Unique key of this result.

        Returns
        -------
        int
        """
        return id(self)


@register_pass(mutates_CFG=True, analysis_only=False)
class VulkanBackend(LoweringPass):
    """Final compiler pass: package the lowering output in a compile result."""

    _name = "vulkan_backend"

    def __init__(self):
        LoweringPass.__init__(self)

    def run_pass(self, state):
        """Create the `VulkanCompileResult`.

        Parameters
        ----------
        state : numba.core.compiler.StateDict
            Compiler state; ``state.cr`` is replaced by the result.

        Returns
        -------
        bool
            Always ``True``.
        """
        lowered = state["cr"]
        signature = typing.signature(state.return_type, *state.args)
        state.cr = VulkanCompileResult(
            **sanitize_compile_result_entries(
                dict(
                    typing_context=state.typingctx,
                    target_context=state.targetctx,
                    typing_error=state.status.fail_reason,
                    type_annotation=state.type_annotation,
                    library=state.library,
                    call_helper=lowered.call_helper,
                    signature=signature,
                    fndesc=lowered.fndesc,
                )
            )
        )
        return True


# Calls that create arrays of a constant shape, by Numba's name for them.
_STATIC_ARRAYS = {
    ("array", "numba_vulkan.shared"),
    ("array", "numba_vulkan.local"),
    ("empty", "numpy"),
    ("zeros", "numpy"),
    ("ones", "numpy"),
    ("full", "numpy"),
}


@register_pass(mutates_CFG=False, analysis_only=False)
class NumberSharedArrays(FunctionPass):
    """Give every call that creates an array an identity.

    Shared arrays become workgroup variables, and local arrays (including
    ``np.zeros`` and its relatives) private variables, which must be the
    same for every execution of the call and different from that of any
    other call.
    The pass passes a literal derived from the function and the position of
    the call as a hidden first argument, which typing turns into the array's
    "binding" (see `numba_vulkan.vkdecl.SharedArray`). Being derived from
    the source, it is the same in every process, as the kernel cache needs.
    """

    _name = "vulkan_number_shared_arrays"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """Add the hidden argument to the calls.

        Parameters
        ----------
        state : numba.core.compiler.StateDict
            Compiler state.

        Returns
        -------
        bool
            Whether the IR changed.
        """
        func_ir = state.func_ir
        changed, seen = False, 0
        for block in func_ir.blocks.values():
            body = []
            for stmt in block.body:
                expr = getattr(stmt, "value", None)
                if (
                    isinstance(stmt, numba_ir.Assign)
                    and isinstance(expr, numba_ir.Expr)
                    and expr.op == "call"
                    and guard(find_callname, func_ir, expr) in _STATIC_ARRAYS
                ):
                    seen += 1
                    site = (
                        f"{func_ir.func_id.modname}.{func_ir.func_id.func_qualname}:"
                        f"{stmt.loc.line}:{stmt.loc.col}:{seen}"
                    )
                    value = zlib.crc32(site.encode()) % (1 << 22)
                    var = numba_ir.Var(
                        block.scope, mk_unique_var("$shared_site"), stmt.loc
                    )
                    body.append(
                        numba_ir.Assign(numba_ir.Const(value, stmt.loc), var, stmt.loc)
                    )
                    expr.args = [var, *expr.args]
                    changed = True
                body.append(stmt)
            block.body = body
        return changed


class VulkanCompiler(CompilerBase):
    """Numba compiler pipeline of the Vulkan target.

    Numba's untyped and typed passes are used unchanged, followed by its
    native lowering and `VulkanBackend`. No machine code is generated.
    """

    def define_pipelines(self):
        """Build the pass pipeline.

        Returns
        -------
        list of numba.core.compiler_machinery.PassManager
        """
        dpb = DefaultPassBuilder
        pm = PassManager(TARGET_NAME)
        pm.passes.extend(dpb.define_untyped_pipeline(self.state).passes)
        pm.add_pass_after(NumberSharedArrays, IRProcessing)
        pm.passes.extend(dpb.define_typed_pipeline(self.state).passes)
        pm.add_pass(IRLegalization, "ensure IR is legal prior to lowering")
        pm.add_pass(AnnotateTypes, "annotate types")
        pm.add_pass(NativeLowering, "native lowering")
        pm.add_pass(VulkanBackend, "vulkan backend")
        pm.finalize()
        return [pm]


_NARROW_HELPERS = {}


@global_compiler_lock
def compile_vulkan(
    pyfunc,
    return_type,
    args,
    narrow_math=False,
    fast_math=False,
    boundscheck=False,
    error_model="numpy",
):
    """Run ``pyfunc`` through Numba's pipeline down to LLVM IR.

    Parameters
    ----------
    pyfunc : function
        The Python function.
    return_type : numba.types.Type or None
        Return type, or ``None`` to infer it.
    args : tuple of numba.types.Type
        Argument types.
    narrow_math : bool
        Compute float64 transcendental functions in float32 when libclc is
        not available.
    fast_math : bool
        Use the device's built-in float32 math functions instead of libclc.
    boundscheck : bool
        Raise ``IndexError`` for array indices that are out of bounds.
    error_model : {'numpy', 'python'}
        Numba's error model: ``'python'`` raises ``ZeroDivisionError`` for
        divisions by zero, ``'numpy'`` gives NumPy's results.

    Returns
    -------
    VulkanCompileResult
    """
    emitter.warm_up()
    flags = Flags()
    flags.boundscheck = boundscheck
    flags.no_compile = True
    flags.no_cpython_wrapper = True
    flags.no_cfunc_wrapper = True
    flags.error_model = error_model
    options = {"narrow_math": narrow_math, "fast_math": fast_math}
    if narrowing.current.floats:
        # Helper functions compiled without float64 are kept apart from the
        # ordinary ones, which Numba caches per target context.
        options["cached_internal_func"] = _NARROW_HELPERS
    targetctx = vulkan_target.target_context.subtarget(**options)
    with target_override(TARGET_NAME):
        cres = compiler.compile_extra(
            typingctx=vulkan_target.typing_context,
            targetctx=targetctx,
            func=pyfunc,
            args=args,
            return_type=return_type,
            flags=flags,
            locals={},
            pipeline_class=VulkanCompiler,
        )
    cres.library.finalize()
    return cres


# Push constants guaranteed by every Vulkan device. Kernels whose arguments
# need more use buffers instead, so that compiled kernels do not depend on
# the device. The last 12 bytes are kept for the size of the grid.
PUSH_CONSTANT_BYTES = 128
_GRID_BYTES = 12


def _push_members(context, argtypes):
    """Plan the push-constant block of a kernel.

    Parameters
    ----------
    context : VulkanTargetContext
        The target context.
    argtypes : sequence of numba.types.Type
        Argument types of the kernel.

    Returns
    -------
    list of tuple or None
        One ``(source, llvm_type)`` per member, in order, where `source`
        is ``("arg", index)`` for a scalar argument or ``("shape", k)``
        for the `k`-th extent of all array arguments. ``None`` if the
        arguments do not fit into `PUSH_CONSTANT_BYTES` or include a
        scalar type the block cannot hold.

    Notes
    -----
    Members of 8 bytes come first, so that no member needs padding. Integers
    narrower than 32 bits are widened to ``i32``, since 8- and 16-bit
    push constants need device features of their own.
    """
    wide, narrow, shapes = [], [], []
    for index, ty in enumerate(argtypes):
        if isinstance(ty, VulkanArray):
            count = _layout_values(ty)
            shapes += [("shape", len(shapes) + k) for k in range(count)]
            continue
        if isinstance(ty, types.Boolean):
            narrow.append((("arg", index), i32))
            continue
        if not isinstance(ty, (types.Integer, types.Float)):
            return None
        if isinstance(ty, types.Float) and ty.bitwidth < 32:
            return None  # half needs a device feature of its own
        elem = buffer_element_type(context, ty)
        if isinstance(ty, types.Integer) and ty.bitwidth < 32:
            elem = i32
        # Sized as stored: a narrowed kernel holds float64 and int64 values in
        # 32 bits (the IR is narrowed later; see `_push_dtype`).
        stored = narrowing.stored_dtype(np.dtype(str(ty)), narrowing.current)
        (wide if stored.itemsize == 8 else narrow).append((("arg", index), elem))
    members = wide + narrow + [(source, i32) for source in shapes]
    size = 8 * len(wide) + 4 * (len(narrow) + len(shapes))
    return members if size <= PUSH_CONSTANT_BYTES - _GRID_BYTES else None


def _layout_values(ty):
    """Number of values that describe the layout of an array argument.

    Its extents; for an array that is not plainly contiguous (layout
    ``"A"``, a view on the host side), also the position of its first
    element and its steps along each axis, in elements.

    Returns
    -------
    int
    """
    return ty.ndim if ty.layout == "C" else 2 * ty.ndim + 1


def _push_dtype(ty, mode):
    """Host type of a scalar argument in the push-constant block.

    Parameters
    ----------
    ty : numba.types.Type
        Type of the argument.
    mode : numba_vulkan.narrowing.Mode
        Mode the kernel is compiled in.

    Returns
    -------
    numpy.dtype
    """
    dtype = narrowing.stored_dtype(np.dtype(str(ty)), mode)
    if dtype.itemsize < 4:
        dtype = np.dtype(np.int32 if dtype.kind in "ib" else np.uint32)
    return dtype


def _load_argument(context, builder, index, ty, shape_offset, push=None):
    """Build the Numba value of one kernel argument.

    Parameters
    ----------
    context : VulkanTargetContext
        The target context.
    builder : llvmlite.ir.IRBuilder
        Builder positioned in the kernel entry point.
    index : int
        Position of the argument.
    ty : numba.types.Type
        Type of the argument.
    shape_offset : int
        Index of the array's first extent in the shape buffer, or among
        the extents in the push-constant block; for arrays of layout
        ``"A"``, the first element and the steps follow. Ignored for
        scalars.
    push : callable, optional
        Loads a member of the push-constant block, given its source as in
        `_push_members`. Without it, scalars and extents come from buffers.

    Returns
    -------
    llvmlite.ir.Value
        For a scalar, its value. For an array, the metadata structure
        filled with its extents.
    """
    intp = context.get_value_type(types.intp)
    if isinstance(ty, VulkanArray):

        def layout_value(k):
            """The `k`-th value describing the array's layout."""
            if push is not None:
                return push(("shape", shape_offset + k))
            return load_element(builder, META_BINDING, i32, i32(shape_offset + k))

        shape = [builder.zext(layout_value(dim), intp) for dim in range(ty.ndim)]
        itemsize = context.get_abi_sizeof(storage_type(context, ty))
        nitems = intp(1)
        for extent in shape:
            nitems = builder.mul(nitems, extent)
        offset = intp(0)
        if ty.layout == "C":
            strides, step = [], intp(itemsize)
            for extent in reversed(shape):
                strides.insert(0, step)
                step = builder.mul(step, extent)
        else:
            # A view: its first element and steps (which may be negative).
            offset = builder.zext(layout_value(ty.ndim), intp)
            strides = [
                builder.mul(builder.sext(layout_value(ty.ndim + 1 + d), intp),
                            intp(itemsize))
                for d in range(ty.ndim)
            ]  # fmt: skip
        proxy = cgutils.create_struct_proxy(ty)(context, builder)
        proxy.nitems = nitems
        proxy.itemsize = intp(itemsize)
        proxy.shape = cgutils.pack_array(builder, shape, ty=intp)
        proxy.strides = cgutils.pack_array(builder, strides, ty=intp)
        proxy.offset = offset
        return proxy._getvalue()
    elem = buffer_element_type(context, ty)
    if push is not None:
        val = push(("arg", index))
        if val.type != elem:
            val = builder.trunc(val, elem)
    else:
        val = load_element(builder, arg_binding(index), elem, i32(0))
    if isinstance(ty, types.Boolean):
        val = builder.icmp_unsigned("!=", val, i32(0))
    return val


_SHARED_GLOBAL = re.compile(r"addrspace\(3\) global \[(\d+) x (\w+)\]")
_TYPE_BYTES = {"i8": 1, "i16": 2, "i32": 4, "i64": 8, "float": 4, "double": 8}


def _shared_bytes(text):
    """Bytes of workgroup-shared memory that LLVM IR declares."""
    return sum(int(n) * _TYPE_BYTES[t] for n, t in _SHARED_GLOBAL.findall(text))


@global_compiler_lock
def compile_kernel(cres, ndim, exact=True, local_size=None):
    """Wrap a compiled function in a shader entry point and emit SPIR-V.

    Parameters
    ----------
    cres : VulkanCompileResult
        The compiled kernel body; it must return ``None``.
    ndim : int
        Dimensionality of the dispatch grid (selects the default workgroup
        size).
    exact : bool
        Whether drivers must keep float arithmetic as written; see
        `numba_vulkan.codegen.mark_exact`.
    local_size : tuple of int, optional
        Workgroup size; by default 64 invocations, shaped by `ndim`.

    Returns
    -------
    CompiledKernel
    """
    context, fndesc = cres.target_context, cres.fndesc
    argtypes = fndesc.argtypes
    if fndesc.restype != types.none:
        raise TypeError(f"kernels must return None, not {fndesc.restype}")

    library = context.codegen().create_library(f"{cres.library.name}_kernel")
    library.add_linking_library(cres.library)
    module = context.create_module("vulkan.kernel.wrapper")
    fnty = context.call_conv.get_function_type(fndesc.restype, argtypes)
    func = ir.Function(module, fnty, fndesc.llvm_func_name)
    wrapper = ir.Function(module, ir.FunctionType(ir.VoidType(), []), ENTRY_POINT)
    builder = ir.IRBuilder(wrapper.append_basic_block("entry"))

    members = _push_members(context, argtypes)
    push = None
    if members:
        position = {source: k for k, (source, _) in enumerate(members)}

        def push(source):
            """Load one member of the push-constant block."""
            k = position[source]
            return load_element(builder, PUSH_BINDING, members[k][1], i32(k))

    # Element 0 of the shape buffer receives the error status; the extents
    # follow it unless they are push constants.
    callargs, shape_offset = [], 0 if push else 1
    for index, ty in enumerate(argtypes):
        callargs.append(_load_argument(context, builder, index, ty, shape_offset, push))
        if isinstance(ty, VulkanArray):
            shape_offset += _layout_values(ty)
    status, _ = context.call_conv.call_function(
        builder, func, fndesc.restype, argtypes, callargs
    )
    # An exception leaves its code for the host to raise after the launch.
    # Invocations do not synchronise: if several fail, one of them wins.
    # Kernels that cannot raise lose this store during optimisation.
    with builder.if_then(status.is_error, likely=False):
        store_element(builder, META_BINDING, i32, i32(STATUS_INDEX), status.code)
    builder.ret_void()

    local = tuple(local_size) if local_size else LOCAL_SIZES[ndim]
    local = local + (1,) * (3 - len(local))
    text, count = re.subn(
        rf'(define void @"?{ENTRY_POINT}"?\(\))', r"\1 #0", str(module), count=1
    )
    assert count == 1
    numthreads = ",".join(map(str, local))
    text += f'\nattributes #0 = {{ "hlsl.numthreads"="{numthreads}" "hlsl.shader"="compute" }}\n'
    library.add_ir_module(text)
    library.first_constant_binding = 1 + len(argtypes)
    library.mode = narrowing.current
    library.local_size = local
    library.push_types = [str(ty) for _, ty in members or ()]
    library.finalize()

    spirv = library.get_spirv(exact)
    push_format, push_sources = _push_packing(
        members or [], argtypes, library.mode, push_constant_layout(spirv)
    )
    return CompiledKernel(
        name=fndesc.qualname,
        spirv=spirv,
        llvm_ir=library.get_optimized_llvm_str(),
        argtypes=tuple(argtypes),
        ndim=ndim,
        local_size=local,
        capabilities=spirv_capabilities(spirv),
        written_bindings=set(library.written_bindings),
        read_bindings=set(library.read_bindings),
        shared_bytes=_shared_bytes(library.get_optimized_llvm_str()),
        print_binding=library.print_binding,
        constants=dict(library.constants),
        mode=library.mode,
        narrowed=library.narrowed,
        push_format=push_format,
        push_sources=push_sources,
        args_pushed=members is not None,
        exact=exact,
    )


# Format characters of `struct` by NumPy kind and size.
_STRUCT_CODES = {
    ("i", 4): "i",
    ("i", 8): "q",
    ("u", 4): "I",
    ("u", 8): "Q",
    ("f", 4): "f",
    ("f", 8): "d",
}


def _push_packing(members, argtypes, mode, layout):
    """How the host packs the push-constant block of a kernel.

    Parameters
    ----------
    members : list of tuple
        The members, from `_push_members`; empty if the arguments are
        passed in buffers. The size of the grid may follow them.
    argtypes : sequence of numba.types.Type
        Argument types of the kernel.
    mode : numba_vulkan.narrowing.Mode
        Mode the kernel was compiled in.
    layout : list of tuple of int
        ``(offset, size)`` of each member in the SPIR-V module, from
        `numba_vulkan.codegen.push_constant_layout`. Narrowing may have
        changed the sizes since `members` was planned.

    Returns
    -------
    format : str
        A `struct` format for the block, padding included.
    sources : tuple of tuple
        The source of each value to pack, as in `_push_members`.

    Raises
    ------
    SpirvCodegenError
        If the module's block does not match the planned one.
    """
    if not layout:
        return "", ()  # the kernel uses none of them
    if len(layout) == len(members) + 3:
        # The size of the grid follows, if the kernel reads it.
        members = members + [(("groups", axis), i32) for axis in range(3)]
    if len(layout) != len(members):
        raise SpirvCodegenError(
            f"the push-constant block has {len(layout)} members, not {len(members)}"
        )
    fmt, end = "<", 0
    for (source, _), (offset, size) in zip(members, layout):
        if source[0] in ("shape", "groups"):
            dtype = np.dtype(np.int32)
        else:
            dtype = _push_dtype(argtypes[source[1]], mode)
        if dtype.itemsize != size or offset < end:
            raise SpirvCodegenError(
                f"push constant {source} is {size} bytes at offset {offset}, "
                f"expected {dtype.itemsize} bytes after offset {end}"
            )
        fmt += "x" * (offset - end) + _STRUCT_CODES[dtype.kind, dtype.itemsize]
        end = offset + size
    return fmt, tuple(source for source, _ in members)
