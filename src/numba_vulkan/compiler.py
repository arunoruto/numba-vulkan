"""Compilation pipeline of the Vulkan target."""

import re

from llvmlite import ir
from numba.core import cgutils, compiler, types, typing
from numba.core.compiler import (
    CompilerBase,
    CompileResult,
    DefaultPassBuilder,
    Flags,
    sanitize_compile_result_entries,
)
from numba.core.compiler_lock import global_compiler_lock
from numba.core.compiler_machinery import LoweringPass, PassManager, register_pass
from numba.core.target_extension import target_override
from numba.core.typed_passes import AnnotateTypes, IRLegalization, NativeLowering

from numba_vulkan.buffers import META_BINDING, arg_binding, i32, load_element
from numba_vulkan.codegen import ENTRY_POINT, CompiledKernel, spirv_capabilities
from numba_vulkan.target import TARGET_NAME, vulkan_target
from numba_vulkan.vkimpl import buffer_element_type
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
        pm.passes.extend(dpb.define_typed_pipeline(self.state).passes)
        pm.add_pass(IRLegalization, "ensure IR is legal prior to lowering")
        pm.add_pass(AnnotateTypes, "annotate types")
        pm.add_pass(NativeLowering, "native lowering")
        pm.add_pass(VulkanBackend, "vulkan backend")
        pm.finalize()
        return [pm]


@global_compiler_lock
def compile_vulkan(pyfunc, return_type, args, narrow_math=False, fast_math=False):
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

    Returns
    -------
    VulkanCompileResult
    """
    flags = Flags()
    flags.no_compile = True
    flags.no_cpython_wrapper = True
    flags.no_cfunc_wrapper = True
    flags.error_model = "numpy"
    targetctx = vulkan_target.target_context.subtarget(
        narrow_math=narrow_math, fast_math=fast_math
    )
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


def _load_argument(context, builder, index, ty, shape_offset):
    """Build the Numba value of one kernel argument from its buffers.

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
        Index of the array's first extent in the shape buffer. Ignored
        for scalars.

    Returns
    -------
    llvmlite.ir.Value
        For a scalar, its value loaded from element 0 of its buffer. For
        an array, the metadata structure filled from the shape buffer.
    """
    intp = context.get_value_type(types.intp)
    if isinstance(ty, VulkanArray):
        shape = []
        for dim in range(ty.ndim):
            extent = load_element(builder, META_BINDING, i32, i32(shape_offset + dim))
            shape.append(builder.zext(extent, intp))
        itemsize = context.get_abi_sizeof(buffer_element_type(context, ty.dtype))
        strides, step = [], intp(itemsize)
        for extent in reversed(shape):
            strides.insert(0, step)
            step = builder.mul(step, extent)
        nitems = step if not shape else builder.udiv(step, intp(itemsize))
        proxy = cgutils.create_struct_proxy(ty)(context, builder)
        proxy.nitems = nitems
        proxy.itemsize = intp(itemsize)
        proxy.shape = cgutils.pack_array(builder, shape, ty=intp)
        proxy.strides = cgutils.pack_array(builder, strides, ty=intp)
        return proxy._getvalue()
    elem = buffer_element_type(context, ty)
    val = load_element(builder, arg_binding(index), elem, i32(0))
    if isinstance(ty, types.Boolean):
        val = builder.icmp_unsigned("!=", val, i32(0))
    return val


@global_compiler_lock
def compile_kernel(cres, ndim, exact=True):
    """Wrap a compiled function in a shader entry point and emit SPIR-V.

    Parameters
    ----------
    cres : VulkanCompileResult
        The compiled kernel body; it must return ``None``.
    ndim : int
        Dimensionality of the dispatch grid (selects the workgroup size).
    exact : bool
        Whether drivers must keep float arithmetic as written; see
        `numba_vulkan.codegen.mark_exact`.

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

    callargs, shape_offset = [], 0
    for index, ty in enumerate(argtypes):
        callargs.append(_load_argument(context, builder, index, ty, shape_offset))
        if isinstance(ty, VulkanArray):
            shape_offset += ty.ndim
    # Errors raised inside a kernel are dropped, as numba.cuda does by default.
    context.call_conv.call_function(builder, func, fndesc.restype, argtypes, callargs)
    builder.ret_void()

    local = LOCAL_SIZES[ndim]
    text, count = re.subn(
        rf'(define void @"?{ENTRY_POINT}"?\(\))', r"\1 #0", str(module), count=1
    )
    assert count == 1
    numthreads = ",".join(map(str, local))
    text += f'\nattributes #0 = {{ "hlsl.numthreads"="{numthreads}" "hlsl.shader"="compute" }}\n'
    library.add_ir_module(text)
    library.finalize()

    spirv = library.get_spirv(exact)
    return CompiledKernel(
        name=fndesc.qualname,
        spirv=spirv,
        llvm_ir=library.get_optimized_llvm_str(),
        argtypes=tuple(argtypes),
        ndim=ndim,
        local_size=local,
        capabilities=spirv_capabilities(spirv),
        written_bindings=set(library.written_bindings),
    )
