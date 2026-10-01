"""Code libraries for the Vulkan target: LLVM IR in, SPIR-V out."""

import os
import re
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

import llvmlite.binding as llvm
from llvmlite import ir
from numba.core.codegen import Codegen, CodeLibrary

from numba_vulkan.buffers import expand_buffer_access
from numba_vulkan.errors import SpirvCodegenError, VulkanUnsupportedError
from numba_vulkan.structurize import structurize

TRIPLE = "spirv1.5-unknown-vulkan1.2-compute"
ENTRY_POINT = "main"

_SPV_CAPABILITIES = {10: "float64", 11: "int64", 22: "int16", 39: "int8"}
_OP_TYPE_POINTER, _OP_SELECT, _OP_PHI = 32, 169, 245
_OP_DECORATE, _NO_CONTRACTION = 71, 42
# OpFNegate, OpFAdd, OpFSub, OpFMul, OpFDiv, OpFRem, OpFMod
_FLOAT_ARITHMETIC = {127, 129, 131, 133, 136, 140, 141}
# Opcodes that end the annotation section: OpUndef, the type and constant
# declarations, OpFunction and OpVariable.
_FIRST_DECLARATIONS = {1, 54, 59} | set(range(19, 40)) | set(range(41, 53))
_FCMP_ORDERING = re.compile(
    r"^(\s*)(%\S+) = fcmp (?:[a-z]+ )*?(uno|ord) (float|double) ([^,]+), (.+)$",
    re.MULTILINE,
)
_COPYSIGN = re.compile(
    r"^(\s*)(%\S+) = (?:tail )?call (float|double) @llvm\.copysign\.f(?:32|64)"
    r"\((?:float|double) (?:noundef )?([^,]+), (?:float|double) (?:noundef )?([^)]+)\).*$",
    re.MULTILINE,
)
_SREM = re.compile(r"^(\s*)(%\S+) = srem (\S+) ([^,]+), (.+)$", re.MULTILINE)

# LLVM passes run on the linked kernel, in order. Everything is inlined into
# the entry point first. instcombine is required: it folds away the
# aggregates Numba builds for arrays and tuples, and the SPIR-V backend
# miscompiles nested insertvalue/extractvalue chains.
PASSES = (
    "always_inliner",
    "global_dead_code_eliminate",
    "sroa",
    "instruction_combine",
    "simplify_cfg",
    "dead_code_elimination",
    "global_dead_code_eliminate",
    # Canonical loops (one latch, dedicated exits) and no phi nodes: the form
    # numba_vulkan.structurize needs to rearrange control flow.
    "loop_simplify",
    "register_to_memory",
)

# Seconds after which the SPIR-V backend is assumed to hang.
EMIT_TIMEOUT = 120

_target_machine = None


def target_machine():
    """The LLVM target machine for Vulkan SPIR-V, created on first use.

    It is only used in this process to drive optimisation passes and to
    query the data layout; code generation happens in a child process.

    Returns
    -------
    llvmlite.binding.TargetMachine
    """
    global _target_machine
    if _target_machine is None:
        llvm.initialize_all_targets()
        llvm.initialize_all_asmprinters()
        _target_machine = llvm.Target.from_triple(TRIPLE).create_target_machine(opt=0)
    return _target_machine


def emit_spirv(llvm_ir):
    """Translate LLVM IR to a SPIR-V binary in a child process.

    Parameters
    ----------
    llvm_ir : str
        Textual LLVM IR of a module containing the shader entry point.

    Returns
    -------
    bytes
        The SPIR-V module.

    Raises
    ------
    SpirvCodegenError
        If the backend fails or the module does not pass `check_spirv`.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "numba_vulkan._emit", TRIPLE],
            input=llvm_ir.encode(),
            capture_output=True,
            timeout=EMIT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise SpirvCodegenError(
            f"LLVM's SPIR-V backend did not finish within {EMIT_TIMEOUT} seconds"
        ) from None
    if proc.returncode != 0 or not proc.stdout:
        lines = proc.stderr.decode(errors="replace").strip().splitlines()
        reason = next(
            (ln for ln in lines if "LLVM ERROR" in ln or "Assertion" in ln), None
        )
        reason = reason or (lines[-1] if lines else f"exit code {proc.returncode}")
        raise SpirvCodegenError(f"LLVM's SPIR-V backend failed: {reason}")
    spirv = mark_exact(proc.stdout)
    check_spirv(spirv)
    return spirv


@dataclass
class CompiledKernel:
    """A kernel specialisation ready to be run by `numba_vulkan.runtime`.

    Attributes
    ----------
    name : str
        Qualified name of the Python function.
    spirv : bytes
        The SPIR-V module.
    llvm_ir : str
        Optimised LLVM IR the module was generated from.
    argtypes : tuple of numba.types.Type
        Argument types, with arrays bound to storage buffers.
    ndim : int
        Dimensionality of the dispatch grid.
    local_size : tuple of int
        Workgroup size along x, y and z.
    capabilities : set of str
        Optional device features the shader needs, out of ``"float64"``,
        ``"int64"``, ``"int16"`` and ``"int8"``.
    written_bindings : set of int
        Bindings whose buffers the shader stores to.
    """

    name: str
    spirv: bytes
    llvm_ir: str
    argtypes: tuple
    ndim: int
    local_size: tuple
    capabilities: set = field(default_factory=set)
    written_bindings: set = field(default_factory=set)

    @property
    def num_bindings(self):
        """Number of storage buffers the kernel uses.

        Returns
        -------
        int
            One for the shape buffer plus one per argument.
        """
        return 1 + len(self.argtypes)


def spirv_capabilities(spirv):
    """Find the optional capabilities a SPIR-V module declares.

    Parameters
    ----------
    spirv : bytes
        A SPIR-V module.

    Returns
    -------
    set of str
        Names matching the feature attributes of
        `numba_vulkan.runtime.DeviceInfo`, for example ``"float64"``.
    """
    words = struct.unpack(f"<{len(spirv) // 4}I", spirv)
    caps, pos = set(), 5
    while pos < len(words):
        count, opcode = words[pos] >> 16, words[pos] & 0xFFFF
        if opcode == 17 and words[pos + 1] in _SPV_CAPABILITIES:
            caps.add(_SPV_CAPABILITIES[words[pos + 1]])
        pos += max(count, 1)
    return caps


def mark_exact(spirv):
    """Decorate all float arithmetic in a SPIR-V module as ``NoContraction``.

    Without the decoration, shader compilers are free to reassociate float
    arithmetic: all tested drivers simplify ``(1 + x) - 1`` to ``x``, which
    destroys compensated algorithms such as the one behind ``log1p``. Numba
    promises IEEE semantics unless ``fastmath`` is requested, so every
    add, subtract, multiply, divide, negate and remainder is marked exact.

    Parameters
    ----------
    spirv : bytes
        A SPIR-V module.

    Returns
    -------
    bytes
        The module with the decorations inserted.
    """
    words = list(struct.unpack(f"<{len(spirv) // 4}I", spirv))
    targets, insert_at, pos = [], None, 5
    while pos < len(words):
        count, opcode = words[pos] >> 16, words[pos] & 0xFFFF
        if insert_at is None and opcode in _FIRST_DECLARATIONS:
            insert_at = pos
        if opcode in _FLOAT_ARITHMETIC:
            targets.append(words[pos + 2])  # the result id follows the type id
        pos += max(count, 1)
    if not targets or insert_at is None:
        return spirv
    decorations = []
    for target in targets:
        decorations += [(3 << 16) | _OP_DECORATE, target, _NO_CONTRACTION]
    words[insert_at:insert_at] = decorations
    return struct.pack(f"<{len(words)}I", *words)


def check_spirv(spirv):
    """Reject modules that are known to crash or confuse drivers.

    Parameters
    ----------
    spirv : bytes
        A SPIR-V module.

    Raises
    ------
    SpirvCodegenError
        If the module selects between pointers (invalid without the
        VariablePointers capability), or, with ``NUMBA_VULKAN_VALIDATE=1``
        in the environment, if ``spirv-val`` rejects it.
    """
    words = struct.unpack(f"<{len(spirv) // 4}I", spirv)
    pointer_types, pos = set(), 5
    while pos < len(words):
        count, opcode = words[pos] >> 16, words[pos] & 0xFFFF
        if opcode == _OP_TYPE_POINTER:
            pointer_types.add(words[pos + 1])
        elif opcode in (_OP_PHI, _OP_SELECT) and words[pos + 1] in pointer_types:
            raise SpirvCodegenError(
                "the kernel selects between array elements by reference, which "
                "Vulkan shaders cannot express; read the values into variables first"
            )
        pos += max(count, 1)
    if os.environ.get("NUMBA_VULKAN_VALIDATE", "0") not in ("", "0"):
        with tempfile.NamedTemporaryFile(suffix=".spv") as tmp:
            tmp.write(spirv)
            tmp.flush()
            proc = subprocess.run(
                ["spirv-val", "--target-env", "vulkan1.2", tmp.name],
                capture_output=True,
                text=True,
            )
        if proc.returncode != 0:
            raise SpirvCodegenError(
                f"generated SPIR-V is invalid: {proc.stderr.strip()}"
            )


def _expand_srem(text):
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
    counter = iter(range(1 << 30))

    def repl(match):
        """Replacement text for one ``srem`` instruction."""
        indent, res, ty, lhs, rhs = match.groups()
        n = next(counter)
        return (
            f"{indent}%srem.q{n} = sdiv {ty} {lhs}, {rhs}\n"
            f"{indent}%srem.m{n} = mul {ty} %srem.q{n}, {rhs}\n"
            f"{indent}{res} = sub {ty} {lhs}, %srem.m{n}"
        )

    return _SREM.sub(repl, text)


def _expand_fcmp_ordering(text):
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
    counter = iter(range(1 << 30))
    layout = {
        "float": ("i32", 0xFF << 23, (1 << 31) - 1),
        "double": ("i64", 0x7FF << 52, (1 << 63) - 1),
    }

    def repl(match):
        """Replacement text for one ``fcmp uno`` or ``fcmp ord``."""
        indent, res, pred, ty, lhs, rhs = match.groups()
        bits, exponent, mask = layout[ty]
        code, flags = [], []
        for operand in (lhs, rhs):
            if not operand.startswith("%"):
                continue  # a finite constant is never NaN
            n = next(counter)
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
            n = next(counter)
            code.append(f"{indent}%nan.o{n} = or i1 {flags[0]}, {flags[1]}")
            unordered = f"%nan.o{n}"
        if pred == "uno":
            code.append(f"{indent}{res} = or i1 {unordered}, false")
        else:
            code.append(f"{indent}{res} = xor i1 {unordered}, true")
        return "\n".join(code)

    return _FCMP_ORDERING.sub(repl, text)


def _expand_copysign(text):
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
    counter = iter(range(1 << 30))

    def repl(match):
        """Replacement text for one ``llvm.copysign`` call."""
        indent, res, ty, magnitude, sign = match.groups()
        bits, width = ("i64", 64) if ty == "double" else ("i32", 32)
        n = next(counter)
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


class VulkanCodeLibrary(CodeLibrary):
    """Holds LLVM IR modules; only kernel libraries are turned into SPIR-V.

    One library is created per compiled function. The library of a kernel
    additionally contains the entry point and links the libraries of all
    functions the kernel calls.

    Parameters
    ----------
    codegen : VulkanCodegen
        The code generator that owns the library.
    name : str
        Name of the library.

    Attributes
    ----------
    written_bindings : set of int
        Bindings the linked code stores to; filled when the library is
        linked.
    """

    def __init__(self, codegen, name):
        super().__init__(codegen, name)
        self._modules = []
        self._linking_libraries = []
        self._linked = None
        self._spirv = None
        self.written_bindings = set()

    def add_ir_module(self, module):
        """Add a module to the library.

        Parameters
        ----------
        module : llvmlite.ir.Module or str
            An IR module, or textual LLVM IR.
        """
        self._raise_if_finalized()
        self._modules.append(module)

    def add_linking_library(self, library):
        """Make the code of another library available to this one.

        Parameters
        ----------
        library : VulkanCodeLibrary
            Library holding functions that this library calls.
        """
        if library is not self and library not in self._linking_libraries:
            self._linking_libraries.append(library)

    def finalize(self):
        """Mark the library complete. Linking is deferred until SPIR-V is needed."""
        self._raise_if_finalized()
        self._finalized = True

    def get_function(self, name):
        """Look up a function defined or declared in this library.

        Parameters
        ----------
        name : str
            Mangled function name.

        Returns
        -------
        llvmlite.ir.Function

        Raises
        ------
        KeyError
            If no module of the library contains the function.
        """
        for module in self._modules:
            fn = module.globals.get(name) if isinstance(module, ir.Module) else None
            if fn is not None:
                return fn
        raise KeyError(f"Function {name} not found")

    def _all_libraries(self, seen=None):
        """This library and everything it links, each exactly once.

        Parameters
        ----------
        seen : list of VulkanCodeLibrary, optional
            Libraries collected so far; used by the recursion.

        Returns
        -------
        list of VulkanCodeLibrary
        """
        seen = [] if seen is None else seen
        if self not in seen:
            seen.append(self)
            for library in self._linking_libraries:
                library._all_libraries(seen)
        return seen

    def get_llvm_str(self):
        """Unoptimised LLVM IR of this library alone.

        Returns
        -------
        str
        """
        return "\n\n".join(str(m) for m in self._modules)

    def _link_and_optimize(self):
        """Link all modules into one and prepare it for the SPIR-V backend.

        The steps are: link; mark every function for inlining into the entry
        point; run a fixed set of LLVM passes; reject recursion; rewrite
        ``srem``; expand the buffer access placeholders.

        Returns
        -------
        llvmlite.binding.ModuleRef
            The linked module. The result is cached.

        Raises
        ------
        VulkanUnsupportedError
            If a function could not be inlined, which means it is recursive.

        Notes
        -----
        The pass list is deliberately short. ``instcombine`` is needed to fold
        the aggregates Numba builds, which the backend miscompiles; the full
        optimisation pipeline produces constructs the backend cannot handle.
        """
        if self._linked is not None:
            return self._linked
        linked = None
        for library in self._all_libraries():
            for module in library._modules:
                parsed = llvm.parse_assembly(str(module))
                if linked is None:
                    linked = parsed
                else:
                    linked.link_in(parsed)
        linked.triple = TRIPLE
        # Shaders cannot keep Numba's calling convention (return pointers,
        # status codes), so everything is inlined into the entry point.
        for fn in linked.functions:
            if not fn.is_declaration and fn.name != ENTRY_POINT:
                fn.linkage = "internal"
                fn.add_function_attribute("alwaysinline")
        # Numba's environment globals are never used by shader code.
        for gv in linked.global_variables:
            gv.linkage = "internal"
        linked.verify()
        machine = target_machine()
        pto = llvm.create_pipeline_tuning_options(speed_level=0)
        builder = llvm.create_pass_builder(machine, pto)
        passes = llvm.create_new_module_pass_manager()
        for name in PASSES:
            getattr(passes, f"add_{name}_pass")()
        passes.run(linked, builder)
        leftover = [
            fn.name
            for fn in linked.functions
            if not fn.is_declaration and fn.name != ENTRY_POINT
        ]
        if leftover:
            # Only recursive functions survive the always-inliner.
            raise VulkanUnsupportedError(
                "recursive functions are not supported on Vulkan "
                f"(could not inline {', '.join(leftover)})"
            )
        text = _expand_copysign(_expand_fcmp_ordering(_expand_srem(str(linked))))
        text = structurize(text)
        text, self.written_bindings = expand_buffer_access(text)
        linked = llvm.parse_assembly(text)
        linked.verify()
        self._linked = linked
        return linked

    def get_optimized_llvm_str(self):
        """LLVM IR of the whole kernel as handed to the SPIR-V backend.

        Returns
        -------
        str
        """
        return str(self._link_and_optimize())

    def get_asm_str(self):
        """Not available; disassemble `get_spirv` with ``spirv-dis`` instead.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("use get_spirv() and spirv-dis")

    def get_spirv(self):
        """SPIR-V binary of the kernel this library holds the entry point of.

        Returns
        -------
        bytes
            The SPIR-V module. The result is cached.

        Raises
        ------
        SpirvCodegenError
            If code generation fails.
        """
        if self._spirv is None:
            self._spirv = emit_spirv(self.get_optimized_llvm_str())
        return self._spirv


class VulkanCodegen(Codegen):
    """Creates the LLVM modules and code libraries of the Vulkan target.

    Parameters
    ----------
    module_name : str
        Name used for diagnostics.
    """

    _library_class = VulkanCodeLibrary

    def __init__(self, module_name):
        self._module_name = module_name
        self._data_layout = str(target_machine().target_data)

    def _create_empty_module(self, name):
        """Create an LLVM module set up for the Vulkan SPIR-V target.

        Parameters
        ----------
        name : str
            Module name.

        Returns
        -------
        llvmlite.ir.Module
        """
        module = ir.Module(name)
        module.triple = TRIPLE
        module.data_layout = self._data_layout
        return module

    def _add_module(self, module):
        """Required by Numba's interface; modules are tracked by their library."""
        pass

    def magic_tuple(self):
        """Values identifying this code generator for Numba's caching.

        Returns
        -------
        tuple
        """
        return (TRIPLE,)
