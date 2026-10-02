"""Code libraries for the Vulkan target: LLVM IR in, SPIR-V out."""

import atexit
import os
import re
import select
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field

import llvmlite.binding as llvm
from llvmlite import ir
from numba.core.codegen import Codegen, CodeLibrary

from numba_vulkan import _emit, libclc
from numba_vulkan.buffers import expand_buffer_access, renumber_constants
from numba_vulkan.errors import SpirvCodegenError, VulkanUnsupportedError
from numba_vulkan.legalize import legalize
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
_LIBCLC_SYMBOL = re.compile(r"_Z\d+[a-z_0-9]+[fdil]+$")
_UNDEFINED_LIBCLC = re.compile(
    r"^declare [^\n]*spir_func [^\n]*@(_Z\w+)\(", re.MULTILINE
)
_AS_OPENCL_CONSTANT, _AS_PRIVATE = "addrspace(2)", "addrspace(10)"

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
    # Two-way branches only, canonical loops (one latch, dedicated exits) and
    # no phi nodes: the form numba_vulkan.structurize needs to rearrange
    # control flow.
    "lower_switch",
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


class Emitter:
    """A child process that runs LLVM's SPIR-V backend.

    The backend aborts the process on input it cannot handle, and it must
    not translate two modules in one process, so it runs outside the
    user's interpreter. Starting a Python process that imports llvmlite
    takes a few tenths of a second; the child is therefore kept, and forks
    for every module (see `numba_vulkan._emit`). It is replaced when it
    dies or hangs. On platforms without ``fork`` a new child is started
    for every module.
    """

    def __init__(self):
        self._proc = None
        self._log = None
        self._lock = threading.Lock()

    def _running(self):
        """Whether the child process exists and has not exited."""
        return self._proc is not None and self._proc.poll() is None

    def _start(self):
        """Start the child process without waiting for it to be ready."""
        self._stop()
        self._log = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(
            # Run as a script, which spares the child importing this package
            # and Numba; -P keeps this directory off its module search path.
            [sys.executable, "-P", _emit.__file__, TRIPLE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            bufsize=0,
            # A group of its own, so that a hanging grandchild can be killed.
            start_new_session=True,
        )

    def _stop(self):
        """Terminate the child process and its children, if any."""
        if self._proc is not None:
            try:
                os.killpg(self._proc.pid, signal.SIGKILL)
            except (AttributeError, ProcessLookupError, PermissionError):
                self._proc.kill()
            self._proc.wait()
            for stream in (self._proc.stdin, self._proc.stdout, self._log):
                stream.close()
        self._proc = self._log = None

    def warm_up(self):
        """Start the child process in the background if it is not running.

        Called when compilation of a function begins, so that the process
        is ready by the time there is a module to translate.
        """
        with self._lock:
            if not self._running():
                self._start()

    def close(self):
        """Terminate the child process; it is restarted when needed."""
        with self._lock:
            self._stop()

    def _read(self, count, deadline):
        """Read `count` bytes of a reply.

        Raises
        ------
        TimeoutError
            If the deadline passes first.
        EOFError
            If the process exits first.
        """
        fd, data = self._proc.stdout.fileno(), b""
        while len(data) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                raise TimeoutError
            chunk = os.read(fd, count - len(data))
            if not chunk:
                raise EOFError
            data += chunk
        return data

    def _failure(self, start):
        """Describe a failure from what the backend wrote since `start`."""
        self._log.seek(start)
        lines = self._log.read().decode(errors="replace").strip().splitlines()
        reason = next(
            (ln for ln in lines if "LLVM ERROR" in ln or "Assertion" in ln), None
        )
        return reason or (lines[-1] if lines else "no message")

    def emit(self, llvm_ir):
        """Translate a module.

        Parameters
        ----------
        llvm_ir : str
            Textual LLVM IR.

        Returns
        -------
        bytes
            The SPIR-V module.

        Raises
        ------
        SpirvCodegenError
            If the backend fails or does not finish within `EMIT_TIMEOUT`
            seconds.
        """
        with self._lock:
            if not self._running():
                self._start()
            start = os.fstat(self._log.fileno()).st_size
            data = llvm_ir.encode()
            deadline = time.monotonic() + EMIT_TIMEOUT
            try:
                self._proc.stdin.write(_emit.HEADER.pack(len(data)) + data)
                header = self._read(_emit.HEADER.size, deadline)
                (size,) = _emit.HEADER.unpack(header)
                if size != _emit.FAILED:
                    return self._read(size, deadline)
                reason = self._failure(start)
            except TimeoutError:
                self._stop()
                raise SpirvCodegenError(
                    f"LLVM's SPIR-V backend did not finish within {EMIT_TIMEOUT} seconds"
                ) from None
            except (EOFError, OSError):
                self._proc.wait()
                reason = self._failure(start)
                self._stop()
            finally:
                if not _emit.CAN_FORK:
                    self._stop()
            raise SpirvCodegenError(f"LLVM's SPIR-V backend failed: {reason}")


emitter = Emitter()
atexit.register(emitter.close)


def emit_spirv(llvm_ir, exact=True):
    """Translate LLVM IR to a SPIR-V binary in a child process.

    Parameters
    ----------
    llvm_ir : str
        Textual LLVM IR of a module containing the shader entry point.
    exact : bool
        Whether float arithmetic is marked exact; see `mark_exact`.

    Returns
    -------
    bytes
        The SPIR-V module.

    Raises
    ------
    SpirvCodegenError
        If the backend fails or the module does not pass `check_spirv`.
    """
    spirv = emitter.emit(llvm_ir)
    if exact:
        spirv = mark_exact(spirv)
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
    constants: dict = field(default_factory=dict)

    @property
    def num_bindings(self):
        """Number of storage buffers the kernel uses.

        Returns
        -------
        int
            One for the shape buffer, one per argument and one per constant
            array.
        """
        return 1 + len(self.argtypes) + len(self.constants)


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


def _link_libclc(module, pass_builder):
    """Link the libclc functions a module calls into it.

    The library is linked in the form `numba_vulkan.libclc.prepared_bitcode`
    provides, from which the linker takes the called functions and what
    they depend on.

    Parameters
    ----------
    module : llvmlite.binding.ModuleRef
        The linked kernel.
    pass_builder : llvmlite.binding.PassBuilder
        Pass builder for the target.

    Returns
    -------
    llvmlite.binding.ModuleRef
        The module with the functions linked in, or `module` itself if it
        calls nothing from libclc.

    Raises
    ------
    VulkanUnsupportedError
        If libclc is needed but not installed.
    """
    wanted = [
        fn.name
        for fn in module.functions
        if fn.is_declaration and _LIBCLC_SYMBOL.match(fn.name)
    ]
    if not wanted:
        return module
    if not libclc.available():
        raise VulkanUnsupportedError(
            f"this kernel needs libclc for {', '.join(wanted)}, but libclc was not "
            f"found; install it or point {libclc.ENV_VAR} at clspv--.bc"
        )
    library = llvm.parse_bitcode(libclc.prepared_bitcode())
    library.triple = module.triple
    library.data_layout = module.data_layout
    module.link_in(library)
    for fn in module.functions:
        if not fn.is_declaration and fn.name != ENTRY_POINT:
            fn.linkage = "internal"
    for gv in module.global_variables:
        gv.linkage = "internal"
    passes = llvm.create_new_module_pass_manager()
    passes.add_global_dead_code_eliminate_pass()
    passes.run(module, pass_builder)
    return module


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
        self._spirv = {}
        self.written_bindings = set()
        self.constants = {}
        self.first_constant_binding = 0

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
        machine = target_machine()
        pto = llvm.create_pipeline_tuning_options(speed_level=0)
        builder = llvm.create_pass_builder(machine, pto)
        linked = _link_libclc(linked, builder)
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
        text = legalize(str(linked))
        missing = _UNDEFINED_LIBCLC.findall(text)
        if missing:
            raise VulkanUnsupportedError(
                "libclc needs helper functions that this target does not provide: "
                + ", ".join(sorted(set(missing)))
            )
        text = structurize(text)
        text, self.constants = renumber_constants(text, self.first_constant_binding)
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

    def get_spirv(self, exact=True):
        """SPIR-V binary of the kernel this library holds the entry point of.

        Parameters
        ----------
        exact : bool
            Whether float arithmetic is marked exact; see `mark_exact`.

        Returns
        -------
        bytes
            The SPIR-V module. The result is cached.

        Raises
        ------
        SpirvCodegenError
            If code generation fails.
        """
        if exact not in self._spirv:
            self._spirv[exact] = emit_spirv(self.get_optimized_llvm_str(), exact)
        return self._spirv[exact]


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
