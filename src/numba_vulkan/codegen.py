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
import numpy as np
from llvmlite import ir
from numba.core.codegen import Codegen, CodeLibrary

from numba_vulkan import _emit, kernelcache, libclc, narrowing
from numba_vulkan.buffers import (
    constant_data,
    constant_order,
    expand_buffer_access,
    renumber_constants,
    renumber_print,
)
from numba_vulkan.errors import SpirvCodegenError, VulkanUnsupportedError
from numba_vulkan.legalize import emulate_fma, emulate_rounding64, legalize
from numba_vulkan.structurize import structurize

TRIPLE = "spirv1.5-unknown-vulkan1.2-compute"
ENTRY_POINT = "main"

_SPV_CAPABILITIES = {
    10: "float64",
    11: "int64",
    9: "float16",
    12: "int64_atomics",
    4433: "storage16",
    22: "int16",
    39: "int8",
    6033: "float32_atomic_add",
}
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

# Before 32-bit integers are introduced into a kernel that keeps 64-bit
# floats: inline and clean up, but keep the control flow as it is.
EARLY_PASSES = (
    "always_inliner",
    "global_dead_code_eliminate",
    "sroa",
    "instruction_combine",
    "simplify_cfg",
    "dead_code_elimination",
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

    With `llc`, LLVM's ``llc`` program translates every module instead, in
    a process of its own. That serves when llvmlite's own backend is
    broken, as it is in llvmlite 0.50's macOS arm64 wheel (see
    ``docs/source/known_issues.md``). Its LLVM must read the IR that
    llvmlite writes, so it should have the same major version.

    Parameters
    ----------
    llc : str, optional
        Path of ``llc``; by default llvmlite's backend is used.
    """

    def __init__(self, llc=None):
        self.llc = llc
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
        if self.llc is not None:
            return
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
        return _failure_reason(self._log.read())

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
        if self.llc is not None:
            return self._emit_with_llc(llvm_ir)
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

    def _emit_with_llc(self, llvm_ir):
        """Translate a module with the ``llc`` program; see `emit`."""
        proc = subprocess.Popen(
            [self.llc, "-O0", "-filetype=obj", f"--spirv-ext={_emit.EXTENSIONS}"]
            + ["-o", "-", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            spirv, log = proc.communicate(llvm_ir.encode(), timeout=EMIT_TIMEOUT)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            raise SpirvCodegenError(
                f"LLVM's SPIR-V backend did not finish within {EMIT_TIMEOUT} seconds"
            ) from None
        if proc.returncode == 0:
            return spirv
        if proc.returncode < 0:
            # A crash ends standard error with a stack dump, not a message.
            reason = f"llc was killed by {signal.Signals(-proc.returncode).name}"
        else:
            reason = _failure_reason(log)
        raise SpirvCodegenError(f"LLVM's SPIR-V backend failed: {reason}")


def _failure_reason(log):
    """The line of the backend's standard error that describes its failure.

    Parameters
    ----------
    log : bytes
        What the backend wrote to standard error.

    Returns
    -------
    str
    """
    lines = log.decode(errors="replace").strip().splitlines()
    reason = next((ln for ln in lines if "LLVM ERROR" in ln or "Assertion" in ln), None)
    return reason or (lines[-1] if lines else "no message")


# Environment variable naming an ``llc`` to use instead of llvmlite's backend.
LLC_ENV_VAR = "NUMBA_VULKAN_LLC"
emitter = Emitter(llc=os.environ.get(LLC_ENV_VAR) or None)
atexit.register(emitter.close)


def emit_spirv(llvm_ir, exact=True, narrow_ints=False):
    """Translate LLVM IR to a SPIR-V binary in a child process.

    Parameters
    ----------
    llvm_ir : str
        Textual LLVM IR of a module containing the shader entry point.
    exact : bool
        Whether float arithmetic is marked exact; see `mark_exact`.
    narrow_ints : bool
        Whether the module is meant for devices without 64-bit integers;
        see `narrow_index_constants`.

    Returns
    -------
    bytes
        The SPIR-V module.

    Raises
    ------
    SpirvCodegenError
        If the backend fails or the module does not pass `check_spirv`.
    """
    spirv = fix_barrier_semantics(fix_compare_exchange(emitter.emit(llvm_ir)))
    spirv = half_storage(spirv)
    if narrow_ints:
        spirv = narrow_index_constants(spirv)
    spirv = strip_unused(spirv)
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
    read_bindings: set = field(default_factory=set)
    shared_bytes: int = 0
    print_binding: int = None
    constants: dict = field(default_factory=dict)
    mode: narrowing.Mode = narrowing.Mode()
    narrowed: narrowing.Mode = narrowing.Mode()

    @property
    def num_bindings(self):
        """Number of storage buffers the kernel uses.

        Returns
        -------
        int
            One for the shape buffer, one per argument, one per constant
            array, and one for the output of ``print``.
        """
        return (
            1
            + len(self.argtypes)
            + len(self.constants)
            + (self.print_binding is not None)
        )


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


# Result-id position of the definitions that `strip_unused` may remove.
_REMOVABLE = {
    21: 1, 22: 1, 23: 1, 28: 1, 29: 1, 30: 1, 32: 1,  # types
    41: 2, 42: 2, 43: 2, 44: 2, 46: 2,  # constants
    59: 2,  # variables
}  # fmt: skip
# Instructions that name or decorate an id without using it.
_ANNOTATIONS = {5, 6, 71, 72}
_OP_CAPABILITY, _OP_TYPE_INT, _OP_TYPE_FLOAT, _OP_VARIABLE = 17, 21, 22, 59
_STORAGE_FUNCTION = 7
_OP_CONSTANT, _ACCESS_CHAINS = 43, {65, 66}
# Instructions whose first operand is not a result type.
_UNTYPED = _ANNOTATIONS | {
    3,
    4,
    7,
    8,
    10,
    11,
    14,
    15,
    16,
    17,
    62,
    63,
    248,
    249,
    250,
    251,
}
_UNTYPED |= set(range(19, 40))
# Capability -> (type opcode, width) that requires it.
_WIDTH_CAPABILITIES = {
    10: (_OP_TYPE_FLOAT, 64),
    11: (_OP_TYPE_INT, 64),
    22: (_OP_TYPE_INT, 16),
    39: (_OP_TYPE_INT, 8),
}


def _instructions(spirv):
    """Split a SPIR-V module into its header and instructions.

    Returns
    -------
    header : list of int
        The five header words.
    instructions : list of tuple of int
    """
    words = struct.unpack(f"<{len(spirv) // 4}I", spirv)
    instructions, pos = [], 5
    while pos < len(words):
        count = max(words[pos] >> 16, 1)
        instructions.append(words[pos : pos + count])
        pos += count
    return list(words[:5]), instructions


def _assemble(header, instructions):
    """Join a header and instructions into a SPIR-V module."""
    out = list(header)
    for inst in instructions:
        out += inst
    return struct.pack(f"<{len(out)}I", *out)


def _id_positions(inst):
    """Word positions of the operands of an instruction that are ids.

    See `_id_operands`; this gives their positions instead of values.

    Returns
    -------
    list of int
    """
    opcode = inst[0] & 0xFFFF
    if opcode in (3, 7, 8, 10, 14, 17):
        return []
    if opcode in (11, 16, 21, 22, 247):
        return [1]
    if opcode in (43, 246):
        return [1, 2]
    if opcode == 81:
        return [1, 2, 3]
    if opcode == 82:
        return [1, 2, 3, 4]
    if opcode == 15:
        end = 3
        while inst[end] >> 24:
            end += 1
        return [2] + list(range(end + 1, len(inst)))
    if opcode == 12:
        return [1, 2, 3] + list(range(5, len(inst)))
    if opcode == 32:
        return [1] + list(range(3, len(inst)))
    if opcode in (54, 59):
        return [1, 2] + list(range(4, len(inst)))
    if opcode == 251:
        return [1, 2] + list(range(4, len(inst), 2))
    return list(range(1, len(inst)))


_OP_ATOMIC_COMPARE_EXCHANGE, _OP_COMPOSITE_INSERT = 230, 82
_OP_CONTROL_BARRIER = 224
# Acquire-release ordering of workgroup and buffer memory, for barriers.
_BARRIER_SEMANTICS = 0x8 | 0x40 | 0x100


def fix_compare_exchange(spirv):
    """Repair the result of compare-and-swap operations.

    LLVM's ``llvm.spv.cmpxchg`` intrinsic yields the old value, but the
    backend packs it with a success flag into a composite that it then
    treats as the integer result. The composite is removed and its uses
    take the old value directly.

    Parameters
    ----------
    spirv : bytes
        A SPIR-V module.

    Returns
    -------
    bytes
        The repaired module; `spirv` itself if it has no compare-and-swap.
    """
    header, instructions = _instructions(spirv)
    results = {
        i[2] for i in instructions if i[0] & 0xFFFF == _OP_ATOMIC_COMPARE_EXCHANGE
    }
    if not results:
        return spirv
    first, replace, drop = {}, {}, set()
    for k, inst in enumerate(instructions):
        if inst[0] & 0xFFFF != _OP_COMPOSITE_INSERT or len(inst) != 6:
            continue
        _, _, result, obj, composite, index = inst
        if obj in results and index == 0:
            first[result] = obj
            drop.add(k)
        elif composite in first and index == 1:
            replace[result] = first[composite]
            drop.add(k)
    out = []
    for k, inst in enumerate(instructions):
        if k in drop:
            continue
        if replace:
            inst = list(inst)
            for at in _id_positions(inst):
                inst[at] = replace.get(inst[at], inst[at])
            inst = tuple(inst)
        out.append(inst)
    return _assemble(header, out)


_OP_LOAD, _OP_STORE, _OP_FCONVERT, _OP_COPY_OBJECT = 61, 62, 115, 83
_CAP_FLOAT16, _CAP_STORAGE16 = 9, 4433


def half_storage(spirv):
    """Declare 16-bit storage instead of 16-bit arithmetic where possible.

    ``float16`` arrays are read and written as ``half`` and computed with
    as ``float``. The backend declares the ``Float16`` capability for any
    use of ``half``, which needs the ``shaderFloat16`` device feature. If
    ``half`` values are only loaded, stored and converted, the capability
    is replaced by ``StorageBuffer16BitAccess``, which more devices offer.

    Parameters
    ----------
    spirv : bytes
        A SPIR-V module.

    Returns
    -------
    bytes
        The module; unchanged if it does not use ``half`` or computes with
        it.
    """
    header, instructions = _instructions(spirv)
    half = {
        i[1] for i in instructions if i[0] & 0xFFFF == _OP_TYPE_FLOAT and i[2] == 16
    }
    if not half:
        return spirv
    values = set()
    for inst in instructions:
        opcode = inst[0] & 0xFFFF
        if len(inst) > 2 and inst[1] in half and opcode not in _UNTYPED:
            if opcode not in (_OP_LOAD, _OP_FCONVERT, _OP_COPY_OBJECT):
                return spirv
            values.add(inst[2])
    for inst in instructions:
        opcode = inst[0] & 0xFFFF
        if opcode in (_OP_STORE, _OP_FCONVERT, _OP_COPY_OBJECT, _OP_DECORATE, 5):
            continue
        used = set(_id_operands(inst))
        if len(inst) > 2 and inst[1] in half:
            used.discard(inst[2])  # the definition itself
        if values & used:
            return spirv
    out = [
        (inst[0], _CAP_STORAGE16)
        if inst[0] & 0xFFFF == _OP_CAPABILITY and inst[1] == _CAP_FLOAT16
        else inst
        for inst in instructions
    ]
    return _assemble(header, out)


def fix_barrier_semantics(spirv):
    """Give barriers memory semantics that Vulkan accepts.

    The backend emits ``OpControlBarrier`` with sequentially consistent
    semantics, which Vulkan forbids. They become acquire-release on
    workgroup and buffer memory, which is what ``syncthreads`` promises.

    Parameters
    ----------
    spirv : bytes
        A SPIR-V module.

    Returns
    -------
    bytes
        The repaired module; `spirv` itself if it has no barrier.
    """
    header, instructions = _instructions(spirv)
    if not any(i[0] & 0xFFFF == _OP_CONTROL_BARRIER for i in instructions):
        return spirv
    uint = next(
        i[1]
        for i in instructions
        if i[0] & 0xFFFF == _OP_TYPE_INT and i[2] == 32 and i[3] == 0
    )
    constant = next(
        (
            i[2]
            for i in instructions
            if i[0] & 0xFFFF == _OP_CONSTANT
            and i[1] == uint
            and i[3] == _BARRIER_SEMANTICS
        ),
        None,
    )
    out = []
    for inst in instructions:
        opcode = inst[0] & 0xFFFF
        if opcode == _OP_CONTROL_BARRIER:
            inst = inst[:3] + (constant,)
        out.append(inst)
        if constant is None and opcode == _OP_TYPE_INT and inst[1] == uint:
            constant = header[3]
            header[3] += 1  # the id bound
            out.append(((4 << 16) | _OP_CONSTANT, uint, constant, _BARRIER_SEMANTICS))
    # A barrier may come before the type was seen; fix those up now.
    out = [
        i[:3] + (constant,) if i[0] & 0xFFFF == _OP_CONTROL_BARRIER else i for i in out
    ]
    return _assemble(header, out)


def _id_operands(inst):
    """The operands of an instruction that are (or may be) ids.

    Literal operands of the common instructions are left out, so that a
    number which happens to equal an id is not taken for a use of it. For
    instructions not known here, every operand is returned.

    Parameters
    ----------
    inst : tuple of int
        The words of one instruction.

    Returns
    -------
    tuple of int
    """
    opcode = inst[0] & 0xFFFF
    if opcode in (3, 7, 8, 10, 14, 17):  # source, strings, extension, capability
        return ()
    if opcode in (11, 16, 21, 22, 247):  # one id, then literals
        return inst[1:2]
    if opcode in (43, 246):  # two ids, then literals
        return inst[1:3]
    if opcode == 81:  # OpCompositeExtract: literal indices at the end
        return inst[1:4]
    if opcode == 82:  # OpCompositeInsert
        return inst[1:5]
    if opcode == 15:  # OpEntryPoint: model, id, name, interface ids
        end = 3
        while inst[end] >> 24:  # the name ends with a zero byte
            end += 1
        return inst[2:3] + inst[end + 1 :]
    if opcode == 12:  # OpExtInst: the instruction number is a literal
        return inst[1:4] + inst[5:]
    if opcode == 32:  # OpTypePointer: storage class
        return inst[1:2] + inst[3:]
    if opcode in (54, 59):  # OpFunction, OpVariable: literal third operand
        return inst[1:3] + inst[4:]
    if opcode == 251:  # OpSwitch: literal, label pairs
        return inst[1:3] + inst[4::2]
    return inst[1:]


def strip_unused(spirv):
    """Remove unused variables, constants and types from a SPIR-V module.

    LLVM's backend wants a name string for every buffer, and emits each as
    a local array of 8-bit integers that nothing reads. Those arrays are
    the only reason most shaders declare the ``Int8`` capability, an
    optional device feature. This removes local variables, constants and
    types that are never used, and then the capabilities for integer and
    float widths that no longer occur.

    Parameters
    ----------
    spirv : bytes
        A SPIR-V module.

    Returns
    -------
    bytes
        The module without the unused definitions.

    Notes
    -----
    Operands of instructions that `_id_operands` does not know all count as
    uses, so a literal can keep a definition alive, but nothing that is
    used is ever removed.
    """
    words = struct.unpack(f"<{len(spirv) // 4}I", spirv)
    instructions, pos = [], 5
    while pos < len(words):
        count = max(words[pos] >> 16, 1)
        instructions.append(words[pos : pos + count])
        pos += count

    while True:
        seen, candidates = {}, {}
        for inst in instructions:
            opcode = inst[0] & 0xFFFF
            if opcode in _ANNOTATIONS:
                continue
            ids = _id_operands(inst)
            for word in ids:
                seen[word] = seen.get(word, 0) + 1
            at = _REMOVABLE.get(opcode)
            if at is not None and (
                opcode != _OP_VARIABLE or inst[3] == _STORAGE_FUNCTION
            ):
                candidates[inst[at]] = ids.count(inst[at])
        # Unused: the id occurs nowhere but in its own definition.
        dead = {i for i, own in candidates.items() if seen[i] == own}
        if not dead:
            break
        instructions = [
            inst
            for inst in instructions
            if not (
                (inst[0] & 0xFFFF) in _ANNOTATIONS
                and inst[1] in dead
                or (inst[0] & 0xFFFF) in _REMOVABLE
                and inst[_REMOVABLE[inst[0] & 0xFFFF]] in dead
            )
        ]

    widths = {
        (inst[0] & 0xFFFF, inst[2])
        for inst in instructions
        if (inst[0] & 0xFFFF) in (_OP_TYPE_INT, _OP_TYPE_FLOAT)
    }
    instructions = [
        inst
        for inst in instructions
        if not (
            (inst[0] & 0xFFFF) == _OP_CAPABILITY
            and inst[1] in _WIDTH_CAPABILITIES
            and _WIDTH_CAPABILITIES[inst[1]] not in widths
        )
    ]
    out = list(words[:5])
    for inst in instructions:
        out += inst
    return struct.pack(f"<{len(out)}I", *out)


def narrow_index_constants(spirv):
    """Turn 64-bit index constants in a SPIR-V module into 32-bit ones.

    The backend indexes into constant tables with 64-bit constants of its
    own making, whatever the width of the indices in the LLVM IR. In a
    kernel compiled for a device without 64-bit integers they are the last
    use of that type.

    Parameters
    ----------
    spirv : bytes
        A SPIR-V module.

    Returns
    -------
    bytes
        The module with the constants retyped, or `spirv` itself if it
        uses 64-bit integers for anything but small constant indices of
        access chains.
    """
    words = struct.unpack(f"<{len(spirv) // 4}I", spirv)
    instructions, pos = [], 5
    while pos < len(words):
        count = max(words[pos] >> 16, 1)
        instructions.append(words[pos : pos + count])
        pos += count
    wide = {i[1] for i in instructions if i[0] & 0xFFFF == _OP_TYPE_INT and i[2] == 64}
    if not wide:
        return spirv
    narrow, constants = None, set()
    for inst in instructions:
        opcode = inst[0] & 0xFFFF
        if opcode == _OP_TYPE_INT and inst[2] == 32 and not constants:
            narrow = inst[1]  # declared before the first constant to retype
        elif opcode == _OP_CONSTANT and inst[1] in wide:
            if narrow is None or inst[4]:
                return spirv
            constants.add(inst[2])
        elif len(inst) > 2 and inst[1] in wide and opcode not in _UNTYPED:
            return spirv  # a 64-bit value is computed
    for inst in instructions:
        opcode = inst[0] & 0xFFFF
        if opcode in _ANNOTATIONS or (opcode == _OP_CONSTANT and inst[2] in constants):
            continue
        used = set(_id_operands(inst)) & constants
        if used and (opcode not in _ACCESS_CHAINS or used & set(inst[1:4])):
            return spirv  # used as something other than an index
    out = list(words[:5])
    for inst in instructions:
        if inst[0] & 0xFFFF == _OP_CONSTANT and inst[2] in constants:
            inst = ((4 << 16) | _OP_CONSTANT, narrow, inst[2], inst[3])
        out += inst
    return struct.pack(f"<{len(out)}I", *out)


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


_LOCAL_SIZE = re.compile(
    r'^(\s*%\S+) = (?:tail )?call i32 @"?numba_vulkan\.local_size"?\(i32 (\d)\).*$',
    re.MULTILINE,
)


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
        self.read_bindings = set()
        self.constants = {}
        self.print_binding = None
        self._placeholders = {}
        self._text = None
        self.first_constant_binding = 0
        self.mode = self.narrowed = narrowing.Mode()
        # Workgroup size of the kernel this library holds the entry point of.
        self.local_size = None

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

    @staticmethod
    def _optimize(linked, builder, passes):
        """Inline everything into the entry point and run `passes`.

        Parameters
        ----------
        linked : llvmlite.binding.ModuleRef
            The module.
        builder : llvmlite.binding.PassBuilder
            Pass builder for the target.
        passes : tuple of str
            Names of llvmlite's ``add_<name>_pass`` methods.

        Returns
        -------
        llvmlite.binding.ModuleRef
            The module, changed in place.
        """
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
        manager = llvm.create_new_module_pass_manager()
        for name in passes:
            getattr(manager, f"add_{name}_pass")()
        manager.run(linked, builder)
        return linked

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
        if self.local_size is not None and "numba_vulkan.local_size" in str(linked):
            linked = llvm.parse_assembly(
                _LOCAL_SIZE.sub(
                    lambda m: (
                        f"{m.group(1)} = add i32 0, {self.local_size[int(m.group(2))]}"
                    ),
                    str(linked),
                )
            )
        machine = target_machine()
        pto = llvm.create_pipeline_tuning_options(speed_level=0)
        builder = llvm.create_pass_builder(machine, pto)
        # With 64-bit floats, only the kernel's own integers are narrowed,
        # before libclc is linked: its float64 functions need 64-bit integers.
        early = self.mode.ints and not self.mode.floats
        user_i64 = False
        if early:
            linked = self._optimize(linked, builder, EARLY_PASSES)
            text = str(linked)
            user_i64 = re.search(r"\bi64\b", text) is not None
            ints = narrowing.Mode(ints=True)
            linked = llvm.parse_assembly(narrowing.narrow_ir(text, ints))
        linked = _link_libclc(linked, builder)
        linked.triple = TRIPLE
        linked = self._optimize(linked, builder, PASSES)
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
        text = str(linked)
        # Workarounds for the device (see numba_vulkan.probes). The float64
        # ones are left out when doubles are narrowed to float anyway.
        if self.mode.soft_fma:
            types = ("float",) if self.mode.floats else ("float", "double")
            text = emulate_fma(text, types)
        if self.mode.soft_rounding and not self.mode.floats:
            text = emulate_rounding64(text)
        if early:
            self.narrowed = narrowing.Mode(ints=user_i64)
            text = legalize(text)
        else:
            self.narrowed = narrowing.Mode(
                self.mode.floats and re.search(r"\bdouble\b", text) is not None,
                self.mode.ints and re.search(r"\bi64\b", text) is not None,
            )  # only the narrowing of types is recorded
            text = legalize(narrowing.narrow_ir(text, self.mode), self.mode.ints)
            # Once more: some of the rewrites above introduce 64-bit indices.
            text = narrowing.narrow_ir(text, self.mode)
        missing = _UNDEFINED_LIBCLC.findall(text)
        if missing:
            raise VulkanUnsupportedError(
                "libclc needs helper functions that this target does not provide: "
                + ", ".join(sorted(set(missing)))
            )
        text = structurize(text)
        text, self._placeholders = renumber_constants(
            text, self.first_constant_binding, constant_order(self._source())
        )
        self._bind_constants()
        binding = self.first_constant_binding + len(self.constants)
        text, prints = renumber_print(text, binding)
        self.print_binding = binding if prints else None
        if constant_order(text):
            # A placeholder binding in a shader crashes drivers.
            raise SpirvCodegenError("a constant array was not given a binding")
        text, self.written_bindings, self.read_bindings = expand_buffer_access(text)
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
        if self._text is None:
            self._text = str(self._link_and_optimize())
        return self._text

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
        if exact in self._spirv:
            return self._spirv[exact]
        source = name = None
        if kernelcache.enabled():
            source = self._source()
            order = constant_order(source)
            # Constant arrays are numbered per process; name them by position.
            canonical = renumber_constants(source, 0, order)[0]
            name = kernelcache.key(
                kernelcache.normalise(canonical),
                exact,
                tuple(self.mode),
                self.first_constant_binding,
                PASSES,
            )
            entry = kernelcache.load(name)
            if entry is not None:
                check_spirv(entry["spirv"])
                self._text = entry["llvm_ir"]
                self.written_bindings = set(entry["written"])
                self.read_bindings = set(entry["read"])
                self.narrowed = narrowing.Mode(*entry["narrowed"])
                self._placeholders = {
                    int(binding): order[slot]
                    for binding, slot in entry["constants"].items()
                }
                self.print_binding = entry.get("print_binding")
                self._bind_constants()
                self._spirv[exact] = entry["spirv"]
                return entry["spirv"]
        spirv = emit_spirv(self.get_optimized_llvm_str(), exact, self.mode.ints)
        self._spirv[exact] = spirv
        if name is not None:
            order = constant_order(source)
            kernelcache.store(
                name,
                {
                    "spirv": spirv,
                    "llvm_ir": self._text,
                    "written": sorted(self.written_bindings),
                    "read": sorted(self.read_bindings),
                    "narrowed": list(self.narrowed),
                    "constants": {
                        binding: order.index(placeholder)
                        for binding, placeholder in self._placeholders.items()
                    },
                    "print_binding": self.print_binding,
                },
            )
        return spirv

    def _source(self):
        """Unoptimised LLVM IR of all modules that make up the kernel.

        Returns
        -------
        str
        """
        return "\n".join(
            str(module)
            for library in self._all_libraries()
            for module in library._modules
        )

    def _bind_constants(self):
        """Look up the data of the kernel's constant arrays.

        They hold the element type the kernel reads, which differs from
        that of the array when the kernel is narrowed.
        """
        self.constants = {}
        for binding, placeholder in self._placeholders.items():
            data = constant_data(placeholder)
            stored = narrowing.stored_dtype(data.dtype, self.mode)
            self.constants[binding] = np.ascontiguousarray(data, dtype=stored)


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

    def magic_tuple(self):
        """Values identifying this code generator for Numba's caching.

        Returns
        -------
        tuple
        """
        return (TRIPLE,)
