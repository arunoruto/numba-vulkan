"""Child process that turns LLVM IR into SPIR-V.

LLVM's SPIR-V backend aborts the whole process on input it cannot handle,
and it keeps state between modules that corrupts the second module
translated by one process. It therefore runs outside the user's
interpreter, and in a process of its own for every module.

Starting Python and importing llvmlite takes much longer than the
translation, so this process does that once and then forks for every
request, where the platform allows it. It serves requests until its
standard input is closed. A request is the length of the IR as an unsigned
64-bit little-endian integer followed by the IR; the reply is framed in the
same way, with the length `FAILED` and no data if the backend failed. Its
message is then on standard error. See `numba_vulkan.codegen.Emitter`.
"""

import os
import struct
import sys

import llvmlite.binding as llvm

HEADER = struct.Struct("<Q")
FAILED = 2**64 - 1
CAN_FORK = hasattr(os, "fork")


def _read(stream, count):
    """Read exactly `count` bytes, or return ``None`` at end of input."""
    data = b""
    while len(data) < count:
        chunk = stream.read(count - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def _translate(target, text, stdout):
    """Translate one module and write the reply."""
    module = llvm.parse_assembly(text.decode())
    spirv = target.create_target_machine(opt=0).emit_object(module)
    stdout.write(HEADER.pack(len(spirv)) + spirv)
    stdout.flush()


def main():
    """Serve translation requests on standard input and output.

    The target triple is taken from the first command line argument.
    Without ``fork``, the process translates a single module itself and
    exits.
    """
    triple = sys.argv[1]
    # Lets the backend use float atomics; they are emitted only for devices
    # that support them (see numba_vulkan.narrowing.Mode).
    llvm.set_option("", "--spirv-ext=+SPV_EXT_shader_atomic_float_add")
    llvm.initialize_all_targets()
    llvm.initialize_all_asmprinters()
    target = llvm.Target.from_triple(triple)
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        header = _read(stdin, HEADER.size)
        if header is None:
            return
        text = _read(stdin, HEADER.unpack(header)[0])
        if text is None:
            return
        if not CAN_FORK:
            _translate(target, text, stdout)
            return
        pid = os.fork()
        if pid == 0:
            status = 1
            try:
                _translate(target, text, stdout)
                status = 0
            except BaseException as exc:
                print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            finally:
                os._exit(status)
        if os.waitpid(pid, 0)[1] != 0:
            stdout.write(HEADER.pack(FAILED))
            stdout.flush()


if __name__ == "__main__":
    main()
