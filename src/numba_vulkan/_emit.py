"""Child process that turns LLVM IR (stdin) into SPIR-V (stdout).

LLVM's SPIR-V backend aborts the whole process on input it cannot handle, so
it is kept out of the user's interpreter.
"""

import sys

import llvmlite.binding as llvm


def main():
    """Translate the LLVM IR on standard input to SPIR-V on standard output.

    The target triple is taken from the first command line argument. On
    failure LLVM writes a message to standard error and aborts, which the
    parent process turns into a `SpirvCodegenError`.
    """
    triple = sys.argv[1]
    llvm.initialize_all_targets()
    llvm.initialize_all_asmprinters()
    module = llvm.parse_assembly(sys.stdin.read())
    machine = llvm.Target.from_triple(triple).create_target_machine(opt=0)
    sys.stdout.buffer.write(machine.emit_object(module))


if __name__ == "__main__":
    main()
