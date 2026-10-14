"""The child process that runs LLVM's SPIR-V backend, and the libclc cache."""

import os
import shutil
import subprocess
import sys

import pytest

from numba_vulkan import _emit, codegen, libclc
from numba_vulkan.errors import SpirvCodegenError

# Without fork (Windows) there is no long-lived child to test.
needs_fork = pytest.mark.skipif(not _emit.CAN_FORK, reason="needs os.fork")

SHADER = f"""
target triple = "{codegen.TRIPLE}"
define void @main() #0 {{
  ret void
}}
attributes #0 = {{ "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }}
"""


@needs_fork
def test_one_process_serves_several_modules():
    emitter = codegen.Emitter()
    try:
        first = emitter.emit(SHADER)
        pid = emitter._proc.pid
        assert first[:4] == b"\x03\x02\x23\x07"  # SPIR-V magic number
        assert emitter.emit(SHADER) == first
        assert emitter._proc.pid == pid
    finally:
        emitter.close()


@needs_fork
def test_failure_is_reported_and_the_process_keeps_serving():
    emitter = codegen.Emitter()
    try:
        emitter.emit(SHADER)
        pid = emitter._proc.pid
        with pytest.raises(SpirvCodegenError, match="SPIR-V backend failed: .+"):
            emitter.emit("this is not LLVM IR")
        assert emitter.emit(SHADER)[:4] == b"\x03\x02\x23\x07"
        assert emitter._proc.pid == pid
    finally:
        emitter.close()


@needs_fork
def test_dead_process_is_replaced():
    emitter = codegen.Emitter()
    try:
        emitter.emit(SHADER)
        emitter._proc.kill()
        emitter._proc.wait()
        assert emitter.emit(SHADER)[:4] == b"\x03\x02\x23\x07"
    finally:
        emitter.close()


def test_hanging_process_is_killed(monkeypatch):
    monkeypatch.setattr(codegen, "EMIT_TIMEOUT", 0)
    emitter = codegen.Emitter()
    with pytest.raises(SpirvCodegenError, match="did not finish within 0 seconds"):
        emitter.emit(SHADER)
    assert emitter._proc is None


@needs_fork
def test_warm_up_and_close():
    emitter = codegen.Emitter()
    emitter.warm_up()
    pid = emitter._proc.pid
    emitter.warm_up()  # already running
    assert emitter._proc.pid == pid
    emitter.close()
    assert emitter._proc is None
    emitter.close()


LLC = os.environ.get(codegen.LLC_ENV_VAR) or shutil.which("llc")


@pytest.mark.skipif(LLC is None, reason="llc is not installed")
def test_llc_translates_and_reports_failures():
    emitter = codegen.Emitter(llc=LLC)
    emitter.warm_up()  # nothing to start
    assert emitter.emit(SHADER)[:4] == b"\x03\x02\x23\x07"
    with pytest.raises(SpirvCodegenError, match="SPIR-V backend failed: .+"):
        emitter.emit("this is not LLVM IR")
    assert emitter._proc is None


@pytest.mark.skipif(sys.platform == "win32", reason="a shell script as llc")
def test_crashing_llc_is_reported(tmp_path):
    llc = tmp_path / "llc"
    llc.write_text('#!/bin/sh\necho "#0 0x0 stack frame" >&2\nkill -SEGV $$\n')
    llc.chmod(0o755)
    with pytest.raises(SpirvCodegenError, match="llc was killed by SIGSEGV"):
        codegen.Emitter(llc=str(llc)).emit(SHADER)


@pytest.mark.skipif(not libclc.available(), reason="libclc is not installed")
def test_prepared_libclc_is_cached_on_disk(tmp_path, monkeypatch):
    monkeypatch.setenv(libclc.CACHE_ENV_VAR, str(tmp_path))
    libclc.prepared_bitcode.cache_clear()
    try:
        prepared = libclc.prepared_bitcode()
        (cached,) = os.listdir(tmp_path)
        assert cached.startswith("libclc-") and cached.endswith(".bc")
        assert (tmp_path / cached).read_bytes() == prepared
        # a second process would read the file instead of preparing again
        libclc.prepared_bitcode.cache_clear()
        monkeypatch.setattr(libclc, "_prepare", None)
        assert libclc.prepared_bitcode() == prepared
    finally:
        libclc.prepared_bitcode.cache_clear()


@pytest.mark.skipif(not libclc.available(), reason="libclc is not installed")
def test_prepared_libclc_works_without_a_writable_cache(monkeypatch):
    monkeypatch.setenv(libclc.CACHE_ENV_VAR, "/proc/numba-vulkan-cannot-write-here")
    libclc.prepared_bitcode.cache_clear()
    try:
        text = str(codegen.llvm.parse_bitcode(libclc.prepared_bitcode()))
        assert "define linkonce_odr" in text and "noinline" not in text
        assert "addrspace(2)" not in text
    finally:
        libclc.prepared_bitcode.cache_clear()


def test_failures_are_saved_for_reproducing(tmp_path, monkeypatch):
    monkeypatch.setenv("NUMBA_VULKAN_CACHE_DIR", str(tmp_path))
    with pytest.raises(SpirvCodegenError) as info:
        codegen.emit_spirv("this is not LLVM IR")
    (note,) = info.value.__notes__
    assert note.startswith("to reproduce: ")
    saved = tmp_path / "failures"
    (ll,) = saved.glob("*.ll")
    assert ll.read_text() == "this is not LLVM IR"
    assert str(ll) in note


def test_translation_as_on_windows(tmp_path):
    """llvmlite appends "-elf" to the triple on Windows unless asked not to
    (issue #53); the backend then aborts. Pretends to be Windows to llvmlite
    only, in a child process, since the backend may abort it."""
    script = tmp_path / "windows.py"
    script.write_text(
        "import io, sys, types\n"
        "import llvmlite.binding as llvm\n"
        "import llvmlite.binding.targets as targets\n"
        "targets.os = types.SimpleNamespace(name='nt')\n"
        f"sys.path.insert(0, {os.path.dirname(codegen.__file__)!r})\n"
        "import _emit\n"
        "llvm.initialize_all_targets()\n"
        "llvm.initialize_all_asmprinters()\n"
        f"target = llvm.Target.from_triple({codegen.TRIPLE!r})\n"
        "out = io.BytesIO()\n"
        "_emit._translate(target, sys.stdin.buffer.read(), out)\n"
        "print(len(out.getvalue()))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-I", str(script)],
        input=SHADER.encode(),
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    assert int(proc.stdout) > 20


@pytest.mark.skipif(shutil.which("spirv-val") is None, reason="needs spirv-val")
def test_spirv_val_rejects_invalid_modules(monkeypatch):
    monkeypatch.setenv("NUMBA_VULKAN_VALIDATE", "1")
    spirv = codegen.emit_spirv(SHADER)
    codegen.check_spirv(spirv)
    # SPIR-V 1.6 is too new for Vulkan 1.2.
    newer = spirv[:4] + (0x00010600).to_bytes(4, "little") + spirv[8:]
    with pytest.raises(SpirvCodegenError, match="invalid"):
        codegen.check_spirv(newer)


def test_translation_without_fork(monkeypatch):
    """Where there is no fork (Windows), each module gets a process of its own;
    reading its reply must not use select, which works only on sockets there."""
    monkeypatch.setattr(codegen._emit, "CAN_FORK", False)
    monkeypatch.setattr(
        codegen.select, "select", lambda *a: pytest.fail("select on a pipe")
    )
    emitter = codegen.Emitter()
    emitter.warm_up()  # starts nothing
    assert emitter._proc is None
    assert emitter.emit(SHADER)[:4] == b"\x03\x02\x23\x07"
    with pytest.raises(SpirvCodegenError, match="SPIR-V backend failed: .+"):
        emitter.emit("this is not LLVM IR")
