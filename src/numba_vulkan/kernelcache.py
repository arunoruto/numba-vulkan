"""On-disk cache of compiled kernels.

Turning a kernel's LLVM IR into SPIR-V (linking libclc, optimising,
restructuring, running the backend) is the larger part of compiling it.
The result depends only on the IR that Numba's lowering produces and on
this package, so it is stored under a hash of both and reused by later
processes. Numba's own pipeline still runs every time, which is what
makes the cache safe: a change to a kernel, to a function it calls or to a
global it reads changes the IR and therefore the hash.

Entries live in ``kernels/`` below `numba_vulkan.libclc.cache_directory`
and can be deleted at any time. Setting ``NUMBA_VULKAN_CACHE=0`` turns the
cache off.

Kernels compiled with ``@nv.jit(cache=True)`` are stored whole as well, in
``functions/``, which skips Numba's pipeline too. They are found by their
bytecode, the timestamps of their source files and those of the ``@nv.jit``
functions they call, as with Numba's ``cache=True``; see
`numba_vulkan.dispatcher.VulkanDispatcher.compile`.
"""

import hashlib
import inspect
import json
import os
import pickle
import re
import shutil
import sys
import tempfile
import zipfile
from functools import lru_cache

import llvmlite

from numba_vulkan import libclc

ENV_VAR = "NUMBA_VULKAN_CACHE"
# Changes whenever the layout of an entry does.
_FORMAT = "3"
# Parts of the IR that differ between processes without changing its
# meaning: the counter Numba appends to function names, and module names.
_UID = re.compile(r"B(\d+)v(\d+)(?=B)")
_MODULE_ID = re.compile(r"^; ModuleID = [^\n]*\n", re.MULTILINE)


def enabled():
    """Whether kernels are cached on disk.

    Returns
    -------
    bool
    """
    return os.environ.get(ENV_VAR, "1") != "0"


@lru_cache
def fingerprint():
    """Hash of everything besides a kernel's IR that shapes its SPIR-V.

    That is the source of this package, the version of llvmlite (and with
    it of LLVM), the libclc file in use, and the ``llc`` program if one
    replaces llvmlite's backend.

    Returns
    -------
    str
    """
    digest = hashlib.sha256(f"{_FORMAT} {llvmlite.__version__}".encode())
    package = os.path.dirname(__file__)
    for name in sorted(os.listdir(package)):
        if name.endswith(".py"):
            with open(os.path.join(package, name), "rb") as fh:
                digest.update(name.encode() + fh.read())
    for path in (libclc.find_bitcode(), _llc()):
        if path is not None:
            stat = os.stat(path)
            digest.update(f"{path} {stat.st_size} {stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def _llc():
    """The ``llc`` that replaces llvmlite's backend, if one is configured.

    Returns
    -------
    str or None
        Its resolved path; see `numba_vulkan.codegen.Emitter`.
    """
    from numba_vulkan import codegen  # imports this module

    llc = codegen.emitter.llc
    return None if llc is None else os.path.realpath(shutil.which(llc) or llc)


def normalise(source):
    """Remove what differs between processes from unoptimised LLVM IR.

    Parameters
    ----------
    source : str
        The IR of all modules of a kernel.

    Returns
    -------
    str
        The IR with Numba's per-process function counters renumbered in
        order of appearance and module names removed.
    """
    seen = {}

    def renumber(match):
        """Replace one counter by its position among those seen."""
        number = seen.setdefault(match.group(2), len(seen))
        return f"B0v{number}"

    return _UID.sub(renumber, _MODULE_ID.sub("", source))


def key(source, *settings):
    """Name of the cache entry for a kernel.

    Parameters
    ----------
    source : str
        Unoptimised LLVM IR of all modules of the kernel, after
        `normalise` and with constant arrays numbered canonically.
    *settings
        Everything else that affects code generation, such as the
        narrowing mode.

    Returns
    -------
    str
        A hexadecimal digest.
    """
    digest = hashlib.sha256(fingerprint().encode())
    digest.update(repr(settings).encode())
    digest.update(source.encode())
    return digest.hexdigest()


def _path(name):
    """File of the entry with the given name."""
    return os.path.join(libclc.cache_directory(), "kernels", name[:2], name + ".zip")


def load(name):
    """Read an entry.

    Parameters
    ----------
    name : str
        Name of the entry, from `key`.

    Returns
    -------
    dict or None
        The entry as passed to `store`, or ``None`` if there is none or
        it is damaged.
    """
    try:
        with zipfile.ZipFile(_path(name)) as archive:
            entry = json.loads(archive.read("meta.json"))
            entry["spirv"] = archive.read("kernel.spv")
            entry["llvm_ir"] = archive.read("kernel.ll").decode()
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return None
    if hashlib.sha256(entry["spirv"]).hexdigest() != entry.pop("sha256", None):
        return None
    return entry


def store(name, entry):
    """Write an entry; failures to write are ignored.

    Parameters
    ----------
    name : str
        Name of the entry, from `key`.
    entry : dict
        ``spirv`` (bytes), ``llvm_ir`` (str) and any JSON-serialisable
        metadata.
    """
    meta = {k: v for k, v in entry.items() if k not in ("spirv", "llvm_ir")}
    meta["sha256"] = hashlib.sha256(entry["spirv"]).hexdigest()
    path = _path(name)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Written under another name first: another process may be reading.
        with (
            tempfile.NamedTemporaryFile(dir=os.path.dirname(path), delete=False) as fh,
            zipfile.ZipFile(fh, "w", zipfile.ZIP_DEFLATED) as archive,
        ):
            archive.writestr("meta.json", json.dumps(meta))
            archive.writestr("kernel.spv", entry["spirv"])
            archive.writestr("kernel.ll", entry["llvm_ir"])
        os.replace(fh.name, path)
    except OSError:
        pass


# -- whole kernels (cache=True) ------------------------------------------------


def source_stamp(function):
    """What identifies a Python function's code across processes.

    Parameters
    ----------
    function : function
        A Python function.

    Returns
    -------
    bytes or None
        Its bytecode with the name, size and modification time of its
        source file, or ``None`` if it has no source file (it was defined
        interactively or with ``exec``), and cannot be cached.
    """
    try:
        path = inspect.getsourcefile(function)
    except TypeError:
        return None
    if not path or not os.path.isfile(path):
        return None
    stat = os.stat(path)
    head = f"{path} {stat.st_size} {stat.st_mtime_ns} {function.__qualname__}"
    return head.encode() + _code_digest(function.__code__)


def _code_digest(code):
    """A digest of a code object that is the same in every process.

    ``marshal`` output is not: it marks objects by their reference counts.

    Returns
    -------
    bytes
    """
    digest = hashlib.sha256(code.co_code)
    for part in (code.co_names, code.co_varnames, code.co_freevars):
        digest.update(repr(part).encode())
    for const in code.co_consts:
        if inspect.iscode(const):
            digest.update(_code_digest(const))
        else:
            digest.update(f"{type(const).__name__}:{const!r}".encode())
    return digest.digest()


def kernel_key(stamps, *settings):
    """Name of the cache entry for a compiled kernel.

    Parameters
    ----------
    stamps : list of bytes
        `source_stamp` of the kernel and of every function it calls.
    *settings
        Argument types, grid, narrowing mode and options.

    Returns
    -------
    str
        A hexadecimal digest.
    """
    digest = hashlib.sha256(f"{fingerprint()} {sys.version}".encode())
    for stamp in stamps:
        digest.update(stamp)
    digest.update(repr(settings).encode())
    return digest.hexdigest()


def _kernel_path(name):
    """File of a cached kernel."""
    return os.path.join(
        libclc.cache_directory(), "functions", name[:2], name + ".pickle"
    )


def load_kernel(name):
    """Read a cached kernel.

    Returns
    -------
    dict or None
        As passed to `store_kernel`, or ``None`` if there is none or it
        cannot be read.
    """
    try:
        with open(_kernel_path(name), "rb") as fh:
            entry = pickle.load(fh)  # written by store_kernel
    except Exception:  # noqa: BLE001 - any damage means: compile again
        return None
    return entry if isinstance(entry, dict) else None


def store_kernel(name, entry):
    """Write a cached kernel; failures to write are ignored.

    Parameters
    ----------
    name : str
        From `kernel_key`.
    entry : dict
        The compiled kernel and what it needs from the compiling process.
    """
    path = _kernel_path(name)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=os.path.dirname(path), delete=False) as fh:
            pickle.dump(entry, fh)
        os.replace(fh.name, path)
    except (OSError, pickle.PicklingError, TypeError, AttributeError):
        pass
