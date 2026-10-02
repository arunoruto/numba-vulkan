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
"""

import hashlib
import json
import os
import re
import tempfile
import zipfile
from functools import lru_cache

import llvmlite

from numba_vulkan import libclc

ENV_VAR = "NUMBA_VULKAN_CACHE"
# Changes whenever the layout of an entry does.
_FORMAT = "1"
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
    it of LLVM), and the libclc file in use.

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
    bitcode = libclc.find_bitcode()
    if bitcode is not None:
        stat = os.stat(bitcode)
        digest.update(f"{bitcode} {stat.st_size} {stat.st_mtime_ns}".encode())
    return digest.hexdigest()


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
        with tempfile.NamedTemporaryFile(dir=os.path.dirname(path), delete=False) as fh:
            with zipfile.ZipFile(fh, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("meta.json", json.dumps(meta))
                archive.writestr("kernel.spv", entry["spirv"])
                archive.writestr("kernel.ll", entry["llvm_ir"])
        os.replace(fh.name, path)
    except OSError:
        pass
