"""Access to libclc, LLVM's implementation of the OpenCL math library.

Vulkan's own math library (GLSL.std.450) is small and 32-bit only. libclc
provides the missing functions, and double precision versions of all of
them, as LLVM bitcode. Its ``clspv`` build is written for Vulkan shaders,
which makes it this target's counterpart of CUDA's libdevice.

Wheels bundle the bitcode; see `find_bitcode` for where it is looked for
and `version` for which one is in use.
"""

import hashlib
import os
import re
import tempfile
from functools import lru_cache

import llvmlite.binding as llvm
from llvmlite import ir

from numba_vulkan import narrowing

CALLING_CONVENTION = "spir_func"
ENV_VAR = "NUMBA_VULKAN_LIBCLC"
CACHE_ENV_VAR = "NUMBA_VULKAN_CACHE_DIR"
# Changes whenever `_prepare` does, so that stale cache files are ignored.
_PREPARE_VERSION = b"1"
_AS_OPENCL_CONSTANT, _AS_PRIVATE = "addrspace(2)", "addrspace(10)"
# Function definitions without an explicit linkage.
_EXTERNAL_DEFINITION = re.compile(
    r"^define (?!(?:private|internal|available_externally|linkonce|linkonce_odr"
    r"|weak|weak_odr|common|appending|extern_weak|external) )",
    re.MULTILINE,
)
_FILE = "clspv--.bc"
_VERSION_FILE = "libclc-version.txt"
# Nix store paths name the package: /nix/store/<hash>-libclc-22.1.8/...,
# or libclc-clspv-22.1.8 for this project's own build (nix/libclc.nix).
_STORE_VERSION = re.compile(r"-libclc(?:-clspv)?-(\d[\w.]*)/")
_SEARCH = (
    os.path.join(os.path.dirname(__file__), "data"),
    "/usr/share/clc",
    "/usr/lib/clc",
    "/usr/lib64/clc",
    "/usr/local/share/clc",
    "/run/current-system/sw/share/clc",
)
_TYPE_CODES = {"float": "f", "double": "d", "i32": "i", "i64": "l"}


@lru_cache
def find_bitcode():
    """Locate libclc's ``clspv--.bc``.

    The ``NUMBA_VULKAN_LIBCLC`` environment variable takes precedence and
    may name the file or the directory containing it. Otherwise the
    package's own ``data`` directory and the usual system locations are
    searched.

    Returns
    -------
    str or None
        Path of the bitcode file, or ``None`` if libclc is not installed.
    """
    candidates = []
    configured = os.environ.get(ENV_VAR)
    if configured:
        candidates += [configured, os.path.join(configured, _FILE)]
    candidates += [os.path.join(directory, _FILE) for directory in _SEARCH]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def available():
    """Whether libclc can be used.

    Returns
    -------
    bool
    """
    return find_bitcode() is not None


def version():
    """Describe the libclc in use.

    The bundled copy carries a ``libclc-version.txt``, written when the
    wheel was built. A copy from the Nix store is identified by its path.

    Returns
    -------
    str or None
        For example ``"22.1.8 (nixpkgs 774debe...)"``; ``None`` if libclc
        is missing or its version is unknown.

    Examples
    --------
    >>> from numba_vulkan import libclc
    >>> libclc.version()  # doctest: +SKIP
    '22.1.8 (nixpkgs 774debe7a0d1b496e35677ad955a1011c6ff74f3)'
    """
    path = find_bitcode()
    if path is None:
        return None
    sidecar = os.path.join(os.path.dirname(path), _VERSION_FILE)
    if os.path.isfile(sidecar):
        with open(sidecar) as fh:
            return fh.read().strip()
    match = _STORE_VERSION.search(os.path.realpath(path))
    return f"{match.group(1)} (nix)" if match else None


@lru_cache
def read_bitcode():
    """Contents of the bitcode file.

    Returns
    -------
    bytes

    Raises
    ------
    FileNotFoundError
        If libclc is not installed.
    """
    path = find_bitcode()
    if path is None:
        raise FileNotFoundError(f"libclc ({_FILE}) not found; set {ENV_VAR}")
    with open(path, "rb") as fh:
        return fh.read()


def cache_directory():
    """Directory for files derived from libclc.

    ``NUMBA_VULKAN_CACHE_DIR`` if set, otherwise ``numba-vulkan`` below
    ``XDG_CACHE_HOME`` or ``~/.cache``.

    Returns
    -------
    str
    """
    configured = os.environ.get(CACHE_ENV_VAR)
    if configured:
        return configured
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(base, "numba-vulkan")


def _prepare(raw):
    """Rewrite libclc so that it can be linked into shaders cheaply.

    Parameters
    ----------
    raw : bytes
        The bitcode as distributed.

    Returns
    -------
    bytes
        Bitcode in which

        - every function has ``linkonce_odr`` linkage, so that the linker
          only takes the functions a kernel uses instead of all of them;
        - no function is ``noinline`` or ``optnone``, because shaders need
          everything inlined into the entry point;
        - lookup tables are in the Private storage class instead of
          OpenCL's constant address space, which would become the
          UniformConstant storage class that Vulkan does not allow for
          initialised globals.
    """
    text = str(llvm.parse_bitcode(raw))
    text = re.sub(r"\b(noinline|optnone) ", "", text)
    text = text.replace(_AS_OPENCL_CONSTANT, _AS_PRIVATE)
    text = _EXTERNAL_DEFINITION.sub("define linkonce_odr ", text)
    return llvm.parse_assembly(text).as_bitcode()


@lru_cache
def prepared_bitcode():
    """libclc in the form that is linked into kernels; see `_prepare`.

    Preparing takes a few seconds, so the result is kept in
    `cache_directory`, named after a hash of the original file. If that
    directory cannot be written, the work is redone once per process.

    Returns
    -------
    bytes

    Raises
    ------
    FileNotFoundError
        If libclc is not installed.
    """
    raw = read_bitcode()
    digest = hashlib.sha256(_PREPARE_VERSION + raw).hexdigest()[:24]
    path = os.path.join(cache_directory(), f"libclc-{digest}.bc")
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        pass
    prepared = _prepare(raw)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Written under another name first: another process may be reading.
        with tempfile.NamedTemporaryFile(dir=os.path.dirname(path), delete=False) as fh:
            fh.write(prepared)
        os.replace(fh.name, path)
    except OSError:
        pass
    return prepared


def mangle(name, types):
    """Symbol of a libclc function.

    libclc uses the Itanium C++ scheme, for example ``_Z3sind`` for
    ``sin(double)``.

    Parameters
    ----------
    name : str
        OpenCL name of the function.
    types : sequence of llvmlite.ir.Type
        Scalar argument types.

    Returns
    -------
    str
    """
    return f"_Z{len(name)}{name}" + "".join(_TYPE_CODES[str(t)] for t in types)


def call(builder, name, args, restype=None):
    """Emit a call to a libclc function.

    Parameters
    ----------
    builder : llvmlite.ir.IRBuilder
        Builder positioned where the code is emitted.
    name : str
        OpenCL name of the function, for example ``"erf"``.
    args : list of llvmlite.ir.Value
        Scalar arguments.
    restype : llvmlite.ir.Type, optional
        Return type; that of the first argument by default.

    Returns
    -------
    llvmlite.ir.Value
        The result.

    Notes
    -----
    The functions use the ``spir_func`` calling convention. Calling them
    with any other convention is undefined behaviour that LLVM turns into
    unreachable code.
    """
    if narrowing.current.floats and any(
        isinstance(a.type, ir.DoubleType) for a in args
    ):
        # The device has no float64: evaluate in float32. The values around
        # the call become float32 as well when the kernel is narrowed.
        single, double = ir.FloatType(), ir.DoubleType()
        args = [narrowing.to_single(builder, a) for a in args]
        result = call(builder, name, args, single if restype == double else restype)
        return narrowing.to_double(builder, result)
    argtypes = [a.type for a in args]
    symbol = mangle(name, argtypes)
    fn = builder.module.globals.get(symbol)
    if fn is None:
        fnty = ir.FunctionType(restype or argtypes[0], argtypes)
        fn = ir.Function(builder.module, fnty, symbol)
        fn.calling_convention = CALLING_CONVENTION
    return builder.call(fn, args, cconv=CALLING_CONVENTION)
