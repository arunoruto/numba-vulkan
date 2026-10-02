"""Get libclc's clspv bitcode for bundling it into a wheel.

numba-vulkan links libclc into kernels for accurate and float64 math (see
docs/source/math_library.md). Wheels carry a copy, together with a
``libclc-version.txt`` that says which one. It comes from

``nix`` (the default where Nix is installed)
    the ``nixpkgs-libclc`` input of devenv.lock, the same package the
    development shell and the tests use. Its version is whatever that
    nixpkgs revision has; moving the input updates it, and reverting the
    lock file goes back.
``conda``
    conda-forge's libclc package, for building without Nix. A plain
    download, so it is pinned by URL and SHA-256 and must be moved by hand
    to follow the Nix version. It needs only Python (3.14 brings zstd;
    older versions need the ``zstandard`` package).

    python scripts/fetch_libclc.py                  # into src/numba_vulkan/data/
    python scripts/fetch_libclc.py --source conda --output x.bc

The LLVM inside llvmlite must be at least as new as the one libclc was built
with: llvmlite 0.50 has LLVM 22, hence ``llvmPackages_22``.
"""

import argparse
import hashlib
import io
import json
import pathlib
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

LLVM = "22"
ROOT = pathlib.Path(__file__).parent.parent
CONDA_VERSION = "22.1.8"
CONDA_PACKAGE = f"libclc-{CONDA_VERSION}-h0f8336f_0"
URL = f"https://conda.anaconda.org/conda-forge/linux-64/{CONDA_PACKAGE}.conda"
SHA256 = "e7ff3677226a3307bf5713f0b4cc58af09be57ad674e231b0be5b87c114dcfab"
MEMBER = "share/clc/clspv--.bc"
DEFAULT = ROOT / "src" / "numba_vulkan" / "data" / "clspv--.bc"
VERSION_FILE = "libclc-version.txt"


def _zstd_decompress(data):
    """Decompress zstd data with whatever is available."""
    try:
        from compression import zstd  # Python 3.14+
    except ImportError:
        try:
            import zstandard
        except ImportError:
            sys.exit(
                "needs Python 3.14 or the 'zstandard' package to unpack the download"
            )
        return zstandard.ZstdDecompressor().decompressobj().decompress(data)
    return zstd.decompress(data)


def _nix(*args):
    """Run a ``nix`` command and return its standard output."""
    return subprocess.run(
        ["nix", "--extra-experimental-features", "nix-command flakes", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def from_nix():
    """Fetch libclc with Nix, from the revision devenv.lock pins.

    Returns
    -------
    data : bytes
        The bitcode.
    version : str
        Description of the version, for ``libclc-version.txt``.
    """
    lock = json.loads((ROOT / "devenv.lock").read_text())
    rev = lock["nodes"]["nixpkgs-libclc"]["locked"]["rev"]
    reference = f"github:NixOS/nixpkgs/{rev}#llvmPackages_{LLVM}.libclc"
    version = _nix("eval", "--raw", f"{reference}.version")
    out = _nix("build", "--no-link", "--print-out-paths", reference).split()[0]
    if version != CONDA_VERSION:
        print(
            f"note: nixpkgs has libclc {version}, the conda-forge fallback "
            f"{CONDA_VERSION}; consider moving the fallback"
        )
    data = (pathlib.Path(out) / MEMBER).read_bytes()
    return data, f"{version} (nixpkgs {rev})"


def from_conda():
    """Download libclc from conda-forge.

    Returns
    -------
    data : bytes
        The bitcode.
    version : str
        Description of the version, for ``libclc-version.txt``.
    """
    with urllib.request.urlopen(URL) as response:
        package = response.read()
    digest = hashlib.sha256(package).hexdigest()
    if digest != SHA256:
        sys.exit(f"checksum mismatch for {URL}: {digest}")
    archive = zipfile.ZipFile(io.BytesIO(package))
    (inner,) = [n for n in archive.namelist() if n.startswith("pkg-")]
    with tarfile.open(fileobj=io.BytesIO(_zstd_decompress(archive.read(inner)))) as tar:
        data = tar.extractfile(MEMBER).read()
    return data, f"{CONDA_VERSION} (conda-forge {CONDA_PACKAGE})"


def main(argv=None):
    """Write the bitcode and its version file, unless they are there."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT)
    parser.add_argument("--force", action="store_true", help="fetch again")
    parser.add_argument(
        "--source",
        choices=("auto", "nix", "conda"),
        default="auto",
        help="where to take libclc from; auto: Nix if it is installed",
    )
    opts = parser.parse_args(argv)
    version_file = opts.output.parent / VERSION_FILE
    if opts.output.exists() and version_file.exists() and not opts.force:
        print(f"{opts.output} exists")
        return
    source = opts.source
    if source == "auto":
        source = "nix" if shutil.which("nix") else "conda"
    data, version = from_nix() if source == "nix" else from_conda()
    if not data.startswith(b"BC\xc0\xde"):
        sys.exit(f"libclc from {source} is not LLVM bitcode")
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    opts.output.write_bytes(data)
    version_file.write_text(version + "\n")
    print(f"wrote libclc {version} ({len(data):,} bytes) to {opts.output}")


if __name__ == "__main__":
    main()
