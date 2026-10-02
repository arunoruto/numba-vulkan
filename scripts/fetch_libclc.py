"""Download libclc's clspv bitcode for bundling it into a wheel.

numba-vulkan links libclc into kernels for accurate and float64 math (see
docs/source/math_library.md). Wheels carry a copy, which this script takes
from conda-forge's libclc package: a pinned version, verified by its
SHA-256, so that every build bundles the same file. It needs only the
standard library (Python 3.14 brings zstd; older versions need the
``zstandard`` package).

    python scripts/fetch_libclc.py              # into src/numba_vulkan/data/
    python scripts/fetch_libclc.py --output x.bc

The LLVM inside llvmlite must be at least as new as the one libclc was built
with: llvmlite 0.50 has LLVM 22.
"""

import argparse
import hashlib
import io
import pathlib
import sys
import tarfile
import urllib.request
import zipfile

VERSION = "22.1.8"
URL = (
    "https://conda.anaconda.org/conda-forge/linux-64/"
    f"libclc-{VERSION}-h0f8336f_0.conda"
)
SHA256 = "e7ff3677226a3307bf5713f0b4cc58af09be57ad674e231b0be5b87c114dcfab"
MEMBER = "share/clc/clspv--.bc"
DEFAULT = pathlib.Path(__file__).parent.parent / "src" / "numba_vulkan" / "data" / "clspv--.bc"


def _zstd_decompress(data):
    """Decompress zstd data with whatever is available."""
    try:
        from compression import zstd  # Python 3.14+
    except ImportError:
        try:
            import zstandard
        except ImportError:
            sys.exit("needs Python 3.14 or the 'zstandard' package to unpack the download")
        return zstandard.ZstdDecompressor().decompressobj().decompress(data)
    return zstd.decompress(data)


def fetch():
    """Download the package and return the bitcode.

    Returns
    -------
    bytes
    """
    with urllib.request.urlopen(URL) as response:
        package = response.read()
    digest = hashlib.sha256(package).hexdigest()
    if digest != SHA256:
        sys.exit(f"checksum mismatch for {URL}: {digest}")
    archive = zipfile.ZipFile(io.BytesIO(package))
    (inner,) = [n for n in archive.namelist() if n.startswith("pkg-")]
    with tarfile.open(fileobj=io.BytesIO(_zstd_decompress(archive.read(inner)))) as tar:
        return tar.extractfile(MEMBER).read()


def main(argv=None):
    """Write the bitcode, unless it is there already."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT)
    parser.add_argument("--force", action="store_true", help="download again")
    opts = parser.parse_args(argv)
    if opts.output.exists() and not opts.force:
        print(f"{opts.output} exists")
        return
    data = fetch()
    if not data.startswith(b"BC\xc0\xde"):
        sys.exit("the download does not contain LLVM bitcode")
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    opts.output.write_bytes(data)
    print(f"wrote libclc {VERSION} ({len(data):,} bytes) to {opts.output}")


if __name__ == "__main__":
    main()
