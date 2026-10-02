"""Check that built distributions carry libclc and its licence.

    python scripts/check_dist.py dist
"""

import pathlib
import sys
import tarfile
import zipfile

REQUIRED = ("numba_vulkan/data/clspv--.bc", "numba_vulkan/data/LICENSE-libclc.txt")


def main(folder):
    """Exit with an error if a distribution lacks a required file."""
    found = list(pathlib.Path(folder).iterdir())
    if not any(p.suffix == ".whl" for p in found):
        sys.exit(f"no wheel in {folder}")
    for path in found:
        if path.suffix == ".whl":
            names = zipfile.ZipFile(path).namelist()
        elif path.name.endswith(".tar.gz"):
            names = tarfile.open(path).getnames()
        else:
            continue
        for required in REQUIRED:
            if not any(n.endswith(required) for n in names):
                sys.exit(f"{path.name} lacks {required}")
        print(f"{path.name}: ok")


if __name__ == "__main__":
    main(sys.argv[1])
