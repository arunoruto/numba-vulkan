# Building and checking numba-vulkan without Nix. Needs uv
# (https://docs.astral.sh/uv/); Vulkan drivers only for `test`.
#
#   make dist     sdist and wheel with libclc bundled, in dist/
#   make check    install the wheel in a clean environment and run a kernel
#   make test     the test suite
#   make docs     the documentation, in docs/build/html
#   make clean

UV ?= uv
LIBCLC := src/numba_vulkan/data/clspv--.bc

.PHONY: dist check test docs clean libclc

libclc: $(LIBCLC)

$(LIBCLC):
	$(UV) run --no-project --with zstandard python scripts/fetch_libclc.py --output $@

dist: $(LIBCLC)
	rm -rf dist
	$(UV) build
	$(UV) run --no-project python scripts/check_dist.py dist

check: dist
	NUMBA_VULKAN_LIBCLC= $(UV) run --isolated --no-project --with dist/*.whl python scripts/smoke_test.py

test: $(LIBCLC)
	NUMBA_VULKAN_LIBCLC=$(LIBCLC) $(UV) run pytest

docs:
	$(UV) run --group docs sphinx-build -W --keep-going -b html docs/source docs/build/html

clean:
	rm -rf dist docs/build $(LIBCLC)
