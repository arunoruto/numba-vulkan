# AGENTS.md

Guidance for AI agents (and humans) working on numba-vulkan.

## What this is

A proof-of-concept Vulkan compute target for Numba. Python kernels go through
Numba's pipeline to LLVM IR, then through LLVM's SPIR-V backend, and run as
Vulkan compute shaders.

## Read first

1. `docs/source/how_it_works.md`: the architecture and why it looks this way.
2. `docs/source/development.md`: layout, commands, debugging, and a list of
   rules learned the hard way. Follow those rules.
3. `docs/source/known_issues.md`: what is broken, why, and where to fix it.

## Commands

Run everything inside the devenv shell (required on NixOS):

```sh
devenv shell -- uv run pytest
devenv shell -- uv run pytest tests/test_known_issues.py -rxX
devenv shell -- uv run python benchmarks/bench.py
devenv shell -- bash -c 'cd docs && uv run sphinx-build -M html ./source ./build -W'
```

## Expectations

- Tests run on every Vulkan device present. A change is not done until the
  whole suite passes on all of them.
- When you fix a known issue, move its case out of
  `tests/test_known_issues.py` into a regular test and remove the entry from
  `docs/source/known_issues.md`. When you find a new one, add both.
- Public and private functions carry NumPy-style docstrings; keep that up.
- Do not state that something works unless a test or a run shows it.
- Never hand unvalidated SPIR-V to a GPU driver, and never run LLVM's SPIR-V
  backend in the main process. Both can kill the interpreter.

## Authorship

This project was written by an AI under human direction and review; see
`docs/source/authorship.md`. Keep the disclosure accurate: note AI-written
contributions as such, and do not merge them without human review. Numba's
[AI tools policy](https://numba.readthedocs.io/en/stable/reference/ai_tools_policy.html)
applies to anything proposed upstream.
