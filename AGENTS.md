# pytorch_wavelets - guide for agents

This file is the reference for this repository. It takes precedence over any
`AGENTS.md` further up the directory tree, which describes other projects.

An internal fork of [fbcotter/pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets):
DWT, SWT, DTCWT and a DTCWT ScatterNet in PyTorch, with custom autograd
Functions for the gradients. Not published on PyPI.

## Layout

- `pytorch_wavelets/` - the package: `dwt/` (DWT, 1D DWT, SWT), `dtcwt/`,
  `scatternet/`, `utils.py`. `dtcwt/data/*.npz` hold the filter coefficients.
- `tests/` - pytest suite; `conftest.py`, `Transform2d_np.py` (numpy reference)
  and `datasets.py` (+ `*.npz` images) support it.
- `examples/` - sphinx-gallery scripts, executed by the docs build.
- `notebooks/` - **generated** from `examples/`; never edit by hand.
- `docs/` - Sphinx. `benchmarks/` - profiling scripts, not tests.
- `tools/` - `make_notebooks.py` and the git hook under `tools/hooks/`.

## Environment and commands

Everything runs in a conda/mamba environment the `Makefile` builds (not uv):

```bash
make dev              # create env (environment.yml) + editable install
make test             # default suite; slow gradchecks excluded
make test-slow        # full-size gradchecks, ~30 min
make test-cov         # with coverage
make lint             # flake8 over package, tests, examples, benchmarks, tools
make docs             # sphinx-build; CI uses -W, warnings are errors
make notebooks        # regenerate notebooks/ after changing examples/
make check-notebooks
make hooks            # install the pre-commit hook for notebooks/
```

Without make: `pip install ".[test]"` then `pytest`. The quality gate is lint,
tests, a warning-free docs build and `make check-notebooks`.

## Conventions

- Tests: small, seeded, deterministic. Gradients are checked with
  `torch.autograd.gradcheck` in float64; checks taking minutes get
  `@pytest.mark.slow`, and the default run must stay fast
  (`test_gradients*.py` show the small-input style). Every bug fix gets a
  regression test that fails on the old code.
- The `dtcwt` reference package pins numpy<2; `tests/conftest.py` leaves out
  the files that need it when it is absent. The library itself must work
  under NumPy 2 (a CI job checks it).
- Style: flake8 with E226/E231 ignored (tight index spacing is the house
  style); match the surrounding code and its comment density.
- Changes that users would notice go in the README's "New in version 1.4.0"
  section - it serves as the changelog.
- Commits: `type: summary` (`fix`, `ci`, `chore`, `docs`, `build`, `tools`),
  with a body explaining the why. Branches: `fix/*`, `ci/*`, `chore/*`, merged
  by pull request with a merge commit (currently into `modernise-library`).
- No secrets or personal data in the repository.
