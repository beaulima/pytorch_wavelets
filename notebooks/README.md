# Notebooks

These are generated from the gallery scripts in [`examples/`](../examples), and
are not edited by hand. Each one is the same demo as the corresponding page of
the [documentation](https://pytorch-wavelets.readthedocs.io/), minus the
figures - run the notebook and they appear.

| Notebook | What it covers |
|---|---|
| [`plot_dwt_basics.ipynb`](plot_dwt_basics.ipynb) | Forward and inverse, coefficient layout, padding schemes, batches and GPU |
| [`plot_directional_selectivity.ipynb`](plot_directional_selectivity.ipynb) | Why the DTCWT's six orientations separate what the DWT's three cannot |
| [`plot_shift_invariance.ipynb`](plot_shift_invariance.ipynb) | What a one pixel shift does to each transform, and what it costs to avoid |
| [`plot_denoising.ipynb`](plot_denoising.ipynb) | Soft thresholding, and why the undecimated transform denoises better |
| [`plot_gradients.ipynb`](plot_gradients.ipynb) | Optimising an image through the transform, and checking the gradients |

## Running them

From a clone of the repository, using the conda environment the Makefile
builds:

```bash
make dev        # create the environment and install the package into it
make kernel     # register it with Jupyter
jupyter notebook notebooks/
```

These notebooks ask for a kernel named `pytorch_wavelets`, which `make kernel`
registers - so once that has been run, opening one selects the right
interpreter no matter where Jupyter itself was started from. Skip that step and
Jupyter will offer to pick a kernel instead; choosing an unrelated one is the
usual cause of `ModuleNotFoundError: No module named 'matplotlib'` here.

They deliberately do not ask for the generic `python3` kernel, which resolves
to whichever python3 the running Jupyter finds first - often another project's
environment, and then the first cell fails on the imports.

Or with pip, into an environment of your own:

```bash
pip install "pytorch_wavelets[examples]" notebook ipykernel
python -m ipykernel install --user --name pytorch_wavelets \
    --display-name "Python (pytorch_wavelets)"
jupyter notebook notebooks/
```

The `examples` extra brings what the demos import - matplotlib and
scikit-image for the test images. Jupyter itself is listed separately on
purpose: the same extra is pulled in when the documentation is built, and a
docs builder has no use for a notebook server.

## Regenerating

After changing anything under `examples/`:

```bash
make notebooks
```

CI runs `make check-notebooks` and fails if the two have drifted apart, so the
notebooks cannot quietly fall behind the scripts the documentation executes.

Opening a notebook in Jupyter is enough to make it drift: saving writes a
trailing empty cell, execution counts and timing metadata back into the file.
Rather than leave that to be tidied up by hand, install the git hook once:

```bash
make hooks
```

It regenerates the notebooks and restages them before each commit that touches
`examples/` or `notebooks/`, and stays out of the way otherwise. Skip it for a
single commit with `git commit --no-verify`, and undo it with
`git config --unset core.hooksPath`.
