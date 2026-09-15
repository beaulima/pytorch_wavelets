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

```bash
pip install "pytorch_wavelets[examples]"
jupyter notebook notebooks/
```

## Regenerating

After changing anything under `examples/`:

```bash
make notebooks
```

CI runs `make check-notebooks` and fails if the two have drifted apart, so the
notebooks cannot quietly fall behind the scripts the documentation executes.
