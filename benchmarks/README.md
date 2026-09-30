# Benchmarks

Scripts for measuring the speed and stability of the transforms. They are not
tests: nothing here runs in CI, and none of it is part of the installed package.

| File | What it does |
|---|---|
| [`profile_dtcwt.py`](profile_dtcwt.py) | Runs the DTCWT forward, inverse or end to end - or a reference 11x11 convolution, FFT convolution or DWT - five times, for a profiler to record. The workloads behind [Notes on Speed](../docs/speed.rst). |
| [`compare_numpy.py`](compare_numpy.py) | Times a forward and inverse transform against PyWavelets (DWT) or the `dtcwt` package (DTCWT). |
| [`measure_of_stability.ipynb`](measure_of_stability.ipynb) | The experiment behind the stability table in the [ScatterNet docs](../docs/scatternet.rst): scattering distances under noise, shifts and deformations over 1000 Tiny ImageNet samples, against KymatIO. |

## Running them

From the repository root, in the environment `make dev` builds:

```bash
python benchmarks/profile_dtcwt.py --help
python benchmarks/profile_dtcwt.py -j 2 --device cpu
nsys profile -o dtcwt_j2 python benchmarks/profile_dtcwt.py -j 2
python benchmarks/compare_numpy.py torch dtcwt -j 3
python benchmarks/compare_numpy.py numpy dwt --wave db4
```

The CUDA timings in `compare_numpy.py` synchronise before stopping the clock,
and one untimed pass runs first, so CUDA start-up is left out.

`measure_of_stability.ipynb` is kept for the record, with its outputs, and does
not run as it stands. It imports `dtcwt_gainlayer` and `plotters`, which are
not published, as well as `kymatio`, and it needs Tiny ImageNet.
