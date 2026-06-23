# Benchmark Datasets

## SSP-3D

SSP-3D is public and can be downloaded from the official repository.

```bash
python tools/benchmark/fetch_benchmark_data.py --ssp3d
python tools/benchmark/bench_discover.py
```

This downloads:

```text
https://github.com/akashsengupta1997/SSP-3D/raw/master/ssp_3d.zip
```

Expected local layout:

```text
datasets/SSP-3D/**/labels.npz
datasets/SSP-3D/**/images/
```

`tools/benchmark/bench_discover.py` searches recursively for `labels.npz`, so both the official
zip layout and `data_ext/ssp_3d` layouts are accepted.

## HBW

HBW is the official SHAPY-style anthropometry benchmark, but it is gated.

Official page:

```text
https://shapy.is.tue.mpg.de/datasets.html
```

The SHAPY site says access requires registration and license acceptance. HBW validation
ground truth is released after access; HBW test ground truth is not public. For test
evaluation, the official SHAPY instructions require submitting an NPZ containing:

- `image_name`
- `v_shaped`

Official evaluation docs:

```text
https://github.com/muelea/shapy/tree/master/regressor/hbw_evaluation
```

Expected local validation layout after manual download:

```text
datasets/HBW
```

Check local status:

```bash
python tools/benchmark/fetch_benchmark_data.py --hbw-info
python tools/benchmark/bench_discover.py
```

## Claim Rule

Do not claim HBW SOTA from SSP-3D alone. SSP-3D is useful sanity evidence, but HBW owns
the official SHAPY-style anthropometry verdict. See `ANTHRO_SOTA_STATUS.md`.
