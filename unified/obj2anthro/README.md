# OBJ to Anthropometry

This stage owns OBJ discovery, anthropometry method selection, backend execution,
and canonical CSV output.

```bash
python -m unified obj2anthro --input Python_slice_2026/OBJ --method slice --units auto
python -m unified.obj2anthro --input person.obj --method all --units auto --out scratch/measurements
```

Methods:

| Method | Implementation |
|---|---|
| `auto` | Current default, equivalent to `all`. |
| `all` | Runs segmentation and slice as separate branches. |
| `segmentation` | Relocated `Python_Fall2025/` implementation. |
| `slice` | Existing root `Python_slice_2026/` implementation through a thin wrapper. |

Raw backend artifacts are written beneath the selected output directory. In the
top-level composed pipeline, that directory is the run stage folder under `runs/`.
