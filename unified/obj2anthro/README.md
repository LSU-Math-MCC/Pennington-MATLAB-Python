# OBJ to Anthropometry

This stage owns OBJ discovery, anthropometry backend selection, backend
execution, and canonical CSV output.

```bash
python -m unified obj2anthro --input data/obj --method slice --units auto
python -m unified.obj2anthro --input data/obj/man.obj --method all --units auto --out runs/measurements
```

Methods:

| Method | Implementation |
|---|---|
| `auto` | Current default, equivalent to `all`. |
| `auto` / `all` | Runs segmentation, slice, avatar, and MATLAB branches. |
| `segmentation` | Anatomical-region landmark backend. |
| `slice` | Slice-based biomarker backend. |
| `avatar` | MATLAB-faithful Python port of `Avatar.m`. |
| `matlab` | Original `Avatar.m` through one persistent MATLAB Engine. |
| `matlab-full` | Explicit `Avatar.m` steps `[1 2 3]`; slower cleaning/repair path. |

Raw backend artifacts are written beneath the selected output directory. In the
top-level composed pipeline, that directory is the run stage folder under
`runs/`.
