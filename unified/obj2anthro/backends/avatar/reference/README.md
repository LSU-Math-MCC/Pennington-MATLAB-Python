# Avatar Reference Outputs

Frozen output from the reference batch run of the MATLAB-faithful port
(2026-08-03, 18 scans, `--scale-to-cm 0.1`). These files are regression
fixtures: `tests/test_avatar_backend.py` replays them against the current
backend, so a change that shifts any measurement is caught immediately.

| File | Contents |
|---|---|
| `all_measurements.csv` | one row per mesh, one column per measurement, plus segment vertex counts |
| `all_landmarks.csv` | long format: file, landmark, x, y, z |
| `failures.csv` | meshes that could not be processed (empty for this run) |
| `batch_summary.json` | run metadata, settings, per-file status and timings |
| `slices.csv` | the slice actually used for each girth: plane, point count, hull size |
| `slices_on_body.png` | front and side views with every measurement slice drawn |
| `slice_cross_sections.png` | one panel per slice, points plus hull outline |
| `body_measurement_reference.pdf` | measurement definitions |

Rows are keyed by `(n_vertices, n_faces, height)`. `(n_vertices, n_faces)`
alone is ambiguous — several of these scans share a vertex count.

The source scans are in `data/obj/`. Note that the reference `file` column
names do **not** reliably identify which scan produced a row: the upload these
tables arrived in had its file contents shuffled across filenames. Match on the
shape/height key instead.

Regenerate with:

```bash
python unified/obj2anthro/backends/avatar/batch_measure.py data/obj \
    --scale-to-cm 0.1 --output <dir>
```
