# Reports

Self-contained HTML, figures embedded — open the file directly, no server needed.

| Report | What it covers |
|---|---|
| [`geometry_report.html`](geometry_report.html) | **Three Ways to Measure a Body.** What each anthropometry backend defines a circumference to be, where each one cuts, how each locates a level (fixed constant vs geometric search), what each costs per scan, and where the four disagree. |

Regenerate with:

```bash
python -m unified obj2anthro --input data/obj --method avatar --units auto --out runs/avatar
python -m unified.compare runs/methods_report/combined_measurements.csv --reference matlab
python -m unified.obj2anthro.geometry_figures data/obj --out runs/methods_report/figures --scale-to-cm 0.1
python -m unified.obj2anthro.slice_levels
python -m unified.obj2anthro.build_geometry_report runs/methods_report
cp runs/methods_report/geometry_report.html docs/
```

The build reads the comparison tables and figures out of `runs/methods_report/`,
which is where `unified.compare` and `geometry_figures` write them. `runs/` is
gitignored, so the tables that back this report are force-added; `docs/` is the
copy meant to be found.
