# Avatar Backend

This directory owns the MATLAB-faithful OBJ anthropometry implementation: a
Python port of `Avatar.m` (`steps = 3`, `SA = 'on'`). It sits alongside the
`segmentation` and `slice` backends and runs under `--method auto`.

Run it through the staged wrapper from the repository root:

```bash
python -m unified obj2anthro --input data/obj --method avatar --units auto
```

The backend code lives in `avatar_conversion/`:

| Module | Contents |
|---|---|
| `mesh_io.py` | OBJ/PLY loading with MATLAB `readObj` semantics |
| `matlab_ops.py` | primitives: `get_v_on_line`, `get_circumference`, `find_minmax`, `sosmooth3`, `get_faces`, `fix_orientation`, `constrained_flood_fill` |
| `matlab_avatar.py` | `MatlabAvatar`; method names mirror `Avatar.m` so the two can be diffed side by side |

The MATLAB reference lives at `MATLAB Code/Pennington/Avatar.m`.

Dependencies are `numpy` and `scipy` only (plus `matplotlib` for `show_slices.py`).
All three are already in the repository's root `requirements.txt` — nothing extra
to install, no MATLAB, no trimesh, no compiler.

## Verify it reproduces the reference

```bash
python unified/obj2anthro/backends/avatar/selftest.py "data/obj/A00-08-4914_A 2025-12-09_12-27-17.obj"
```

Expected:

```text
loaded A00-08-4914_A 2025-12-09_12-27-17.obj: 5005 vertices, 9998 faces
--- measurements ---
  32/32 measurements match
--- landmarks ---
  14/14 landmarks match

PASS - this port reproduces the MATLAB reference exactly.
```

`tests/test_avatar_backend.py` does the same check across every OBJ that can be
matched to a row in `reference/`, so `pytest unified/obj2anthro/tests` covers it
too. Run the selftest after any change to `avatar_conversion/`.

## Standalone tools

These are the port's original CLIs, kept next to the library. They work without
the unified wrapper, which is useful when you want the native `Avatar.m` names
rather than the canonical column names.

```bash
cd unified/obj2anthro/backends/avatar

# one combined table for a cohort; --scale-to-cm 0.1 reports mm meshes in cm
python batch_measure.py ../../../../data/obj --scale-to-cm 0.1 --output out/
python batch_measure.py scans/ --recursive --jobs 8      # parallel; identical output
python batch_measure.py scans/ --recursive --resume      # appends to existing results

# one mesh at a time, into out/<name>/
python run_avatar.py scan.obj --output out --scale-to-cm 0.1
python run_avatar.py scan.obj --json

# draw where every girth was measured
python show_slices.py scan.obj --output out
```

A bad mesh never aborts a batch — it lands in `failures.csv` and the run
continues. `--scale-to-cm` multiplies lengths by the given factor, areas by its
square and volumes by its cube; without it, values come out in raw mesh units.
There is no unit inference here (the unified backend does that part).

Each row of `all_measurements.csv` carries a `warnings` column from cheap sanity
checks: lopsided leg segmentation, crotch height outside 40-52% of stature, and
left/right asymmetry above 1.5x in thigh, calf or arm length. These are warnings
only — the numbers are still written — so a silently wrong mesh cannot hide in a
batch of three hundred.

## Use as a library

```python
from unified.obj2anthro.backends.avatar.avatar_conversion import MatlabAvatar, load_obj

v, f = load_obj("scan.obj")
avatar = MatlabAvatar(v, f).run()

avatar.measurements["chestGirth"]   # float, raw mesh units
avatar.landmarks["crotch"]          # np.array([x, y, z])
avatar.segments["left_leg"]         # vertex indices
avatar.slices["rWrist"]             # the slice the girth was taken from
avatar.v                            # oriented vertices
```

Pass `MatlabAvatar(v, f, orient=False)` to skip `fixOrientation` if the mesh is
already in the Styku frame (z up, feet towards -y).

## How this differs from the earlier Python pipeline

The previous Python code was not a buggy version of these algorithms — it was a
*different* set of algorithms producing plausible-looking but wrong numbers.
Three structural mismatches, all resolved here:

- **Leg segmentation.** The old code used a connectivity flood fill; on a
  5005-vertex mesh the legs touch, so the fill merged them (`left_leg: 2210`,
  `right_leg: 271`). MATLAB uses no connectivity at all, and this port now gets
  **989 / 905**.
- **Circumferences.** The old code computed exact mesh cross-section loops.
  MATLAB computes a convex-hull perimeter over a band of vertices — a different
  quantity entirely.
- **Orientation.** The old code did a PCA alignment, which tilted the frame and
  shifted every landmark. MATLAB only applies 90/180-degree turns.

## What "faithful" means here

Output reproduces the MATLAB reference rather than improving on it:

- **Girths are convex-hull perimeters**, not exact mesh cross-sections.
  `getCircumference` calls MATLAB's `boundary(x, y, 0)` over a *band* of
  vertices near each plane, gathered by `getVOnLine` with no interpolation.
  This slightly overestimates concave sections such as the waist.
- **Leg segmentation uses no connectivity.** `getLegs` cuts two lines in the
  (x, z) plane, crotch to each hip, and keeps everything below them.
- **Orientation is axis-aligned only.** `fixOrientation` applies 90/180-degree
  turns, so the resulting height equals one of the original bounding-box
  extents exactly. It is not a PCA alignment.

## MATLAB bugs reproduced on purpose

Each is flagged with a `MATLAB BUG` comment in the source. Where a corrected
value is meaningful, it is exposed under a `*_corrected_cm` canonical column
alongside the faithful one.

| Bug | Effect | Corrected column |
|---|---|---|
| `getAnkleGirth` returns `[lAnkle, rAnkle, lAnkleGirth, rAnkleGirth]` but the constructor assigns slots 3-4 to `r_ankle_girth, l_ankle_girth` | the two ankle girths land on the wrong sides | `ankle_circumference_{left,right}_corrected_cm` |
| `getTrunkLength` and `getCollarScalpLength` use `collar(1,2)` — the **Y** component — where **Z** is meant | `collarScalpLength` inflates to ~97% of stature | `trunk_length_corrected_cm`, `collar_to_scalp_length_corrected_cm` |
| `getLegLength` treats `x > 0` as "right", opposite to every other routine | leg lengths mirrored | — |
| `getFaces` ORs its three columns despite a docstring claiming "all 3 vertices" | segment areas include a ring of boundary faces and sum to more than the whole-body total | — |
| `getArmGirth` references `armMaxL(2)` in the right-arm angle | small right-arm girth error | — |
| `sosmooth3` pads its tail with a constant index rather than a mirrored one | minor smoothing artefact at the signal tail | — |

## Deliberate improvement

`adjustCrotch` calls MATLAB's `kmeans`, which uses random initialisation. This
port substitutes an exact 1-D 2-means solved by scanning every split of the
sorted values — the global optimum for one-dimensional data, so it is
deterministic and never worse. It is the only intentional algorithmic
deviation and does not change the result on any tested mesh.

## Raw artifacts

Per subject, under the run's `raw/avatar/<artifact_id>/`:

| File | Contents |
|---|---|
| `measurements.csv` | native `Avatar.m` measurement names and values |
| `landmarks.csv` | name, x, y, z in the oriented frame |
| `segments.csv` | segment name, vertex count |
| `summary.json` | everything above plus provenance and plausibility warnings |

## Known limitations

- **Mesh resolution.** This port corresponds to MATLAB's `steps = 3` branch and
  does no mesh cleaning. `Avatar.m`'s cleaning stage is calibrated for scans of
  roughly 100k vertices; at ~5k vertices its degeneracy threshold collapses
  almost every face.
- **Scope.** Not ported: mesh cleaning/repair (`steps` 1-2), CPD template
  fitting, ellipse fitting, partial hole-filling volumes, plotting and marker
  export.
- **Reference provenance.** Reference values were generated by running
  `Avatar.m` under GNU Octave 8.4 with exact shims for three functions Octave
  lacks (`boundary(x,y,0)` → `convhull`; `incenter` → the standard incenter
  formula; two-argument `round`).
