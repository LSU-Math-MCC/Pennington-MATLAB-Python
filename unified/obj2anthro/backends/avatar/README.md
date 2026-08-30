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

## Verify it reproduces MATLAB

The reference that matters is `runs/matlab_ground_truth/raw/matlab/` — the
recorded output of `Avatar.m` (steps=3, Vol_SA=on) driven through the MATLAB
Engine on R2023b. `tests/test_avatar_matches_matlab.py` replays the port against
every scan in it:

```bash
python -m pytest unified/obj2anthro/tests/test_avatar_matches_matlab.py
```

On the 20 scans MATLAB measured, the port matches all 41 measurements to
floating-point noise on **19**. The two exceptions are properties of the
reference, not gaps in the port:

| Scan | What differs | Why |
|---|---|---|
| `A00-09-0254 2025-12-10_10-38-56` | crotch landmark, and the 16 measurements derived from it | `adjustCrotch` calls MATLAB `kmeans`, which is randomly seeded. This scan's `delta_v2` has no dominant outlier, so several Lloyd fixed points are reachable, and MATLAB's recorded answer is one it lands on roughly a quarter of the time. |
| `cancan07_A 2026-01-28_11-48-32` | `rCalfGirth`, by 5 µm on 42.6 cm | `calfGirth` maximises a hull perimeter over a discretised plane sweep. No plane anywhere in the search range reproduces MATLAB's value, so the two runs sample marginally different bands. |

`reference/` holds a frozen snapshot of this port's *own* output, replayed by
`tests/test_avatar_backend.py`. That catches drift, but cannot catch a shared
mistake — score against `test_avatar_matches_matlab.py` for that. Regenerate the
snapshot with `batch_measure.py` whenever a deliberate change moves a number.

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
- **Hull loops start where MATLAB's start.** `boundary` begins its trace at the
  lowest-numbered input point and repeats it to close the loop; scipy starts
  elsewhere. Perimeters do not care, but `getWrist` *averages* the closed loop,
  so the duplicated point moves the wrist centroid and with it the arm length.
- **Face windings are reconciled before the volume pass.** `Vol_SA='on'` runs
  `fixFaceOrientation2` on `self.f` first. Surface area is winding-invariant, so
  this shows up only in signed volume — but two of these scans carry the same
  triangle twice with opposite winding, and without the fix the two
  contributions cancel.

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

## Standing in for a random reference

`adjustCrotch` calls MATLAB's `kmeans`, which is randomly seeded (`Start='plus'`)
and converges to a *local* optimum. Its result is therefore not reproducible even
by MATLAB itself, so the port has to model it.

This port seeds Lloyd's algorithm at `(min, max)` and iterates to convergence.
For k = 2 that is the deterministic limit of k-means++ seeding: the second centre
is drawn with probability proportional to squared distance from the first, and
these `delta_v2` vectors carry one dominant outlier, so the draw lands on the
extremes with overwhelming probability. It reproduces MATLAB's partition on 19 of
the 20 reference scans.

An earlier version instead solved the 2-means problem *exactly*, on the reasoning
that the global optimum could never be worse. That was the wrong target: MATLAB
frequently does not reach the global optimum, and the exact solver disagreed with
the reference on 6 of the 20 scans.

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
- **Reference provenance.** The scoring reference is a real MATLAB R2023b run
  (`runs/matlab_ground_truth/`). An earlier Octave 8.4 stand-in, which shimmed
  `boundary(x,y,0)` with `convhull`, is no longer used: that shim silently
  changed where the hull loop starts, which is exactly one of the differences
  resolved above.
