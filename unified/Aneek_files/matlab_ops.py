# Avatar MATLAB → Python port

A Python reimplementation of the landmark-detection and body-measurement logic in
`Avatar.m`. Output is **bit-identical** to the MATLAB reference: 34 measurements
and 14 landmarks agree exactly across five test meshes.

---

## Install

Requires Python 3.9+ and two packages:

```bash
pip install -r requirements.txt
```

That is `numpy` and `scipy` — nothing else. No trimesh, no MATLAB, no compiler.

## Verify it works

```bash
python selftest.py path/to/A00-08-4914_A_2025-12-09_12-27-17__1_.obj
```

Expected:

```
loaded ...obj: 5005 vertices, 9998 faces
--- measurements ---
  32/32 measurements match
--- landmarks ---
  14/14 landmarks match

PASS - this port reproduces the MATLAB reference exactly.
```

Run this again any time you modify `avatar_conversion/`.

## Run it — many meshes at once

`batch_measure.py` is the one you want for a cohort. It writes **one combined
table**: one row per mesh, one column per measurement.

```bash
# every .obj in a folder (add --recursive for subfolders)
python batch_measure.py scans/ --recursive

# millimetre meshes reported in centimetres
python batch_measure.py scans/ --recursive --scale-to-cm 0.1

# specific files, or shell globs
python batch_measure.py a.obj b.obj "cohort/*/scan.obj"

# use 8 cores
python batch_measure.py scans/ --recursive --jobs 8

# long batch you may need to restart; appends to existing results
python batch_measure.py scans/ --recursive --resume
```

Outputs into `--output` (default `./batch_output`):

| File | Contents |
|---|---|
| `all_measurements.csv` | one row per mesh, one column per measurement, plus segment sizes |
| `all_landmarks.csv` | long format: file, landmark, x, y, z |
| `failures.csv` | any mesh that could not be processed, with the reason |
| `batch_summary.json` | run metadata, settings, per-file status and timings |

A bad mesh never aborts the batch — it lands in `failures.csv` and the run
continues.

### Plausibility warnings

Each row gets a `warnings` column from cheap sanity checks: lopsided leg
segmentation, crotch height outside 40–52% of stature, and left/right asymmetry
above 1.5× in thigh, calf or arm length. These are warnings only; the numbers are
still written. They exist so a silently wrong mesh doesn't hide in a batch of
three hundred. For reference, the old broken pipeline's output trips all three:

```
leg segmentation lopsided (2210 vs 271)
crotch at 55% of height (expect ~45)
thigh asymmetry (159 vs 1093)
```

Parallel and serial runs produce byte-identical output; `--jobs` only affects speed.

## Run it — a single mesh

```bash
# single mesh, results land in ./output/<name>/
python run_avatar.py scan.obj

# choose an output directory
python run_avatar.py scan.obj --output results

# a whole folder
python run_avatar.py scans/ --recursive

# your mesh is in millimetres and you want centimetres out
python run_avatar.py scan.obj --scale-to-cm 0.1

# machine-readable
python run_avatar.py scan.obj --json
```

Each mesh produces a folder containing:

| File | Contents |
|---|---|
| `measurements.csv` | name, value, units |
| `landmarks.csv` | name, x, y, z (in the oriented frame) |
| `segments.csv` | segment name, vertex count, vertex indices |
| `summary.json` | everything above plus provenance notes |

### Units

The pipeline does no unit inference. `--scale-to-cm` multiplies lengths by the
factor you give (areas by its square, volumes by its cube) and relabels the
output. Your supplied scan is in **millimetres**, so `--scale-to-cm 0.1` gives
centimetres — height 154.66 cm, chest 91.29 cm. Without the flag, values come out
in raw mesh units and are labelled "mesh units".

## Use as a library

```python
from avatar_conversion import MatlabAvatar, load_obj

v, f = load_obj("scan.obj")
avatar = MatlabAvatar(v, f).run()

avatar.measurements["chestGirth"]   # float
avatar.landmarks["crotch"]          # np.array([x, y, z])
avatar.segments["left_leg"]         # vertex indices
avatar.v                            # oriented vertices
```

Pass `MatlabAvatar(v, f, orient=False)` to skip `fixOrientation` if your mesh is
already in the Styku frame (z up, feet towards −y).

---

## What this port does differently from the earlier Python pipeline

The previous Python code wasn't a buggy version of these algorithms — it was a
*different* set of algorithms that produced plausible-looking but wrong numbers.
Three structural mismatches, all fixed here:

**Leg segmentation.** MATLAB uses no connectivity at all. `getLegs` cuts two
lines in the (x, z) plane — crotch→right hip and crotch→left hip — and keeps
everything below them, then splits left/right on a plane at the crotch's x. The
old code used a connectivity flood fill; on a 5005-vertex mesh the legs touch, so
the fill merged them (`left_leg: 2210`, `right_leg: 271`). Now: **989 / 905**.

**Circumferences.** `getCircumference` calls `boundary(x, y, 0)`, which is the
**convex-hull perimeter** over a *band* of vertices near the plane, gathered by
`getVOnLine` — every vertex of every face straddling the plane, with no
interpolation. The old code computed exact mesh cross-section loops. Different
quantity entirely. Note this means girths slightly overestimate concave sections
such as the waist; that is the reference behaviour, preserved deliberately.

**Orientation.** `fixOrientation` only ever applies 90°/180° axis rotations, so
the resulting height equals one of the original bounding-box extents exactly. The
old code did a PCA alignment, which tilted the frame and shifted every landmark.

## MATLAB bugs reproduced on purpose

Matching the reference means reproducing its bugs. Each is flagged with a
`MATLAB BUG` comment in the source, and where a corrected value is meaningful it
is exposed under a `*_fixed` key alongside the faithful one.

| Bug | Effect | Corrected key |
|---|---|---|
| `getAnkleGirth` returns `[lAnkle, rAnkle, lAnkleGirth, rAnkleGirth]` but the constructor assigns slots 3–4 to `r_ankle_girth, l_ankle_girth` | the two ankle girths land on the wrong sides | `rAnkleGirth_fixed`, `lAnkleGirth_fixed` |
| `getTrunkLength` and `getCollarScalpLength` use `collar(1,2)` — the **Y** component — where **Z** is meant | `collarScalpLength` inflates to ~97% of stature | `trunkLength_fixed`, `collarScalpLength_fixed` |
| `getLegLength` treats `x > 0` as "right", opposite to every other routine in the file | leg lengths mirrored | — |
| `getFaces` ORs its three columns despite a docstring claiming "all 3 vertices" | segment areas include a ring of boundary faces and sum to more than the whole-body total | — |
| `getArmGirth` references `armMaxL(2)` in the right-arm angle | small right-arm girth error | — |
| `sosmooth3` pads its tail with a constant index rather than a mirrored one | minor smoothing artefact at the signal tail | — |

**If you want correct anthropometry rather than MATLAB compatibility, read the
`*_fixed` keys and be aware of the four unfixable-without-divergence items above.**

## Deliberate improvement

`adjustCrotch` calls MATLAB's `kmeans`, which uses random initialisation and is
therefore non-deterministic in principle. This port substitutes an exact 1-D
2-means, solved by scanning every split of the sorted values. For one-dimensional
data that is the global optimum, so it is deterministic and never worse than what
Lloyd iterations converge to. This is the only intentional algorithmic deviation
and it does not change the result on any tested mesh.

---

## Known limitations

**Mesh resolution.** `Avatar.m`'s cleaning stage (`steps` 1 and 2) is calibrated
for scans of roughly 100k vertices. In `omitBadShapedFaces`:

```matlab
thr_div = round(length(v) ./ 5000);   % for meshes under 30k vertices
thr     = ave_len_edge ./ thr_div;
```

At 5005 vertices `thr_div` rounds to **1**, making the degeneracy threshold equal
to the *mean* edge length — so every face whose three edges are all below average
gets collapsed. On your scan that deleted 9215 of 9998 faces. This port
corresponds to MATLAB's `steps = 3` branch and does no mesh cleaning, which is why
it works. If you feed it high-resolution scans you may want cleaning back; that
threshold needs fixing first.

**Mesh loading.** `load_obj` deliberately does *not* merge duplicate vertices.
Do not swap it for `trimesh.load(..., process=True)` — trimesh's default
processing renumbers every index, which breaks correspondence with MATLAB and
shifts landmark results.

**Reference provenance.** The reference values were generated by running
`Avatar.m` under **GNU Octave 8.4**, not MathWorks MATLAB, using exact shims for
three functions Octave lacks (`boundary(x,y,0)` → `convhull`, which MATLAB
*defines* as the convex hull at shrink factor 0; `incenter`/`triangulation` → the
standard incenter formula; two-argument `round`). These are mathematically exact
for the cases the code uses, but if you have MATLAB access, one confirmation run
is worth doing.

**Scope.** This covers the `steps = 3` landmark and measurement branch with
`SA = 'on'`. Not ported: mesh cleaning/repair (`steps` 1–2), CPD template
fitting, ellipse fitting (`circumference`, `'ellipse'`/`'cpd'`), partial
hole-filling volumes (`Vol_SA`), plotting, and marker export.

## File map

```
avatar_matlab_port/
├── batch_measure.py                 batch CLI -> one combined table
├── run_avatar.py                    single-mesh CLI -> per-mesh folders
├── selftest.py                      verification against reference values
├── requirements.txt
└── avatar_conversion/
    ├── __init__.py
    ├── mesh_io.py                   OBJ/PLY loading, MATLAB readObj semantics
    ├── matlab_ops.py                primitives: get_v_on_line, get_circumference,
    │                                find_minmax, sosmooth3, get_faces,
    │                                fix_orientation, constrained_flood_fill
    └── matlab_avatar.py             MatlabAvatar; method names mirror Avatar.m
                                     so the two can be diffed side by side
```

`MatlabAvatar.run()` reproduces the call order of the MATLAB constructor's
`steps == 3` branch exactly, so you can read the two in parallel.
