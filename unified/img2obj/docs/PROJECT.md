# Meshmap — Project Plan & Engineering Record

> Single durable record of mission, framework, architecture, state, benchmarks, and
> lessons. Consolidates the former `docs/plans/*` and `docs/reports/*` so the goals
> survive context compaction. Update in place as the work moves.
>
> **Active stack (post-cleanup):** CameraHMR + BLADE + SHAPY + LHM + HaMeR + DECA/FLAME
> + Blender + the default MeshMap `src/pipeline`. The earlier HMR2/4D-Humans, 3DDFA_V2,
> GFPGAN, PiFUHD, and SMPLitex repo-side runners were pruned; where this doc cites work
> that originally used them, it is marked **(historical)** and points at the kept
> equivalent.

## Contents
1. Mission & Deliverable
2. Design Intent (north star)
3. Sensor-Fusion Estimation Framework
4. The Crux — Abdomen Contour Accuracy
5. Pipeline Architecture (stages, presets, HMR backends)
6. Current Validated State
7. Texture & Appearance
8. Face & Detail Fidelity
9. Benchmarking & Beating SHAPY
10. Pennington Integration (measurement lab)
11. Open Defects & Prioritized TODO
12. External Tools & Models Audit
13. Postmortem — the early splat pipeline
14. Run Cheatsheet

---

## 1. Mission & Deliverable

Reconstruct humans from **legacy media (single image first, then multi-view)** to enable
**longitudinal anthropometric back-tracing**, benchmarked against a **3D scanner**.

**Our deliverable = the A-pose reconstruction + accurate joint placement.** The
post-A-pose anthropometric *measurement* extraction is owned by **another lab**
(Pennington — see §10). We keep girth/length calculation only as an internal sanity check.

Operating principles (stated by the user):
- **Single image perfected first**, then multi-view to debug. Use every tool to max capability.
- **Use vision circuits** to inspect outputs and iterate to quality.
- **Don't ask** for things resolvable algorithmically (e.g., gender from face).
- **Fast AND perfect** fusion. Cache expensive steps. Modular per-stage pipeline.
- Outputs in offline-loadable formats (3DGS `.ply`, GLB mesh, relight); Blender is the user's.

---

## 2. Design Intent (north star — adhere relentlessly)

Output = **clean SMPL-X A-pose body** (smooth geometry, correct joints incl. fingers/toes/
face) **carrying the real person's appearance at pixel resolution**: every observed pixel
(veins, tattoos, blemishes, moles) mapped to its **anatomically correct** canonical location
(correct L/R foot, correct finger/toe) via **dense SMPL-X UV correspondence**, with
**normal + displacement maps** for micro-relief, and **metric-anchored depth** fixing the
abdomen contour. NO Poisson-from-monocular-depth (that made "tumors"). Detail lives in the
**texture / normal / displacement maps**, not lumpy vertices. EVERY view + EVERY estimator
contributes texels/measurements (information filter; nothing discarded).

Pipeline target:
1. SMPL-X fit per view (betas, full pose incl. MANO hands + FLAME face/expr, v3d, K).
2. Multi-view shape fusion (betas) + pose; A-pose canonicalization + joints (done).
3. **Texture bake**: project each view onto posed SMPL-X, per-texel visibility/quality-weighted
   accumulation into the shared UV atlas → high-res albedo (L/R + digits correct by UV).
4. **Normal/displacement bake**: per-texel surface detail from shading/Sapiens-normals + the
   metric-anchored depth residual → micro-relief (veins/blemishes) without geometry tumors.
5. Abdomen x-z / y-x corrected by metric-anchored depth displacement on the torso.
6. Export A-pose mesh + UV albedo + normal + displacement (+ joints) for the other lab.

### Multi-SOTA fusion architecture (the thesis)
Every SOTA tool gives independent information; combine via MLE / information-filter (in beta
space, pixel space, keypoint space) weighted by calibrated uncertainty. Tool → role:
- **Silhouette/matte**: BiRefNet (clean human matting) + SAM2 (instance) → precise person
  contour for shape fitting (replaces rough YOLO mask).
- **2D pose**: ViTPose + MediaPipe(33) → fused keypoints → joint lift.
- **3D shape+pose prior**: Multi-HMR / CameraHMR (SMPL-X betas+pose+v3d, metric prior).
- **Metric shape baseline**: SHAPY (attribute/silhouette regression; separate env, gated ckpt).
- **Shape refinement (ours)**: silhouette-fit betas vs the clean matte (`fit_silhouette`).
- **Geometry/texture**: LHM gaussians (per-texel UV bake) + DECA/FLAME face.
- **Depth ensemble**: Depth-Anything S/B/L → info-filter fuse (terrain low-freq).
- **Normals/relief**: Sapiens (encoder bundled; normal head gated) for fine relief.
- **Face**: MediaPipe FaceLandmarker(478) + FLAME (in SMPL-X).

**The fusion**: final SMPL-X betas = inverse-variance fusion of {prior, SHAPY, silhouette-fit};
depth = info-filter over estimators; pose = fused ViTPose+MediaPipe. Each estimator carries an
a-priori sigma; **mis-stated sigma poisons the estimate** — calibrate.

---

## 3. Sensor-Fusion Estimation Framework

(The user is the sensor-fusion authority; the CV is ours.)

Treat every **(depth-estimator n, pixel m)** as a measurement `z_nm = h(x)+ε, ε~N(0,σ_nm²)`.
With **calibrated a-priori σ**, the MLE is the **information filter**:

```
info(m) = Σ 1/σ_nm²        x̂ = Σ (z/σ²) / info        posterior var = 1/info
```

Fisher information is **additive across estimators AND views** → posterior σ only shrinks.
σ calibration is load-bearing (mis-stated R poisons the estimate, like a killchain track).
- **Width (x)**: directly observed by silhouette → low σ.
- **Front depth (z)**: multi-estimator MLE depth fusion → calibrated σ.
- **Back depth (z)**: unobserved single-view → σ→∞, filled by symmetry prior or another view.

SMPL-X body = **prior/regularizer**; fit abdomen to obs under MAP
`argmin Σ‖obs−proj(x)‖²/σ² + ‖x−x_prior‖²/σ_prior²`.

Core implementation: `src/pipeline/geometry/depth_fusion.py` (+ `tests/test_depth_fusion.py`):
robust affine calibration, per-pixel σ, info-filter fuse.

**Key calibration finding:** the 3-estimator MLE depth fusion gives a confident (σ≈6 mm) but
z-metrically WRONG cross-section (~0.9 m front-to-side vs ~0.15 m real). Root cause = the
classic *mis-stated R*: the monocular estimators **share a systematic depth-scale bias**
(correlated sensors), so low *inter-estimator* σ is NOT low error. Fix = **external metric
anchor** = rendered SMPL-X depth:
```
fused = SMPL-X prior depth (metric anchor) ⊕ info-filtered residual(estimator − prior)
```
Affine-calibrate each estimator to the SMPL-X depth (kills z-scale bias), fuse the residuals
(true person−model detail) with honest σ; back-half by symmetry / another view.

---

## 4. The Crux — Abdomen Contour Accuracy

The A-pose must be correct at the **abdomen contour level**:
- **x-z (transverse)** cross-section at navel — width × depth aspect must be right.
- **y-x (coronal/silhouette)** lateral-width-vs-height profile.

Also required: **finger- and toe-level resolution** (SMPL-X MANO hands + FLAME face fuse
natively) and **face fidelity** (§8).

**Resolved (vision-verified):** abdomen aspect ratio was inverted (24 W × 32 D, deeper than
wide). Root cause: SMPL-X rest-pose arms hang into the navel band (no gap). Fix: extract
contours on an **arms-raised measurement mesh** (`smplx_measure_mesh`) → s1 now
**W=36 × D=15 cm**, wider-than-deep. Also fixed: full-body `trimesh.section` (→ vertex-band),
arm gap-detection, anatomical girth levels from the central section curve, arm/leg-contamination.

**Open defect:** single-image abdomen x-z still too DEEP / not wide enough on un-anchored runs;
the metric-anchor fusion in §3 is the fix (TODO P0, §11).

### Abdomen terrain (user wants a 2D MLE terrain, not a 1D contour)
`tools/geometry/abdomen_terrain.py`. v1 (monocular-depth high-pass) = NOISE (navel relief is
below monocular depth resolution; metric calib collapses estimators to a≈0.07; high-pass
amplifies griddata/estimator noise; σ misleadingly small from correlated estimators).
**Correct method: fine relief from NORMAL integration (Frankot-Chellappa)** — navel/linea/
muscle live in SHADING, not cm-resolution depth. Low freq anchored by fused metric depth;
high freq from normals. v2 = shading-normal integration. Refine with the **Sapiens-Normal**
model (purpose-built human normals) + multi-view normal fusion (info filter over normals).

---

## 5. Pipeline Architecture

### 5.1 Modular stages
The `src/pipeline` flow is a set of independently swappable stages. Pick an implementation per
stage on the CLI; a `--backend` preset just sets defaults you can override.

```bash
# preset + per-stage overrides (mix freely)
python -m pipeline.run single --image img.jpg --out runs/x \
  --backend real --depth depth-anything-large --pose mediapipe --seg yolo

# geometry from LHM (GPU/WSL), everything else as-is
python -m pipeline.run single --image img.jpg --out runs/x --backend lhm
```

Adding a new implementation = one line in `src/pipeline/backends/registry.py`
(`stage -> {impl_name: factory}`) plus a backend class behind the interface in
`backends/base.py`. The geometry/fusion core and tests never change.

| Stage | impl (flag) | tool | status |
|-------|-------------|------|--------|
| **gs** (geometry/scene) | `dummy` | synthetic | ✅ |
| | `depth-lift` / `depth-lift-large` | Depth-Anything backproject (2.5D) | ✅ |
| | `lhm` / `lhm-500m` / `lhm-1b` | LHM → canonical 3D Gaussians (A-pose) | 🔧 wiring (GPU/WSL) |
| **seg** (instances) | `dummy` | synthetic | ✅ |
| | `yolo` | YOLOv8-seg (multi-person, hole-filled) | ✅ |
| | `sam2` / `birefnet` | SAM2 / BiRefNet matting (LHM env) | 🔭 available to add |
| **pose** | `dummy` | synthetic | ✅ |
| | `mediapipe` | MediaPipe PoseLandmarker (33) | ✅ |
| | `yolo` | YOLOv8-pose (17) | ✅ |
| | `smplx-3d` | LHM/ROMP SMPL-X 3D pose | 🔭 available to add |
| **face** | `dummy` / `facemesh` | MediaPipe FaceLandmarker (478) | ✅ |
| **depth** | `dummy` | synthetic | ✅ |
| | `depth-anything[-base/-large]` | Depth-Anything-V2 S/B/L | ✅ |
| **assoc** | `dummy` / `color-hist` | color histogram embedding | ✅ |
| | `reid` / `insightface` | OSNet re-ID / face embeddings | 🔭 available to add |

✅ implemented · 🔧 wiring in progress · 🔭 enabled by installed tools, ready to register

| preset | gs | seg | pose | face | depth | assoc |
|--------|----|----|------|------|-------|-------|
| `dummy` | dummy | dummy | dummy | dummy | dummy | dummy |
| `real`/`auto` | depth-lift | yolo | mediapipe | facemesh | depth-anything | color-hist |
| `lhm` | lhm | yolo | mediapipe | facemesh | depth-anything | color-hist |

A real impl that fails to construct in a preset falls back to `dummy` for that stage (unless
explicitly requested with a flag, which then surfaces the error).

### 5.2 HMR backend stack
The anthropometry path runs SMPL-X regressors behind a uniform interface
(`tools/smplx/backends.py`): each backend is `(name, label, conda env, runner script)` and
writes a normalized `schema.npz` (`tools/smplx/schema.py`). Backends:
- **CameraHMR** — `tools/hmr/camerahmr/` (current SOTA single-image; env `camerahmr`).
- **BLADE** — `tools/hmr/blade/` (close-range perspective SMPL-X; env `blade_env`).
- **SHAPY** — `tools/hmr/shapy/` (metric anthropometry baseline/reference).
- **LHM** — `src/pipeline/backends/lhm_backend.py` + `--backend lhm` (whole-body gaussians).

Shared anthropometry/measurement libraries live in `tools/anthro/`:
- `tools/anthro/lhm_anthropometry.py` — multi-view SMPL-X shape fusion + measurement (`as A`).
- `tools/anthro/shapy_measure.py` — reproduced-SHAPY metric measurement (`as SM`).

See `docs/setup/HMR_BACKENDS_SETUP.md` for env build/setup details.

---

## 6. Current Validated State

- Modular per-stage pipeline + registry; **51 tests green** (incl. MLE depth-fusion math).
- **All 16 models (single+subject) canonicalized** to aligned arms-down A-pose + 22 joints.
  (Fixed: arms-up "Y" pose from a wrong `apose_body_pose` sign + collar split.)
- **Multi-HMR/CameraHMR → SMPL-X betas** per view → robust MAD-trimmed fusion → A-pose mesh +
  joints. `tools/anthro/lhm_anthropometry.py` (fast, ~3.5 GB VRAM, no gaussian gen).
- **LHM** (WSL2 + CUDA, RTX 3080 Ti 16GB) single image → real 3D A-pose gaussians (validated
  on s5; cached in `runs/.lhm_cache/`). pytorch3d / diff-gauss / simple-knn built.
- **Gender allocation**: CLIP zero-shot per view, log-posterior fused across views, neutral
  abstain; debuggable `gender.json` (s1 → female, conf 0.9999).
- **Per-view camera registration is correct** (projected verts span the body). Consistent-shape
  registration (all views → one shared re-posed body) is a marginal sharpness gain, not dramatic.
- **Textured A-pose per subject** (`runs/uv_s1..s5`): coverage s2=83% s3=88% s4=91% s1=67%
  s5=37% (scales with view diversity). Clean SMPL-X geometry + per-texel UV (face/garment/
  tattoo land on correct anatomy). No tumors.
- **Abdomen terrain** (`runs/terrain_s1..s5`): good on clean frontal (s1); fragile on
  oblique/partial torsos (garment+arm edges) — needs Sapiens-Normal + frontal-view select.
- Hi-fi export: full 3DGS `.ply` + Poisson GLB (normals) + relight studio;
  `tools/render/inspect_cloud.py`.
- Workflow hardened: multi-step WSL work via Python batch drivers (`tools/**/batch_*.py`) to
  avoid `$`-mangling (which silently failed loops).

### Contour fitting (the crux mechanism) — working
`tools/geometry/fit_silhouette.py`: optimize SMPL-X betas so the posed-mesh silhouette matches
the person's segmentation contour (pixel-space per-row width profile; camera-frame translation
offset was the key fix — without it the projected mesh sat at origin → zero gradient).
Residual reduction: s1 61%, s2 37%, s4 32%, s5 27%, s3 0% (FAIL — auto-selected view is
reclining/oblique; per-row width assumes upright → needs pose-robust objective or better view
pick). Fitted betas: `runs/fit_sX/fitted_betas.npy`.

### Honest dominant limits (data/model-bound, NOT code bugs)
- **Coverage is bounded by the source views.** Symmetry fill (L↔R) only helps asymmetric
  capture; it cannot fill the BACK from frontal-only views, nor invent unphotographed regions
  (s5 legs are gray because s5's photos are upper-body). True full coverage needs back/oblique/
  leg views.
- **Pixel-level veins/blemishes on fingers/face** need higher-res *close* views + Sapiens-Normal.
- **Abdomen fine relief** needs Sapiens-Normal (albedo-invariant); shading works only on clean
  frontal skin.
- **Body shape too heavy vs slim subjects**: monocular regressors over-estimate girth and
  monocular silhouette is scale-ambiguous; the real fix is SHAPY-style metric/clothing-robust
  regression (§9). Flagged so they aren't mistaken for unfinished work.

---

## 7. Texture & Appearance

### 7.1 Bake, fusion, delighting
- **K-UV [done]**: per-TEXEL UV-atlas bake (`tools/texture/texture_uv_bake.py`) over SMPL-X UV
  (`smplx_uv.obj`). s1: ~380k texels/view, 66.9% atlas coverage, 1024². Real face/bikini/hands
  land at correct UV. Inverse bake (atlas→image) reuses our fitted pinhole.
- **K-OCC [done]**: vertex z-buffer occlusion test (texel depth ≤ nearest-surface +3cm) → no
  back-bleed. Refined with a triangle-rasterized z-buffer (visibility against triangle interiors,
  not only sparse projected vertices).
- **K-NRM [done v1]**: tangent normal map from albedo high-freq luminance → micro-relief under
  light. Refine: Sapiens normals + metric-anchored depth-residual displacement.
- **Texture fusion**: weighted MULTI-VIEW blend (atlas_sum/atlas_w) replaces hard best-view → no
  seams; anti-streak = 4px silhouette-edge erosion + facing gate 0.35→0.5. Flags `--no-delight`,
  `--best-view`.
- **SOTA delighting**: integrated compphoto/Intrinsic (Careaga & Aksoy, SIGGRAPH'23/'24; open
  weights). `texture_uv_bake.sota_albedo()` → diffuse `hr_alb`; retinex `delight()` fallback.
  NOTE: Intrinsic install upgrades numpy→2.x which breaks the lhm stack → pin numpy<2 in lhm
  (Intrinsic still runs under 1.26.4).
- **Skin-only fusion** (`--skin-only`): bakes only texels projecting onto SKIN pixels (`skin_mask`:
  YCrCb Cr 133-180 / Cb 77-130 OR HSV warm-hue, opened). Garments left honestly unmapped (grey
  0.5), never fabricated. Person matte = SAM2 box-prompt ∩ depth-foreground. s1 coverage 46.7%
  skin, zero background/garment bleed.

### 7.2 Texture production rules (from the singleton diagnosis, 2026-06-14)
- Use **observed-alpha overlays** for acceptance, not completed RGB atlases. Every bake writes
  `atlas_observed_rgba.png` + `atlas_observed_mask.png`; overlay reports record `atlas_mode`.
- Keep `--no-coherent-face` on until the face warp has its own passing overlay metric (the worst
  face artifacts were tied to the coherent-face alignment/overwrite + head completion path).
- Treat completion as **avatar fill only**; never let it count as photo agreement.
- For multi-view, fix fusion next: per-texel confidence + view selection must operate on observed
  texels before any fill/repair stage.
- Diagnosis: the singleton path was not mainly failing on atlas resolution or UV projection
  (2048 helps: s6 6.52→5.57 mean abs RGB diff); the big failures came from rendering
  completion/repair output as if it were observed photo texture.

---

## 8. Face & Detail Fidelity

Goal: "look EXACTLY like the photo, with all detail and more."

Priority goals:
1. **G1 Coherent face** — bake the entire head region from the SINGLE best frontal view (no
   per-texel patchwork). **Done**: 298k head texels, no more patchwork; real coherence gain BUT
   still not photo-exact (eyes dark, nose smudgy) because generic SMPL-X head geometry ≠ person.
2. **G2 Face-shape fit** — align face texture to the photo via landmarks. **Done**: FAN gives 68
   landmarks; dlib k → SMPL-X joint 76+(k−17) direct map; TPS warp (51 pts) aligns the chosen
   view; head baked from that single coherent aligned view. Better placement, still not exact.
3. **G3 High-res / detailed face** — the real ceiling-raiser. The SMPL-X head is low-detail with
   SEPARATE eyeball geometry; **no texture work fixes this** — it needs detailed face GEOMETRY.
   Current face-detail work lives in the **DECA/FLAME** scripts under `tools/face/` (FLAME fit +
   photo back-projection welded to SMPL-X via exact correspondence). *(Historical: the original
   G3 used 3DDFA_V2 dense face verts; that repo-side tooling was pruned.)*
4. **G4 Hands/feet (MANO)** — same coherent-view + fit principle for fingers/toes (HaMeR slim
   runner under `tools/hands/`).

Method notes: head texel mask = SMPL-X vertices above the neck joint; best frontal face view =
argmax over views of sum(facing·in_person) over head texels; FLAME landmarks via SMPL-X face
keypoints / `lmk_faces_idx`; verify every step with vision vs the source crop (no claims without
a render). Env note: face-alignment + mediapipe installed in lhm; numpy re-pinned <2.

---

## 9. Benchmarking & Beating SHAPY

This section is the honest win-condition for the metric body-shape claim. Keep
appearance/texture/face success **separate** from any body-shape claim.

### Bottom line
> Meshmap is **broader** than SHAPY (A-pose canonicalization, SMPL-X meshes, UV texture, skin
> transfer, 3DGS/GLB export, face, abdomen contours, Pennington markers), but it has **not yet
> beaten SHAPY** on validated metric body-shape regression. SHAPY's core claim is narrower and
> harder: metric body shape from in-the-wild images, validated against scanner-grounded shape.

### SHAPY targets to beat (published; update only on a proven newer official source)

HBW Test (paper Table 3):

| Method | Model | Height mm | Chest mm | Waist mm | Hips mm | P2P20K mm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SHAPY | SMPL-X | 51 | 65 | 69 | 57 | 21 |

Primary: beat SHAPY on mean P2P20K AND on ≥3 of 4 measurement errors; do not regress any
measurement by >10% unless P2P20K improves by >20% and the failure is explained.

MMTS (paper Table 4):

| Method | Model | Height mm | Chest mm | Waist mm | Hips mm |
| --- | --- | ---: | ---: | ---: | ---: |
| SHAPY | SMPL-X | 71 | 64 | 98 | 74 |

Secondary: beat average measurement error; prefer beating waist and chest.

SSP-3D (paper Table 5): SHAPY SMPL-X **19.2 mm** PVE-T-SC. Use SSP-3D as a **sanity** benchmark,
not the main victory condition (it is scale-corrected and hides absolute size error). Published
PVE-T-SC reference: CameraHMR 11.6 | Sengupta 13.6 | STRAPS 15.9 | SHAPY 19.2.

### Autonomous operating rules
1. SHAPY gated checkpoints missing → don't block; use published numbers, write
   `baseline_source="published"`.
2. SHAPY checkpoints present locally → run as additional baseline, `baseline_source="local_shapy"`.
3. HBW missing → do not claim "beats SHAPY"; run SSP-3D + smoke only, emit
   `verdict="insufficient_ground_truth"`.
4. HBW present → HBW owns the verdict.
5. Keep appearance metrics separate from body-shape metrics.
6. Every run produces machine-readable JSON + markdown summary + worst-case visual panels.
7. Any variant failing >5% of images reports both "all images" and "successful only" (the main
   number is all images).
8. Cache predictions keyed by image hash, code version, model version, method config.
9. Never silently skip subjects/images; missing predictions count as failure unless the manifest
   marks the sample invalid.
10. Do not hand-edit generated benchmark outputs.

### Metrics to implement
- **Shape**: `P2P20K` (sampled surface point-to-point mm, main HBW metric — avoids SMPL-X vertex
  density bias), `V2V`, `PA-V2V` (diagnostic only).
- **Measurement** (mean abs error mm): height, chest, waist, hip circumference. Measurement
  extraction must be **identical across methods** (do not use one impl for Meshmap, another for
  SHAPY).
- **Failure**: image/subject counts, prediction success rate, invalid count, median runtime,
  GPU memory, cache hit rate.

### Candidate method variants
`meshmap_silhouette` (silhouette-fit only, reliability-gated, may abstain), `meshmap_full`
(info-fused betas from prior + CLIP correction + silhouette fit + local SHAPY if present, each
estimator writing its sigma), `published_shapy` (table numbers), `local_shapy` (only when local
gated checkpoints exist), plus the current backend runners (`camerahmr_sota`,
`meshmap_camerahmr_skf`, `blade_smplx`). *(Historical: `meshmap_hmr2` / `meshmap_hmr2_clip` were
HMR2-based; HMR2 was pruned — use the kept backends in `tools/hmr/{camerahmr,blade,shapy}/`.)*

### How to beat SHAPY honestly
The route is **calibrated metric shape fusion**, not texture/face/prettier renders:
1. Build the HBW/MMTS harness (without it there is no honest win condition).
2. Reproduce the SSP-3D result after cleaning command/config/cache.
3. Implement HBW metrics; run the simplest method first for the floor.
4. Add reliability-gated silhouette fitting (never force it on reclining/oblique/heavily-clothed/
   truncated people — give it σ=∞ when it carries no information).
5. Calibrate fusion sigmas on a validation split from residuals — not by intuition. Save curves.
6. Use SHAPY's own insight: semantic body-shape attributes. A CLIP attribute predictor
   (`tools/geometry/clip_shape.py`) approximates SHAPY's linguistic signal without gated weights;
   calibrate against ground truth, not vibes.
7. Optimize measurement errors AND P2P20K together.
8. Add multi-view only as a separate, clearly-labelled stronger setting (`meshmap_multiview`),
   never as a single-image replacement claim.

### Harness layout & verdict schema
Benchmark code/config lives under `benchmarks/` (manifests, configs, results) and
`tools/benchmark/` (`bench_discover.py`, `bench_metrics.py`, `bench_all.py`). Dataset discovery
searches `datasets/{HBW,SSP-3D}`, `~/HBW`, `~/SSP-3D`, `~/datasets/*`, `D:/`, `E:/`; SHAPY
checkpoints under `~/shapy/data/{trained_models,expose_release,utility_files}`. Never request
credentials; if absent, use published numbers.

Every run writes a verdict JSON whose `verdict` is exactly one of: `beats_shapy_on_hbw`,
`does_not_beat_shapy_on_hbw`, `beats_shapy_only_on_ssp3d`, `insufficient_ground_truth`.

```bash
python tools/benchmark/bench_all.py --dataset ssp3d \
  --methods camerahmr_sota meshmap_full published_shapy --out benchmarks/results/ssp3d_smoke
```
If data is missing the command finishes without input and writes
`status="blocked_missing_dataset"` with the exact searched paths.

### SHAPY measurement reproduced WITHOUT gated weights
`tools/anthro/shapy_measure.py` reproduces SHAPY's virtual-measurement *method* with OPEN parts:
published SMPL-X anthropometry landmarks (DavidBoja/SMPL-Anthropometry: HeadTop 8976, heel 8847,
nipple 3572, navel 5939, pubic 5949); height = headtop−heel; mass = mesh-volume × 985;
girth = sum of horizontal plane-intersection segments (= SHAPY `compute_peripheries`, CPU not
CUDA since system CUDA12 vs torch-cu117 blocks the C++ ext). Arm-contamination removed via
central torso-loop clustering. Outputs `runs/fit_sX/shapy_measurements.json`,
`runs/SHAPY_measurements.json`.

**Honest finding:** monocular regressors MODE-COLLAPSE to ~average shape (reading slim and heavy
subjects at similar weight). Fusing another mode-collapsed regressor cannot fix slimness. The
discriminative body-shape signal is the **silhouette** (correctly separates s5=111kg from
s3=69kg) and **semantic attributes** (SHAPY's real mechanism). The measurement tool quantifies
every change.

### SHAPY baseline env (gated)
`tools/hmr/shapy/wsl_setup_shapy.sh` → conda env `shapy` (py3.8, torch 1.13.1+cu117), smplx,
SMPL-X/SMPL linked, attributes pkg + mesh-mesh-intersection CUDA built. Local
`~/shapy/data/trained_models/shapy/SHAPY_A` (1.4 GiB) now exists, so the remaining question is
**runner validation**, not missing checkpoints. `tools/hmr/shapy/run_shapy.py` →
`shapy_betas.npy` → `fuse_betas` picks it up (σ 0.5, tightest estimator).

---

## 10. Pennington Integration (the measurement lab)

The Pennington (MATLAB→Python) team places 17 anatomical markers on a body mesh via
`src.body.Body`. They are the post-A-pose MEASUREMENT lab. Integration = OUR CLIP-corrected
A-pose mesh → THEIR marker pipeline.
- `tools/workflows/export_for_penn.py` → `runs/penn_integration/<s>_apose.obj` +
  `<s>_markers_smplx.json` (17 markers from exact SMPL-X joints/landmarks; meters, Y-up/Z-fwd).
- Scratch lives in their git-ignored `.venv/` (their hard rule).
- Findings: (a) their arm-marker COLLAPSE (shoulder==wrist) happens on NOISY SCANS; on our clean
  parametric A-pose mesh the wrist separates correctly (wrist-span/H 0.32 ≈ ours 0.33) — our mesh
  IMPROVES their output. (b) Two independent marker methods CROSS-VALIDATE: mean
  |Pennington − meshmap| scale-invariant ratio diff = 0.023 (2.3% of height) over 5 subjects ×
  {wrist-span, shoulder-w, hip-w}. (c) Cross-val exposed our hip-marker bug (joint vs surface) →
  fixed to lateral pelvis surface (arm-excluded).

---

## 11. Open Defects & Prioritized TODO

1. **[P0] Metric-anchored abdomen fusion** (the fix):
   a. Replicate the Multi-HMR/CameraHMR inner call → get `v3d`, `j3d`, `K`/camera per view.
   b. Rasterize `v3d` (SMPL-X faces) with pytorch3d → metric prior depth, aligned to image.
   c. `fuse_estimators(depth_maps, mask, reference=smplx_depth, ref_sigma≈0.02 m)` →
      metric-anchored fused front surface (residual carries observed detail).
   d. Combine with silhouette (x width, low σ) → corrected abdomen x-z (front) + y-x.
   e. Validate aspect ratio is now wider-than-deep; report posterior σ. Back half: symmetry/MV.
2. **[P0] Validate** corrected x-z aspect ratio against anatomical expectation (wider than deep).
3. **[P1] Max single-image fidelity** — LHM-1B/500M (VRAM permitting); Sapiens normal-map bake.
4. **[P1] Multi-view fusion** — align views in canonical frame, accumulate Fisher info across
   views to fill the back-half x-z and tighten σ; debug abdomen with multi-view.
5. **[P2] Fingers/toes** — per-view MANO hand pose + FLAME face params, fused like betas; export
   full SMPL-X hand/foot joints; per-digit alignment.
6. **[P2] Calibration** — `--stature-cm` / in-frame reference for absolute scale; bench vs scanner.
7. **[P3] Per-view visibility-weighted** girth/contour slicing; gender-specific refinements.
8. **[fusion] Calibrate estimator covariance** against ground truth (current beta fusion lacks it).

Note `runs/bench_single/fit_report.json` once reported `improvement_pct=-4.1` (silhouette fit
made a sample worse) — any estimator that can regress into failure must be reliability-gated
before it joins the autonomous benchmark.

---

## 12. External Tools & Models Audit

Snapshot measured 2026-06-13 from the workstation; records the pre-cleanup footprint. The
selected product stack is now CameraHMR + BLADE + SHAPY + LHM + HaMeR + DECA/FLAME + Blender +
the default MeshMap pipeline. 3DDFA, GFPGAN, PiFUHD, SMPLitex, and HMR2/4D-Humans repo-side
tooling and their external installs/caches were removed.

| Area | Footprint | Notes |
| --- | ---: | --- |
| Repo checkout (excl. WSL/home envs) | ~6.05 GiB | Mostly `vendor/`, `runs/`, local weights, datasets. |
| `vendor/` | 3.03 GiB | Blender portable + SMPL archive/extracted. Not needed by default `pipeline.run`. |
| `runs/` | 2.45 GiB | Generated output/cache. Safe to delete if outputs not needed. |
| Local weights (root/`models`) | ~500 MiB | Only ~46.1 MiB needed for default real pipeline. |
| `datasets/` | 176 MiB | SSP-3D zip + extracted. Benchmark-only. |
| WSL/home repos/envs/caches | ~160.5 GiB | LHM, CameraHMR, BLADE, SHAPY, HaMeR, DECA, conda envs, HF/torch caches. |

**Default `--backend real` needs only**: YOLOv8n-seg (6.74 MiB), YOLOv8n-pose (6.52 MiB, pose
fallback), MediaPipe PoseLandmarker Heavy (29.24 MiB), MediaPipe FaceLandmarker (3.58 MiB),
Depth-Anything-V2-Small (HF cache, outside repo), + OpenCV/scipy/numpy/pillow/trimesh
(open3d only for relightable Poisson GLB). That is ~46.1 MiB of local model assets.

**Unit tests / dummy smoke need**: no learned weights — just `numpy, scipy, pillow, trimesh`.

**Key external installs** repo scripts target (WSL `/home/clint/...`): `LHM` (41 GiB,
`--backend lhm` + LHM tools), `CameraHMR` (8.5 GiB; checkpoint `camerahmr_checkpoint_cleaned.ckpt`
7.5 GiB), `blade` (9.5 GiB), `hamer` (12 GiB), `DECA` (1.7 GiB), `shapy` (4.3 GiB incl. local
`SHAPY_A`). Referenced model subpaths still in use: `~/LHM/pretrained_models/sam2/
sam2.1_hiera_large.pt` (857 MiB, texture matte), `~/LHM/.../smplx` (404 MiB),
`~/LHM/.../mano` (7.3 MiB), `~/CameraHMR/data/{pretrained-models,models/SMPL}`.

**Cleanup candidates** (keeping default `--backend real` working): archive/delete `runs/`
(~2.45 GiB); `vendor/blender/` (1.34 GiB) unless local rendering; `vendor/SMPL_*.zip` (315 MiB)
if extracted models kept; `datasets/*.zip` if extracted kept; the large `yolov8x*.pt` (~268 MiB)
unless using texture matte tools.

**Packaging caveats**: `pyproject.toml`'s `[real]` extra lists only `ultralytics, mediapipe,
opencv-python` but the code imports `transformers`+`torch` for Depth-Anything (incomplete); many
optional tools import `open3d, matplotlib, pyrender, imageio, smplx, torch` without declaration.
The LHM backend intentionally raises if the WSL CUDA env is not importable. WSL recheck used
explicit `/home/clint/...` paths because PowerShell expands `$HOME` before WSL sees it.

---

## 13. Postmortem — the early splat pipeline

**Historical, but the lessons govern current design.** The original `src/pipeline` ran
end-to-end (single/folder/multi-person/subject) with 0 crashes and 40/40 tests green, yet the
3D output was **not a reconstruction** — it was a flat, color-mapped depth relief ("billboard")
of each photo. The failure was architectural, not cosmetic.

Two headline failures:
1. **Output was not in A-pose** — reposing/skinning was never implemented. `CANONICAL_APOSE`
   (`geometry/canonicalize.py`) was dead code; `canonicalize_splats()` only applied a single
   global rigid + uniform-scale (`world_to_canonical`) — a coordinate change, not a repose. A
   person mid-jump came out re-centered mid-jump. A-pose normalization is the entire point.
2. **3D pose was fabricated** — "3D joints" came from sampling non-metric monocular depth at 2D
   keypoints (`lift_joints_3d`), so they were near-coplanar and their relative depths were noise.

Root cause: **monocular depth-lift is 2.5D, not 3D**. `RealGS` remapped relative Depth-Anything
disparity to fake "metres" (`1.5 + (1-disp)*2.5`), backprojected through an invented pinhole
(`default_camera(fov=55)`), yielding a front-surface height-field in a made-up frame. The
canonical `front_axis = cross(hip_axis, spine_axis)` was therefore dominated by depth noise (bad,
sign-flipping Z), so nothing aligned and same-subject fusion stacked misaligned cards into a blob.

What was real and kept: repository structure, typed dataclasses, config/cache/parallel, backend
interfaces; the 40/40 geometry/fusion/export math (projection, Procrustes, quaternion averaging,
voxel fusion, right-handed frames); the mask-gates-depth selection; multi-person instance
separation (YOLOv8-seg); dense 2D markers (MediaPipe 33/478); debug artifacts, manifests, A-Frame
export, thread-safe model loading.

The fix (now the design): a parametric human model (**SMPL-X via a regressor**, multi-view-fit for
subject folders) gives accurate 3D pose, a body natively defined in canonical pose space that
**reposes to A-pose by construction**, and real 3D — with monocular depth demoted to surface
relief and true GPU 3DGS gated behind hardware.

Process lessons (load-bearing):
- **Validate the hardest invariant first, not last.** Render an orthogonal side view of any
  canonical cloud on day one; a paper-thin side profile = a billboard.
- **"Tests pass" measured plumbing, not product** — synthetic-data tests were correct by
  construction and asserted nothing about real-image 3D plausibility.
- **A permitted fallback is not a free pass** — monocular depth was allowed as a fallback, not a
  foundation; escalate to a real reconstructor and flag 2.5D output explicitly.
- Don't over-optimize the visible/cheap (overlays, mask holes) while the central problem (real 3D)
  goes unaddressed.

Add hard quality gates to the dev loop: a **billboard gate** (`z_range/y_range > τ` from a side
render), an **A-pose gate** (reposed wrist/elbow/ankle match the A-pose target within tolerance),
and a **pose gate** (minimum high-confidence joints per subject, else flag).

---

## 14. Run Cheatsheet

```bash
# Live demo, all models (WSL lhm): A-pose + joints + abdomen x-z/y-x + demo PNG per model
python tools/workflows/demo_all.py

# Anthropometry + A-pose (WSL lhm): fused betas → A-pose mesh + joints
python tools/anthro/lhm_anthropometry.py --subject <dir> --out <dir>     # --from-betas <fused_betas.npy> to re-measure fast

# Depth-fusion abdomen contour
python tools/geometry/chest_contour.py --image <img> --out <dir> [--shoulder-cm N]

# Inspect a cloud (ply | npz | glb)
python tools/render/inspect_cloud.py <path> <out>

# SSP-3D benchmark summary
python tools/benchmark/bench_all.py --dataset ssp3d --methods camerahmr_sota meshmap_full published_shapy --out benchmarks/results/ssp3d_smoke

# Overlay final fused geometry back onto source photos
python tools/render/overlay_final_mesh.py runs/subject_s1

# LHM single image (WSL): conda activate lhm; cd ~/LHM
python -m LHM.launch infer.human_lrm model_name=LHM-MINI image_input=<folder> export_mesh=True motion_seqs_dir=None
```

See also: `docs/setup/HMR_BACKENDS_SETUP.md` (backend envs), `docs/setup/DATASETS.md` (benchmark
data), `docs/archive/` (point-in-time status snapshots).
