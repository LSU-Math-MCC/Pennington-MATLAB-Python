> Relocation note: This project was moved verbatim from `Python_img_to_obj/` to `unified/img2obj/`.
> Its scientific implementation and internal organization were intentionally
> preserved. See `unified/RELOCATION_MAP.md`.
> Old native command: `cd Python_img_to_obj && python -m pipeline.run single --image IMG --out OUT`
> New native command: `cd unified/img2obj && $env:PYTHONPATH="src"; python -m pipeline.run single --image IMG --out OUT`
> New ergonomic unified command: `python -m unified img2obj --input IMG --out OUT`

# MeshMap - Image -> Canonical A-pose Human -> Anthropometry

Reconstruct people from legacy media (single image first, then multi-view) into a clean
**SMPL-X A-pose body with accurate joints**, carrying the real person's appearance, for
**longitudinal anthropometric back-tracing** (benchmarked against a 3D scanner). The
post-A-pose measurement extraction is owned by a separate lab; we keep girth/length only as
an internal sanity check.

Full design, framework, benchmarks, and history: **`docs/PROJECT.md`**.

## Honest Status

Two related but different paths live here:

- **`src/pipeline`** - visual canonical splat reconstruction (A-Frame demos, inspection).
  Useful, but debug artifacts still show background/depth residue and occasional held-object
  leakage; do not use those visual splats alone as anthropometric proof.
- **`tools/anthro/`, `tools/hmr/{camerahmr,blade,shapy}/`, `tools/benchmark/`** - the
  anthropometry path: multi-view evidence fused into one canonical A-pose body, then measured.
  Single-image CameraHMR numbers are baselines.

Benchmark claims are labelled by setting. SSP-3D is a sanity benchmark; HBW remains the
SHAPY-style verdict dataset when available. Multi-view / LOSO-calibrated results are not
single-image leaderboard claims. See `docs/PROJECT.md` Section 9.

---

## Install

### 1. Core + visual pipeline (Windows or Linux)

```bash
python -m pip install -e .            # core (numpy/scipy/pillow/trimesh) -> dummy backends + tests
python -m pip install -e ".[real]"    # real models: ultralytics, mediapipe, opencv, transformers, torch
```

Real backends (CPU-capable):
- **YOLOv8-seg** - multi-person instance masks + boxes
- **YOLOv8-pose** - 17-keypoint 2D pose (fallback)
- **MediaPipe FaceLandmarker** - 478 dense face landmarks + hull mask (model in `models/`)
- **Depth-Anything-V2-Small** (via `transformers`) - monocular depth
- point-splat scene lifted from monocular depth

Weights auto-download on first use (face model is fetched to `models/face_landmarker.task`).

### 2. HMR backend stack (WSL2 + CUDA)

The anthropometry backends each run in their **own conda env** (incompatible torch/CUDA/mmcv),
so they are invoked via `conda run -n <env> python <runner>` and reduce native output to one
normalized `schema.npz`. Build them per `docs/setup/HMR_BACKENDS_SETUP.md`:

| Backend | env | setup |
|---|---|---|
| CameraHMR | `camerahmr` | `tools/hmr/camerahmr/setup_camerahmr_env.sh`, `fetch_camerahmr.sh` |
| BLADE | `blade_env` | `tools/hmr/blade/setup_blade_env.sh`, `fetch_blade.sh` |
| SHAPY | `shapy` | `tools/hmr/shapy/wsl_setup_shapy.sh` |
| LHM | `lhm` | `tools/hmr/lhm/wsl_setup_lhm.sh` |

Datasets (SSP-3D, HBW) are set up per `docs/setup/DATASETS.md`.

---

## Use

### Visual pipeline (`src/pipeline`)

```bash
# single image
python -m pipeline.run single  --image datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_000029.png --out runs/ssp3d_bodybuilder_single --backend real --quick

# folder of images (parallel) -> aggregate viewer
python -m pipeline.run folder  --images "datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_*.png" --out runs/ssp3d_bodybuilder_folder --backend real --quick --workers auto

# same-subject fusion -> single fused A-Frame scene
python -m pipeline.run subject --subject "datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_*.png" --out runs/ssp3d_bodybuilder_subject --backend real --quick
```

`--backend dummy` runs the whole flow with synthetic, GPU-free data (used by tests).
Mix per-stage overrides freely, e.g. `--depth depth-anything-large --pose mediapipe`
(stage matrix in `docs/PROJECT.md` Section 5).

### Anthropometry & benchmark (WSL)

```bash
# fused betas -> A-pose mesh + joints (WSL lhm env)
python tools/anthro/lhm_anthropometry.py --subject <dir> --out <dir>

# SSP-3D benchmark summary
python tools/benchmark/bench_all.py --dataset ssp3d \
  --methods camerahmr_sota meshmap_full published_shapy --out benchmarks/results/ssp3d_smoke

# overlay the final fused geometry back on the source photos
python tools/render/overlay_final_mesh.py runs/subject_s1
```

### Notebooks (run with a WSL kernel)

The HMR notebooks (`notebooks/hmr_backends_demo.ipynb`, etc.) use WSL paths (`/mnt/c/...`,
`/home/clint/...`) and shell into the conda envs above - **they must run on a WSL kernel, not a
Windows Python kernel.** One-time setup, then connect VS Code to WSL:

```bash
# register the env as a Jupyter kernel (inside WSL)
wsl -e bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate camerahmr && \
  pip install -q ipykernel && python -m ipykernel install --user --name camerahmr \
  --display-name "Python (camerahmr/WSL)"'
```

In VS Code: install the **WSL** extension (`ms-vscode-remote.remote-wsl`), run
**"WSL: Reopen Folder in WSL"** (bottom-left should show `WSL: Ubuntu`), open the notebook, and
select the **"Python (camerahmr/WSL)"** kernel.

---

## Outputs

Every `pipeline.run` writes `manifest.json`, `index.html` (open locally),
`canonical_splats.ply`, and a `debug/` folder of step-by-step PNGs:

```
debug/input.png  mask.png  pose_overlay.png  depth_preview.png
selected_depth_overlay.png  splat_projection_overlay.png
face_region_mask.png  face_landmarks_overlay.png  canonical_preview.png
```

Subject mode additionally writes `assets/fused_canonical_splats.ply`,
`assets/fused_proxy_mesh.glb`, and `debug/{fusion_report,alignment_report}.json`.
`overlay_final_mesh.py` writes `runs/<subject>/overlays/` incl. `overlay_report.json`
(alignment + projection residuals).

## Tests

```bash
make test     # 59 tests: projection, mask-depth, assignment, face, instances, partial
              # visibility, canonical frame, fusion, export, CLI smoke, anthropometry metrics
make smoke    # dummy single-image run + smoke assertions
```

## Layout

`src/pipeline/{backends,geometry,export}` - backends behind interfaces; geometry/fusion are
model-free and fully unit-tested; export writes PLY/GLB/A-Frame.
`tools/{anthro,hmr,benchmark,texture,face,hands,geometry,render,smplx,workflows}` - the research
stack. `docs/PROJECT.md` is the consolidated project record; point-in-time snapshots live in
`docs/archive/`.
> Its scientific implementation and internal organization were intentionally
> preserved. See `unified/RELOCATION_MAP.md`.
>
> Old native command: `cd Python_img_to_obj && python -m pipeline.run single --image IMG --out OUT`
> New native command: `cd unified/img2obj && python -m pipeline.run single --image IMG --out OUT`
> New ergonomic unified command: `python -m unified img2obj --input IMG --out OUT`


