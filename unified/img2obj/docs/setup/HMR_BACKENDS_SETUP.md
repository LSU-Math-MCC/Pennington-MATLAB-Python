# HMR backends (CameraHMR · BLADE · SHAPY) — zero-shot setup & usage

Single-image human mesh recovery behind one **polymorphic interface**, plus the figures they
produce. This doc is written so a fresh agent can reproduce the whole thing without re-deriving the
pitfalls. **Read the "gotchas" per backend — most setup time is lost to gated/aux files, not code.**

Everything runs in **WSL** (Ubuntu) with conda at `~/miniconda3`. The repo lives on the Windows
side at `/mnt/c/Users/Clint/OneDrive/Desktop/meshmap` (call it `$REPO`). GPU: RTX 3080 Ti 16 GB.

> WSL note: a background process started with `nohup ... &` from a one-shot `wsl -e bash -lc`
> gets **killed** when the launching shell exits (WSL tears down the instance). Keep the process
> attached (run the whole script synchronously) instead of detaching.

---

## 0. Architecture (the polymorphic layer)

The three repos have **mutually incompatible deps** (different torch/CUDA/mmcv), so they cannot
coexist in one Python process. The polymorphism boundary is therefore a **per-backend runner
invoked via `conda run -n <env> python <runner>`**; each runner reduces its model's native output
to ONE normalized contract.

```
tools/hmr_compare/
  schema.py         normalized per-image npz: people verts (camera space), faces, focal, img size
  backends.py       Backend registry; .run(images, out) shells into the right conda env
  render.py         topology-agnostic multi-person renderer (uniform style across methods)
  run_camerahmr.py  runner (env: camerahmr) -> schema npz of geometry
  run_blade.py      runner (env: blade_env) -> BLADE's native overlay png (it renders its own)
tools/smplx/plots.py comparison   rows=methods x cols=images scene-overlay figure
tools/hmr/shapy/plots.py         per-vertex shape-error heatmap figure (SHAPY-style)
tools/hmr/shapy/run_shapy_ssp3d.py           SHAPY image->shape on SSP-3D (keypoint remap + demo)
```

A backend cell is **either** geometry (we render with `render.py`) **or** a pre-rendered overlay
png (the model rendered it). The compositor prefers geometry, falls back to the png.

| Backend | env | Body model | Detector | Output for figure | Strength |
|---|---|---|---|---|---|
| CameraHMR | `camerahmr` | SMPL | public detectron2 COCO | geometry npz (all people) | crowds, SOTA shape |
| BLADE | `blade_env` | SMPL-X | Sapiens/RTMDet (built-in) | native overlay png (primary person) | close-range, true perspective camera |
| SHAPY | `shapy` | SMPL-X | needs OpenPose keypoints | betas npz (shape) | semantic/metric body shape |

---

## 1. CameraHMR (env `camerahmr`, repo `~/CameraHMR`)

Already installed. Official forward = HumanFoV focal (`FLNet`) + `CameraHMR` regressor.

**Gotcha — gated detector:** `mesh_estimator.HumanMeshEstimator` uses CameraHMR's *gated*
detectron2 weights (`data/pretrained-models/model_final_f05665.pkl`). We **avoid the gate** by
using the public COCO detector (`faster_rcnn_X_101_32x8d_FPN_3x`, from `detectron2.model_zoo`), as
`run_camerahmr.py` and `tools/hmr/camerahmr/camerahmr_subjects.py` do. No download needed.

Weights present: `camerahmr_checkpoint_cleaned.ckpt`, `cam_model_cleaned.ckpt`, `SMPL_NEUTRAL.pkl`,
`smpl_mean_params.npz` (fetched once via `tools/hmr/camerahmr/fetch_camerahmr.sh` with MPI creds).

Run: `conda run -n camerahmr python tools/smplx/plots.py comparison --images ... --backends camerahmr`

---

## 2. BLADE (env `blade_env`, repo `~/blade` = NVlabs/blade, CVPR'25)

The hard one. CUDA-ops compile + a long tail of aux files. Scripts:
`tools/hmr/blade/setup_blade_env.sh` (build), `tools/hmr/blade/fetch_blade.sh` (weights),
`tools/smplx/make_smpl_uv_decomr.py` (UV asset). Logs land in `~/blade/*.log`.

### 2a. Environment / build nuances (all in `setup_blade_env.sh`)
- Python 3.9.19, `torch==2.0.1+cu118`, pytorch3d py39_cu118_pyt201 wheel.
- **`set -u` breaks it**: conda's own activate/deactivate hooks reference unbound vars and abort
  the script. Do NOT use `set -u`.
- **CUDA toolchain must be a *consistent* 11.8.** `conda install -c nvidia cuda-toolkit=11.8.0`
  leaves `cuda-nvcc` at 11.8 but pulls `cudart`/`cusparse`/`nvrtc` at **12.9** → link/ABI mismatch.
  Use the **version-label channel**: `conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit`
  (pins every component to 11.8). Also install conda `gcc_linux-64=11`/`gxx_linux-64=11`.
  Guard the toolkit install on **`bin/nvcc`** existing, not on gcc.
- **Header path**: conda puts CUDA headers/libs under `$CONDA_PREFIX/targets/x86_64-linux/{include,lib}`
  but torch's `cpp_extension` looks in `$CUDA_HOME/{include,lib}`. Add the `targets/...` dirs to
  `CPATH`/`CPLUS_INCLUDE_PATH` and `LIBRARY_PATH`/`LD_LIBRARY_PATH`, else mmcv fails with
  `fatal error: cusparse.h: No such file` (compile) then `cannot find -lcudart` (link).
  Do **not** blindly `ln -sf targets/include/* into include/` — it shadows real headers; prefer paths.
- **Editable installs fail** (`mmcv`, `sapiens/*`, `blade`): the env's legacy setuptools lacks the
  PEP 660 `build_editable` hook → install **non-editable** (`pip install .`, not `-e .`), with
  **`--no-build-isolation`** (so mmcv's ops build sees the env's torch + nvcc), and **`--no-deps`**
  on the `sapiens`/`blade` packages so they don't pull `mmcv` from PyPI and clobber the compiled ops.
- `chumpy` (old) breaks pip lines under build isolation → install it on its own line:
  `pip install --no-build-isolation chumpy || true`.
- **`--no-deps` leaves mm* runtime deps missing:** install `xtcocotools` (mmpose; `BLADE_API`
  import fails without it) and `shapely` (mmdet; without it `from mmdet.apis import ...` fails deep
  in `mmdet.structures.mask`, so BLADE's `has_mmdet=False` → "Please install mmdet to run the demo").
  `terminaltables` too.
- **mediapipe must keep the legacy `solutions` API**: BLADE calls `mp.solutions.pose.Pose(...)`.
  mediapipe ≥0.10.15 (e.g. 0.10.35) removed `mp.solutions` → `AttributeError: module 'mediapipe'
  has no attribute 'solutions'` at detection time. Pin **`mediapipe==0.10.14`** — the last version
  with `solutions` that *also* uses protobuf 4 (4.25.x). Do NOT use 0.10.9: it forces protobuf
  3.20.3, which then breaks `wandb` (BLADE imports it) with
  `cannot import name 'Imports' from wandb.proto...`. 0.10.14 keeps mediapipe + protobuf4 + wandb happy.
- Build order: mmcv → sapiens(engine,pretrain,pose,det,seg) → blade → aios ops
  (`aios_repo/models/aios/ops/setup.py`) → `torch-trust-ncg`. Verify: `from mmcv.ops import nms`.

### 2b. Weights — **all public on HuggingFace, no login needed** (`fetch_blade.sh`)
- `McMvMc/BLADE` → `epoch_2.pth` (2.4 GB; fallback `McMvMc/BLADE_backup`).
- `depth-anything/Depth-Anything-V2-Metric-Hypersim-Large` → `depth_anything_v2_metric_hypersim_vitl.pth`.
- `ttxskk/AiOS` → `aios_checkpoint.pth`.
- `facebook/sapiens-pose-bbox-detector` → `rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth`.
- `facebook/sapiens-pose-1b` → `sapiens_1b_goliath_best_goliath_AP_639.pth` (4.4 GB).
  (The `facebook/sapiens-*` repos are *not* gated in practice; downloaded unauthenticated fine.)
All land in `~/blade/pretrained/...` per `fetch_blade.sh`.

### 2c. Body-model aux files BLADE/AiOS load at init (the slow tail)
BLADE errors **one missing file per full model-reload**, so stage them ALL up front
(grep for paths: `grep -rhoE '(transfer_data|pretrained|body_models|config_files|data)/[A-Za-z0-9_./-]+\.(pkl|npz|npy|yaml|json|ckpt|pth)' ~/blade/{blade,aios_repo,smplx_repo}`):

| File (under `~/blade/body_models/...`) | Where we got it |
|---|---|
| `smpl/SMPL_{NEUTRAL,MALE,FEMALE}.pkl`, `smplx/SMPLX_*.npz` | symlinked from `~/shapy` and `~/LHM` (no re-download) |
| `smpl/smpl_uv_decomr.npz` | **generated** from `smpl_uv.obj` (in `smpl_uv_20200910.zip`) via `tools/smplx/make_smpl_uv_decomr.py` — keys `verts_uv, faces_uv, faces, vt2v` (mmhuman3d layout) |
| `smpl/J_regressor_extra.npy`, `smpl/J_regressor_h36m.npy` | symlinked from `~/STRAPS-3DHumanShapePose/additional/` |
| `smpl/smpl_mean_params.npz` | symlinked from `~/LHM/.../smpl_mean_params.npz` |
| `smplx/SMPLX_to_J14.pkl`, `MANO_SMPLX_vertex_ids.pkl`, `SMPL-X__FLAME_vertex_ids.npy`, `smplx_flip_correspondences.npz` | symlinked the **whole** `~/LHM/pretrained_models/human_model_files/smplx/` set |
| `pretrained/transfer_data/smplx2smpl_deftrafo_setup.pkl` | **synthesized** nearest-vertex matrix `{ 'mtx': csr (6890,10475) }`. BLADE builds the smplx→smpl transfer unconditionally in `build_body_models_and_conversion()` but only *uses* it when `convert_to_smpl=True` (we keep it False), so a loadable placeholder suffices. The real (gated) file is on the SMPL-X site's transfer_model data — only needed if you want BLADE to emit SMPL. |

### 2d. Other gotchas
- **`demo_images/*.jpg` are git-LFS pointer stubs** (132-byte text) after a `--depth 1` clone — BLADE
  errors `cannot identify image file`. Use real images (e.g. SSP-3D pngs) or `git lfs pull`.
- `single_gpu_test`/`forward_test` **swallow exceptions** (try/except) → failures look like "no
  output". Check that an overlay/`.pth` was actually written.
- The state_dict "unexpected/missing key … body_model.*" warnings on load are **benign** (body-model
  buffers); BLADE runs fine.
- BLADE's side-by-side jpg = `[ full original padded image (top) ; gray SMPL-X mesh on white
  (bottom) ]`. `run_blade.py` extracts the gray mesh (low-saturation, non-white) and composites it
  onto the BLADE input image, marking the subject with a **blue box** by default (`--mark outline`
  keeps the older contour style).
- BLADE is close-range **single-subject**; the comparison registry crops BLADE inputs first. For
  SSP-3D, `tools/benchmark/ssp3d_target_crops.py` uses the GT target box to choose the target
  detector box, then writes the largest crop that contains that target while excluding separable
  other subjects. The old padded GT crop remains available via `run_blade_ssp3d.py --crop-mode gt-pad`.
- **Never construct `BLADE_API` twice in one process** → `BN is already registered in
  torch.nn.modules.batchnorm` (mmcv registry double-registration). So we **load BLADE once and pass
  ALL images in one `batch_list`** (`samples_per_gpu=1`): `id_list == batch_list keys`, so `.pth`
  files are `{stem}.pth`; overlay jpgs are ~60 s apart so distinct timestamps map by mtime order.
  Per-image *reload* was the bottleneck (Sapiens-1B is 4.4 GB) — one load handles N images.

Run: `conda run -n blade_env python tools/hmr/blade/run_blade.py --images <real.png> --out <dir>`

Batch SSP-3D subject-shape run (after checking the crops visually):
```bash
conda run -n camerahmr python tools/benchmark/ssp3d_target_crops.py \
  --all-subjects --out runs/ssp3d_target_crops

conda run -n blade_env python tools/hmr/blade/run_blade_ssp3d.py \
  --all-subjects --crop
```

---

## 3. SHAPY (env `shapy`, repo `~/shapy`)

`~/shapy/regressor` image→shape regressor. Body models + the A2S attribute models are present;
the **image regressor checkpoint is gated**.

### Gated data — `shapy_data.zip` (~2 GB)
From an account at **https://shapy.is.tue.mpg.de/** (accept license), via
`~/shapy/data/download_data.sh` (prompts user/pass) — or the website's download page. Companion
sets `ModelAgencyData.zip` (attributes/keypoints) and `HBW_low_resolution.zip` (HBW val images).
Unzip into `~/shapy/data/` → gives `data/trained_models/shapy/SHAPY_A/checkpoints/best_checkpoint`
(1.43 GB, the image regressor) plus `b2a`/`a2b` models. Demo cfg: `configs/b2a_expose_hrnet_demo.yaml`,
`output_folder=../data/trained_models/shapy/SHAPY_A`.

### Gotcha — the `shapy` env was never run for this demo (missing deps + omegaconf)
- **omegaconf must be 2.0.6.** SHAPY's config uses custom `StringTuple`/`FloatTuple` NewTypes and
  nested-tuple defaults that omegaconf ≥2.1 (env had 2.3.0) rejects
  (`Unexpected type annotation: StringTuple`, `Cannot convert 'tuple' to string`). pip ≥24.1
  refuses omegaconf 2.0.6's legacy metadata, so: `pip install "pip<24.1"` then
  `pip install omegaconf==2.0.6`. (Also relaxed `groups: Tuple[str]` → `Any` in
  `human_shape/config/network_defaults.py`, harmless either way.)
- **Missing runtime deps** (install into `shapy`): `jpeg4py opencv-python-headless fvcore nflows
  yacs torchgeometry smplx pyrender kornia "pytorch-lightning<2" torchmetrics`.
- **`body_measurements` / `mesh_mesh_intersection`**: `body_measurements` (in
  `~/shapy/mesh-mesh-intersection/body_measurements`) imports the CUDA `mesh_mesh_intersection` at
  module load, but that op only computes measurements (not betas). **Stub it** instead of building:
  put a no-op `MeshMeshIntersection(nn.Module)` at `site-packages/mesh_mesh_intersection/__init__.py`
  and symlink `body_measurements` into site-packages **so the import succeeds**, AND disable the
  measurement path so the op is never actually called: pass
  `--exp-opts network.smplx.compute_measurements=False` to `demo.py` (the regressor *does* call
  MeshMeshIntersection in its forward otherwise). Betas come from the regressor head regardless.
- **numpy must be <1.24** (env had ≥1.24): SHAPY code uses removed `np.int`. Pin `numpy==1.23.5`
  (still ships `np.int`; fine with torch 1.13).
- The regressor demo's output arg is `--output-folder` (NOT `--demo_output_folder`, which is the
  *attributes* demo).

### Gotcha — needs OpenPose BODY-25 keypoints
SHAPY's demo `openpose` dataset reads per-image `pose_keypoints_2d` json. SSP-3D ships COCO-17
`joints2D`, so `tools/hmr/shapy/run_shapy_ssp3d.py` **remaps COCO-17 → BODY-25** (neck = shoulder mid, midhip =
hip mid), lays out `{images/, openpose/}`, runs the demo with `--save-params`, and collects each
image's SMPL-X `betas` into `runs/SHAPY_ssp3d.npz`.

Run: `cd ~/shapy/regressor && conda run -n shapy python $REPO/tools/hmr/shapy/run_shapy_ssp3d.py --n-subjects 6`

---

## 4. Figures

```bash
# Scene overlay (rows = methods, cols = images) — CameraHMR + BLADE
conda run -n camerahmr python tools/smplx/plots.py comparison \
  --images <img1> <img2> --backends camerahmr blade --out runs/HMR_COMPARE.png

# Body-shape heatmap teaser (input | GT | SHAPY | CameraHMR | BLADE), per-vertex error to GT surface
conda run -n camerahmr python tools/hmr/shapy/plots.py --methods SHAPY CameraHMR BLADE --n 6 --out runs/SHAPY_TEASER.png
conda run -n camerahmr python tools/hmr/shapy/plots.py --skinny --methods SHAPY CameraHMR BLADE --n 6 --out runs/SHAPY_TEASER_skinny.png

# Score the dataset (mean vertex-to-GT-surface mm + CameraHMR PVE-T-SC) -> runs/SSP3D_SCORES.json
conda run -n camerahmr python tools/benchmark/score_ssp3d.py
```
The teaser colors each predicted mesh by **point-to-surface distance to the GT mesh** (mm), which is
topology-agnostic — SMPL (CameraHMR) and SMPL-X (SHAPY, BLADE) are directly comparable; **every
vertex incl. hands/face is colored** (clamps to the top color past 50 mm). Subjects are selected by
`tools/benchmark/ssp3d_subjects.py`: farthest-point sampling for diversity, or `select_skinny` (smallest
volume/height³) for the skinny panel — the same selector the runners use so SHAPY/BLADE betas align.

**Scores (SSP-3D, vertex-to-GT-surface mm):** CameraHMR 9.0 (PVE-T-SC 11.47 ≈ published 11.6) <
SHAPY 25.1; cropped BLADE is 25.3 on one representative frame for each of the 62 SSP-3D subjects
(`62/62` subjects, `62/311` rows). CameraHMR is SOTA shape; BLADE is a close-range pose/camera
method, and a full 311-frame BLADE run is a larger frame-variance pass.

Notebook: `notebooks/hmr_backends_demo.ipynb` (run from a WSL VS Code window with the native
`camerahmr` kernel; BLADE/SHAPY are
invoked as subprocesses). Sections: CameraHMR · BLADE · scene compare · teaser (diverse + skinny) ·
per-person geometry · dataset score.
