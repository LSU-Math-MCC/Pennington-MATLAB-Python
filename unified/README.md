# Unified Pennington Tool

`unified` is the staged entry point for image-to-OBJ, OBJ-to-anthropometry,
and ML experiment helpers.

```bash
python -m unified --input IMAGE_OR_OBJ_OR_DIR --image-method auto --anthro-method auto --units auto
```

Stage commands are available too:

```bash
python -m unified img2obj --input path/to/person.png --method auto
python -m unified obj2anthro --input path/to/person.obj --method auto --units auto
python -m unified.img2obj --input path/to/person.png --method auto
python -m unified.obj2anthro --input path/to/person.obj --method auto --units auto
```

## Install

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Artifacts

Every run writes beneath `runs/<run_id>/` unless `--out` is provided:

```text
runs/<run_id>/
    manifest.json
    img2obj/
    obj2anthro/
        <source_method>/
            <subject_id>/
                <anthro_method>/
                    results.csv
                    raw/
```

## Example OBJ Runs

```powershell
.\.venv\Scripts\python.exe -m unified --input "data\obj\CanCan01_A 2025-10-27_11-10-43.obj" --anthro-method slice --units auto --out runs\verify_cancan01_a
.\.venv\Scripts\python.exe -m unified obj2anthro --input data\obj --method slice --units auto --out runs\slice_all
```

## Example Image Runs

Two SSP-3D sports frames are included as lightweight image examples:

```powershell
.\.venv\Scripts\python.exe -m unified img2obj --input unified\docs\assets\ssp3d_beach_volleyball_frame.png --method dummy --out runs\img_dummy
```

Real image backends need external model weights and, for some methods, WSL or
conda environments. See `img2obj/docs/setup/`.

## Layout

| Stage | Path | Notes |
|---|---|---|
| Image to OBJ | `unified/img2obj/` | Native `src/pipeline/` package is preserved. |
| OBJ to anthropometry | `unified/obj2anthro/` | Orchestrates backend selection and canonical CSV output. |
| Segmentation backend | `unified/obj2anthro/backends/segmentation/` | Anatomical-region landmark backend. |
| Slice backend | `unified/obj2anthro/backends/slice/` | Slice-based biomarker backend. |
| ML experiments | `unified/ml/experiment/` | Historical ML/PCA/DOE and Tkinter GUI code. |
| Core OBJ data | `data/obj/` | Local OBJ examples for smoke runs and demos. |

## GUI Surfaces

- The ML Tkinter GUI is launched from `unified/ml/experiment` with
  `..\..\.venv\Scripts\python.exe python\ml_GUI.py`.
- The PCA/Ganger launcher is launched from `unified/ml/experiment` with
  `..\..\.venv\Scripts\python.exe python\PCA_App\run_ganger.py`.
- Slice and image workflows produce browser-viewable HTML artifacts under
  `runs/...`.
- The segmentation backend can open a `trimesh` viewer with
  `python -m unified obj2anthro --input data\obj\man.obj --method segmentation --show`.
