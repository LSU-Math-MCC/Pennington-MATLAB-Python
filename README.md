# Pennington MATLAB to Python

This repository combines the Pennington body-scanning Python work into one
staged project. The main entry point is `python -m unified`.

## Reports

[**Three Ways to Measure a Body**](docs/geometry_report.html) ([PDF](docs/geometry_report.pdf)) — what each
anthropometry backend defines a circumference to be, where each one cuts, how
each locates an anatomical level (a fixed fraction of stature or a geometric
search), what each costs per scan, and where the four disagree. Open the file
directly; it is self-contained. Index and rebuild steps in
[`docs/`](docs/README.md).

## Why Python?

The original work was tied to MATLAB workflows. Python makes the project easier
to share, test, automate, and run without MATLAB licensing. It also gives the
team access to the scientific Python ecosystem for mesh processing, machine
learning, image processing, notebooks, and future GUIs.

The goal is not to erase the research history. The goal is to keep the useful
algorithms, data, and experiments in one layout that a new student can install,
run, test, and extend.

## One-Time Setup

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use the venv Python for the commands below:

```powershell
$py = ".\.venv\Scripts\python.exe"
```

If you are using macOS or Linux, replace `.\.venv\Scripts\python.exe` with
`.venv/bin/python`.

## Project Layout

| Path | Purpose |
|---|---|
| `unified/` | Main staged wrapper and package entry point. |
| `unified/img2obj/` | Image-to-OBJ tooling and visual reconstruction tests. |
| `unified/obj2anthro/` | OBJ-to-anthropometry orchestration. |
| `unified/obj2anthro/backends/segmentation/` | Landmark/region segmentation backend. |
| `unified/obj2anthro/backends/slice/` | Slice-based biomarker backend. |
| `unified/obj2anthro/backends/avatar/` | Pure-Python port of `Avatar.m`, faithful to the MATLAB reference. |
| `unified/obj2anthro/backends/matlab/` | Drives the real `Avatar.m` through the MATLAB Engine. |
| `unified/ml/experiment/` | Historical ML, PCA, DOE, and Tkinter GUI code. |
| `data/obj/` | Core OBJ files used for local runs and smoke tests. |

Generated outputs go under `runs/`. Every run folder gets a
`combined_measurements.csv`: one row per (subject, anthropometry method) with
the method, its runtime, and every measurement column side by side.

## Fast Smoke Tests

Run the test suites from this folder:

```powershell
& $py -m pytest unified\tests unified\obj2anthro\tests -q
pushd unified\img2obj
..\..\.venv\Scripts\python.exe -m pytest tests -q
popd
pushd unified\obj2anthro\backends\segmentation
..\..\..\..\.venv\Scripts\python.exe -m pytest tests -q
popd
```

## Run OBJ to Measurements

Process one OBJ through the slice backend:

```powershell
& $py -m unified obj2anthro --input "data\obj\CanCan01_A 2025-10-27_11-10-43.obj" --method slice --units auto --out runs\slice_demo
```

Process all core OBJ examples:

```powershell
& $py -m unified obj2anthro --input data\obj --method slice --units auto --out runs\slice_all
```

Run every anthropometry backend:

```powershell
& $py -m unified obj2anthro --input "data\obj\CanCan01_A 2025-10-27_11-10-43.obj" --method all --units auto --out runs\all_backends_demo --no-images --no-aligned-obj
```

Every backend over every core OBJ, into one comparison table:

```powershell
& $py -m unified --input data --anthro-method auto --units auto
```

Then open `combined_measurements.csv` in the new `runs/<run_id>/` folder.
`auto` expands to `AUTO_BACKENDS` in `unified/obj2anthro/pipeline.py`. The
`matlab` method needs the MATLAB Engine for Python (CPython 3.9-3.11); on other
interpreters it records a failed row with that reason instead of aborting.

## Run the Unified Wrapper

For a direct OBJ:

```powershell
& $py -m unified --input "data\obj\CanCan01_A 2025-10-27_11-10-43.obj" --anthro-method slice --units auto --out runs\unified_obj_demo
```

For images, the lightweight dummy/test path is covered by tests. Real image
backends need their own model weights and setup. Start with:

```powershell
& $py -m unified img2obj --help
```

Then read `unified/img2obj/README.md` for the model-backed options.

## GUI Interfaces

Use these from an activated or explicit venv. Desktop GUIs must be run in the
foreground; do not launch them detached, because the viewer window owns the
process lifetime.

### ML Tkinter GUI

The ML GUI is a historical Tkinter application. Launch it from the ML experiment
folder so its relative asset paths resolve:

```powershell
pushd unified\ml\experiment
..\..\.venv\Scripts\python.exe python\ml_GUI.py
popd
```

In the window:

1. Pick the Styku, DXA, Blood, Questionnaire, and Manual input files.
2. Choose an output folder and output name.
3. Select features, targets, and a regressor.
4. Click the run button.

The GUI expects the old ShapeUp-style spreadsheets and CSVs. If a file picker
opens in the wrong place, browse to `unified/ml/experiment/python/data/`.

### PCA/Ganger Launcher

The PCA app includes a Tkinter folder picker that launches the bundled
`ganger.exe` fitting tool for `.ply`/`.mkr` batches. This is Windows-only and
expects the Ganger files already present under `python/PCA_App/ganger/`.

```powershell
pushd unified\ml\experiment
..\..\.venv\Scripts\python.exe python\PCA_App\run_ganger.py
popd
```

In the folder picker, select the directory containing the `.ply` files and their
matching `.mkr` marker files. The script creates a `fitted/` output folder next
to the selected inputs.

### Legacy Tkinter Prototype

There is also an older toy Tkinter prototype at
`unified/ml/experiment/old_ML/ML2019Summer/GUI.py`. It only opens a basic window
and is useful as historical reference, not as a production workflow:

```powershell
pushd unified\ml\experiment
..\..\.venv\Scripts\python.exe old_ML\ML2019Summer\GUI.py
popd
```

### Image/3D Viewers

Several pipelines write HTML or viewer artifacts rather than opening a desktop
GUI directly. After running a command, look under the selected `runs/...` folder
for files such as `interactive_3d.html`, `index.html`, or relight viewer HTML.
Open those files in a browser.

Slice backend browser viewer:

```powershell
& $py -m unified obj2anthro --input "data\obj\CanCan01_A 2025-10-27_11-10-43.obj" --method slice --units auto --out runs\slice_viewer
```

Then open `runs/slice_viewer/.../interactive_3d.html`.

Image-to-OBJ browser viewer:

```powershell
pushd unified\img2obj
..\..\.venv\Scripts\python.exe -m pipeline.run single --image tests\fixtures\person_stub.png --out ..\..\runs\img2obj_dummy --backend dummy --quick
popd
```

Then open `runs/img2obj_dummy/index.html`.

### Segmentation Mesh Viewer

The segmentation backend can open an interactive mesh window when `--show` is
passed:

```powershell
& $py -m unified obj2anthro --input "data\obj\man.obj" --method segmentation --units auto --show --out runs\segmentation_viewer
```

The window is controlled by the `trimesh` viewer. Close the viewer to return to
the terminal.

## Notes

- The MATLAB files are preserved under `MATLAB Code/` for comparison and
  reference.
- External HMR research stacks require model-specific setup. They are documented
  in `unified/img2obj/docs/setup/`.
- Keep new datasets under `data/` and generated outputs under `runs/`.
