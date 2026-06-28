# Marker Demo

`src/main.py` loads a body mesh, locates the 17 landmarks, and opens an
interactive `trimesh` window with body parts, marker spheres, and measurement
paths. It also prints the head, trunk, arm, and leg measurements to the console.

## Install

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

From this backend directory:

```powershell
cd unified\obj2anthro\backends\segmentation
..\..\..\..\.venv\Scripts\python.exe -m src.main
```

Use the venv interpreter by full path, not a bare `python`, and run it as a
module. On macOS or Linux the interpreter is `.venv/bin/python`.

`man.obj` is about 24k vertices, so landmark detection takes 15-40 seconds
before the window opens. Left-drag rotates, scroll zooms, ctrl-drag pans, `w`
toggles wireframe, `z` resets, and `q` or Esc closes. The terminal blocks until
the window closes.
