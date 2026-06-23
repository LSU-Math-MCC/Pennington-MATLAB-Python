# Marker Demo

`src/main.py` loads a body mesh, locates the 17 landmarks, and opens an interactive
trimesh window with the body parts, marker spheres, and measurement paths. It also
prints the head, trunk, arm, and leg measurements to the console. There is nothing to
write; the demo already exists. It runs on man.obj and produces 17 markers.

## Run

Work from `Python_Fall2025` (the directory with `src/`, `model_files/`, and
`requirements.txt`):

```powershell
cd Python_Fall2025
.venv\Scripts\python.exe -m src.main
```

Use the venv interpreter by full path, not a bare `python`, and run it as a module, not
as `python src/main.py`. On macOS or Linux the interpreter is `.venv/bin/python`.

man.obj is ~24k vertices, so landmark detection takes 15-40 seconds before the window
opens. Left-drag rotates, scroll zooms, ctrl-drag pans, w toggles wireframe, z resets,
q or Esc closes. The terminal blocks until the window closes.

## Install

Skip this if `.venv` already exists.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

requirements.txt pins pyglet 1.5.31, which the viewer requires. A fresh clone reproduces
the demo by creating its own venv and installing from requirements.txt.

## Expected output

A standing figure with semi-transparent body parts, 17 marker spheres (nose; collar,
crotch, two armpits, two hips; shoulder, wrist, and highest point per arm; foot and ankle
per leg), and black measurement paths. After the window closes, the head, trunk, arm, and
leg measurements print in cm.

## Other meshes

`src/main.py` line 71 selects the input:

```python
body = Body("model_files/man.obj")
```

Changing the quoted path to another file in `model_files/` reruns the same pipeline on
that mesh. man.obj is the canonical input and the one to verify against. penn-mesh-1 also
runs but is a T-pose, so its arm markers land on the shoulders. penn-mesh-2 does not run
as-is; it is in millimeter scale and the landmark code expects roughly unit scale.

Restore line 71 to man.obj when done so the tree stays clean.

## Notes

No window means the global Python ran instead of the venv. The global install is usually
pyglet 2, which does not open the window. Use the venv interpreter.

Do not launch the demo detached. pyglet exits immediately when fire-and-forgotten. Run it
in the foreground and let it block.

No console output during the run is expected. The prints flush when the process exits,
i.e. when the window closes.
