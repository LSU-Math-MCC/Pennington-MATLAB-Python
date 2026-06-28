# ML Experiment Code

This folder contains historical body-shape ML, PCA, DOE, GUI, and MATLAB/Octave
experiment code. It is retained for research continuity, not as a polished
prediction service.

## What Is Here

- `python/`: machine-learning, data processing, PCA, GUI, and experiment code.
- `Matlab/`: MATLAB and Octave body-shape scripts.
- `misc/`: older exploratory PCA/ML notebooks and support files.
- `old_ML/`: earlier ML scripts retained for reference.
- `documentation/`: imported project documentation.
- `requirements.txt`: package list from the imported experiment environment.

## Tkinter GUI

Launch from this folder so relative GUI asset paths resolve:

```powershell
..\..\.venv\Scripts\python.exe python\ml_GUI.py
```

The GUI expects ShapeUp-style Styku, DXA, Blood, Questionnaire, and Manual data
files. Use the file-picker buttons to choose inputs, choose an output folder,
select features/targets/regressor settings, then run.

## PCA/Ganger Launcher

The PCA app includes an interactive folder picker and Ganger launcher:

```powershell
..\..\.venv\Scripts\python.exe python\PCA_App\run_ganger.py
```

Select a folder containing `.ply` meshes and matching `.mkr` marker files. The
script creates a `fitted/` folder and launches the bundled `ganger.exe` in
batches. This workflow is Windows-only.

## Legacy GUI Prototype

The older `old_ML/ML2019Summer/GUI.py` script opens a minimal Tkinter window.
It is retained as historical reference:

```powershell
..\..\.venv\Scripts\python.exe old_ML\ML2019Summer\GUI.py
```

## Octave Notes

Packages to install and load when using Octave to run `Avatar.m`:

```text
pkg install -forge geometry
pkg install -forge statistics
pkg load geometry
pkg load statistics
```

Example call from the `Matlab/` folder:

```text
Avatar Styku_01.obj
```
