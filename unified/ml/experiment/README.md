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
