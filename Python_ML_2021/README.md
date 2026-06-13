# Python_ML_2021

This folder contains the cleaned import of the old GitLab repository:

https://gitlab.com/kiarashws/body-shape-ml.git

It was moved into this GitHub repository under `Python_ML_2021/` with `git subtree`, using the GitLab `master` branch as the source. The intent is to keep the useful project history while placing the old MATLAB/Python body-shape work inside the current Pennington MATLAB/Python repository.

## What Is Here

- `python/`: Python machine-learning, data processing, PCA, GUI, and experiment code.
- `Matlab/`: MATLAB and Octave body-shape scripts.
- `misc/`: older exploratory PCA/ML notebooks and support files that were small enough to keep.
- `old_ML/`: earlier ML scripts retained for reference.
- `documentation/`: imported project documentation.
- `requirements.txt`: Python package list from the original project.

## What Was Cleaned

The original GitLab history contained large generated artifacts and local environment files that made the import too large for GitHub and noisy for future work. These were removed from the imported history instead of using Git LFS.

Removed from history:

- Large NHANES spreadsheets over GitHub's file-size limit.
- Large generated notebook/model artifacts, including `MeshFit.nb` and `LDL_risk_threshold_clf_1909261611.joblib`.
- Generated PCA/MKR output meshes and marker files under `python/PCA_App/process/MKR/output/`.
- Generated report artifacts under `python/reports/` and `python/PCA_App/reports/`.
- Jupyter checkpoint folders.
- Embedded virtual environments under `misc/PCA/venv/` and `misc/PCA2/pyt3/`.
- Zip archives of duplicated working folders.

The cleanup preserves source code, small reference data, notebooks, documentation, and the old project structure where practical.

## Octave Notes From The Original README

Packages to install and load when using Octave to run `Avatar.m`:

```text
pkg install -forge geometry
pkg install -forge statistics
pkg load geometry
pkg load statistics
```

Example call:

```text
Avatar Styku_01.obj
```

Octave compatibility notes:

- Avoid MATLAB's `round` with precision arguments; plain `round` is okay.
- Replace `incenter(triangulation([1,2,3],p),1)` with `centroid(delaunay(p),1)`.
- Leg volume logic was commented out in the original code because it still needed debugging.
