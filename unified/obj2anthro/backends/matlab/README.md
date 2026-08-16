# MATLAB Avatar backend

This backend is a thin Python wrapper around the original
`MATLAB Code/Pennington/Avatar.m` implementation. It starts MATLAB lazily and
keeps one MATLAB Engine alive for every OBJ in a `run_pipeline` batch.

The normal MATLAB branch uses the legacy batch settings:

```matlab
Avatar(obj, 'steps', [3], 'Vol_SA', 'on')
```

Use it directly with:

```powershell
.\.venv\Scripts\python.exe -m unified obj2anthro `
  --input data\obj --method matlab --units auto `
  --out runs\matlab
```

`auto` and `all` run MATLAB alongside the segmentation, slice, and Python
`avatar` branches. `--method matlab-full` explicitly selects
`steps = [1 2 3]`; the cleaning/repair stage is substantially slower on the
small scan meshes in this repository and is intentionally not part of `auto`.

MATLAB R2023b's Engine package supports Python 3.9-3.11. Install the Engine
from the MATLAB installation's `extern/engines/python` directory into the
Python environment used to run the pipeline. The optional
`PENNINGTON_MATLAB_DIR` environment variable can point to another directory
containing `Avatar.m`.

`py2mat_avatar_measure.m` is only a marshaling adapter. It reads the Avatar
properties directly instead of calling `extractValues`, whose legacy
`surfaceArea.lLeg` field name does not match the constructor's `surfaceArea.lleg`
field. The unified CSV keeps MATLAB-faithful values and exposes the known
corrected ankle/length companions under the existing `*_corrected_cm` columns.

Before the call, the wrapper fan-triangulates OBJ quad faces into a temporary
copy when necessary; source OBJ files are never rewritten. If a mesh completes
landmark extraction but the legacy segmented volume closure fails, the adapter
keeps the MATLAB landmark measurements and computes whole-mesh surface area and
volume from the triangular faces. The fallback passes numeric `Vol_SA=0` to
avoid an old string-option comparison that is invalid in current MATLAB.
