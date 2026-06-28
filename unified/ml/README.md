# ML

`experiment/` contains the historical ML, PCA, DOE, and Tkinter GUI code.

Training and cross-validation experiments are separate from inference runs under
`runs/`. Future prediction integration belongs in the staged
`python -m unified --input ...` pipeline, but this stage does not define a
standardized prediction API yet.

Use:

```bash
python -m unified ml
```

To launch the Tkinter GUI, run from the repository root:

```powershell
pushd unified\ml\experiment
..\..\.venv\Scripts\python.exe python\ml_GUI.py
popd
```

The PCA/Ganger launcher is also interactive:

```powershell
pushd unified\ml\experiment
..\..\.venv\Scripts\python.exe python\PCA_App\run_ganger.py
popd
```

It opens a folder picker, then launches the bundled Ganger executable for
`.ply`/`.mkr` batches. This path is Windows-only.
