# Unified Pipeline Notebooks

`image_to_anthro_pipeline.ipynb` demonstrates the staged wrapper from a user
perspective:

```bash
python -m unified --input IMAGE_OR_OBJ_OR_DIR --image-method auto --anthro-method auto --units auto
```

The default notebook path is safe to execute without heavy local models. It uses
lightweight monkeypatched stage backends, but still calls
`unified.pipeline.run_pipeline()`, the same wrapper path used by the CLI.

The notebook was executed in-place with:

```text
C:\Users\Clint\AppData\Local\Programs\Python\Python312\python.exe
```

Use the executable printed by the notebook's first code cell when reproducing a
run.
