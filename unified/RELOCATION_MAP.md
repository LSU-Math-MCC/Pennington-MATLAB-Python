# Unified Layout Map

This file records the current source layout for the combined project.

| Area | Path |
|---|---|
| Staged wrapper | `unified/` |
| Image to OBJ | `unified/img2obj/` |
| OBJ to anthropometry | `unified/obj2anthro/` |
| Segmentation backend | `unified/obj2anthro/backends/segmentation/` |
| Slice backend | `unified/obj2anthro/backends/slice/` |
| ML experiments and GUI | `unified/ml/experiment/` |
| Core OBJ data | `data/obj/` |

## Run Contract

Default layout:

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

- Direct OBJ inputs skip image processing and use source method `direct`.
- Image inputs must produce concrete `.obj` handoffs before anthropometry runs.
- Mixed directories are valid: direct OBJs go straight to `obj2anthro`; images go
  through `img2obj` first.
- Root CLI exit code is `0` only for `success`; `partial` and `failed` are
  nonzero for automation.
- `runs/` stays untracked.

## Path Bridges

- `unified/obj2anthro/backend_registry.py` resolves the segmentation backend at
  `unified/obj2anthro/backends/segmentation/`.
- `unified/obj2anthro/backend_registry.py` resolves the slice backend at
  `unified/obj2anthro/backends/slice/slice.py`.
- `unified/img2obj/__init__.py` adds `unified/img2obj/src` to `sys.path` before
  delegating to `pipeline.run.main`.
- Core OBJ examples live under `data/obj/`.
