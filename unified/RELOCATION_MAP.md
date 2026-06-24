# Pennington Unified Relocation Map

This map records the preservation-first relocation into `unified/`.

## Moved Roots

| Former path | New path | README now lives at | Treatment |
|---|---|---|---|
| `Python_img_to_obj/` | `unified/img2obj/` | `unified/img2obj/README.md` | Moved verbatim; native `src/pipeline/` name preserved. |
| `Python_Fall2025/` | `unified/obj2anthro/backends/segmentation/` | `unified/obj2anthro/backends/segmentation/README.md` | Moved verbatim as the segmentation backend. |
| `Python_ML_2021/` | `unified/ml/experiment/` | `unified/ml/experiment/README.md` | Moved verbatim; historical ML/DOE code was not redesigned. |
| `Python_slice_2026/` | unchanged | unchanged | Root implementation remains unmoved and is exposed through `unified/obj2anthro/backends/slice/`. |

## Current `unified` Re-home

| Former path | New path |
|---|---|
| `unified/pipeline.py` | `unified/obj2anthro/pipeline.py` |
| `unified/schema.py` | `unified/obj2anthro/schema.py` |
| `unified/backends.py` | `unified/obj2anthro/backend_registry.py` |
| `unified/__main__.py` | `unified/obj2anthro/cli.py` |
| `unified/tests/` | `unified/obj2anthro/tests/` |
| `unified/results/` | `unified/obj2anthro/results/` |
| `unified/README.md` | `unified/obj2anthro/README.md` |

New top-level files `unified/__main__.py`, `unified/cli.py`, and
`unified/pipeline.py` now represent the staged tool. The default artifact root is
`runs/<run_id>/`, not a package-local results directory.

## Command Migration

| Prior command/context | New natural equivalent | Verification |
|---|---|---|
| `python -m unified --input OBJ --backend all --units auto` | `python -m unified obj2anthro --input OBJ --method all --units auto` | Help path tested; use `--method` in new docs. |
| Existing OBJ through complete tool | `python -m unified --input OBJ --anthro-method slice --units auto --out runs/verify_cancan01_a` | Tested on `Python_slice_2026/OBJ/CanCan01_A 2025-10-27_11-10-43.obj`. |
| `cd Python_img_to_obj && python -m pipeline.run single --image IMG --out OUT` | `cd unified/img2obj && $env:PYTHONPATH="src"; python -m pipeline.run single --image IMG --out OUT` | Tested with the dummy fixture. |
| Image via unified wrapper | `python -m unified img2obj --input IMG --method auto` | Help path and monkeypatched OBJ handoff contract tested. |
| Direct image stage module | `python -m unified.img2obj --input IMG --method dummy --out OUT` | Tested with `unified/img2obj/tests/fixtures/person_stub.png`. |
| Image, OBJ, or mixed directory through complete wrapper | `python -m unified --input IMAGE_OR_OBJ_OR_DIR --image-method auto --anthro-method auto --units auto` | Monkeypatched image-to-OBJ-to-anthropometry path tested; real image success requires a backend that emits or can be bridged to OBJ. |
| `cd Python_Fall2025 && python -m src.main ...` | `cd unified/obj2anthro/backends/segmentation && python -m src.main ...` | Native project preserved; wrapper tests cover relocated adapter paths. |
| Historical ML path under `Python_ML_2021/...` | Same relative path beneath `unified/ml/experiment/...` | Path verified; ML inference API intentionally not invented. |
| Native slice command under `Python_slice_2026` | Unchanged | Root project remains unmoved. |
| Slice through unified stage | `python -m unified obj2anthro --input OBJ --method slice` | Tested on `CanCan01_B`. |

## Unified Run Contract

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
- Duplicate subject/backend branches receive distinct output directories.
- Root CLI exit code is `0` only for `success`; `partial` and `failed` are
  nonzero for automation.
- `runs/` stays untracked.

## Path Bridges

- `unified/obj2anthro/backend_registry.py` resolves the relocated segmentation root
  at `unified/obj2anthro/backends/segmentation/`.
- `unified/obj2anthro/backend_registry.py` resolves the unmoved slice
  implementation at `Python_slice_2026/slice.py`.
- `unified/img2obj/__init__.py` adds `unified/img2obj/src` to `sys.path` before
  delegating to `pipeline.run.main`.
- CameraHMR scripts and the HMR backend registry resolve `CAMERAHMR_ROOT`, defaulting
  to `~/CameraHMR`.

## Intentional Compatibility Limits

- No global `pipeline` alias was added. Native image commands use the relocated
  project root with installation or `PYTHONPATH=src`, while the repository-root
  command is `python -m unified img2obj --input ...`.
- No global `src` alias was added for segmentation. Native segmentation commands are
  natural from `unified/obj2anthro/backends/segmentation/`.
- The old ML code was moved as historical experiment code; no standardized inference
  prediction API was invented in this relocation.
