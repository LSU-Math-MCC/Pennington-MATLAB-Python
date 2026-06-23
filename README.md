# Pennington MATLAB to Python

The Pennington MATLAB to Python project is a collaborative undergraduate and
graduate research effort led by the LSU Department of Mathematics and the
Pennington Biomedical Research Center. The goal is to translate and preserve
body-scanning and metabolic analysis workflows in Python.

## Unified Python Tool

The Python projects are now organized as staged compartments under `unified/`.
The primary interface is:

```bash
python -m unified --input INPUT
```

Stage-local commands remain available:

```bash
python -m unified img2obj --input IMAGE_OR_IMAGE_DIR
python -m unified obj2anthro --input OBJ_OR_OBJ_DIR --method slice --units auto
python -m unified ml
```

| Stage | Path | Purpose |
|---|---|---|
| Image to OBJ | `unified/img2obj/` | Relocated `Python_img_to_obj/` image preprocessor. |
| OBJ to anthropometry | `unified/obj2anthro/` | OBJ measurement orchestration and backend adapters. |
| Segmentation backend | `unified/obj2anthro/backends/segmentation/` | Relocated `Python_Fall2025/`. |
| Slice backend | `Python_slice_2026/` | Unmoved root implementation exposed through `unified/obj2anthro/backends/slice/`. |
| ML experiments | `unified/ml/experiment/` | Relocated `Python_ML_2021/`; historical ML/DOE code, not redesigned as inference. |

See `unified/README.md` for the staged tool and `unified/RELOCATION_MAP.md` for
old-to-new command mappings.

## Why Python?

- Eliminate MATLAB licensing requirements where possible.
- Improve collaboration through Python's scientific ecosystem.
- Modernize documentation and execution paths while preserving established research
  implementations.

## Code Objectives

The codebase calculates metrics from human body scans and related datasets:

- 3D measurement from triangularized body-scan mesh objects (`.obj`).
- Biometric calculations such as body-section circumferences using geometric
  methods.
- Historical ML and DOE experiments retained for future staged integration.

As of Summer 2026, this project is being conducted under the guidance of Dr. Peter
R. Wolenski, Russell B. Long Professor of Mathematics at LSU.
