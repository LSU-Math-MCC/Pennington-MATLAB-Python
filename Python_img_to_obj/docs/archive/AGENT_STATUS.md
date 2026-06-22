# Agent Status

## completed
- project skeleton (src/ layout, pyproject, Makefile, conftest)
- geometry core: camera project/backproject, transforms (quat/Procrustes), tests
- mask-gated depth selection (the central bridge) + tests
- splat assignment (projection + nearest-sample score, ambiguous resolution) + tests
- face mapping: region gating, 3D anchor lifting, face canonical frame + tests
- subject instances + tracks, multi-person association, partial-visibility handling + tests
- canonical body frame (3D joint lift -> world_to_canonical) + tests
- voxel fusion (face/body resolution split, confidence gating) + tests
- A-Frame viewer (self-contained, inline points) + aggregate viewer + tests
- single-image vertical slice + manifest + CLI smoke test (dummy backends)
- REAL backends:
  - segmentation + boxes: Ultralytics YOLOv8-seg (multi-person)
  - 2D pose (17 kpts): Ultralytics YOLOv8-pose
  - dense face landmarks (478): MediaPipe Tasks FaceLandmarker + landmark-hull mask
  - monocular depth: Depth-Anything-V2-Small (transformers)
  - 3DGS scene: colored point splats lifted from monocular depth
  - association: color-histogram appearance embedding
- AVIF/HEIF input support (pillow-avif-plugin / pi-heif)
- folder mode (parallel) + aggregate viewer
- same-subject fusion (Procrustes canonical alignment + voxel fusion + proxy mesh)

## test status
- `make test`: 40/40 passing (geometry + fusion + export + CLI smoke, all dummy-backed)

## next / possible refinements
- limb-local (region-aware) transforms beyond global canonical
- learned face/body re-ID embeddings for subject association
- native gaussian-splat WebGL rendering in the viewer (currently point rendering)
