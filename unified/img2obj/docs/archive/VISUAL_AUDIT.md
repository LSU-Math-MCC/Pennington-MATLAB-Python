# Visual Audit

Date: 2026-06-13

Inspected local artifacts:

- `runs/single_s5/debug/steps_single.png`
- `runs/single_s5/debug/canonical_3d.png`
- `runs/subject_s1/debug/steps_s1_1.png` through `steps_s1_5.png`
- `runs/subject_s1/debug/canonical_ortho.png`
- `runs/anthro_s1/apose_inspect.png`
- `runs/AUDIT/AUDIT_matrix.png`

## Stage Findings

- Input, person mask, pose, face region, and face landmarks are generally coherent on the inspected real runs.
- Depth and mask-gated depth are nonblank and aligned, but monocular depth creates visible sheets behind the subject in some views.
- Splat assignment is visually aligned with the person mask, but it inherits mask mistakes. Held objects and cloth can become person-owned geometry.
- Per-view canonical 3D previews are useful for debugging, not clean enough to use as anthropometric evidence by themselves.
- Subject fusion preserves useful body signal, but `runs/subject_s1/debug/canonical_ortho.png` shows background/pose residue and partial-view artifacts.
- Partial-body views (`subject_s1` views 3 and 4) should be down-weighted or excluded for stature/lower-body measurements.
- The anthropometry A-pose inspection (`runs/anthro_s1/apose_inspect.png`) is much cleaner than the visual splat fusion path: body pose and cross-section plots look coherent.
- The audit matrix reports `166/187` pass. `pifuhd` fails across the sampled set; a few face/fuse failures appear on difficult single images.

## Honest Interpretation

The visual splat pipeline is good enough for inspection/demo output and for finding failure modes. It is not, as currently visualized, proof of better anthropometry. The anthropometry claim should stay tied to reliability-gated multi-view canonical A-pose measurement, with held-object/background/partial-body views either rejected or explicitly down-weighted.
