"""Pipeline configuration: thresholds, quick-mode flags, backend selection."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field


@dataclass
class Config:
    backend: str = "auto"          # preset: auto | dummy | real | lhm
    # per-stage implementation overrides (stage -> impl name); empty = use preset.
    # stages: gs, seg, pose, face, depth, assoc
    stage_impls: dict = field(default_factory=dict)
    quick: bool = True
    workers: int = 1               # CPU workers for folder mode (resolved from "auto")
    workers_gpu: int = 1

    # downscale longest image edge in quick mode (0 = no downscale)
    max_image_edge: int = 1280

    # depth selection
    depth_min: float = 1e-3
    depth_max: float = 1e6
    depth_conf_thresh: float = 0.1

    # splat assignment
    depth_tau: float = 0.05        # body depth agreement (canonical/world units fraction)
    tau_3d: float = 0.1            # nearest-sample distance for body
    person_threshold: float = 0.35

    # face (stricter)
    tau_face_depth: float = 0.03
    tau_face_3d: float = 0.05
    face_margin_px: int = 6

    # fusion voxel sizes (canonical units)
    body_voxel: float = 0.02
    face_voxel: float = 0.006

    # subject association
    assoc_iou_thresh: float = 0.3
    ambiguous_margin: float = 0.15  # min score gap to disambiguate overlap splats

    # misc
    seed: int = 0
    extra: dict = field(default_factory=dict)

    def hash(self) -> str:
        payload = json.dumps(self._hashable(), sort_keys=True).encode()
        return hashlib.sha1(payload).hexdigest()[:12]

    def _hashable(self) -> dict:
        d = asdict(self)
        # workers do not affect output content
        d.pop("workers", None)
        d.pop("workers_gpu", None)
        return d

    def to_dict(self) -> dict:
        return asdict(self)
