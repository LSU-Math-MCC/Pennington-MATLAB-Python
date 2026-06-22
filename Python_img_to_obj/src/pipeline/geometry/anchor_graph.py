"""AnchorGraph: a flexible set of geometric anchors that does not assume a full
skeleton. The canonical-frame estimator consumes this graph and returns the
highest-confidence transform available, following the anchor hierarchy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class AnchorNode:
    name: str
    type: str                       # face|head|torso|limb|silhouette|clothing|depth_cluster
    position_2d: Optional[tuple] = None
    position_3d: Optional[np.ndarray] = None
    orientation: Optional[np.ndarray] = None
    scale_cue: Optional[float] = None
    confidence: float = 0.0
    visibility: float = 0.0
    source_image: Optional[str] = None


@dataclass
class AnchorGraph:
    nodes: dict = field(default_factory=dict)      # name -> AnchorNode
    edges: list = field(default_factory=list)      # (name_a, name_b)

    def add(self, node: AnchorNode):
        self.nodes[node.name] = node

    def connect(self, a: str, b: str):
        if a in self.nodes and b in self.nodes:
            self.edges.append((a, b))

    def has(self, *names) -> bool:
        return all(n in self.nodes for n in names)

    def pos(self, name):
        n = self.nodes.get(name)
        return n.position_3d if n is not None else None


DEFAULT_EDGES = [
    ("left_eye", "right_eye"), ("nose", "mouth_center"),
    ("left_shoulder", "right_shoulder"), ("shoulder_mid", "pelvis"),
    ("left_hip", "right_hip"), ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"), ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
]


def build_anchor_graph(joints_3d: dict, face_anchors: dict, source_image: str = "") -> AnchorGraph:
    g = AnchorGraph()
    for name, v in joints_3d.items():
        atype = "torso" if name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip") else "limb"
        if name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear"):
            atype = "head"
        g.add(AnchorNode(name=name, type=atype, position_3d=np.array(v[:3]),
                         confidence=float(v[3]) if len(v) > 3 else 0.5,
                         visibility=1.0, source_image=source_image))
    for name, v in (face_anchors or {}).items():
        g.add(AnchorNode(name=name, type="face", position_3d=np.array(v[:3]),
                         confidence=float(v[3]) if len(v) > 3 else 0.5,
                         visibility=1.0, source_image=source_image))
    for a, b in DEFAULT_EDGES:
        g.connect(a, b)
    return g


def anchor_tier(graph: AnchorGraph) -> str:
    """Report which anchor tier is best available (for logging / debug)."""
    if graph.has("left_eye", "right_eye") and (graph.has("nose") or graph.has("mouth_center")):
        return "face"
    if graph.has("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
        return "torso"
    if graph.has("left_shoulder", "right_shoulder"):
        return "shoulders"
    if graph.has("left_hip", "right_hip"):
        return "hips"
    if any(n.type == "limb" for n in graph.nodes.values()):
        return "limb"
    return "silhouette"
