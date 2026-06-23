"""Multi-person subject instance + subject track association.

Every visible person becomes a separate SubjectInstance. Across images we group
instances into SubjectTracks. This module keeps the deterministic geometry/appearance
association logic that tests exercise; learned embeddings plug in via the backend.
"""
from __future__ import annotations

import numpy as np

from ..types import SubjectInstance, SubjectTrack


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def bbox_center(bbox):
    x0, y0, x1, y1 = bbox
    return np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])


def bbox_area(bbox):
    x0, y0, x1, y1 = bbox
    return max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))


def associate_tracks(
    per_image_instances: list[list[SubjectInstance]],
    embeddings: dict | None = None,
    target_hint: str | None = None,
    same_subject: bool = False,
):
    """Group instances across images into subject tracks.

    Heuristic deterministic association used for tests and as a fallback:
      - within a single image every instance is a distinct subject;
      - across images, link instances by appearance embedding cosine similarity when
        provided, else by relative bbox-center proximity + area similarity.
    In same_subject mode we additionally bias toward the largest recurring person.

    Returns list[SubjectTrack]. Ambiguous links are kept separate (never silently
    merged).
    """
    tracks: list[SubjectTrack] = []

    def emb(inst):
        return embeddings.get(inst.instance_id) if embeddings else None

    def similarity(inst, track):
        # appearance first
        e = emb(inst)
        if e is not None and track.instances:
            te = emb(track.instances[-1])
            if te is not None:
                cs = float(np.dot(e, te) / (np.linalg.norm(e) * np.linalg.norm(te) + 1e-9))
                return cs
        # geometry fallback: normalized bbox center distance + area ratio
        last = track.instances[-1]
        c1 = bbox_center(inst.bbox)
        c2 = bbox_center(last.bbox)
        d = np.linalg.norm(c1 - c2)
        a1, a2 = bbox_area(inst.bbox), bbox_area(last.bbox)
        ar = min(a1, a2) / (max(a1, a2) + 1e-9)
        scale = np.sqrt(max(a1, a2)) + 1e-6
        geo = float(np.exp(-(d / (2 * scale)) ** 2)) * ar
        return geo

    link_thresh = 0.5
    for img_instances in per_image_instances:
        used_tracks = set()
        for inst in img_instances:
            best_tr, best_s = None, -1.0
            for ti, tr in enumerate(tracks):
                if ti in used_tracks:
                    continue
                s = similarity(inst, tr)
                if s > best_s:
                    best_s, best_tr = s, tr
            if best_tr is not None and best_s >= link_thresh:
                best_tr.instances.append(inst)
                used_tracks.add(tracks.index(best_tr))
            else:
                tr = SubjectTrack(subject_id=f"subject_{len(tracks):03d}", instances=[inst])
                tracks.append(tr)

    for tr in tracks:
        tr.confidence = float(np.mean([i.association_confidence or 0.5 for i in tr.instances]))

    if same_subject and tracks:
        # choose target: manual hint -> largest recurring person
        if target_hint is not None:
            tracks.sort(key=lambda t: 0 if any(target_hint in i.instance_id for i in t.instances) else 1)
        else:
            tracks.sort(key=lambda t: -(len(t.instances) * np.mean(
                [bbox_area(i.bbox) for i in t.instances])))
    return tracks
