"""Mesh loading with MATLAB readObj semantics.

IMPORTANT: do not swap this for ``trimesh.load(..., process=True)``.  trimesh's
default processing merges duplicate vertices, which silently renumbers every
index.  MATLAB's readObj does no such thing, so merging makes vertex-index
comparison with the reference impossible and shifts landmark results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse an OBJ into ``(vertices Nx3 float, faces Mx3 int0based)``.

    Handles ``f v``, ``f v/vt``, ``f v//vn`` and ``f v/vt/vn`` forms, plus
    CRLF line endings and trailing whitespace.  Polygons with more than three
    corners are fan-triangulated, matching how the reference data is stored.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh does not exist: {path}")

    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if not parts:
            continue
        tag = parts[0]
        if tag == "v":
            vertices.append([float(t) for t in parts[1:4]])
        elif tag == "f":
            corners = [int(t.split("/")[0]) for t in parts[1:]]
            # Negative OBJ indices are relative to the end of the vertex list.
            corners = [c if c > 0 else len(vertices) + 1 + c for c in corners]
            for k in range(1, len(corners) - 1):
                faces.append([corners[0], corners[k], corners[k + 1]])

    if not vertices:
        raise ValueError(f"No vertices parsed from {path}")
    if not faces:
        raise ValueError(f"No faces parsed from {path}")

    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64) - 1  # OBJ/MATLAB are 1-based

    if f.min() < 0 or f.max() >= len(v):
        raise ValueError(
            f"Face indices out of range in {path}: "
            f"got [{f.min()}, {f.max()}] for {len(v)} vertices"
        )
    return v, f


def load_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Minimal ASCII PLY reader (vertex x/y/z + triangular faces)."""
    path = Path(path)
    lines = path.read_text(errors="ignore").splitlines()

    n_vertices = n_faces = 0
    header_end = 0
    for i, line in enumerate(lines):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format" and parts[1] != "ascii":
            raise ValueError("Only ASCII PLY is supported; convert to OBJ instead.")
        if parts[0] == "element" and parts[1] == "vertex":
            n_vertices = int(parts[2])
        elif parts[0] == "element" and parts[1] == "face":
            n_faces = int(parts[2])
        elif parts[0] == "end_header":
            header_end = i + 1
            break

    body = [ln.split() for ln in lines[header_end:] if ln.strip()]
    v = np.array([[float(c) for c in row[:3]] for row in body[:n_vertices]], dtype=float)

    faces: list[list[int]] = []
    for row in body[n_vertices:n_vertices + n_faces]:
        count = int(row[0])
        corners = [int(c) for c in row[1:count + 1]]
        for k in range(1, len(corners) - 1):
            faces.append([corners[0], corners[k], corners[k + 1]])
    return v, np.asarray(faces, dtype=np.int64)


def load_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch on file extension."""
    path = Path(path)
    if path.suffix.lower() == ".ply":
        return load_ply(path)
    return load_obj(path)
