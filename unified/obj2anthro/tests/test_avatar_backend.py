"""The avatar backend must keep reproducing the frozen MATLAB reference."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from unified.obj2anthro.backend_registry import BACKENDS, PipelineOptions
from unified.obj2anthro.pipeline import AUTO_BACKENDS
from unified.obj2anthro.schema import AVATAR_FIELD_MAP, CANONICAL_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[3]
OBJ_DIR = REPO_ROOT / "data" / "obj"
REFERENCE_DIR = (
    REPO_ROOT / "unified" / "obj2anthro" / "backends" / "avatar" / "reference"
)

# The reference run reported centimetres from millimetre meshes.
SCALE = 0.1

AREA_KEYS = {
    "SA_total", "SA_head", "SA_trunk", "SA_rArm", "SA_lArm",
    "SA_rleg", "SA_lleg", "SA_legs",
}
VOLUME_KEYS = {"VOL_total"}

SEGMENT_COLUMNS = {
    "seg_left_arm": "left_arm",
    "seg_right_arm": "right_arm",
    "seg_legs": "legs",
    "seg_head": "head",
    "seg_trunk": "trunk",
    "seg_left_leg": "left_leg",
    "seg_right_leg": "right_leg",
}

NON_MEASUREMENT = {"file", "n_vertices", "n_faces", "units", "warnings", *SEGMENT_COLUMNS}


def scale_for(key: str) -> float:
    if key in AREA_KEYS:
        return SCALE ** 2
    if key in VOLUME_KEYS:
        return SCALE ** 3
    return SCALE


def load_reference_rows() -> dict[tuple[int, int, float], dict[str, str]]:
    path = REFERENCE_DIR / "all_measurements.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # Several scans share a vertex/face count, so height disambiguates them.
    return {
        (int(row["n_vertices"]), int(row["n_faces"]), round(float(row["height"]), 2)): row
        for row in rows
    }


def avatar_modules():
    pytest.importorskip("scipy")
    from unified.obj2anthro.backends.avatar.avatar_conversion import MatlabAvatar, load_obj
    from unified.obj2anthro.backends.avatar.avatar_conversion.matlab_ops import fix_orientation

    return MatlabAvatar, load_obj, fix_orientation


def matched_cases():
    """Pair each source OBJ with its reference row via (vertices, faces, height)."""
    reference = load_reference_rows()
    if not reference or not OBJ_DIR.is_dir():
        return []

    MatlabAvatar, load_obj, fix_orientation = avatar_modules()
    cases = []
    for obj in sorted(OBJ_DIR.glob("*.obj")):
        try:
            vertices, faces = load_obj(obj)
        except Exception:
            continue
        oriented = fix_orientation(vertices, faces)
        height = round(float(oriented[:, 2].max() - oriented[:, 2].min()) * SCALE, 2)
        row = reference.get((len(vertices), len(faces), height))
        if row is not None:
            cases.append((obj, row))
    return cases


def test_avatar_is_registered_and_runs_under_auto():
    assert "avatar" in BACKENDS
    assert "avatar" in AUTO_BACKENDS


def test_avatar_columns_are_canonical():
    for target in AVATAR_FIELD_MAP.values():
        for column in (target if isinstance(target, tuple) else (target,)):
            assert column in CANONICAL_COLUMNS


@pytest.mark.parametrize("obj_path,reference", matched_cases(), ids=lambda item: getattr(item, "stem", ""))
def test_matches_frozen_reference(obj_path, reference):
    MatlabAvatar, load_obj, _ = avatar_modules()
    vertices, faces = load_obj(obj_path)
    avatar = MatlabAvatar(vertices, faces).run()

    for key, expected in reference.items():
        if key in NON_MEASUREMENT:
            continue
        got = avatar.measurements[key] * scale_for(key)
        want = float(expected)
        # The reference CSV is printed at ~10 significant digits.
        assert got == pytest.approx(want, rel=1e-8), key

    for column, segment in SEGMENT_COLUMNS.items():
        assert len(avatar.segments[segment]) == int(reference[column]), column


def test_backend_emits_raw_artifacts(tmp_path):
    cases = matched_cases()
    if not cases:
        pytest.skip("no reference-matched OBJ available")
    obj_path, _ = cases[0]

    options = PipelineOptions(units="auto", output_dir=tmp_path, save_images=False)
    row = BACKENDS["avatar"].run(obj_path, options)

    assert row["height_cm"] > 0
    artifact_dir = next((tmp_path / "avatar").iterdir())
    for name in ("measurements.csv", "landmarks.csv", "segments.csv", "summary.json"):
        assert (artifact_dir / name).is_file(), name


def test_landmarks_match_frozen_reference():
    path = REFERENCE_DIR / "all_landmarks.csv"
    cases = matched_cases()
    if not path.is_file() or not cases:
        pytest.skip("no reference landmarks available")

    with path.open(encoding="utf-8") as handle:
        by_file: dict[str, dict[str, tuple[float, float, float]]] = {}
        for row in csv.DictReader(handle):
            by_file.setdefault(row["file"], {})[row["landmark"]] = (
                float(row["x"]), float(row["y"]), float(row["z"])
            )

    MatlabAvatar, load_obj, _ = avatar_modules()
    checked = 0
    for obj_path, reference in cases:
        expected = by_file.get(reference["file"])
        if not expected:
            continue
        vertices, faces = load_obj(obj_path)
        avatar = MatlabAvatar(vertices, faces).run()
        for name, point in expected.items():
            got = np.asarray(avatar.landmarks[name], dtype=float) * SCALE
            assert np.allclose(got, point, rtol=1e-6, atol=1e-6, equal_nan=True), (
                f"{reference['file']} {name}"
            )
        checked += 1

    assert checked > 0
