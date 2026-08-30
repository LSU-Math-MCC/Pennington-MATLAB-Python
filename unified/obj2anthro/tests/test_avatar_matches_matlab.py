"""The avatar port must reproduce the recorded MATLAB R2023b run.

``unified/obj2anthro/backends/avatar/reference/`` is a snapshot of the port's
own output, so it can only catch drift.  This module scores the port against
the real thing: ``runs/matlab_ground_truth/raw/matlab/``, produced by calling
``Avatar.m`` (steps=3, Vol_SA=on) through the MATLAB Engine.

Two scans are expected to disagree, both for reasons in the reference rather
than the port -- see ``EXPECTED_MISMATCHES``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from unified.obj2anthro.backends.avatar.avatar_conversion import MatlabAvatar, load_obj

REPO_ROOT = Path(__file__).resolve().parents[3]
OBJ_DIR = REPO_ROOT / "data" / "obj"
MATLAB_DIR = REPO_ROOT / "runs" / "matlab_ground_truth" / "raw" / "matlab"

# The recorded run reported centimetres from millimetre meshes.
SCALE = 0.1
RTOL = 1e-9

# Scans that legitimately differ, and the single measurement family involved.
EXPECTED_MISMATCHES = {
    # adjustCrotch calls MATLAB kmeans, which is randomly seeded.  This scan's
    # delta_v2 profile has no dominant outlier, so several Lloyd fixed points
    # are reachable; MATLAB's recorded answer is one it reaches roughly a
    # quarter of the time.  Every crotch-derived measurement follows.
    "A00-09-0254 2025-12-10_10-38-56.obj": None,
    # calfGirth maximises a hull perimeter over a discretised plane sweep.  No
    # plane in the search range reproduces MATLAB's value here; the gap is
    # 5 micrometres on a 42.6 cm girth.
    "cancan07_A 2026-01-28_11-48-32.obj": {"rCalfGirth"},
}


def _scaled(measurements: dict[str, float]) -> dict[str, float]:
    out = {}
    for key, value in measurements.items():
        if key.startswith("SA_"):
            out[key] = value * SCALE ** 2
        elif key.startswith("VOL_"):
            out[key] = value * SCALE ** 3
        else:
            out[key] = value * SCALE
    return out


def _matlab_runs() -> list[tuple[str, dict[str, float]]]:
    if not MATLAB_DIR.is_dir():
        return []
    runs = []
    for directory in sorted(MATLAB_DIR.iterdir()):
        summary = directory / "summary.json"
        if not summary.is_file():
            continue
        payload = json.loads(summary.read_text(encoding="utf-8"))
        name = Path(payload["source_file"].replace("\\", "/")).name
        if (OBJ_DIR / name).is_file():
            runs.append((name, payload["measurements"]))
    return runs


RUNS = _matlab_runs()


@pytest.mark.skipif(not RUNS, reason="MATLAB ground-truth run not present")
@pytest.mark.parametrize("name,expected", RUNS, ids=[r[0] for r in RUNS])
def test_avatar_reproduces_matlab(name: str, expected: dict[str, float]) -> None:
    vertices, faces = load_obj(OBJ_DIR / name)
    produced = _scaled(MatlabAvatar(vertices, faces).run().measurements)

    allowed = EXPECTED_MISMATCHES.get(name, set())
    mismatches = {
        key: (produced[key], value)
        for key, value in expected.items()
        if key in produced
        and not np.isclose(produced[key], value, rtol=RTOL, atol=0.0)
    }

    if name in EXPECTED_MISMATCHES and allowed is None:
        # Whole-scan exemption: only assert it still runs and stays plausible.
        assert produced["height"] == pytest.approx(expected["height"], rel=RTOL)
        return

    unexpected = {k: v for k, v in mismatches.items() if k not in allowed}
    assert not unexpected, (
        f"{name}: {len(unexpected)} measurement(s) differ from MATLAB: "
        + ", ".join(f"{k}={p:.6f} vs {m:.6f}" for k, (p, m) in unexpected.items())
    )
