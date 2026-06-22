"""Run the pipeline over the SSP-3D bodybuilder example frames in a single process
(so the heavy models load only once), then build step montages for each run.

  folder  SSP-3D bodybuilder frames -> runs/ssp3d_bodybuilder_folder
  subject SSP-3D bodybuilder frames -> runs/ssp3d_bodybuilder_subject
"""

from __future__ import annotations
# --- tool-path bootstrap (capability-tree reorg): make sibling tool modules importable ---
import os as _os, sys as _sys
_repo = _os.path.dirname(_os.path.abspath(__file__))
while _repo != _os.path.dirname(_repo) and not _os.path.exists(_os.path.join(_repo, "pyproject.toml")):
    _repo = _os.path.dirname(_repo)
for _sub in ("smplx", "texture", "benchmark", "geometry", "anthro", "render"):
    _p = _os.path.join(_repo, "tools", _sub)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---


import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pipeline.config import Config
from pipeline.run import run_folder, run_subject
from galleries import montage_run as montage_main  # tools/render/galleries.py


def banner(msg):
    print(f"\n{'='*70}\n{msg}\n{'='*70}", flush=True)


def main():
    cfg = Config(backend="real", quick=True, workers="auto")
    ssp_bodybuilder = (
        "datasets/SSP-3D/ssp_3d/images/"
        "bodybuilding_vid_009_clip_000_person_001_frame_*.png"
    )
    results = {}

    banner("FOLDER: SSP-3D bodybuilder frames -> runs/ssp3d_bodybuilder_folder")
    t = time.time()
    m = run_folder(ssp_bodybuilder, "runs/ssp3d_bodybuilder_folder", cfg)
    results["ssp3d_bodybuilder_folder"] = (m["status"], len(m["failures"]))
    montage_main("runs/ssp3d_bodybuilder_folder")
    print("ssp3d_bodybuilder_folder", m["status"], "in", round(time.time() - t), "s", flush=True)

    banner("SUBJECT: SSP-3D bodybuilder frames -> runs/ssp3d_bodybuilder_subject")
    t = time.time()
    m = run_subject(ssp_bodybuilder, "runs/ssp3d_bodybuilder_subject", cfg)
    results["ssp3d_bodybuilder_subject"] = (m["status"], len(m["failures"]))
    montage_main("runs/ssp3d_bodybuilder_subject")
    print("ssp3d_bodybuilder_subject", m["status"], "in", round(time.time() - t), "s", flush=True)

    banner("SUMMARY")
    for k, (st, fails) in results.items():
        print(f"  {k:18s} {st:8s} failures={fails}", flush=True)


if __name__ == "__main__":
    main()
