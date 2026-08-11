#!/usr/bin/env python3
"""Measure many meshes at once and collect everything into one table.

One row per mesh, one column per measurement -- the format you want for a cohort
of scans you're going to open in Excel or load into pandas.

Examples
--------
    # every .obj in a folder
    python batch_measure.py scans/

    # recurse into subfolders, report centimetres (input is millimetres)
    python batch_measure.py scans/ --recursive --scale-to-cm 0.1

    # specific files, or shell globs
    python batch_measure.py a.obj b.obj "cohort/*/scan.obj"

    # use 8 cores
    python batch_measure.py scans/ --recursive --jobs 8

    # long-running batch you may need to restart
    python batch_measure.py scans/ --recursive --resume

Outputs (into --output, default ./batch_output):
    all_measurements.csv   one row per mesh, one column per measurement
    all_landmarks.csv      long format: file, landmark, x, y, z
    failures.csv           any mesh that could not be processed, with the reason
    batch_summary.json     run metadata, settings and per-file status
"""

from __future__ import annotations

import argparse
import csv
import glob as globlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from avatar_conversion.matlab_avatar import MatlabAvatar
from avatar_conversion.mesh_io import load_mesh

MESH_SUFFIXES = {".obj", ".ply"}

AREA_KEYS = {"SA_total", "SA_trunk", "SA_lleg", "SA_rleg", "SA_legs",
             "SA_head", "SA_rArm", "SA_lArm"}
VOLUME_KEYS = {"VOL_total"}

# Preferred column order; anything not listed is appended alphabetically so new
# measurements never silently disappear from the output.
COLUMN_ORDER = [
    "height",
    "chestGirth", "waistGirth", "hipGirth",
    "rThighGirth", "lThighGirth",
    "rCalfGirth", "lCalfGirth",
    "rAnkleGirth", "lAnkleGirth",
    "rBicepGirth", "lBicepGirth",
    "rForearmGirth", "lForearmGirth",
    "rWristGirth", "lWristGirth",
    "rArmLength", "lArmLength",
    "rLegLength", "lLegLength",
    "trunkLength", "crotchHeight", "collarScalpLength",
    "SA_total", "SA_head", "SA_trunk", "SA_rArm", "SA_lArm",
    "SA_rleg", "SA_lleg", "SA_legs",
    "VOL_total",
    "rAnkleGirth_fixed", "lAnkleGirth_fixed",
    "trunkLength_fixed", "collarScalpLength_fixed",
]


# ----------------------------------------------------------------------------
# discovery
# ----------------------------------------------------------------------------
def collect_meshes(inputs: list[str], recursive: bool) -> list[Path]:
    """Expand files, directories and glob patterns into a sorted unique list."""
    found: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            found.extend(
                p for p in path.glob(pattern)
                if p.is_file() and p.suffix.lower() in MESH_SUFFIXES
            )
        elif path.is_file():
            found.append(path)
        else:
            # Treat as a glob pattern. recursive=True enables '**'.
            matches = [Path(m) for m in globlib.glob(raw, recursive=True)]
            matches = [m for m in matches
                       if m.is_file() and m.suffix.lower() in MESH_SUFFIXES]
            if not matches:
                print(f"warning: nothing matched {raw!r}", file=sys.stderr)
            found.extend(matches)

    unique = sorted({p.resolve() for p in found})
    return unique


# ----------------------------------------------------------------------------
# per-mesh work (must be top-level so multiprocessing can pickle it)
# ----------------------------------------------------------------------------
def scale_value(name: str, value: float, scale: float) -> float:
    if name in AREA_KEYS:
        return value * scale ** 2
    if name in VOLUME_KEYS:
        return value * scale ** 3
    return value * scale


def measure_one(args: tuple[str, float]) -> dict:
    """Measure a single mesh. Returns a result dict; never raises."""
    path_str, scale = args
    path = Path(path_str)
    started = time.time()
    try:
        v, f = load_mesh(path)
        avatar = MatlabAvatar(v, f).run()

        measurements = {
            name: scale_value(name, float(value), scale)
            for name, value in avatar.measurements.items()
        }
        landmarks = {
            name: [float(c) * scale for c in np.asarray(point, dtype=float)[:3]]
            for name, point in avatar.landmarks.items()
        }
        segments = {
            name: int(len(np.asarray(idx)))
            for name, idx in avatar.segments.items()
        }
        return {
            "status": "ok",
            "file": path.name,
            "path": str(path),
            "n_vertices": int(len(avatar.v)),
            "n_faces": int(len(avatar.f)),
            "measurements": measurements,
            "landmarks": landmarks,
            "segments": segments,
            "seconds": round(time.time() - started, 2),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "file": path.name,
            "path": str(path),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=5),
            "seconds": round(time.time() - started, 2),
        }


# ----------------------------------------------------------------------------
# sanity checking
# ----------------------------------------------------------------------------
def sanity_flags(result: dict) -> list[str]:
    """Cheap plausibility checks, so a silently wrong mesh doesn't hide in a batch.

    These are warnings, not errors -- the numbers are still written out.
    """
    flags: list[str] = []
    m = result["measurements"]
    seg = result["segments"]

    left = seg.get("left_leg", 0)
    right = seg.get("right_leg", 0)
    if left and right:
        ratio = max(left, right) / max(min(left, right), 1)
        if ratio > 2.0:
            flags.append(f"leg segmentation lopsided ({left} vs {right})")

    height = m.get("height", 0)
    if height > 0:
        crotch_pct = m.get("crotchHeight", 0) / height * 100
        if not 40 <= crotch_pct <= 52:
            flags.append(f"crotch at {crotch_pct:.0f}% of height (expect ~45)")

    for a, b, label in [
        ("rThighGirth", "lThighGirth", "thigh"),
        ("rCalfGirth", "lCalfGirth", "calf"),
        ("rArmLength", "lArmLength", "arm length"),
    ]:
        va, vb = m.get(a, 0), m.get(b, 0)
        if va > 0 and vb > 0 and max(va, vb) / min(va, vb) > 1.5:
            flags.append(f"{label} asymmetry ({va:.0f} vs {vb:.0f})")

    return flags


# ----------------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------------
def ordered_columns(all_names: set[str]) -> list[str]:
    cols = [c for c in COLUMN_ORDER if c in all_names]
    cols += sorted(all_names - set(cols))
    return cols


def read_prior(out_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load previously written rows so --resume appends instead of clobbering."""
    meas: list[dict] = []
    lms: list[dict] = []
    mp_ = out_dir / "all_measurements.csv"
    lp = out_dir / "all_landmarks.csv"
    if mp_.exists():
        with open(mp_, newline="") as fh:
            meas = list(csv.DictReader(fh))
    if lp.exists():
        with open(lp, newline="") as fh:
            lms = list(csv.DictReader(fh))
    return meas, lms


def write_outputs(results: list[dict], out_dir: Path, units: str,
                  scale: float, elapsed: float,
                  prior_meas: list[dict] | None = None,
                  prior_lms: list[dict] | None = None) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = [r for r in results if r["status"] == "ok"]
    bad = [r for r in results if r["status"] != "ok"]
    prior_meas = prior_meas or []
    prior_lms = prior_lms or []
    # Rows measured in this run supersede any earlier row for the same file.
    fresh = {r["file"] for r in ok}
    prior_meas = [row for row in prior_meas if row.get("file") not in fresh]
    prior_lms = [row for row in prior_lms if row.get("file") not in fresh]

    # -- wide measurement table -------------------------------------------
    names: set[str] = set()
    for r in ok:
        names.update(r["measurements"])
    prior_extra = {k for row in prior_meas for k in row}
    names |= {k for k in prior_extra
              if not k.startswith("seg_")
              and k not in ("file", "n_vertices", "n_faces", "units", "warnings")}
    cols = ordered_columns(names)

    seg_names: list[str] = []
    for r in ok:
        for s in r["segments"]:
            if s not in seg_names:
                seg_names.append(s)
    for row in prior_meas:
        for k in row:
            if k.startswith("seg_") and k[4:] not in seg_names:
                seg_names.append(k[4:])

    meas_path = out_dir / "all_measurements.csv"
    with open(meas_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["file", "n_vertices", "n_faces", "units", "warnings"]
            + cols
            + [f"seg_{s}" for s in seg_names]
        )
        for row in prior_meas:
            w.writerow(
                [row.get("file", ""), row.get("n_vertices", ""),
                 row.get("n_faces", ""), row.get("units", units),
                 row.get("warnings", "")]
                + [row.get(c, "") for c in cols]
                + [row.get(f"seg_{s}", "") for s in seg_names]
            )
        for r in ok:
            flags = "; ".join(sanity_flags(r))
            row = [r["file"], r["n_vertices"], r["n_faces"], units, flags]
            row += [f"{r['measurements'][c]:.10g}" if c in r["measurements"] else ""
                    for c in cols]
            row += [r["segments"].get(s, "") for s in seg_names]
            w.writerow(row)

    # -- long landmark table ----------------------------------------------
    lm_path = out_dir / "all_landmarks.csv"
    with open(lm_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "landmark", "x", "y", "z"])
        for row in prior_lms:
            w.writerow([row.get("file", ""), row.get("landmark", ""),
                        row.get("x", ""), row.get("y", ""), row.get("z", "")])
        for r in ok:
            for name, point in r["landmarks"].items():
                w.writerow([r["file"], name] + [f"{c:.10g}" for c in point])

    # -- failures ----------------------------------------------------------
    fail_path = out_dir / "failures.csv"
    with open(fail_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "path", "error"])
        for r in bad:
            w.writerow([r["file"], r["path"], r["error"]])

    # -- run metadata ------------------------------------------------------
    with open(out_dir / "batch_summary.json", "w") as fh:
        json.dump({
            "run_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "n_total": len(results),
            "n_succeeded": len(ok),
            "n_failed": len(bad),
            "units": units,
            "scale_applied": scale,
            "measurement_columns": cols,
            "files": [
                {
                    "file": r["file"],
                    "path": r["path"],
                    "status": r["status"],
                    "seconds": r["seconds"],
                    **({"warnings": sanity_flags(r)} if r["status"] == "ok" else {}),
                    **({"error": r["error"]} if r["status"] != "ok" else {}),
                }
                for r in results
            ],
            "notes": [
                "Values reproduce MATLAB Avatar.m (steps=3) exactly.",
                "Girths are convex-hull perimeters over a band of vertices near "
                "each plane, not exact mesh cross-sections.",
                "Columns ending in _fixed correct known Avatar.m bugs; the "
                "unsuffixed columns reproduce MATLAB behaviour.",
            ],
        }, fh, indent=2)

    return len(ok) + len(prior_meas), len(bad)


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Batch-measure many meshes into one combined table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("inputs", nargs="+",
                    help="Files, directories, or glob patterns")
    ap.add_argument("--output", type=Path, default=Path("batch_output"),
                    help="Output directory (default: ./batch_output)")
    ap.add_argument("--recursive", action="store_true",
                    help="Recurse into subdirectories")
    ap.add_argument("--scale-to-cm", type=float, default=1.0, dest="scale",
                    help="Multiply lengths by this. Use 0.1 for millimetre "
                         "meshes to report centimetres. Default 1.0.")
    ap.add_argument("--units", default=None,
                    help="Unit label (default 'cm' if --scale-to-cm given, "
                         "else 'mesh units')")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Parallel worker processes. Use 0 for all cores.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip meshes already present in an existing "
                         "all_measurements.csv in the output directory")
    ap.add_argument("--quiet", action="store_true", help="Less output")
    args = ap.parse_args()

    units = args.units or ("cm" if args.scale != 1.0 else "mesh units")

    meshes = collect_meshes(args.inputs, args.recursive)
    if not meshes:
        print("error: no .obj/.ply files found. Check the path, and pass "
              "--recursive if they are in subfolders.", file=sys.stderr)
        return 2

    done: set[str] = set()
    prior_meas: list[dict] = []
    prior_lms: list[dict] = []
    if args.resume:
        prior_meas, prior_lms = read_prior(args.output)
        prior = args.output / "all_measurements.csv"
        if prior.exists():
            with open(prior, newline="") as fh:
                done = {row["file"] for row in csv.DictReader(fh)}
            before = len(meshes)
            meshes = [p for p in meshes if p.name not in done]
            print(f"resume: skipping {before - len(meshes)} already-measured mesh(es)")

    if not meshes:
        print("Nothing left to do.")
        return 0

    n_jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    n_jobs = max(1, min(n_jobs, len(meshes)))

    print(f"Measuring {len(meshes)} mesh(es) with {n_jobs} worker(s)...")
    started = time.time()
    payload = [(str(p), args.scale) for p in meshes]
    results: list[dict] = []

    if n_jobs == 1:
        for i, item in enumerate(payload, 1):
            r = measure_one(item)
            results.append(r)
            if not args.quiet:
                report(i, len(payload), r, units)
    else:
        with mp.Pool(n_jobs) as pool:
            for i, r in enumerate(pool.imap_unordered(measure_one, payload), 1):
                results.append(r)
                if not args.quiet:
                    report(i, len(payload), r, units)

    results.sort(key=lambda r: r["file"])
    elapsed = time.time() - started
    n_ok, n_bad = write_outputs(results, args.output, units, args.scale,
                                elapsed, prior_meas, prior_lms)

    print(f"\nDone in {elapsed:.1f}s -- {n_ok} succeeded, {n_bad} failed.")
    flagged = [r for r in results if r["status"] == "ok" and sanity_flags(r)]
    if flagged:
        print(f"{len(flagged)} mesh(es) have plausibility warnings "
              f"(see the 'warnings' column):")
        for r in flagged[:10]:
            print(f"  {r['file']}: {'; '.join(sanity_flags(r))}")
        if len(flagged) > 10:
            print(f"  ...and {len(flagged) - 10} more")
    if n_bad:
        print(f"See {args.output / 'failures.csv'} for errors.")
    print(f"\nResults -> {args.output.resolve()}")
    print(f"  all_measurements.csv  ({n_ok} rows)")
    print(f"  all_landmarks.csv")
    return 1 if n_bad else 0


def report(i: int, total: int, r: dict, units: str) -> None:
    if r["status"] == "ok":
        m = r["measurements"]
        flags = sanity_flags(r)
        mark = "!" if flags else " "
        print(f"[{i}/{total}]{mark} {r['file']}  "
              f"h={m.get('height', 0):.1f} chest={m.get('chestGirth', 0):.1f} "
              f"waist={m.get('waistGirth', 0):.1f} hip={m.get('hipGirth', 0):.1f} "
              f"{units}  ({r['seconds']}s)")
        for f in flags:
            print(f"          warning: {f}")
    else:
        print(f"[{i}/{total}]X {r['file']}  FAILED: {r['error']}")


if __name__ == "__main__":
    raise SystemExit(main())
