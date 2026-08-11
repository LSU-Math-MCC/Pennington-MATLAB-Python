#!/usr/bin/env python3
"""Run the MATLAB-faithful avatar pipeline on one or more meshes.

Examples
--------
    python run_avatar.py scan.obj
    python run_avatar.py scan.obj --output results
    python run_avatar.py scans/ --recursive --scale-to-cm 0.1
    python run_avatar.py scan.obj --json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from avatar_conversion.matlab_avatar import MatlabAvatar
from avatar_conversion.mesh_io import load_mesh

MESH_SUFFIXES = {".obj", ".ply"}

# Measurements that scale as area / volume rather than length.
AREA_KEYS = {"SA_total", "SA_trunk", "SA_lleg", "SA_rleg", "SA_legs",
             "SA_head", "SA_rArm", "SA_lArm"}
VOLUME_KEYS = {"VOL_total"}


def collect_meshes(target: Path, recursive: bool) -> list[Path]:
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(f"No such file or directory: {target}")
    pattern = "**/*" if recursive else "*"
    found = sorted(
        p for p in target.glob(pattern)
        if p.is_file() and p.suffix.lower() in MESH_SUFFIXES
    )
    if not found:
        raise FileNotFoundError(f"No .obj/.ply files found under {target}")
    return found


def scaled_measurements(avatar: MatlabAvatar, scale: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, value in avatar.measurements.items():
        value = float(value)
        if name in AREA_KEYS:
            out[name] = value * scale ** 2
        elif name in VOLUME_KEYS:
            out[name] = value * scale ** 3
        elif name in ("nVertices", "nFaces"):
            out[name] = value
        else:
            out[name] = value * scale
    return out


def unit_for(name: str, units: str) -> str:
    if name in AREA_KEYS:
        return f"{units}^2"
    if name in VOLUME_KEYS:
        return f"{units}^3"
    return units


def process(path: Path, out_dir: Path, scale: float, units: str,
            emit_json: bool, quiet: bool) -> dict:
    v, f = load_mesh(path)
    avatar = MatlabAvatar(v, f).run()

    measurements = scaled_measurements(avatar, scale)
    stem = path.stem
    subject_dir = out_dir / stem
    subject_dir.mkdir(parents=True, exist_ok=True)

    with open(subject_dir / "measurements.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "value", "units"])
        for name, value in measurements.items():
            w.writerow([name, f"{value:.10g}", unit_for(name, units)])
        w.writerow(["nVertices", len(avatar.v), "count"])
        w.writerow(["nFaces", len(avatar.f), "count"])

    with open(subject_dir / "landmarks.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "x", "y", "z"])
        for name, point in avatar.landmarks.items():
            w.writerow([name] + [f"{float(c) * scale:.10g}" for c in np.asarray(point)[:3]])

    with open(subject_dir / "segments.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["segment", "n_vertices", "vertex_indices"])
        for name, idx in avatar.segments.items():
            idx = np.asarray(idx, dtype=int)
            w.writerow([name, len(idx), " ".join(map(str, idx.tolist()))])

    summary = {
        "source_file": str(path),
        "num_vertices": int(len(avatar.v)),
        "num_faces": int(len(avatar.f)),
        "units": units,
        "scale_applied": scale,
        "measurements": {
            k: {"value": v, "units": unit_for(k, units)}
            for k, v in measurements.items()
        },
        "landmarks": {
            k: [float(c) * scale for c in np.asarray(p)[:3]]
            for k, p in avatar.landmarks.items()
        },
        "segment_sizes": {k: int(len(np.asarray(x))) for k, x in avatar.segments.items()},
        "notes": [
            "Values reproduce MATLAB Avatar.m (steps=3) exactly.",
            "Circumferences are convex-hull perimeters over a band of vertices "
            "near each plane (MATLAB getCircumference/getVOnLine), not exact "
            "mesh cross-sections.",
            "Keys ending in _fixed are corrected versions of known Avatar.m bugs; "
            "the unsuffixed keys reproduce the MATLAB behaviour.",
        ],
    }
    with open(subject_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    if emit_json:
        print(json.dumps(summary, indent=2))
    elif not quiet:
        seg = {k: int(len(np.asarray(x))) for k, x in avatar.segments.items()}
        print(f"  vertices={len(avatar.v)} faces={len(avatar.f)}")
        print(f"  segments: {seg}")
        print(f"  height={measurements['height']:.2f} {units}   "
              f"chest={measurements['chestGirth']:.2f}   "
              f"waist={measurements['waistGirth']:.2f}   "
              f"hip={measurements['hipGirth']:.2f}")
        print(f"  -> {subject_dir}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(
        description="MATLAB-faithful avatar measurement pipeline (port of Avatar.m)."
    )
    ap.add_argument("input", type=Path, help="An .obj/.ply file, or a directory of them")
    ap.add_argument("--output", type=Path, default=Path("output"),
                    help="Output directory (default: ./output)")
    ap.add_argument("--recursive", action="store_true",
                    help="Recurse into subdirectories when input is a directory")
    ap.add_argument("--scale-to-cm", type=float, default=1.0, dest="scale",
                    help="Multiply lengths by this factor. Use 0.1 for a "
                         "millimetre mesh to report centimetres. Default 1.0 "
                         "(report in raw mesh units).")
    ap.add_argument("--units", default=None,
                    help="Unit label for output. Defaults to 'cm' when "
                         "--scale-to-cm is given, else 'mesh units'.")
    ap.add_argument("--json", action="store_true", help="Print the summary as JSON")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-file output")
    args = ap.parse_args()

    units = args.units or ("cm" if args.scale != 1.0 else "mesh units")

    try:
        meshes = collect_meshes(args.input, args.recursive)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    failures = 0
    for i, path in enumerate(meshes, 1):
        if not args.quiet and not args.json:
            print(f"[{i}/{len(meshes)}] {path.name}")
        try:
            process(path, args.output, args.scale, units, args.json, args.quiet)
        except Exception as exc:  # keep going through a batch
            failures += 1
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            if not args.quiet:
                traceback.print_exc(limit=3)

    if failures:
        print(f"\n{failures} of {len(meshes)} mesh(es) failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
