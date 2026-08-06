#!/usr/bin/env python3
"""Show and print the measurement slices taken on a mesh.

For every girth the pipeline computes, this prints where the slice was taken,
how many surface points it caught and the resulting girth -- then draws them on
the body and plots each cross-section with its convex hull.

    python show_slices.py scan.obj
    python show_slices.py scan.obj --scale-to-cm 0.1
    python show_slices.py scan.obj --dump-points        # every point, to CSV
    python show_slices.py scan.obj --only chest waist hip

Outputs into --output (default ./slice_output):
    slices_on_body.png     front and side views with every slice drawn
    slice_cross_sections.png   one panel per slice, points + hull outline
    slice_summary.csv      one row per slice
    slice_points.csv       every point of every slice (with --dump-points)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from avatar_conversion.matlab_avatar import MatlabAvatar
from avatar_conversion.mesh_io import load_mesh

# Draw order / grouping for the report.
GROUPS = [
    ("Torso", ["chest", "waist", "hip"]),
    ("Legs", ["lThigh", "rThigh", "lCalf", "rCalf", "lAnkle", "rAnkle"]),
    ("Arms", ["lBicep", "rBicep", "lForearm", "rForearm", "lWrist", "rWrist"]),
]
COLORS = {
    "chest": "#D85A30", "waist": "#BA7517", "hip": "#993C1D",
    "lThigh": "#1D9E75", "rThigh": "#0F6E56",
    "lCalf": "#5DCAA5", "rCalf": "#085041",
    "lAnkle": "#9FE1CB", "rAnkle": "#04342C",
    "lBicep": "#7F77DD", "rBicep": "#534AB7",
    "lForearm": "#AFA9EC", "rForearm": "#3C3489",
    "lWrist": "#CECBF6", "rWrist": "#26215C",
}


def print_report(avatar: MatlabAvatar, scale: float, units: str) -> None:
    print(f"\nMesh: {len(avatar.v)} vertices, {len(avatar.f)} faces")
    print(f"Slices recorded: {len(avatar.slices)}\n")

    header = f"{'slice':<10}{'points':>7}{'plane':>12}{'girth':>12}  frame"
    print(header)
    print("-" * len(header))
    for group, names in GROUPS:
        shown = [n for n in names if n in avatar.slices]
        if not shown:
            continue
        print(f"{group}:")
        for name in shown:
            s = avatar.slices[name]
            frame = s["frame"] or "upright"
            print(f"  {name:<8}{s['n_points']:>7}{s['plane'] * scale:>12.2f}"
                  f"{s['girth'] * scale:>12.2f}  {frame}")
    print(f"\n(plane and girth in {units}; 'frame' is the coordinate system the "
          f"slice was cut in --\n limb slices are cut perpendicular to the limb "
          f"axis, so their plane value is\n in the rotated frame, not a height "
          f"above the floor.)")


def write_csvs(avatar: MatlabAvatar, out: Path, scale: float,
               units: str, dump_points: bool) -> None:
    with open(out / "slice_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["slice", "n_points", "plane", "girth", "units",
                    "frame", "hull_points"])
        for name, s in avatar.slices.items():
            w.writerow([name, s["n_points"], f"{s['plane'] * scale:.6f}",
                        f"{s['girth'] * scale:.6f}", units,
                        s["frame"] or "upright", len(s["hull"]) - 1])

    if dump_points:
        with open(out / "slice_points.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["slice", "vertex_index", "x", "y", "z",
                        "hull_u", "hull_w", "on_hull"])
            for name, s in avatar.slices.items():
                on_hull = set(s["hull"].tolist())
                for k, vi in enumerate(s["indices"]):
                    p = s["points3d"][k] * scale
                    u, ww = s["uw"][k] * scale
                    w.writerow([name, int(vi), f"{p[0]:.5f}", f"{p[1]:.5f}",
                                f"{p[2]:.5f}", f"{u:.5f}", f"{ww:.5f}",
                                int(k in on_hull)])


def make_plots(avatar: MatlabAvatar, out: Path, scale: float,
               units: str, names: list[str]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed - skipping plots. "
              "Install with: pip install matplotlib")
        return

    v = avatar.v * scale

    # ---- slices drawn on the body -------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 9))
    for ax, (h, vt, title) in zip(axes, [(0, 2, "Front (x-z)"),
                                         (1, 2, "Side (y-z)")]):
        ax.scatter(v[:, h], v[:, vt], s=0.6, c="#d8d6cf", linewidths=0)
        for name in names:
            s = avatar.slices[name]
            p = s["points3d"] * scale
            if not len(p):
                continue
            ax.scatter(p[:, h], p[:, vt], s=9,
                       c=COLORS.get(name, "#D85A30"), label=name, linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel(f"{'xy'[h]} ({units})")
        ax.set_ylabel(f"z ({units})")
    axes[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5),
                   frameon=False)
    fig.suptitle("Measurement slices on the body")
    fig.tight_layout()
    fig.savefig(out / "slices_on_body.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- one cross-section panel per slice ----------------------------
    n = len(names)
    ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, name in zip(axes, names):
        s = avatar.slices[name]
        uw = s["uw"] * scale
        col = COLORS.get(name, "#D85A30")
        if len(uw):
            ax.scatter(uw[:, 0], uw[:, 1], s=14, c=col, linewidths=0, zorder=3)
            loop = uw[s["hull"]]
            ax.plot(loop[:, 0], loop[:, 1], "-", c=col, lw=1.4, alpha=0.75)
        ax.set_title(f"{name}\n{s['girth'] * scale:.1f} {units} "
                     f"({s['n_points']} pts)", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Cross-sections with convex hull (the measured girth)")
    fig.tight_layout()
    fig.savefig(out / "slice_cross_sections.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlots written to {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Show the measurement slices.")
    ap.add_argument("input", type=Path)
    ap.add_argument("--output", type=Path, default=Path("slice_output"))
    ap.add_argument("--scale-to-cm", type=float, default=1.0, dest="scale")
    ap.add_argument("--units", default=None)
    ap.add_argument("--only", nargs="+", default=None,
                    help="Only these slices, e.g. --only chest waist hip")
    ap.add_argument("--dump-points", action="store_true",
                    help="Write every point of every slice to slice_points.csv")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    units = args.units or ("cm" if args.scale != 1.0 else "mesh units")
    args.output.mkdir(parents=True, exist_ok=True)

    v, f = load_mesh(args.input)
    avatar = MatlabAvatar(v, f).run()

    names = [n for _, g in GROUPS for n in g if n in avatar.slices]
    names += [n for n in avatar.slices if n not in names]
    if args.only:
        missing = [n for n in args.only if n not in avatar.slices]
        if missing:
            print(f"unknown slice(s): {', '.join(missing)}", file=sys.stderr)
            print(f"available: {', '.join(names)}", file=sys.stderr)
            return 2
        names = list(args.only)

    print_report(avatar, args.scale, units)
    write_csvs(avatar, args.output, args.scale, units, args.dump_points)
    if not args.no_plots:
        make_plots(avatar, args.output, args.scale, units, names)

    print(f"\nCSVs written to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
