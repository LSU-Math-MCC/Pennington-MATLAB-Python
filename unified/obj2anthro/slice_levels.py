"""Cohort chart: the height, as a fraction of stature, at which each method slices.

    python -m unified.obj2anthro.slice_levels

Each method exposes its levels differently, so each needs its own recovery:
  avatar        - the plane is recorded on the slice object directly
  slice         - the level is recovered by matching the reported value back
                  against sum_perimeter in the backend's own height profile
  segmentation  - the level is written to the per-subject log as *_level: z=...,
                  in a centred decimetre frame, so z -> (z + H/20) / (H/10)
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from unified.obj2anthro.geometry_figures import (MatlabAvatar, load_obj, METHOD_COLORS,
                                                 INK, _plt, _style, _save)

OUT = ROOT / "runs/methods_report/figures"
MEASUREMENTS = ["chest", "waist", "hip"]
SEG_KEYS = {"chest": "chest_full_level", "waist": "natural_waist_level",
            "hip": "hip_full_level"}


def collect() -> pd.DataFrame:
    comb = pd.read_csv(ROOT / "runs/methods_report/combined_measurements.csv")
    comb = comb[comb.status == "success"]
    sl = comb[comb.anthro_method == "slice"].set_index("subject_id")
    sg = comb[comb.anthro_method == "segmentation"].set_index("subject_id")

    rows = []
    for obj in sorted((ROOT / "data/obj").glob("*.obj")):
        subject = obj.stem.replace(" ", "_")
        if subject not in sl.index:
            continue

        # --- avatar: planes recorded on the run ---------------------------
        v, f = load_obj(obj)
        avatar = MatlabAvatar(v, f).run()
        z_lo, z_hi = avatar.v[:, 2].min(), avatar.v[:, 2].max()
        for name in MEASUREMENTS:
            s = avatar.slices.get(name)
            if s:
                rows.append({"subject": subject, "method": "avatar", "m": name,
                             "pct": (s["plane"] - z_lo) / (z_hi - z_lo) * 100,
                             "value": s["girth"] * 0.1})

        # --- slice: recover the level from its own profile ----------------
        found = glob.glob(str(ROOT / "runs/python_slice_full/raw/*/slice"
                              / f"data_obj_{subject}/slices/*_slices.csv"))
        if found:
            profile = pd.read_csv(found[0])
            for name in MEASUREMENTS:
                target = sl.loc[subject, f"{name}_circumference_cm"] * 10
                i = int(np.argmin(np.abs(profile.sum_perimeter - target)))
                if abs(profile.sum_perimeter.iloc[i] - target) < 1:
                    rows.append({"subject": subject, "method": "slice", "m": name,
                                 "pct": float(profile.height_percent.iloc[i]),
                                 "value": target / 10})

        # --- segmentation: parse the per-subject log ----------------------
        logs = glob.glob(str(ROOT / "runs/timing/segmentation_r1/raw/*/segmentation/logs"
                             / f"data_obj_{subject}.txt"))
        if logs and subject in sg.index:
            text = Path(logs[0]).read_text(errors="ignore")
            height = float(sg.loc[subject, "height_cm"])
            for name, key in SEG_KEYS.items():
                hit = re.search(rf"{key}: z=(-?[\d.]+)", text)
                if hit:
                    z = float(hit.group(1))
                    rows.append({"subject": subject, "method": "segmentation", "m": name,
                                 "pct": (z + height / 20) / (height / 10) * 100,
                                 "value": float(sg.loc[subject,
                                                       f"{name}_circumference_cm"])})
    return pd.DataFrame(rows)


def strip_chart(df: pd.DataFrame):
    """One row per measurement; a dot per scan per method, on a stature axis."""
    plt = _plt()
    methods = ["avatar", "segmentation", "slice"]
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    fig.patch.set_alpha(0)

    lanes = 0.26
    for row, name in enumerate(MEASUREMENTS):
        base = len(MEASUREMENTS) - 1 - row
        for k, method in enumerate(methods):
            g = df[(df.m == name) & (df.method == method)]
            if g.empty:
                continue
            y = base + (k - 1) * lanes
            ax.scatter(g.pct, np.full(len(g), y), s=26,
                       c=METHOD_COLORS[method], alpha=.8, linewidths=0, zorder=3)
            median = float(np.median(g.pct))
            ax.plot([median, median], [y - .09, y + .09], "-",
                    c=METHOD_COLORS[method], lw=2.4, zorder=4)
            ax.annotate(f"{median:.0f}%", (median, y + .12), color=METHOD_COLORS[method],
                        fontsize=7.5, ha="center", va="bottom")
        ax.axhline(base - 0.5, color=INK, lw=.4, alpha=.25)

    ax.set_yticks(range(len(MEASUREMENTS)))
    ax.set_yticklabels(list(reversed(MEASUREMENTS)))
    ax.set_ylim(-0.55, len(MEASUREMENTS) - 0.35)
    ax.set_xlim(20, 90)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=6,
                          color=METHOD_COLORS[m], label=m) for m in methods]
    ax.legend(handles=handles, fontsize=8, frameon=False, labelcolor=INK,
              loc="lower right", ncol=3)
    _style(ax, "height of the cut (% of stature)", "",
           "Where every method slices, one dot per scan")
    fig.tight_layout()
    return _save(fig, OUT, "levels_cohort.png")


def spread_table(df: pd.DataFrame) -> dict:
    out = {}
    for name in MEASUREMENTS:
        out[name] = {}
        for method in df.method.unique():
            g = df[(df.m == name) & (df.method == method)]
            if g.empty:
                continue
            out[name][method] = {"median": float(np.median(g.pct)),
                                 "iqr": float(np.percentile(g.pct, 75)
                                              - np.percentile(g.pct, 25)),
                                 "min": float(g.pct.min()), "max": float(g.pct.max()),
                                 "n": int(len(g))}
    return out


if __name__ == "__main__":
    df = collect()
    df.to_csv(ROOT / "runs/methods_report/slice_levels.csv", index=False)
    strip_chart(df)
    summary = spread_table(df)
    (ROOT / "runs/methods_report/level_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")
    print(f"{'measurement':10s} {'method':14s} {'median%':>8s} {'IQR':>6s} {'range':>14s} {'n':>3s}")
    for name, bym in summary.items():
        for method, s in bym.items():
            print(f"{name:10s} {method:14s} {s['median']:8.1f} {s['iqr']:6.1f} "
                  f"{s['min']:6.1f}-{s['max']:5.1f} {s['n']:3d}")
