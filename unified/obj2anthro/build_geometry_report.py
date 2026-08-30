#!/usr/bin/env python3
"""Build the geometric comparison report from a run's tables and figures.

Reads the comparison tables written by ``python -m unified.compare`` plus the
PNGs written by ``geometry_figures.py``, embeds the figures as data URIs, and
writes one self-contained HTML page.

    python -m unified.obj2anthro.build_geometry_report runs/methods_report
"""

from __future__ import annotations

import argparse
import base64
import glob
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE = "matlab"
SERIES = ["avatar", "segmentation", "slice"]
SCALE_MM_TO_CM = 0.1


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load(run: Path) -> dict:
    combined = pd.read_csv(run / "combined_measurements.csv")
    detail = pd.read_csv(run / "comparison_detail.csv")
    by_method = pd.read_csv(run / "comparison_by_method.csv")
    coverage = pd.read_csv(run / "comparison_coverage.csv")
    ok = combined[combined.status == "success"]
    return {
        "combined": combined, "ok": ok, "detail": detail,
        "by_method": by_method, "coverage": coverage,
        "measurement_columns": [c for c in combined.columns
                                if c.endswith(("_cm", "_cm2", "_cm3"))],
    }


def pairwise(data: dict) -> dict[str, dict]:
    """Symmetric disagreement between every pair, with neither treated as truth."""
    columns = data["measurement_columns"]
    frames = {m: g.set_index("subject_id")[columns]
              for m, g in data["ok"].groupby("anthro_method")}
    out = {}
    for a, b in itertools.combinations([REFERENCE] + SERIES, 2):
        A, B = frames[a], frames[b]
        index = A.index.intersection(B.index)
        X, Y = A.loc[index], B.loc[index]
        both = (X.notna() & Y.notna()).values
        percent = ((X - Y).abs() / ((X.abs() + Y.abs()) / 2) * 100).values
        keep = both & np.isfinite(percent)
        identical = np.isclose(X.values[both], Y.values[both], rtol=1e-9)
        out[f"{a}|{b}"] = {
            "pct": float(np.mean(percent[keep])),
            "n": int(keep.sum()),
            "identical": float(identical.mean() * 100),
        }
    return out


def slice_levels(run_root: Path, data: dict) -> dict:
    """Recover which height the slice pipeline reported each girth from.

    The backend writes a full height profile but not a label->height map. Its
    reported value is the *sum* of every closed loop at one level, so matching
    the value back against ``sum_perimeter`` recovers the level exactly.
    """
    combined = data["ok"]
    reference = combined[combined.anthro_method == REFERENCE].set_index("subject_id")
    sliced = combined[combined.anthro_method == "slice"].set_index("subject_id")
    rows = []
    for subject in reference.index:
        found = glob.glob(str(run_root / "runs/python_slice_full/raw/*/slice"
                              / f"data_obj_{subject}/slices/*_slices.csv"))
        if not found or subject not in sliced.index:
            continue
        profile = pd.read_csv(found[0])
        entry = {"subject": subject}
        for label in ("chest", "waist", "hip"):
            column = f"{label}_circumference_cm"
            target = sliced.loc[subject, column] * 10
            i = int(np.argmin(np.abs(profile.sum_perimeter - target)))
            if abs(profile.sum_perimeter.iloc[i] - target) > 1:
                continue
            entry[label] = {
                "pct_height": float(profile.height_percent.iloc[i]),
                "loops": int(profile.num_loops.iloc[i]),
                "sum": float(profile.sum_perimeter.iloc[i]) * SCALE_MM_TO_CM,
                "max": float(profile.max_perimeter.iloc[i]) * SCALE_MM_TO_CM,
                "reference": float(reference.loc[subject, column]),
            }
        if all(k in entry for k in ("chest", "waist", "hip")):
            rows.append(entry)
    if not rows:
        raise SystemExit(
            "No slice height profiles found. This report needs the slice backend's "
            "per-subject *_slices.csv, which is not committed. Regenerate with: "
            "python -m unified obj2anthro --input data/obj --method slice "
            "--units auto --out runs/python_slice_full")
    return {"rows": rows}


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------
def girth_decomposition(run: Path) -> dict:
    """Split the avatar girth's offset from the true section into its two causes.

    With C the true torso loop on the same plane:

        G_av - |C|  =  (G_av - |conv C|)  +  (|conv C| - |C|)
                        band widening        hull shortcut

    The second term is <= 0 for every closed curve, because the convex hull is the
    shortest closed curve enclosing a point set. Convexity can only pull a girth
    down; anything pushing it up came from the band.
    """
    path = run / "girth_decomposition.json"
    if not path.is_file():
        return {"band": float("nan"), "shortcut": float("nan"),
                "net": float("nan"), "n": 0}
    rows = json.loads(path.read_text(encoding="utf-8"))
    clean = [r for r in rows if not (r["m"] == "chest" and r["n_loops"] < 3)]
    return {
        "band": float(np.mean([r["band"] for r in clean])),
        "shortcut": float(np.mean([r["shortcut"] for r in clean])),
        "net": float(np.mean([r["avatar"] - r["true"] for r in clean])),
        "n": len(clean),
    }


def timing_table(detail: dict) -> str:
    """Per-scan cost, folder cost, throughput, and the multiple against the fastest."""
    rows = []
    for method in sorted(detail, key=lambda m: detail[m]["mean"]):
        d = detail[method]
        rows.append(
            f"<tr><th scope='row'><span class='dot' style='background:"
            f"var(--c-{method}, var(--ref))'></span>{esc(method)}</th>"
            f"<td class='num'>{d['mean']:.2f}</td>"
            f"<td class='num'>{d['median']:.2f}</td>"
            f"<td class='num'>{d['max']:.2f}</td>"
            f"<td class='num'>{d['total']:.0f}</td>"
            f"<td class='num'>{d['per_min']:.0f}</td>"
            f"<td class='num'>{d['vs_avatar']:.1f}&times;</td></tr>")
    return ('<div class="scroll"><table><thead><tr><th>Method</th>'
            '<th class="num">mean s</th><th class="num">median s</th>'
            '<th class="num">slowest s</th><th class="num">folder s</th>'
            '<th class="num">scans/min</th><th class="num">vs fastest</th>'
            '</tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


# How each backend locates a level, from a line-by-line audit of all three.
# CONSTANT = a fixed fraction of stature or of a segment; SEARCH = an unrestricted
# extremum/derivative/optimisation criterion; HYBRID = a search boxed inside a
# hard-coded window.
ASSUMPTIONS = {
    "avatar": {
        "counts": (5, 3, 7),
        "orientation": "Axis permutation only &mdash; 90&deg;/180&deg; turns chosen from "
                       "sorted bounding-box extents. Never PCA, so stature equals one "
                       "bounding-box extent exactly.",
        "limbs": "Arms by constrained flood fill from the fingertip, gated at the armpit. "
                 "Legs by a pure geometric cut &mdash; crotch through each hip in the "
                 "(x,&nbsp;z) plane, with no connectivity at all.",
        "tell": "Waist and chest are literal midpoints, the mean and median of the armpit "
                "and hip heights, with no fullness search. Thigh is a flat "
                "<code>0.75</code> of hip-to-ankle.",
    },
    "segmentation": {
        "counts": (4, 3, 14),
        "orientation": "Rule-based whole-body axis choice, then a genuine OBB/PCA "
                       "alignment applied <em>per limb</em> after segmentation.",
        "limbs": "The richest of the three: boolean subtraction, a cutting plane at the "
                 "armpit and hip-crotch line, connected-component selection scored by "
                 "outward-vertex fraction, with a geodesic flood-fill fallback.",
        "tell": "Chest is a weighted blend of front-depth, perimeter and area "
                "(<code>0.42/0.24/0.20/0.14</code>) with <code>find_peaks</code> inside a "
                "shoulder-relative band. It is looking for the bust, not for a midpoint.",
    },
    "slice": {
        "counts": (4, 0, 13),
        "orientation": "Genuine PCA &mdash; covariance eigendecomposition, principal axis "
                       "onto z. No azimuthal correction, so left/right is inherited from "
                       "the file.",
        "limbs": "None. No boolean subtraction, no cutting plane, no connectivity. Every "
                 "measurement is read off height-percentage bands of the whole-body "
                 "slice stack.",
        "tell": "Not one unrestricted search anywhere &mdash; every extremum is boxed "
                "inside a fixed percent-of-stature window. Arm length is "
                "<code>0.30&nbsp;&times;&nbsp;height</code> and outside leg length "
                "<code>0.53&nbsp;&times;&nbsp;height</code>, unconditionally, with no "
                "geometry consulted.",
    },
}


def assumptions_table() -> str:
    """Constants-vs-search profile per backend, and what each gets structurally."""
    head = "".join(
        f"<th class='num'><span class='dot' style='background:var(--c-{m})'></span>"
        f"{esc(m)}</th>" for m in SERIES)

    bars = []
    for m in SERIES:
        const, search, hybrid = ASSUMPTIONS[m]["counts"]
        total = const + search + hybrid
        parts = []
        for n, cls in ((const, "const"), (hybrid, "hybrid"), (search, "search")):
            if n:
                parts.append(f"<span class='seg {cls}' style='flex:{n}' "
                             f"title='{n} of {total}'></span>")
        bars.append(f"<td class='wrap'><div class='mix'>{''.join(parts)}</div>"
                    f"<span class='mixkey'>{const} constant &middot; {hybrid} hybrid "
                    f"&middot; {search} search</span></td>")

    def row(label, field):
        cells = "".join(f"<td class='wrap'>{ASSUMPTIONS[m][field]}</td>" for m in SERIES)
        return f"<tr><th scope='row'>{label}</th>{cells}</tr>"

    return ('<div class="scroll"><table class="assump"><thead><tr><th></th>'
            + head + "</tr></thead><tbody>"
            + f"<tr><th scope='row'>How levels are located</th>{''.join(bars)}</tr>"
            + row("Orientation", "orientation")
            + row("Limb separation", "limbs")
            + row("The tell", "tell")
            + "</tbody></table></div>"
            + '<p class="lede" style="margin-top:-.7rem">Bars show how each backend finds '
              'its anatomical levels: <span class="key const"></span>a fixed fraction of '
              'stature, <span class="key hybrid"></span>a search inside a fixed window, '
              '<span class="key search"></span>an unrestricted search. Counts are from a '
              'line-by-line audit of all three backends.</p>')


def data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def bar_chart(rows, unit="%", width=680, row_height=34, decimals=2, sub=None):
    """Horizontal bars: label, value, optional sub-label. One series, no legend."""
    pad_left, pad_right = 178, 96
    inner = width - pad_left - pad_right
    top = 26
    height = top + len(rows) * row_height + 12
    peak = max((v for _, v, _ in rows), default=1) or 1
    ticks = [0, peak / 2, peak]
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" '
             f'role="img" aria-label="{esc(sub or "")}">']
    for t in ticks:
        x = pad_left + inner * (t / peak)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top - 8}" '
                     f'x2="{x:.1f}" y2="{height - 14}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{top - 13}" '
                     f'text-anchor="middle">{t:.0f}{unit}</text>')
    for i, (label, value, meta) in enumerate(rows):
        y = top + i * row_height
        w = max(inner * (value / peak), 2)
        colour = f"var(--c-{meta['key']})" if "key" in meta else "var(--ref)"
        parts.append(
            f'<rect class="bar" x="{pad_left}" y="{y + 5}" width="{w:.1f}" '
            f'height="{row_height - 15}" rx="4" fill="{colour}">'
            f'<title>{esc(meta.get("title", ""))}</title></rect>')
        parts.append(f'<text class="rowlabel" x="{pad_left - 12}" '
                     f'y="{y + row_height / 2 - 1}" text-anchor="end">{esc(label)}</text>')
        parts.append(f'<text class="value" x="{pad_left + w + 9:.1f}" '
                     f'y="{y + row_height / 2 - 1}">{value:.{decimals}f}{unit}</text>')
    parts.append("</svg>")
    return "".join(parts)


def matrix_table(pairs):
    """Pairwise disagreement as a tinted matrix. Sequential tint, ink-token text."""
    names = [REFERENCE] + SERIES
    peak = max(v["pct"] for v in pairs.values())
    head = "".join(f"<th class='num'>{esc(n)}</th>" for n in names)
    body = []
    for a in names:
        cells = []
        for b in names:
            if a == b:
                cells.append('<td class="num diag">—</td>')
                continue
            key = f"{a}|{b}" if f"{a}|{b}" in pairs else f"{b}|{a}"
            value = pairs[key]
            alpha = 0.08 + 0.62 * (value["pct"] / peak)
            cells.append(
                f'<td class="num tint" style="--a:{alpha:.3f}" '
                f'title="{value["n"]} paired values, {value["identical"]:.1f}% identical">'
                f'{value["pct"]:.2f}%</td>')
        body.append(f"<tr><th scope='row'>{esc(a)}</th>{''.join(cells)}</tr>")
    return (f'<div class="scroll"><table class="matrix"><thead><tr><th></th>{head}</tr>'
            f'</thead><tbody>{"".join(body)}</tbody></table></div>')


def figure(src: str, caption: str, wide: bool = True) -> str:
    cls = "fig wide" if wide else "fig"
    return (f'<figure class="{cls}"><img src="{src}" alt="{esc(caption)}" loading="lazy">'
            f'<figcaption>{caption}</figcaption></figure>')


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------
def build(run: Path, repo_root: Path, out: Path) -> Path:
    data = load(run)
    pairs = pairwise(data)
    levels = slice_levels(repo_root, data)
    figures = run / "figures"

    def fig_uri(name):
        return data_uri(figures / name)

    combined, detail = data["ok"], data["detail"]
    by_method = data["by_method"].set_index("method")

    # --- headline numbers -------------------------------------------------
    avatar_misses = detail[(detail.method == "avatar") & (~detail.matches)]
    odd_scan = "A00-09-0254_2025-12-10_10-38-56"
    misses_on_odd = int((avatar_misses.subject_id == odd_scan).sum())

    runtime = (combined.groupby("anthro_method")["runtime_seconds"]
               .agg(["mean", "median", "max", "sum", "count"]))

    # slice: sum-of-loops against largest-loop, at the level it chose
    slice_rows = levels["rows"]
    slice_summary = {}
    for label in ("chest", "waist", "hip"):
        s = np.array([r[label]["sum"] for r in slice_rows])
        m = np.array([r[label]["max"] for r in slice_rows])
        t = np.array([r[label]["reference"] for r in slice_rows])
        slice_summary[label] = (float(np.mean(np.abs(s - t) / t) * 100),
                                float(np.mean(np.abs(m - t) / t) * 100))
    inverted = sum(1 for r in slice_rows
                   if r["waist"]["pct_height"] <= r["hip"]["pct_height"])
    merged = sum(1 for r in slice_rows if r["chest"]["loops"] < 3)

    # coverage
    coverage_counts = {m: int(data["combined"][data["combined"].anthro_method == m]
                              [data["measurement_columns"]].notna().any().sum())
                       for m in [REFERENCE] + SERIES}

    decomposition = girth_decomposition(run)

    payload = {
        "decomposition": decomposition,
        "n_scans": int(combined[combined.anthro_method == REFERENCE].shape[0]),
        "avatar_pct_exact": float(by_method.loc["avatar", "pct_exact"]),
        "avatar_mean_pct": float(by_method.loc["avatar", "mean_abs_pct_error"]),
        "avatar_misses": int(len(avatar_misses)),
        "avatar_misses_on_odd": misses_on_odd,
        "pairs": pairs, "slice_summary": slice_summary,
        "inverted": inverted, "merged": merged, "n_slice_rows": len(slice_rows),
        "coverage": coverage_counts,
        "runtime": {m: [float(runtime.loc[m, "mean"]), float(runtime.loc[m, "sum"])]
                    for m in runtime.index},
        "runtime_detail": {m: {"mean": float(runtime.loc[m, "mean"]),
                               "median": float(runtime.loc[m, "median"]),
                               "max": float(runtime.loc[m, "max"]),
                               "total": float(runtime.loc[m, "sum"]),
                               "per_min": 60.0 / float(runtime.loc[m, "mean"]),
                               "vs_avatar": float(runtime.loc[m, "mean"])
                                            / float(runtime.loc["avatar", "mean"])}
                           for m in runtime.index},
    }
    (run / "report_data.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    html = PAGE.format(
        css=CSS,
        n_scans=payload["n_scans"],
        avatar_exact=f"{payload['avatar_pct_exact']:.1f}",
        avatar_pct=f"{payload['avatar_mean_pct']:.3f}",
        avatar_misses=payload["avatar_misses"],
        misses_on_odd=misses_on_odd,
        accuracy_chart=bar_chart(
            [(m, float(by_method.loc[m, "mean_abs_pct_error"]),
              {"key": m, "title": f"{m}: {by_method.loc[m, 'mean_abs_pct_error']:.2f}% mean "
                                  f"absolute difference over "
                                  f"{int(by_method.loc[m, 'n_comparisons'])} paired values, "
                                  f"{by_method.loc[m, 'pct_exact']:.1f}% identical"})
             for m in SERIES],
            sub="Mean absolute percent difference from Avatar.m, by method"),
        matrix=matrix_table(pairs),
        timing_chart=bar_chart(
            [(m, payload["runtime"][m][0],
              dict({"key": m} if m in SERIES else {},
                   title=f"{m}: {payload['runtime'][m][0]:.2f}s per scan, "
                         f"{payload['runtime'][m][1]:.0f}s for the folder"))
             for m in sorted(payload["runtime"], key=lambda m: payload["runtime"][m][0])],
            unit="s", decimals=2,
            sub="Mean wall-clock seconds per scan"),
        slice_chart=bar_chart(
            [(f"{label} — sum of loops", value[0], {"key": "slice",
              "title": f"{label}: reporting the sum of every loop is "
                       f"{value[0]:.1f}% from Avatar.m"})
             for label, value in slice_summary.items()]
            + [(f"{label} — largest loop", value[1], {"key": "avatar",
                "title": f"{label}: reporting the largest single loop instead is "
                         f"{value[1]:.1f}% from Avatar.m"})
               for label, value in slice_summary.items()],
            decimals=1, sub="Slice pipeline: sum of all loops vs largest single loop"),
        headline=headline_table(detail),
        assumptions=assumptions_table(),
        timing_table=timing_table(payload["runtime_detail"]),
        band_cm=f"{decomposition['band']:+.2f}",
        shortcut_cm=f"{decomposition['shortcut']:+.2f}",
        net_cm=f"{decomposition['net']:+.2f}",
        n_sections=decomposition["n"],
        fig_segments=figure(fig_uri("A00-09-0254_2025-12-10_10-38-56_segments.png"),
            "The outlier, and why it is one. <strong>Left:</strong> the left-arm segment "
            "holds 82 vertices against the right arm's 732 — one arm is fused to the torso "
            "and never separates. <strong>Right:</strong> the landmarks that follow. "
            "<code>lShoulder</code> lands 37&nbsp;cm below <code>rShoulder</code>, and the "
            "arm lengths come out 27.3 and 44.5&nbsp;cm. On a clean scan the same segments "
            "hold 549 and 557 vertices and the shoulders sit 1&nbsp;cm apart."),
        fig_segments_ok=figure(fig_uri("CanCan10_A_2026-02-27_09-49-56_segments.png"),
            "The same two panels on a scan that separates cleanly. Every downstream "
            "measurement inherits this decomposition, so a scan that fails here fails "
            "for every method that depends on limb separation — which is all three."),
        avatar_s=f"{payload['runtime_detail']['avatar']['mean']:.2f}",
        slice_s=f"{payload['runtime_detail']['slice']['mean']:.2f}",
        matlab_s=f"{payload['runtime_detail']['matlab']['mean']:.2f}",
        seg_s=f"{payload['runtime_detail']['segmentation']['mean']:.1f}",
        seg_x=f"{payload['runtime_detail']['segmentation']['vs_avatar']:.0f}",
        matlab_x=f"{payload['runtime_detail']['matlab']['vs_avatar']:.1f}",
        avatar_per_min=f"{payload['runtime_detail']['avatar']['per_min']:.0f}",
        seg_total=f"{payload['runtime_detail']['segmentation']['total']:.0f}",
        avatar_total=f"{payload['runtime_detail']['avatar']['total']:.0f}",
        pair_seg=f"{pairs['matlab|segmentation']['pct']:.1f}",
        pair_slice=f"{pairs['matlab|slice']['pct']:.0f}",
        pair_seg_slice=f"{pairs['segmentation|slice']['pct']:.0f}",
        coverage_row=" · ".join(f"<strong>{m}</strong> {n}"
                                for m, n in coverage_counts.items()),
        coverage_row_seg=coverage_counts["segmentation"],
        inverted=inverted, merged=merged, n_slice_rows=len(slice_rows),
        chest_sum=f"{slice_summary['chest'][0]:.0f}",
        chest_max=f"{slice_summary['chest'][1]:.0f}",
        waist_sum=f"{slice_summary['waist'][0]:.0f}",
        waist_max=f"{slice_summary['waist'][1]:.0f}",
        fig_body=figure(fig_uri("CanCan10_A_2026-02-27_09-49-56_body.png"),
            "Every girth <strong>Avatar.m</strong> measures on CanCan10_A, drawn on the scan it "
            "came from. Each colour is one measurement; the points are the vertices that "
            "went into it. Limb girths sit at an angle because they are cut perpendicular "
            "to the limb axis, not to the floor."),
        fig_xsec=figure(fig_uri("CanCan10_A_2026-02-27_09-49-56_xsec.png"),
            "The same measurements seen face-on. Points are the vertex band; the outline is "
            "the convex hull whose perimeter is reported. Where the hull bridges a hollow — "
            "clearest at the waist — the reported girth exceeds the surface it was taken from."),
        fig_girth_sep=figure(fig_uri("A00-08-4914_B_2025-12-09_12-31-25_girthdef.png"),
            "Chest height on A00-08-4914_B. <strong>Left:</strong> the true cross-section is "
            "three separate closed loops — torso 79.0&nbsp;cm, arms 32.1 and 31.5&nbsp;cm. "
            "The slice pipeline adds all three and reports 141.4&nbsp;cm. "
            "<strong>Right:</strong> Avatar.m collects a band of nearby vertices and takes "
            "the convex hull, reporting 82.3&nbsp;cm — the torso, plus about 3&nbsp;cm of "
            "convexity where the hull cuts across the armpit hollows."),
        fig_girth_merged=figure(fig_uri("CanCan01_A_2025-10-27_11-10-43_girthdef.png"),
            "The same height on CanCan01_A, where the arms rest against the torso. Now there "
            "is only <em>one</em> loop: the cross-section walks out around each arm and back, "
            "so the sum and the largest loop are the same 183.5&nbsp;cm. Avatar.m's hull "
            "bridges straight across both armpits and returns 133.1&nbsp;cm. No aggregation "
            "rule recovers the torso here — the loops would have to be cut apart first."),
        fig_levels=figure(fig_uri("CanCan10_A_2026-02-27_09-49-56_levels.png"),
            "Where each pipeline placed the same three named girths on CanCan10_A, with the "
            "value it reported. The chest lines coincide almost exactly — the disagreement "
            "there is entirely about <em>what</em> to measure. Waist and hip are a different "
            "problem: the slice pipeline puts its waist below its hip.", wide=False),
        fig_levels_2=figure(fig_uri("CanCan01_A_2025-10-27_11-10-43_levels.png"),
            "CanCan01_A, the largest scan in the folder. Here the two pipelines disagree "
            "about height as well as definition, and the slice waist again lands below the "
            "slice hip.", wide=False),
        fig_profile=figure(fig_uri("A00-08-4914_B_2025-12-09_12-31-25_profile.png"),
            "The slice pipeline's own height profile for A00-08-4914_B, which it computes and "
            "writes out in full. The reported curve is the sum of every loop; the faint curve "
            "is the largest single loop. Avatar.m's values sit on the faint curve, not the "
            "reported one — the number the pipeline needs is already in the file it wrote."),
        fig_profile_2=figure(fig_uri("CanCan10_A_2026-02-27_09-49-56_profile.png"),
            "CanCan10_A. Below roughly 34% of stature the two curves separate cleanly: that "
            "is where the legs part into two loops. The step at 41% is the arms entering the "
            "section."),
        fig_odd=figure(fig_uri("A00-09-0254_2025-12-10_10-38-56_body.png"),
            "A00-09-0254 2025-12-10_10-38-56 — the one scan where the port and MATLAB still "
            "disagree. Its arms are captured at very different lengths (44&nbsp;cm against "
            "27&nbsp;cm), and the notch profile that <code>adjustCrotch</code> clusters has "
            "no dominant outlier, so MATLAB's randomly seeded k-means has several answers "
            "available to it."),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def headline_table(detail: pd.DataFrame) -> str:
    wanted = [
        ("height_cm", "height"),
        ("chest_circumference_cm", "chest circumference"),
        ("waist_circumference_cm", "waist circumference"),
        ("hip_circumference_cm", "hip circumference"),
        ("thigh_circumference_left_cm", "thigh circumference (L)"),
        ("calf_circumference_left_cm", "calf circumference (L)"),
        ("arm_length_left_cm", "arm length (L)"),
        ("surface_area_total_cm2", "total surface area"),
        ("volume_cm3", "total volume"),
    ]
    grouped = detail.groupby(["measurement", "method"]).agg(
        pct=("abs_pct_error", "mean"), n=("matches", "size"), exact=("matches", "sum"))
    rows = []
    for column, label in wanted:
        cells = []
        for method in SERIES:
            if (column, method) not in grouped.index:
                cells.append('<td class="num dim">—</td>')
                continue
            r = grouped.loc[(column, method)]
            strong = " strong" if r["pct"] < 0.5 else ""
            cells.append(
                f'<td class="num{strong}">{r["pct"]:.2f}%'
                f'<span class="sub"> · {int(r["exact"])}/{int(r["n"])}</span></td>')
        rows.append(f"<tr><th scope='row'>{esc(label)}</th>{''.join(cells)}</tr>")
    head = "".join(f"<th class='num'><span class='dot' style='background:var(--c-{m})'>"
                   f"</span>{esc(m)}</th>" for m in SERIES)
    return (f'<div class="scroll"><table><thead><tr><th>Measurement</th>{head}</tr>'
            f'</thead><tbody>{"".join(rows)}</tbody></table></div>')


CSS = """
:root{
  --ground:#EEF1F2; --surface:#FFFFFF; --surface-2:#F6F9FA;
  --ink:#14181B; --ink-2:#434C52; --ink-3:#727B81;
  --rule:#D7DDDF; --rule-2:#E8ECEE;
  --c-avatar:#0F6BC4; --c-segmentation:#BF6200; --c-slice:#933F79; --ref:#5A6469;
  --good:#1E7A47; --warn:#A8650C; --crit:#A32E2E;
  --shadow:0 1px 2px rgba(20,24,27,.05), 0 10px 28px -18px rgba(20,24,27,.3);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0F1214; --surface:#181C1F; --surface-2:#1E2327;
    --ink:#E5E9EB; --ink-2:#A8B0B6; --ink-3:#7C858B;
    --rule:#293034; --rule-2:#222829;
    --c-avatar:#4A97DC; --c-segmentation:#D07F22; --c-slice:#B4629A; --ref:#8A9298;
    --good:#4FA97A; --warn:#D3922F; --crit:#D96A6A;
    --shadow:0 1px 2px rgba(0,0,0,.45), 0 10px 28px -18px rgba(0,0,0,.85);
  }
}
:root[data-theme="dark"]{
  --ground:#0F1214; --surface:#181C1F; --surface-2:#1E2327;
  --ink:#E5E9EB; --ink-2:#A8B0B6; --ink-3:#7C858B;
  --rule:#293034; --rule-2:#222829;
  --c-avatar:#4A97DC; --c-segmentation:#D07F22; --c-slice:#B4629A; --ref:#8A9298;
  --good:#4FA97A; --warn:#D3922F; --crit:#D96A6A;
  --shadow:0 1px 2px rgba(0,0,0,.45), 0 10px 28px -18px rgba(0,0,0,.85);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Serif",Georgia,"Iowan Old Style",serif;
  font-size:17px; line-height:1.62; -webkit-font-smoothing:antialiased;
}
.cond{font-family:"IBM Plex Sans Condensed","Arial Narrow",ui-sans-serif,sans-serif}
.mono{font-family:"IBM Plex Mono",ui-monospace,Consolas,monospace}
.wrap{max-width:66rem; margin:0 auto; padding:0 1.5rem 6rem}
.prose{max-width:40rem}
.prose p{margin:0 0 1.05rem}

header{border-bottom:1px solid var(--rule); margin-bottom:3rem; padding:3.5rem 0 2.25rem}
.eyebrow{
  font-family:"IBM Plex Sans Condensed","Arial Narrow",ui-sans-serif,sans-serif;
  text-transform:uppercase; letter-spacing:.17em; font-size:.74rem;
  color:var(--ink-3); margin:0 0 1rem; font-weight:600;
}
h1{
  font-family:"IBM Plex Sans Condensed","Arial Narrow",ui-sans-serif,sans-serif;
  font-size:clamp(2.2rem,5.4vw,3.4rem); line-height:1.03; letter-spacing:-.005em;
  font-weight:700; margin:0 0 1.25rem; text-wrap:balance; max-width:20ch;
}
.standfirst{font-size:1.15rem; color:var(--ink-2); max-width:44rem; margin:0}

section{margin:0 0 4rem}
/* Section rule as a dimension line: end ticks, like a drafted measurement. */
.sechead{display:flex; align-items:baseline; gap:1rem; margin:0 0 1.1rem}
.sechead h2{
  font-family:"IBM Plex Sans Condensed","Arial Narrow",ui-sans-serif,sans-serif;
  font-size:1.62rem; font-weight:700; margin:0; letter-spacing:-.003em;
  text-wrap:balance; flex:0 0 auto;
}
.sechead .dim{flex:1 1 auto; height:9px; position:relative; opacity:.85}
.sechead .dim::before{content:""; position:absolute; left:0; right:0; top:4px;
  border-top:1px solid var(--rule)}
.sechead .dim::after{content:""; position:absolute; right:0; top:0; bottom:0;
  border-right:1px solid var(--rule)}
h3{
  font-family:"IBM Plex Sans Condensed","Arial Narrow",ui-sans-serif,sans-serif;
  text-transform:uppercase; letter-spacing:.13em; font-size:.78rem;
  color:var(--ink-3); margin:2rem 0 .7rem; font-weight:600;
}
.lede{color:var(--ink-2); margin:0 0 1.75rem; max-width:41rem}

.cards{display:grid; grid-template-columns:repeat(auto-fit,minmax(14rem,1fr));
  gap:1rem; margin:0 0 2rem}
.card{background:var(--surface); border:1px solid var(--rule); border-radius:10px;
  padding:1.15rem 1.25rem 1.25rem; box-shadow:var(--shadow)}
.card .name{
  font-family:"IBM Plex Sans Condensed","Arial Narrow",ui-sans-serif,sans-serif;
  text-transform:uppercase; letter-spacing:.12em; font-size:.74rem; color:var(--ink-3);
  margin:0 0 .55rem; display:flex; align-items:center; gap:.45rem; font-weight:600;
}
.card .big{font-family:"IBM Plex Mono",ui-monospace,Consolas,monospace;
  font-size:2.1rem; line-height:1; font-variant-numeric:tabular-nums;
  letter-spacing:-.03em; margin:0 0 .3rem; font-weight:500}
.card .unit{font-size:.95rem; color:var(--ink-3)}
.card .note{font-size:.88rem; color:var(--ink-2); margin:.45rem 0 0; line-height:1.45}
.dot{display:inline-block; width:.6rem; height:.6rem; border-radius:2px; flex:0 0 auto}

figure.fig{margin:1.6rem 0; background:var(--surface); border:1px solid var(--rule);
  border-radius:10px; padding:1.2rem 1.3rem 1rem; box-shadow:var(--shadow)}
figure.fig img{display:block; width:100%; height:auto; border-radius:4px}
figure.fig.wide{max-width:none}
figure.fig:not(.wide){max-width:34rem}
figcaption{font-size:.89rem; color:var(--ink-2); margin:.9rem 0 0; padding-top:.8rem;
  border-top:1px solid var(--rule-2); line-height:1.55}
.figpair{display:grid; grid-template-columns:repeat(auto-fit,minmax(20rem,1fr)); gap:1.1rem}
.figpair figure.fig{margin:0; max-width:none}

.chart{display:block; width:100%; height:auto; overflow:visible}
.grid{stroke:var(--rule-2); stroke-width:1}
.tick{fill:var(--ink-3); font-size:11px;
  font-family:"IBM Plex Mono",ui-monospace,Consolas,monospace;
  font-variant-numeric:tabular-nums}
.rowlabel{fill:var(--ink-2); font-size:13px;
  font-family:"IBM Plex Sans Condensed","Arial Narrow",ui-sans-serif,sans-serif}
.value{fill:var(--ink); font-size:13px;
  font-family:"IBM Plex Mono",ui-monospace,Consolas,monospace;
  font-variant-numeric:tabular-nums}
.bar{transition:opacity .12s ease}
.bar:hover{opacity:.78}
figure.chartbox{margin:0 0 1.4rem; background:var(--surface); border:1px solid var(--rule);
  border-radius:10px; padding:1.25rem 1.35rem 1rem; box-shadow:var(--shadow)}

.scroll{overflow-x:auto; border:1px solid var(--rule); border-radius:10px;
  background:var(--surface); box-shadow:var(--shadow); margin:0 0 1.4rem}
table{border-collapse:collapse; width:100%; font-size:.9rem;
  font-family:"IBM Plex Sans Condensed","Arial Narrow",ui-sans-serif,sans-serif}
th,td{padding:.6rem .9rem; text-align:left; border-bottom:1px solid var(--rule-2);
  white-space:nowrap}
thead th{background:var(--surface-2); font-size:.72rem; text-transform:uppercase;
  letter-spacing:.1em; color:var(--ink-3); font-weight:700}
thead th .dot{margin-right:.4rem; vertical-align:baseline}
tbody tr:last-child td, tbody tr:last-child th{border-bottom:none}
td.num, th.num{font-family:"IBM Plex Mono",ui-monospace,Consolas,monospace;
  font-variant-numeric:tabular-nums; text-align:right}
th.num{font-family:"IBM Plex Sans Condensed",ui-sans-serif,sans-serif}
td.num .sub{color:var(--ink-3); font-size:.82em}
td.num.strong{color:var(--ink); font-weight:600}
td.dim{color:var(--ink-3)}
table.matrix td.tint{background:color-mix(in srgb, var(--c-avatar) calc(var(--a) * 100%), var(--surface))}
table.matrix td.diag{color:var(--ink-3)}
table.matrix tbody th{font-weight:600; color:var(--ink-2)}

.callout{border-left:3px solid var(--bar,var(--c-avatar)); background:var(--surface);
  border-radius:0 10px 10px 0; padding:1.1rem 1.35rem; margin:0 0 1.5rem;
  box-shadow:var(--shadow); max-width:44rem}
.callout p{margin:0 0 .65rem}
.callout p:last-child{margin:0}
.callout.seg{--bar:var(--c-segmentation)}
.callout.slice{--bar:var(--c-slice)}

code{font-family:"IBM Plex Mono",ui-monospace,Consolas,monospace; font-size:.87em;
  background:var(--surface-2); border:1px solid var(--rule-2); border-radius:4px;
  padding:.08em .35em}
pre{font-family:"IBM Plex Mono",ui-monospace,Consolas,monospace; font-size:.83rem;
  background:var(--surface); border:1px solid var(--rule); border-radius:8px;
  padding:.9rem 1.05rem; overflow-x:auto; color:var(--ink-2); line-height:1.6;
  box-shadow:var(--shadow)}
pre code{background:none; border:none; padding:0}
ul.plain{margin:0 0 1.2rem; padding-left:1.1rem; max-width:41rem}
ul.plain li{margin:0 0 .5rem; color:var(--ink-2)}
ul.plain li strong{color:var(--ink)}

footer{border-top:1px solid var(--rule); padding-top:1.75rem; color:var(--ink-3);
  font-size:.88rem}
footer p{margin:0 0 .5rem; max-width:44rem}
a{color:var(--c-avatar)}
:focus-visible{outline:2px solid var(--c-avatar); outline-offset:3px; border-radius:3px}

/* --- definition cards, equations, assumption mix bars ------------------- */
.defs{display:grid; grid-template-columns:repeat(auto-fit,minmax(17rem,1fr)); gap:1rem;
  margin:0 0 1.6rem}
.def{background:var(--surface); border:1px solid var(--rule); border-radius:10px;
  padding:1.1rem 1.2rem 1.2rem; box-shadow:var(--shadow); border-top:3px solid var(--bar)}
.def .name{font-family:"IBM Plex Sans Condensed",ui-sans-serif,sans-serif; font-weight:700;
  font-size:.86rem; letter-spacing:.02em; margin:0 0 .7rem; color:var(--ink);
  display:flex; align-items:center; gap:.45rem}
.def .math{margin:0 0 .8rem; padding:.55rem .2rem; overflow-x:auto;
  border-top:1px solid var(--rule-2); border-bottom:1px solid var(--rule-2)}
.def .note{font-size:.88rem; color:var(--ink-2); margin:0; line-height:1.55}
math{font-size:1.03em; color:var(--ink)}
.eq{background:var(--surface); border:1px solid var(--rule); border-radius:10px;
  padding:1.2rem 1.3rem .9rem; box-shadow:var(--shadow); margin:0 0 1.4rem; overflow-x:auto}
.eq math{font-size:1.12em}
.eqkey{display:flex; justify-content:space-around; gap:1rem; margin:.5rem 0 0;
  font-family:"IBM Plex Sans Condensed",ui-sans-serif,sans-serif; font-size:.74rem;
  text-transform:uppercase; letter-spacing:.11em; color:var(--ink-3)}
.mix{display:flex; height:9px; border-radius:5px; overflow:hidden; gap:2px;
  background:var(--rule-2); margin:0 0 .4rem; min-width:8rem}
.mix .seg{display:block}
.seg.const, .key.const{background:var(--c-slice)}
.seg.hybrid, .key.hybrid{background:var(--c-segmentation)}
.seg.search, .key.search{background:var(--c-avatar)}
.mixkey{font-family:"IBM Plex Mono",ui-monospace,Consolas,monospace; font-size:.72rem;
  color:var(--ink-3); white-space:nowrap}
.key{display:inline-block; width:.62rem; height:.62rem; border-radius:2px;
  margin:0 .3rem 0 .5rem; vertical-align:baseline}
table.assump td.wrap, table td.wrap{white-space:normal; min-width:14rem; max-width:22rem;
  font-size:.85rem; line-height:1.5; color:var(--ink-2); vertical-align:top}
table.assump tbody th{white-space:normal; max-width:11rem; vertical-align:top;
  font-weight:600; color:var(--ink)}
@media (prefers-reduced-motion:reduce){*{transition:none!important; animation:none!important}}
@media (max-width:640px){
  body{font-size:16px} .wrap{padding:0 1.1rem 4rem}
  .sechead{flex-wrap:wrap} .sechead .dim{display:none}
}
"""


PAGE = """<title>Three Ways to Measure a Body</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@400;600;700&family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&display=swap">
<style>{css}</style>

<div class="wrap">
<header>
  <p class="eyebrow">Pennington OBJ-to-anthropometry · four pipelines, one folder of 21 scans</p>
  <h1>Three Ways to Measure a Body</h1>
  <p class="standfirst">On the same scan the pipelines return chest circumferences
  <strong>80&nbsp;cm</strong> apart. None is malfunctioning. Each answers a different
  question about the same mesh, and the answers cost between 0.6 and 7 seconds. This is
  what each one assumes, what that buys, and what it costs.</p>
</header>

<section>
  <div class="sechead"><h2>The three definitions</h2><span class="dim"></span></div>
  <p class="lede">Everything below follows from one choice: at height <i>z</i>, which curve
  are you measuring the length of?</p>

  <div class="defs">
    <div class="def" style="--bar:var(--c-avatar)">
      <p class="name"><span class="dot" style="background:var(--c-avatar)"></span>avatar — hull of a band</p>
      <p class="math"><math display="block"><mrow>
        <msub><mi>G</mi><mtext>av</mtext></msub><mo>(</mo><mi>z</mi><mo>)</mo><mo>=</mo>
        <mrow><mo>|</mo><mo>&#8706;</mo><mi>conv</mi><mspace width="3px"></mspace>
        <mi>&#960;</mi><mo>(</mo><msub><mi>B</mi><mi>z</mi></msub><mo>)</mo><mo>|</mo></mrow>
      </mrow></math></p>
      <p class="note">Not a cross-section. <math><msub><mi>B</mi><mi>z</mi></msub></math>
      is every vertex of every face that straddles the plane — points above and below
      alike. They are projected to the plane
      (<math><mi>&#960;</mi></math>) and the convex-hull perimeter is reported. Nothing is
      interpolated, so the girth is read off a slab, not a slice.</p>
    </div>

    <div class="def" style="--bar:var(--c-slice)">
      <p class="name"><span class="dot" style="background:var(--c-slice)"></span>slice — sum of every loop</p>
      <p class="math"><math display="block"><mrow>
        <msub><mi>G</mi><mtext>sl</mtext></msub><mo>(</mo><mi>z</mi><mo>)</mo><mo>=</mo>
        <munderover><mo>&#8721;</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow>
        <mrow><mi>K</mi><mo>(</mo><mi>z</mi><mo>)</mo></mrow></munderover>
        <mrow><mo>|</mo><msub><mi>C</mi><mi>k</mi></msub><mo>(</mo><mi>z</mi><mo>)</mo><mo>|</mo></mrow>
      </mrow></math></p>
      <p class="note">A true planar cross-section, giving
      <math><mi>K</mi><mo>(</mo><mi>z</mi><mo>)</mo></math> closed loops — at chest height
      usually three: one torso, two arms. The reported girth is their sum. The per-loop
      lengths are computed and written to disk; only the aggregation collapses them.</p>
    </div>

    <div class="def" style="--bar:var(--c-segmentation)">
      <p class="name"><span class="dot" style="background:var(--c-segmentation)"></span>segmentation — region first</p>
      <p class="math"><math display="block"><mrow>
        <msub><mi>G</mi><mtext>sg</mtext></msub><mo>(</mo><mi>z</mi><mo>)</mo><mo>=</mo>
        <mrow><mo>|</mo><mo>&#8706;</mo><mo>(</mo>
        <msub><mi>M</mi><mtext>trunk</mtext></msub><mo>&#8745;</mo>
        <msub><mi>P</mi><mi>z</mi></msub><mo>)</mo><mo>|</mo></mrow>
      </mrow></math></p>
      <p class="note">The mesh is cut into anatomical regions first, so the section
      <math><msub><mi>P</mi><mi>z</mi></msub></math> is taken of the trunk alone and the
      arms never enter it. It solves directly what the other two work around, and pays for
      it in time.</p>
    </div>
  </div>
</section>

<section>
  <div class="sechead"><h2>Where the avatar girth comes from</h2><span class="dim"></span></div>
  <div class="prose">
    <p>The band-and-hull definition has two effects that pull in opposite directions, and
    they separate exactly. Writing <math><mi>C</mi></math> for the true torso loop on the
    same plane:</p>
  </div>
  <div class="eq">
    <math display="block"><mrow>
      <msub><mi>G</mi><mtext>av</mtext></msub><mo>&#8722;</mo>
      <mrow><mo>|</mo><mi>C</mi><mo>|</mo></mrow>
      <mo>=</mo>
      <mrow><mo>(</mo><msub><mi>G</mi><mtext>av</mtext></msub><mo>&#8722;</mo>
      <mrow><mo>|</mo><mo>&#8706;</mo><mi>conv</mi><mspace width="3px"></mspace><mi>C</mi><mo>|</mo></mrow><mo>)</mo></mrow>
      <mo>+</mo>
      <mrow><mo>(</mo><mrow><mo>|</mo><mo>&#8706;</mo><mi>conv</mi><mspace width="3px"></mspace><mi>C</mi><mo>|</mo></mrow>
      <mo>&#8722;</mo><mrow><mo>|</mo><mi>C</mi><mo>|</mo></mrow><mo>)</mo></mrow>
    </mrow></math>
    <p class="eqkey"><span>net offset</span><span>band widening</span><span>hull shortcut</span></p>
  </div>
  <div class="prose">
    <p>The second term is <strong>never positive</strong>. The convex hull is the shortest
    closed curve enclosing a point set, so
    <math><mrow><mo>|</mo><mo>&#8706;</mo><mi>conv</mi><mspace width="3px"></mspace><mi>C</mi><mo>|</mo></mrow><mo>&#8804;</mo><mrow><mo>|</mo><mi>C</mi><mo>|</mo></mrow></math>
    for every closed <math><mi>C</mi></math>. Convexity can only pull a girth
    <em>down</em>. Anything pushing it up came from the band — a slab of finite thickness
    through a tapering body, whose projection is wider than any single section inside it.</p>
    <p>Over {n_sections} sections that separate into distinct loops: band widening
    <strong>{band_cm}&nbsp;cm</strong>, hull shortcut <strong>{shortcut_cm}&nbsp;cm</strong>,
    net <strong>{net_cm}&nbsp;cm</strong>. The band dominates. Worth stating precisely,
    because the intuition runs the other way — a convex hull is usually assumed to
    over-read a waist, and here that is not where the excess comes from.</p>
  </div>

  {fig_girth_sep}
  {fig_girth_merged}

  <div class="prose">
    <p>The second figure is the case no aggregation rule survives. With the arms against
    the torso the section is a <em>single</em> loop that already contains them, so
    <math><mi>K</mi><mo>(</mo><mi>z</mi><mo>)</mo><mo>=</mo><mn>1</mn></math> and the sum
    and the largest loop are both 183.5&nbsp;cm. The hull is indifferent — it bridges the
    armpits either way. That indifference is what buys Avatar.m its robustness, and what
    makes it approximate.</p>
  </div>
</section>

<section>
  <div class="sechead"><h2>Constants or search</h2><span class="dim"></span></div>
  <p class="lede">The second axis: how each pipeline decides <em>where</em> a level sits. A
  fixed fraction of stature is fast and reproducible and wrong on atypical bodies; a
  geometric search adapts, and can fail outright.</p>
  {assumptions}
  {fig_body}
</section>

<section>
  <div class="sechead"><h2>Where they cut</h2><span class="dim"></span></div>
  <div class="figpair">
    {fig_levels}
    {fig_levels_2}
  </div>
  <div class="prose">
    <p>On CanCan10_A the chest heights coincide, which isolates that disagreement to
    definition alone. It does not hold everywhere — on CanCan01_A the heights differ too.
    Waist and hip fail differently again: on {inverted} of {n_slice_rows} scans the slice
    pipeline places its waist at or below its hip, an ordering problem upstream of any
    measurement.</p>
  </div>
  {fig_profile}
</section>

<section>
  <div class="sechead"><h2>What the mesh decides</h2><span class="dim"></span></div>
  <p class="lede">One scan is an outlier for every method at once — segmentation 128%,
  slice 180%, and the only scan where the port and MATLAB disagree. The cause sits upstream
  of all three.</p>
  {fig_segments}
  {fig_segments_ok}
  <div class="prose">
    <p>Limb separation is the step every later measurement inherits. When it fails on one
    side, the shoulder, armpit, wrist and arm length on that side fail with it, the torso
    absorbs the missing arm, and the notch profile <code>adjustCrotch</code> clusters
    becomes degenerate — which is why MATLAB's randomly seeded <code>kmeans</code> is
    unstable on precisely this scan and no other. One bad capture propagates through every
    pipeline, because all three assume the limbs can be told apart.</p>
  </div>
</section>

<section>
  <div class="sechead"><h2>How far apart they land</h2><span class="dim"></span></div>
  <p class="lede">Scored against <code>Avatar.m</code> as run in MATLAB R2023b — the
  reference because it is the implementation the others were written from, not because it
  is known to be right.</p>

  <figure class="chartbox">
    {accuracy_chart}
    <figcaption>Mean absolute percent difference from Avatar.m. Only pairs where both
    produced a value are scored. Hover a bar for its counts.</figcaption>
  </figure>

  <h3>Every pair, with neither treated as truth</h3>
  {matrix}
  <p class="lede" style="margin-top:-.6rem">Each cell is the mean difference between two
  methods as a percentage of their average, so it assumes neither is correct. Segmentation
  sits closer to the MATLAB pair ({pair_seg}%) than slice sits to anything
  ({pair_slice}% from MATLAB, {pair_seg_slice}% from segmentation).</p>

  <h3>The measurements people ask for</h3>
  {headline}
  <div class="prose">
    <p>Three rows behave differently. <strong>Height</strong> and <strong>total surface
    area</strong> are identical for segmentation as well as the port, and
    <strong>volume</strong> nearly so — they are functionals of the mesh alone, with no
    landmark in them. Every row that requires deciding <em>where</em> diverges. That
    boundary is the result: the pipelines agree exactly on whatever they do not have to
    interpret.</p>
  </div>
</section>

<section>
  <div class="sechead"><h2>Cost, and the trade space</h2><span class="dim"></span></div>
  {timing_table}
  <figure class="chartbox">
    {timing_chart}
    <figcaption>Mean wall-clock seconds per scan, same folder and same machine. MATLAB
    excludes Engine start-up, a one-time ~20&nbsp;s per batch. Slice is measurement only;
    its optional per-subject renders are not included.</figcaption>
  </figure>

  <div class="prose"><p>Three axes, and no method wins on all of them.</p></div>
  <ul class="plain">
    <li><strong>avatar — {avatar_s}&nbsp;s/scan, {avatar_per_min} scans/min.</strong>
    Reproduces MATLAB to floating-point noise at {matlab_x}&times; less cost than MATLAB
    itself, with no licence and no Engine install. It inherits every approximation in the
    reference, including the band-and-hull girth, and emits the fewest columns.</li>
    <li><strong>slice — {slice_s}&nbsp;s/scan.</strong> Essentially free, and the only one
    computing true cross-sections. The height profile is sound; the reporting layer above
    it is unfinished.</li>
    <li><strong>segmentation — {seg_s}&nbsp;s/scan, {seg_x}&times; avatar.</strong> Buys
    the most columns and the only genuinely region-aware girths, and is the only method
    that has to solve limb separation properly rather than route around it. The folder
    costs {seg_total}&nbsp;s against avatar's {avatar_total}&nbsp;s — which matters at a
    few hundred scans, not at twenty.</li>
  </ul>
  <div class="prose">
    <p>Plainly: if the requirement is <em>reproduce the historical numbers</em>, avatar is
    strictly better than running MATLAB. If the requirement is <em>measure the body
    correctly</em>, none of these is finished, and segmentation is the one whose
    assumptions sit closest to the anatomy — at roughly twelve times the cost.</p>
  </div>

  <h3>Coverage</h3>
  <p class="lede">Measurement columns each backend produces, of 52 canonical columns:
  {coverage_row}. Breadth and agreement are independent. The port sits six columns below
  MATLAB because the segment volumes come from a partial hole-filling routine that is not
  ported; whole-body volume and every segment area are.</p>
</section>

<section>
  <div class="sechead"><h2>Slice: what is unfinished</h2><span class="dim"></span></div>
  <div class="callout slice">
    <p><strong>The geometry is sound; the reporting layer is not.</strong> The backend
    computes a full height profile per subject — every level, loop count, per-loop
    perimeter and area, left/right splits — and writes it out. The selections made on top
    of that are the gap.</p>
  </div>
  <ul class="plain">
    <li><strong>Aggregation.</strong> A reported girth is <code>sum_perimeter</code>.
    <code>max_perimeter</code> sits in the same file. Switching moves chest from
    {chest_sum}% to {chest_max}% and waist from {waist_sum}% to {waist_max}%, with no
    change to the geometry underneath.</li>
    <li><strong>Loop separation.</strong> On {merged} of {n_slice_rows} scans the arms
    touch the torso, so <math><mi>K</mi><mo>=</mo><mn>1</mn></math> and the largest-loop
    rule cannot help either. Those need the loops cut apart.</li>
    <li><strong>Landmarks.</strong> Stature runs ~2.9&nbsp;cm short and waist/hip levels
    are sometimes inverted — both upstream of the aggregation question.</li>
  </ul>
  <figure class="chartbox">
    {slice_chart}
    <figcaption>The same levels, aggregated two ways. Upper three bars: what the pipeline
    reports today. Lower three: what its own profile already contains. The residual is
    concentrated in the scans where loops merge.</figcaption>
  </figure>
</section>

<section>
  <div class="sechead"><h2>The port now matches</h2><span class="dim"></span></div>
  <div class="prose">
    <p>The avatar backend is a port of <code>Avatar.m</code>, so it should return MATLAB's
    numbers exactly. It now matches {avatar_exact}% of 720 paired values, up from 65.4%,
    after four fixes. None was arithmetic:</p>
  </div>
  <div class="scroll"><table>
    <thead><tr><th>Cause</th><th>Where</th><th>Effect</th></tr></thead>
    <tbody>
      <tr><th scope="row">Do-while trim loops</th><td>adjustCrotch</td>
        <td>MATLAB drops an element <em>before</em> testing, at both ends. Testing first
        left one extra element and shifted the whole notch profile.</td></tr>
      <tr><th scope="row">k-means modelling</th><td>adjustCrotch</td>
        <td>The port solved 2-means exactly. MATLAB seeds randomly and stops at a
        <em>local</em> optimum — frequently not the global one.</td></tr>
      <tr><th scope="row">Hull loop start</th><td>getCircumference &#8594; getWrist</td>
        <td><code>boundary</code> starts at the lowest-numbered point and repeats it to
        close the loop. <code>getWrist</code> averages that loop, so the duplicated point
        moved the wrist centroid and every arm length with it.</td></tr>
      <tr><th scope="row">Face winding</th><td>fixFaceOrientation2</td>
        <td>Never ported. Two scans carry the same triangle twice with opposite winding;
        unreconciled, the two signed volumes cancel.</td></tr>
    </tbody>
  </table></div>
  <div class="prose">
    <p>{avatar_misses} values still differ, {misses_on_odd} of them on the single
    bad-capture scan above, where MATLAB's own <code>kmeans</code> is not reproducible. The
    last is a calf girth off by five micrometres.</p>
  </div>
</section>

<section>
  <div class="sechead"><h2>Reproducing this</h2><span class="dim"></span></div>
  <pre><code>python -m unified obj2anthro --input data/obj --method avatar --units auto --out runs/avatar
python -m unified.compare runs/methods_report/combined_measurements.csv --reference matlab
python -m pytest unified/obj2anthro/tests/test_avatar_matches_matlab.py
python -m unified.obj2anthro.geometry_figures data/obj --out runs/methods_report/figures --scale-to-cm 0.1
python -m unified.obj2anthro.build_geometry_report runs/methods_report</code></pre>
</section>

<footer>
  <p>MATLAB values are the recorded run in <code>runs/matlab_ground_truth/</code>:
  <code>Avatar.m</code>, <code>steps=3</code>, <code>Vol_SA=on</code>, MATLAB
  23.2.0.2515942 (R2023b) Update 7, via the MATLAB Engine for Python on CPython 3.10. One
  mesh of 21 (<code>man.obj</code>) fails inside <code>Avatar.m</code> and is excluded from
  every comparison. Python backends were re-run over the same folder for this report.</p>
  <p>Tables beside this page in <code>runs/methods_report/</code>:
  <code>combined_measurements.csv</code>, <code>comparison_detail.csv</code>,
  <code>comparison_by_measurement.csv</code>, <code>comparison_by_subject.csv</code>,
  <code>comparison_coverage.csv</code>, <code>girth_decomposition.json</code>,
  <code>report_data.json</code>.</p>
</footer>
</div>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path, help="run directory holding the comparison tables")
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.run / "geometry_report.html")
    written = build(args.run, args.repo_root, out)
    print(f"Wrote {written}  ({written.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
