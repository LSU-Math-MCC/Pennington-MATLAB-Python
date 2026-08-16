"""Render the MATLAB-ground-truth comparison report as a self-contained page."""
from __future__ import annotations

import html
import json
from pathlib import Path

REPO = Path(r"C:/Users/Clint/OneDrive/Desktop/py2mat/Pennington-MATLAB-Python")
DATA = json.loads((REPO / "runs/matlab_ground_truth/chart_data.json").read_text())
OUT = Path(__file__).resolve().parent / "matlab_comparison.html"

METHOD_ORDER = ["avatar", "segmentation", "slice"]
LABEL = {"avatar": "avatar", "segmentation": "segmentation", "slice": "slice", "matlab": "MATLAB"}
SERIES_VAR = {"avatar": "--c-avatar", "segmentation": "--c-seg", "slice": "--c-slice"}

by_method = {m["method"]: m for m in DATA["methods"]}
runtime = {r["method"]: r for r in DATA["runtime"]}
short = lambda s: s.replace("_2025", " 2025").replace("_2026", " 2026")


# Demo geometry shipped for smoke tests, not body scans; they sit at a different
# scale and would compress every real point into a corner.
DEMO_MESHES = {"man", "penn-mesh-1", "penn-mesh-2"}


def esc(t) -> str:
    return html.escape(str(t))


def nice_ticks(top: float, count: int = 4) -> list[float]:
    """Round tick values covering 0..top, so axes read 0/25/50 not 0/17/33."""
    if top <= 0:
        return [0.0]
    raw = top / count
    magnitude = 10 ** int(f"{raw:e}".split("e")[1])
    for step in (1, 2, 2.5, 5, 10):
        if magnitude * step >= raw:
            nice = magnitude * step
            break
    else:
        nice = magnitude * 10
    ticks, value = [], 0.0
    while value <= top + nice * 0.001:
        ticks.append(round(value, 10))
        value += nice
    return ticks


def fmt(x, n=2):
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return esc(x)
    if v != v:
        return "—"
    return f"{v:,.{n}f}"


def pretty(measure: str) -> str:
    name = measure
    for suffix, unit in (("_cm2", " (cm²)"), ("_cm3", " (cm³)"), ("_cm", " (cm)")):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.replace("_", " ")


# ---------------------------------------------------------------- charts
def bar_agreement() -> str:
    """Mean |% error| against MATLAB, one bar per method."""
    rows = [(m, by_method[m]) for m in METHOD_ORDER]
    ticks = nice_ticks(max(r["mean_abs_pct_error"] for _, r in rows))
    top = ticks[-1]
    w, row_h, gap, left = 720, 46, 16, 128
    h = len(rows) * (row_h + gap) + 44
    plot = w - left - 96

    parts = [
        f'<svg viewBox="0 0 {w} {h}" role="img" class="chart" '
        f'aria-label="Mean absolute percent error against MATLAB by method">'
    ]
    for tick in ticks:
        x = left + plot * tick / top
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="8" x2="{x:.1f}" y2="{h-32:.1f}"/>')
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{h-14:.0f}" text-anchor="middle">'
            f'{tick:g}%</text>'
        )
    for i, (name, r) in enumerate(rows):
        y = 8 + i * (row_h + gap)
        bar = max(plot * r["mean_abs_pct_error"] / top, 2.5)
        parts.append(
            f'<text class="serieslabel" x="{left-14}" y="{y+row_h*0.46:.1f}" '
            f'text-anchor="end">{esc(LABEL[name])}</text>'
        )
        parts.append(
            f'<rect class="bar" style="fill:var({SERIES_VAR[name]})" x="{left}" y="{y}" '
            f'width="{bar:.1f}" height="{row_h*0.52:.1f}" rx="4">'
            f'<title>{esc(LABEL[name])}: {r["mean_abs_pct_error"]:.2f}% mean absolute error, '
            f'{r["pct_exact"]:.1f}% of {int(r["n_comparisons"])} comparisons exact</title></rect>'
        )
        parts.append(
            f'<text class="value" x="{left+bar+10:.1f}" y="{y+row_h*0.42:.1f}">'
            f'{r["mean_abs_pct_error"]:.2f}%</text>'
        )
        share = f'{r["pct_exact"]:.0f}%' if r["pct_exact"] >= 10 else f'{r["pct_exact"]:.1f}%'
        parts.append(
            f'<text class="sub" x="{left}" y="{y+row_h*0.94:.1f}">'
            f'{share} of {int(r["n_comparisons"])} values identical to MATLAB</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def scatter_chest() -> str:
    """Each method's chest circumference against MATLAB's, with the identity line."""
    pts = [p for p in DATA["scatter_chest"] if p["subject_id"] not in DEMO_MESHES]
    top = max(max(p["reference_value"] for p in pts), max(p["value"] for p in pts))
    ticks = nice_ticks(top, 5)
    lo, hi = 0.0, ticks[-1]

    w, h, left, bottom, top_m, right = 720, 460, 62, 52, 14, 16
    pw, ph = w - left - right, h - top_m - bottom
    sx = lambda v: left + pw * (v - lo) / (hi - lo)
    sy = lambda v: top_m + ph * (1 - (v - lo) / (hi - lo))

    parts = [
        f'<svg viewBox="0 0 {w} {h}" role="img" class="chart" '
        f'aria-label="Chest circumference from each method plotted against MATLAB">'
    ]
    for v in ticks:
        parts.append(f'<line class="grid" x1="{sx(v):.1f}" y1="{top_m}" x2="{sx(v):.1f}" y2="{top_m+ph}"/>')
        parts.append(f'<line class="grid" x1="{left}" y1="{sy(v):.1f}" x2="{left+pw}" y2="{sy(v):.1f}"/>')
        parts.append(f'<text class="tick" x="{sx(v):.1f}" y="{top_m+ph+20}" text-anchor="middle">{v:g}</text>')
        parts.append(f'<text class="tick" x="{left-10}" y="{sy(v)+4:.1f}" text-anchor="end">{v:g}</text>')

    parts.append(
        f'<line class="identity" x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" '
        f'x2="{sx(hi):.1f}" y2="{sy(hi):.1f}"/>'
    )
    # Below the diagonal is empty: every method reads at or above MATLAB here.
    parts.append(
        f'<text class="identity-label" x="{left+pw-6:.1f}" y="{top_m+ph-14:.1f}" '
        f'text-anchor="end">dashed line = agrees with MATLAB</text>'
    )
    for name in METHOD_ORDER:
        for p in [q for q in pts if q["method"] == name]:
            parts.append(
                f'<circle class="dot" style="fill:var({SERIES_VAR[name]})" '
                f'cx="{sx(p["reference_value"]):.1f}" cy="{sy(p["value"]):.1f}" r="5.5">'
                f'<title>{esc(short(p["subject_id"]))} — {esc(LABEL[name])} '
                f'{p["value"]:.1f} cm vs MATLAB {p["reference_value"]:.1f} cm</title></circle>'
            )
    parts.append(
        f'<text class="axis" x="{left+pw/2:.0f}" y="{h-10}" text-anchor="middle">'
        f'MATLAB chest circumference (cm)</text>'
    )
    parts.append(
        f'<text class="axis" transform="translate(16,{top_m+ph/2:.0f}) rotate(-90)" '
        f'text-anchor="middle">method chest circumference (cm)</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def bar_avatar_measurements() -> str:
    """Where the Python port drifts from MATLAB, measurement by measurement."""
    rows = [r for r in DATA["avatar_by_measurement"] if r["n_exact"] < r["n_subjects"]]
    rows = sorted(rows, key=lambda r: -r["mean_abs_pct_error"])
    top = max(r["mean_abs_pct_error"] for r in rows)
    row_h, gap, left, w = 21, 7, 258, 720
    h = len(rows) * (row_h + gap) + 34
    plot = w - left - 92

    parts = [
        f'<svg viewBox="0 0 {w} {h}" role="img" class="chart" '
        f'aria-label="Mean absolute percent error against MATLAB for each avatar measurement">'
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        x = left + plot * tick
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="4" x2="{x:.1f}" y2="{h-26:.1f}"/>')
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{h-8:.0f}" text-anchor="middle">{top*tick:.2f}%</text>'
        )
    for i, r in enumerate(rows):
        y = 4 + i * (row_h + gap)
        bar = max(plot * r["mean_abs_pct_error"] / top, 2)
        parts.append(
            f'<text class="rowlabel" x="{left-12}" y="{y+row_h*0.75:.1f}" text-anchor="end">'
            f'{esc(pretty(r["measurement"]))}</text>'
        )
        parts.append(
            f'<rect class="bar" style="fill:var(--c-avatar)" x="{left}" y="{y}" '
            f'width="{bar:.1f}" height="{row_h*0.72:.1f}" rx="4">'
            f'<title>{esc(pretty(r["measurement"]))}: {r["mean_abs_pct_error"]:.3f}% mean, '
            f'{r["mean_abs_error"]:.3f} absolute; identical on '
            f'{int(r["n_exact"])} of {int(r["n_subjects"])} scans</title></rect>'
        )
        parts.append(
            f'<text class="value small" x="{left+bar+8:.1f}" y="{y+row_h*0.72:.1f}">'
            f'{r["mean_abs_pct_error"]:.2f}% · {int(r["n_exact"])}/{int(r["n_subjects"])} exact</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def bar_subjects() -> str:
    """Per-scan agreement, with the six crotch-divergent scans called out."""
    rows = sorted(DATA["avatar_by_subject"], key=lambda r: r["pct_exact"], reverse=True)
    flagged = set(DATA["crotch_mismatch_subjects"])
    w, left, row_h, gap = 720, 250, 19, 6
    h = len(rows) * (row_h + gap) + 32
    plot = w - left - 118

    parts = [
        f'<svg viewBox="0 0 {w} {h}" role="img" class="chart" '
        f'aria-label="Share of measurements identical to MATLAB for each scan">'
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        x = left + plot * tick
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="4" x2="{x:.1f}" y2="{h-24:.1f}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{h-6:.0f}" text-anchor="middle">{tick*100:.0f}%</text>')
    for i, r in enumerate(rows):
        y = 4 + i * (row_h + gap)
        bar = max(plot * r["pct_exact"] / 100.0, 2)
        hit = r["subject_id"] in flagged
        parts.append(
            f'<text class="rowlabel{" flagged" if hit else ""}" x="{left-12}" '
            f'y="{y+row_h*0.75:.1f}" text-anchor="end">{esc(short(r["subject_id"]))}</text>'
        )
        parts.append(
            f'<rect class="bar" style="fill:var(--c-avatar)" '
            f'x="{left}" y="{y}" width="{bar:.1f}" height="{row_h*0.72:.1f}" rx="4">'
            f'<title>{esc(short(r["subject_id"]))}: {int(r["n_exact"])} of '
            f'{int(r["n_measurements"])} measurements identical to MATLAB'
            f'{" — crotch landmark differs" if hit else ""}</title></rect>'
        )
        tag = " ← crotch differs" if hit else ""
        parts.append(
            f'<text class="value small" x="{left+bar+8:.1f}" y="{y+row_h*0.72:.1f}">'
            f'{int(r["n_exact"])}/{int(r["n_measurements"])}{esc(tag)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def bar_runtime() -> str:
    """Seconds per scan, including MATLAB itself."""
    order = ["avatar", "matlab", "slice", "segmentation"]
    top = max(runtime[m]["mean"] for m in order)
    w, row_h, gap, left = 720, 34, 14, 128
    h = len(order) * (row_h + gap) + 30
    plot = w - left - 130

    parts = [
        f'<svg viewBox="0 0 {w} {h}" role="img" class="chart" '
        f'aria-label="Mean seconds per scan for each method">'
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        x = left + plot * tick
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="4" x2="{x:.1f}" y2="{h-22:.1f}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{h-6:.0f}" text-anchor="middle">{top*tick:.0f}s</text>')
    for i, m in enumerate(order):
        y = 4 + i * (row_h + gap)
        r = runtime[m]
        bar = max(plot * r["mean"] / top, 2)
        fill = f"var({SERIES_VAR[m]})" if m in SERIES_VAR else "var(--reference)"
        parts.append(
            f'<text class="serieslabel" x="{left-14}" y="{y+row_h*0.62:.1f}" '
            f'text-anchor="end">{esc(LABEL[m])}</text>'
        )
        parts.append(
            f'<rect class="bar" style="fill:{fill}" x="{left}" y="{y}" width="{bar:.1f}" '
            f'height="{row_h*0.68:.1f}" rx="4"><title>{esc(LABEL[m])}: {r["mean"]:.2f}s per scan, '
            f'{r["sum"]:.1f}s for all {int(r["count"])}</title></rect>'
        )
        parts.append(
            f'<text class="value" x="{left+bar+10:.1f}" y="{y+row_h*0.58:.1f}">'
            f'{r["mean"]:.2f}s <tspan class="dim">· {r["sum"]:.0f}s total</tspan></text>'
        )
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------- tables
def table_methods() -> str:
    head = (
        "<tr><th>Method</th><th>Comparisons</th><th>Measurements</th>"
        "<th>Identical</th><th>Mean |error|</th><th>Median |error|</th>"
        "<th>Mean |%|</th><th>Worst absolute</th></tr>"
    )
    rows = []
    for m in METHOD_ORDER:
        r = by_method[m]
        rows.append(
            f'<tr><td><span class="swatch" style="background:var({SERIES_VAR[m]})"></span>'
            f"{esc(LABEL[m])}</td>"
            f'<td class="num">{int(r["n_comparisons"]):,}</td>'
            f'<td class="num">{int(r["n_measurements"])}</td>'
            f'<td class="num">{int(r["n_exact"]):,} <span class="dim">({r["pct_exact"]:.1f}%)</span></td>'
            f'<td class="num">{fmt(r["mean_abs_error"], 3)}</td>'
            f'<td class="num">{fmt(r["median_abs_pct_error"], 3)}%</td>'
            f'<td class="num">{fmt(r["mean_abs_pct_error"], 2)}%</td>'
            f'<td class="num">{fmt(r["max_abs_error"], 1)}</td></tr>'
        )
    return f'<div class="scroll"><table>{head}{"".join(rows)}</table></div>'


def table_key() -> str:
    keys = sorted({k["measurement"] for k in DATA["key_measurements"]})
    lookup = {(k["method"], k["measurement"]): k for k in DATA["key_measurements"]}
    head = "<tr><th>Measurement</th>" + "".join(
        f'<th class="num"><span class="swatch" style="background:var({SERIES_VAR[m]})"></span>'
        f"{esc(LABEL[m])}</th>"
        for m in METHOD_ORDER
    ) + "</tr>"
    rows = []
    for measure in keys:
        cells = []
        for m in METHOD_ORDER:
            k = lookup.get((m, measure))
            if not k:
                cells.append('<td class="num dim">not produced</td>')
                continue
            exact = f'{int(k["n_exact"])}/{int(k["n_subjects"])}'
            cells.append(
                f'<td class="num">{fmt(k["mean_abs_pct_error"], 2)}% '
                f'<span class="dim">· {exact} exact</span></td>'
            )
        rows.append(f"<tr><td>{esc(pretty(measure))}</td>{''.join(cells)}</tr>")
    return f'<div class="scroll"><table>{head}{"".join(rows)}</table></div>'


# ---------------------------------------------------------------- page
avatar, seg, slc = (by_method[m] for m in METHOD_ORDER)
mat_run = DATA["matlab_run"]
n_sub = int(avatar["n_subjects"])
n_exact_all = sum(1 for r in DATA["avatar_by_measurement"] if r["n_exact"] == r["n_subjects"])
n_measures = len(DATA["avatar_by_measurement"])

CSS = """
:root{
  --ground:#F5F5F1; --surface:#FFFFFF; --surface-2:#FAFAF7;
  --ink:#191C1F; --ink-2:#4A5257; --ink-3:#767D82;
  --rule:#DEE0DA; --rule-2:#EBEDE7;
  --c-avatar:#0F6BC4; --c-seg:#BF6200; --c-slice:#933F79;
  --reference:#4A5257;
  --good:#1E7A47; --warn:#A8650C; --crit:#A32E2E;
  --shadow:0 1px 2px rgba(25,28,31,.05), 0 8px 24px -16px rgba(25,28,31,.28);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#131518; --surface:#1B1E22; --surface-2:#202429;
    --ink:#E8EAEB; --ink-2:#AFB6BB; --ink-3:#828A90;
    --rule:#2C3136; --rule-2:#24282C;
    --c-avatar:#4A97DC; --c-seg:#D07F22; --c-slice:#B4629A;
    --reference:#8A9298;
    --good:#4FA97A; --warn:#D3922F; --crit:#D96A6A;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.8);
  }
}
:root[data-theme="dark"]{
  --ground:#131518; --surface:#1B1E22; --surface-2:#202429;
  --ink:#E8EAEB; --ink-2:#AFB6BB; --ink-3:#828A90;
  --rule:#2C3136; --rule-2:#24282C;
  --c-avatar:#4A97DC; --c-seg:#D07F22; --c-slice:#B4629A;
  --reference:#8A9298;
  --good:#4FA97A; --warn:#D3922F; --crit:#D96A6A;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.8);
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:Georgia,"Iowan Old Style","Source Serif 4",serif;
  font-size:17px; line-height:1.62; -webkit-font-smoothing:antialiased;
}
.narrow{font-family:"Arial Narrow","Roboto Condensed","Liberation Sans Narrow",ui-sans-serif,sans-serif}
.mono{font-family:ui-monospace,"Cascadia Mono",Consolas,"DejaVu Sans Mono",monospace}

.wrap{max-width:64rem; margin:0 auto; padding:0 1.5rem 6rem}
.prose{max-width:40rem}
.prose p{margin:0 0 1rem}

header{border-bottom:1px solid var(--rule); margin-bottom:3rem; padding:3.5rem 0 2.25rem}
.eyebrow{
  font-family:"Arial Narrow","Roboto Condensed","Liberation Sans Narrow",ui-sans-serif,sans-serif;
  text-transform:uppercase; letter-spacing:.16em; font-size:.75rem;
  color:var(--ink-3); margin:0 0 1rem;
}
h1{
  font-family:"Arial Narrow","Roboto Condensed","Liberation Sans Narrow",ui-sans-serif,sans-serif;
  font-size:clamp(2.1rem,5vw,3.1rem); line-height:1.06; letter-spacing:-.01em;
  font-weight:700; margin:0 0 1.25rem; text-wrap:balance;
}
.standfirst{font-size:1.16rem; color:var(--ink-2); max-width:42rem; margin:0}

section{margin:0 0 3.75rem}
h2{
  font-family:"Arial Narrow","Roboto Condensed","Liberation Sans Narrow",ui-sans-serif,sans-serif;
  font-size:1.6rem; letter-spacing:-.005em; font-weight:700;
  margin:0 0 .5rem; text-wrap:balance;
}
h3{
  font-family:"Arial Narrow","Roboto Condensed","Liberation Sans Narrow",ui-sans-serif,sans-serif;
  text-transform:uppercase; letter-spacing:.12em; font-size:.8rem;
  color:var(--ink-3); margin:0 0 .75rem; font-weight:700;
}
.lede{color:var(--ink-2); margin:0 0 1.75rem; max-width:40rem}

.cards{display:grid; grid-template-columns:repeat(auto-fit,minmax(15rem,1fr)); gap:1rem; margin:0 0 2rem}
.card{
  background:var(--surface); border:1px solid var(--rule); border-radius:10px;
  padding:1.15rem 1.25rem 1.25rem; box-shadow:var(--shadow); position:relative; overflow:hidden;
}
.card::before{content:""; position:absolute; inset:0 auto 0 0; width:3px; background:var(--bar)}
.card .name{
  font-family:"Arial Narrow","Roboto Condensed","Liberation Sans Narrow",ui-sans-serif,sans-serif;
  text-transform:uppercase; letter-spacing:.11em; font-size:.76rem; color:var(--ink-3);
  margin:0 0 .5rem; display:flex; align-items:center; gap:.45rem;
}
.card .big{
  font-family:ui-monospace,"Cascadia Mono",Consolas,monospace;
  font-size:2.15rem; line-height:1; font-variant-numeric:tabular-nums;
  letter-spacing:-.02em; margin:0 0 .3rem;
}
.card .unit{font-size:1rem; color:var(--ink-3)}
.card .note{font-size:.9rem; color:var(--ink-2); margin:.4rem 0 0; line-height:1.45}

.swatch{display:inline-block; width:.62rem; height:.62rem; border-radius:2px; margin-right:.4rem; vertical-align:baseline}

figure{margin:0 0 1.25rem; background:var(--surface); border:1px solid var(--rule);
  border-radius:10px; padding:1.25rem 1.35rem 1rem; box-shadow:var(--shadow)}
figcaption{font-size:.9rem; color:var(--ink-2); margin:.85rem 0 0; padding-top:.8rem;
  border-top:1px solid var(--rule-2); line-height:1.5}
.chart{display:block; width:100%; height:auto; overflow:visible}
.grid{stroke:var(--rule-2); stroke-width:1}
.tick{fill:var(--ink-3); font-size:11px; font-family:ui-monospace,Consolas,monospace;
  font-variant-numeric:tabular-nums}
.axis{fill:var(--ink-3); font-size:11.5px; letter-spacing:.06em; text-transform:uppercase;
  font-family:"Arial Narrow","Roboto Condensed",ui-sans-serif,sans-serif}
.serieslabel{fill:var(--ink); font-size:15px;
  font-family:"Arial Narrow","Roboto Condensed",ui-sans-serif,sans-serif; font-weight:700}
.rowlabel{fill:var(--ink-2); font-size:12.5px;
  font-family:"Arial Narrow","Roboto Condensed",ui-sans-serif,sans-serif}
.rowlabel.flagged{fill:var(--ink); font-weight:700}
.value{fill:var(--ink); font-size:13.5px; font-family:ui-monospace,Consolas,monospace;
  font-variant-numeric:tabular-nums}
.value.small{font-size:11.5px; fill:var(--ink-2)}
.value .dim, .dim{fill:var(--ink-3); color:var(--ink-3)}
.sub{fill:var(--ink-3); font-size:11.5px;
  font-family:"Arial Narrow","Roboto Condensed",ui-sans-serif,sans-serif}
.bar{transition:opacity .12s ease}
.bar:hover{opacity:.78}
.bar.hatched{stroke:var(--surface); stroke-width:0}
.dot{stroke:var(--surface); stroke-width:2; transition:r .12s ease}
.dot:hover{r:8}
.identity{stroke:var(--reference); stroke-width:2; stroke-dasharray:7 5}
.identity-label{fill:var(--ink-3); font-size:11.5px; letter-spacing:.06em; text-transform:uppercase;
  font-family:"Arial Narrow","Roboto Condensed",ui-sans-serif,sans-serif}

.legend{display:flex; flex-wrap:wrap; gap:1.1rem; margin:0 0 1rem; padding:0; list-style:none;
  font-family:"Arial Narrow","Roboto Condensed",ui-sans-serif,sans-serif; font-size:.88rem;
  letter-spacing:.03em; color:var(--ink-2)}
.legend li{display:flex; align-items:center; gap:.45rem}
.legend .rule{width:1.1rem; height:0; border-top:2px dashed var(--reference)}

.scroll{overflow-x:auto; border:1px solid var(--rule); border-radius:10px; background:var(--surface);
  box-shadow:var(--shadow)}
table{border-collapse:collapse; width:100%; font-size:.9rem;
  font-family:"Arial Narrow","Roboto Condensed",ui-sans-serif,sans-serif}
th,td{padding:.6rem .85rem; text-align:left; border-bottom:1px solid var(--rule-2); white-space:nowrap}
th{background:var(--surface-2); font-size:.74rem; text-transform:uppercase; letter-spacing:.09em;
  color:var(--ink-3); font-weight:700; position:sticky; top:0}
tbody tr:last-child td, tr:last-child td{border-bottom:none}
td.num{font-family:ui-monospace,Consolas,monospace; font-variant-numeric:tabular-nums; text-align:right}
th.num{text-align:right}

.callout{border-left:3px solid var(--bar,var(--c-avatar)); background:var(--surface);
  border-radius:0 10px 10px 0; padding:1.1rem 1.35rem; margin:0 0 1.5rem;
  box-shadow:var(--shadow); max-width:44rem}
.callout p{margin:0 0 .65rem}
.callout p:last-child{margin:0}
.callout strong{font-weight:700}
.trace{font-size:.82rem; line-height:1.5; color:var(--ink-2); background:var(--surface-2);
  border:1px solid var(--rule-2); border-radius:6px; padding:.6rem .75rem; overflow-x:auto}

code{font-family:ui-monospace,"Cascadia Mono",Consolas,monospace; font-size:.88em;
  background:var(--surface-2); border:1px solid var(--rule-2); border-radius:4px; padding:.08em .35em}

.chips{display:flex; flex-wrap:wrap; gap:.4rem; margin:.9rem 0 0; padding:0; list-style:none}
.chip{font-family:ui-monospace,Consolas,monospace; font-size:.78rem; padding:.24rem .55rem;
  border:1px solid var(--rule); border-radius:999px; color:var(--ink-2); background:var(--surface-2)}

footer{border-top:1px solid var(--rule); padding-top:1.75rem; color:var(--ink-3); font-size:.9rem}
footer p{margin:0 0 .5rem}
a{color:var(--c-avatar)}
:focus-visible{outline:2px solid var(--c-avatar); outline-offset:3px; border-radius:3px}
@media (prefers-reduced-motion:reduce){*{transition:none!important; animation:none!important}}
@media (max-width:640px){ body{font-size:16px} .wrap{padding:0 1.1rem 4rem} }
"""

BODY = f"""
<div class="wrap">
<header>
  <p class="eyebrow">Pennington OBJ-to-anthropometry · {mat_run["ok"]} of {mat_run["n"]} meshes measured in MATLAB R2023b</p>
  <h1>Scoring the Python Backends Against Avatar.m</h1>
  <p class="standfirst">MATLAB R2023b now runs inside the pipeline, so <code>Avatar.m</code>
  can be the reference rather than a memory. Every other method is scored against what
  MATLAB actually returned — run as designed, with nothing caught and retried behind
  the scenes.</p>
</header>

<section>
  <h2>What the reference run itself did</h2>
  <p class="lede">The comparison is only worth as much as the reference. This is what
  <code>Avatar.m</code> did when it ran, unedited and unassisted.</p>

  <div class="callout" style="--bar:var(--crit)">
    <p><strong>{mat_run["n"] - mat_run["ok"]} of {mat_run["n"]} meshes failed outright.</strong>
    <code>man.obj</code> throws inside <code>Avatar.m</code> itself:</p>
    <p class="mono trace">Unable to perform assignment because the size of the left side is
    1-by-1 and the size of the right side is 0-by-1<br>
    &nbsp;&nbsp;in Avatar.adjustCrotch — Avatar.m line 749</p>
    <p>Line 749 is <code>[mx_v_bot(2,i), Idx_v2_bot] = max(v2_bot);</code>. The
    quarter-band filter two lines above leaves <code>v2_bot</code> empty on that mesh,
    and <code>max</code> of an empty array returns nothing to assign. It is a
    robustness gap in the reference, not in the harness. That mesh is recorded as
    failed and excluded from every comparison below.</p>
  </div>

  <div class="callout" style="--bar:var(--warn)">
    <p><strong>Every body scan takes Avatar.m's secondary OBJ reader.</strong> The fast
    path raises, and Avatar.m falls back on its own:</p>
    <p class="mono trace">Warning: An error occured with the message: The logical indices
    contain a true value outside of the array bounds.<br>
    Attempting to read obj file with regexp. You may want to preprocess the obj file.<br>
    &nbsp;&nbsp;In Avatar&gt;readObj (line 3845)</p>
    <p>This is Avatar.m's own designed fallback, so the values it produces are the
    reference's real answer and are left untouched. It is worth knowing that the fast
    reader never succeeds on these files.</p>
  </div>

  <p class="lede">Nothing else is caught. <code>Avatar.m</code> is called once per mesh as
  <code>steps=3, Vol_SA='on'</code>, with the OBJ passed through exactly as it sits on
  disk, and any mesh it cannot process is recorded as a failure rather than retried on a
  reduced configuration.</p>
</section>

<section>
  <h2>How far each method sits from MATLAB</h2>
  <p class="lede">Mean absolute error across every measurement the method and MATLAB
  both produced. A value counts as identical when it matches to floating-point noise.</p>

  <div class="cards">
    <div class="card" style="--bar:var(--c-avatar)">
      <p class="name"><span class="swatch" style="background:var(--c-avatar)"></span>avatar</p>
      <p class="big">{avatar['mean_abs_pct_error']:.2f}<span class="unit">%</span></p>
      <p class="note">{int(avatar['n_exact']):,} of {int(avatar['n_comparisons']):,} values
      identical to MATLAB ({avatar['pct_exact']:.0f}%).</p>
    </div>
    <div class="card" style="--bar:var(--c-seg)">
      <p class="name"><span class="swatch" style="background:var(--c-seg)"></span>segmentation</p>
      <p class="big">{seg['mean_abs_pct_error']:.1f}<span class="unit">%</span></p>
      <p class="note">{int(seg['n_exact']):,} of {int(seg['n_comparisons']):,} identical
      ({seg['pct_exact']:.0f}%). A different algorithm, not a port.</p>
    </div>
    <div class="card" style="--bar:var(--c-slice)">
      <p class="name"><span class="swatch" style="background:var(--c-slice)"></span>slice</p>
      <p class="big">{slc['mean_abs_pct_error']:.0f}<span class="unit">%</span></p>
      <p class="note">{int(slc['n_exact']):,} of {int(slc['n_comparisons']):,} identical
      ({slc['pct_exact']:.1f}%). See the caveat below.</p>
    </div>
  </div>

  <figure>
    {bar_agreement()}
    <figcaption>Mean absolute percent error against MATLAB. Only pairs where both the
    method and MATLAB produced a value are scored, so no method is penalised for
    measurements it does not implement.</figcaption>
  </figure>

  {table_methods()}
</section>

<section>
  <h2>The port reproduces MATLAB; the other two do not</h2>
  <p class="lede">Plotting one measurement against MATLAB's own value makes the three
  methods separate cleanly. Points on the dashed line agree with the reference.</p>

  <ul class="legend">
    <li><span class="swatch" style="background:var(--c-avatar)"></span>avatar</li>
    <li><span class="swatch" style="background:var(--c-seg)"></span>segmentation</li>
    <li><span class="swatch" style="background:var(--c-slice)"></span>slice</li>
    <li><span class="rule"></span>agrees with MATLAB</li>
  </ul>

  <figure>
    {scatter_chest()}
    <figcaption>Chest circumference, one point per body scan per method. The avatar port lands on the
    identity line. Segmentation tracks the trend with a consistent offset — it measures a
    different anatomical slice. Slice sits far above every MATLAB value. The three demo meshes shipped for smoke tests (<code>man</code>, <code>penn-mesh-1/2</code>) are left out here because they sit at a different scale.</figcaption>
  </figure>

  <div class="callout" style="--bar:var(--c-slice)">
    <p><strong>The slice backend needs a look before its numbers are used.</strong>
    It reports a {slc['mean_abs_pct_error']:.0f}% mean deviation, and on individual scans
    returns a chest circumference near 174&nbsp;cm and a thigh near 121&nbsp;cm where both
    MATLAB and the port agree on roughly 107&nbsp;cm and 62&nbsp;cm. Its stature is also
    about 3&nbsp;cm short of MATLAB's.</p>
    <p>This is pre-existing behaviour, reproduced by running <code>--method slice</code>
    on its own against untouched code. It is not a regression from this work — but three
    methods in one table is what made it obvious.</p>
  </div>
</section>

<section>
  <h2>Where the port still drifts</h2>
  <p class="lede">The port is exact on {n_exact_all} of {n_measures} measurements across every
  scan MATLAB measured. The rest disagree, and the pattern points at a single cause.</p>

  <figure>
    {bar_avatar_measurements()}
    <figcaption>Every measurement where the port and MATLAB differ on at least one scan,
    worst first. Bars show the mean absolute percent error; the label to the right gives
    how many scans matched exactly.</figcaption>
  </figure>

  <div class="callout" style="--bar:var(--c-avatar)">
    <p><strong>One root cause explains the largest group.</strong> Trunk length, inseam,
    leg length, hip, waist and torso surface area are all exact on exactly 14 of {n_sub} scans —
    the same six scans fail every one of them. Those six are precisely the scans where the
    port's crotch landmark differs from MATLAB's.</p>
    <p><code>adjustCrotch</code> is the one place the port deliberately diverges: MATLAB
    calls <code>kmeans</code> with random initialisation, and the port substitutes an exact
    1-D 2-means. The port's own README claims this "does not change the result on any tested
    mesh" — but that was five meshes. On these 21 it moves the crotch on six.</p>
    <p>The remaining differences — calf, forearm, bicep, arm length — are sub-1% and come
    from angle searches where a tiny numeric difference tips an <code>acos</code> or an
    iteration bound. They are noise, not a different answer.</p>
  </div>

  <figure>
    {bar_subjects()}
    <figcaption>Share of measurements identical to MATLAB, per scan. The six labelled
    scans are those where the crotch landmark differs; they are the same six that drag
    every crotch-derived measurement.</figcaption>
  </figure>

  <h3>Scans where the crotch landmark differs</h3>
  <ul class="chips">
    {"".join(f'<li class="chip">{esc(short(s))}</li>' for s in DATA["crotch_mismatch_subjects"])}
  </ul>
</section>

<section>
  <h2>Agreement on the measurements people ask for</h2>
  <p class="lede">Mean absolute percent error against MATLAB for the headline measurements,
  with the count of scans that matched exactly.</p>
  {table_key()}
</section>

<section>
  <h2>What each method costs</h2>
  <p class="lede">Mean wall-clock seconds per scan. MATLAB's figure excludes engine
  start-up, which is a one-time cost of roughly 20 seconds per batch.</p>
  <figure>
    {bar_runtime()}
    <figcaption>The port returns MATLAB-grade numbers about seven times faster than
    MATLAB itself, and needs no MATLAB licence or Engine install.</figcaption>
  </figure>
</section>

<footer>
  <p>Generated from <code>runs/matlab_ground_truth/</code>. Reference method:
  <code>matlab</code> — Avatar.m, steps=3, Vol_SA=on, MATLAB 23.2.0.2515942 (R2023b) Update 7,
  driven through the MATLAB Engine for Python on CPython 3.10.</p>
  <p>Underlying tables: <code>comparison_by_method.csv</code>,
  <code>comparison_by_measurement.csv</code>, <code>comparison_by_subject.csv</code>,
  <code>comparison_detail.csv</code>, <code>comparison_coverage.csv</code>. Regenerate with
  <code>python -m unified.compare &lt;combined tables&gt; --reference matlab</code>.</p>
</footer>
</div>
"""

OUT.write_text(
    f"<title>Avatar.m as Ground Truth</title>\n<style>{CSS}</style>\n{BODY}",
    encoding="utf-8",
)
print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")
