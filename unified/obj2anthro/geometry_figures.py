"""Figures showing *where* each pipeline measures, not just what it returns.

Three methods disagree on a chest circumference by 80 cm on the same mesh. A
table can only say that they disagree; these figures say why. Each function
returns a matplotlib ``Figure`` and, given ``out``, also writes a PNG.

Figures are drawn on a transparent background with mid-grey furniture so the
same PNG reads correctly on a light or a dark page.

CLI::

    python -m unified.obj2anthro.geometry_figures data/obj --out figures/ \\
        --subjects "CanCan01_A 2025-10-27_11-10-43" --scale-to-cm 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .backends.avatar.avatar_conversion import MatlabAvatar, load_obj

# Shared with the comparison report so a colour means the same thing everywhere.
METHOD_COLORS = {
    "avatar": "#0F6BC4",
    "matlab": "#5A6469",
    "segmentation": "#BF6200",
    "slice": "#933F79",
}
INK = "#8A9298"      # axis furniture, legible on either theme
BODY = "#B9BEC2"     # the point cloud / silhouette

# Avatar slice groups, in draw order.
SLICE_GROUPS = [
    ("Torso", ["chest", "waist", "hip"]),
    ("Legs", ["lThigh", "rThigh", "lCalf", "rCalf", "lAnkle", "rAnkle"]),
    ("Arms", ["lBicep", "rBicep", "lForearm", "rForearm", "lWrist", "rWrist"]),
]
SLICE_COLORS = {
    "chest": "#D85A30", "waist": "#BA7517", "hip": "#993C1D",
    "lThigh": "#1D9E75", "rThigh": "#0F6E56",
    "lCalf": "#5DCAA5", "rCalf": "#085041",
    "lAnkle": "#9FE1CB", "rAnkle": "#04342C",
    "lBicep": "#7F77DD", "rBicep": "#534AB7",
    "lForearm": "#AFA9EC", "rForearm": "#3C3489",
    "lWrist": "#CECBF6", "rWrist": "#26215C",
}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax, xlabel="", ylabel="", title=""):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK)
    ax.tick_params(colors=INK, labelsize=8)
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10)
    ax.set_facecolor("none")


def _save(fig, out: Path | None, name: str, dpi: int = 140):
    if out is None:
        return fig
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True)
    return fig


# --------------------------------------------------------------------------
# True mesh cross-sections (what the slice pipeline works with)
# --------------------------------------------------------------------------
def cross_section_loops(v: np.ndarray, f: np.ndarray, z: float,
                        tol: float = 1e-9) -> list[np.ndarray]:
    """Exact planar cross-section at ``z``, returned as closed loops.

    Intersects every triangle straddling the plane, then chains the resulting
    segments end to end. This is a real cross-section, unlike the *band* of
    nearby vertices that ``Avatar.m``'s ``getVOnLine`` collects.
    """
    v = np.asarray(v, dtype=float)
    f = np.asarray(f, dtype=np.int64)
    zf = v[f, 2]
    straddles = (zf.min(axis=1) <= z) & (zf.max(axis=1) >= z)

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for tri in f[straddles]:
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            p, q = v[tri[a]], v[tri[b]]
            if (p[2] - z) * (q[2] - z) > 0:
                continue
            denominator = q[2] - p[2]
            if abs(denominator) < tol:
                continue
            t = (z - p[2]) / denominator
            if -tol <= t <= 1 + tol:
                pts.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
        if len(pts) >= 2:
            segments.append((pts[0], pts[1]))
    if not segments:
        return []

    # Chain segments into loops on rounded endpoints, so shared corners match.
    def key(p):
        return (round(p[0], 6), round(p[1], 6))

    adjacency: dict[tuple, list[tuple]] = {}
    for p, q in segments:
        adjacency.setdefault(key(p), []).append(key(q))
        adjacency.setdefault(key(q), []).append(key(p))

    loops, seen = [], set()
    for start in adjacency:
        if start in seen:
            continue
        loop, node, previous = [start], start, None
        seen.add(start)
        while True:
            nxt = [n for n in adjacency[node] if n != previous and n not in seen]
            if not nxt:
                break
            previous, node = node, nxt[0]
            seen.add(node)
            loop.append(node)
        if len(loop) > 2:
            loops.append(np.array(loop + [loop[0]], dtype=float))
    return sorted(loops, key=lambda L: -_perimeter(L))


def _perimeter(loop: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(loop, axis=0), axis=1)))


# --------------------------------------------------------------------------
# Figure 1 -- where the avatar pipeline cuts
# --------------------------------------------------------------------------
def avatar_slices_on_body(avatar: MatlabAvatar, scale: float = 1.0,
                          units: str = "cm", out: Path | None = None,
                          name: str = "avatar_slices_on_body.png",
                          title: str = ""):
    """Front and side views with every measured girth drawn on the body."""
    plt = _plt()
    v = avatar.v * scale
    names = [n for _, group in SLICE_GROUPS for n in group if n in avatar.slices]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 7.2))
    fig.patch.set_alpha(0)
    for ax, (h, label) in zip(axes, [(0, "Front (x-z)"), (1, "Side (y-z)")]):
        ax.scatter(v[:, h], v[:, 2], s=0.5, c=BODY, linewidths=0, alpha=0.55)
        for n in names:
            s = avatar.slices[n]
            p = s["points3d"] * scale
            if not len(p):
                continue
            ax.scatter(p[:, h], p[:, 2], s=7, c=SLICE_COLORS.get(n, "#D85A30"),
                       label=n, linewidths=0)
        ax.set_aspect("equal")
        _style(ax, f"{'xy'[h]} ({units})", f"z ({units})", label)
    axes[1].legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.02, 0.5),
                   frameon=False, labelcolor=INK)
    if title:
        fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout()
    return _save(fig, out, name)


# --------------------------------------------------------------------------
# Figure 2 -- what a girth actually is, for the avatar pipeline
# --------------------------------------------------------------------------
def avatar_cross_sections(avatar: MatlabAvatar, only: list[str] | None = None,
                          scale: float = 1.0, units: str = "cm",
                          out: Path | None = None,
                          name: str = "avatar_cross_sections.png"):
    """One panel per girth: the vertex band, and the hull whose perimeter is reported."""
    plt = _plt()
    names = only or [n for _, g in SLICE_GROUPS for n in g if n in avatar.slices]
    names = [n for n in names if n in avatar.slices]

    ncol = min(4, len(names))
    nrow = (len(names) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow))
    fig.patch.set_alpha(0)
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, n in zip(axes, names):
        s = avatar.slices[n]
        uw = s["uw"] * scale
        colour = SLICE_COLORS.get(n, "#D85A30")
        if len(uw):
            ax.scatter(uw[:, 0], uw[:, 1], s=11, c=colour, linewidths=0, zorder=3)
            loop = uw[s["hull"]]
            ax.plot(loop[:, 0], loop[:, 1], "-", c=colour, lw=1.5, alpha=0.8)
        _style(ax, title=f"{n} — {s['girth'] * scale:.1f} {units}\n"
                         f"{s['n_points']} pts, {len(s['hull']) - 1} on hull")
        ax.set_aspect("equal")
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    return _save(fig, out, name)


# --------------------------------------------------------------------------
# Figure 3 -- the same height, three definitions of "circumference"
# --------------------------------------------------------------------------
def girth_definition_figure(avatar: MatlabAvatar, slice_name: str = "chest",
                            slice_z: float | None = None, scale: float = 1.0,
                            units: str = "cm", out: Path | None = None,
                            name: str = "girth_definition.png",
                            slice_method_value: float | None = None):
    """Side-by-side: the true cross-section loops vs the avatar hull band.

    ``slice_z`` is the height the slice pipeline reported its value from, in the
    same units as the mesh; when given, the left panel is drawn there.
    """
    plt = _plt()
    s = avatar.slices[slice_name]
    z = slice_z if slice_z is not None else s["plane"]

    loops = cross_section_loops(avatar.v, avatar.f, z)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))
    fig.patch.set_alpha(0)

    ax = axes[0]
    total = 0.0
    for i, loop in enumerate(loops):
        per = _perimeter(loop) * scale
        total += per
        ax.plot(loop[:, 0] * scale, loop[:, 1] * scale, "-",
                c=METHOD_COLORS["slice"], lw=1.6,
                alpha=1.0 if i == 0 else 0.5,
                label=f"loop {i + 1}: {per:.1f} {units}")
    ax.set_aspect("equal")
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK, loc="best")
    reported = slice_method_value if slice_method_value is not None else total
    _style(ax, f"x ({units})", f"y ({units})",
           f"True cross-section at z = {z * scale:.1f} {units}\n"
           f"{len(loops)} closed loops, summing to {reported:.1f} {units}")

    ax = axes[1]
    uw = s["uw"] * scale
    ax.scatter(uw[:, 0], uw[:, 1], s=13, c=METHOD_COLORS["avatar"],
               linewidths=0, zorder=3, label="vertex band")
    loop = uw[s["hull"]]
    ax.plot(loop[:, 0], loop[:, 1], "-", c=METHOD_COLORS["avatar"], lw=1.8,
            label=f"convex hull: {s['girth'] * scale:.1f} {units}")
    ax.set_aspect("equal")
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK, loc="best")
    _style(ax, f"x ({units})", f"y ({units})",
           f"Avatar.m band at z = {s['plane'] * scale:.1f} {units}\n"
           f"{s['n_points']} vertices, hull perimeter reported")
    fig.tight_layout()
    return _save(fig, out, name)


# --------------------------------------------------------------------------
# Figure 4 -- where each method placed the same named landmark
# --------------------------------------------------------------------------
def method_levels_figure(avatar: MatlabAvatar, levels: dict[str, dict[str, float]],
                         values: dict[str, dict[str, float]] | None = None,
                         scale: float = 1.0, units: str = "cm",
                         out: Path | None = None,
                         name: str = "method_levels.png", title: str = ""):
    """Body silhouette with each method's chest/waist/hip height drawn across it.

    ``levels`` maps method -> {measurement: z in mesh units}; ``values`` optionally
    maps the same keys to the girth reported there, which is annotated at the end
    of each line. Methods that do not expose a height are simply absent.

    Lines are staggered in length so that two methods agreeing on a height (which
    happens) still both show, rather than one hiding the other.
    """
    plt = _plt()
    v = avatar.v * scale
    fig, ax = plt.subplots(figsize=(6.6, 7.4))
    fig.patch.set_alpha(0)
    ax.scatter(v[:, 0], v[:, 2], s=0.5, c=BODY, linewidths=0, alpha=0.55)

    x_lo, x_hi = v[:, 0].min(), v[:, 0].max()
    span = x_hi - x_lo
    styles = {"chest": "-", "waist": "--", "hip": ":"}
    for i, (method, by_name) in enumerate(levels.items()):
        colour = METHOD_COLORS.get(method, INK)
        pad = (i + 1) * 0.09 * span
        for measurement, z in by_name.items():
            zz = z * scale
            ax.plot([x_lo - pad, x_hi + pad], [zz] * 2,
                    styles.get(measurement, "-"), c=colour, lw=1.6, alpha=0.9)
            reported = (values or {}).get(method, {}).get(measurement)
            if reported is not None:
                # Nudge per method so labels stay apart where levels coincide.
                dy = (i - (len(levels) - 1) / 2) * 8.0
                ax.annotate(f"{reported:.0f}", (x_hi + pad, zz), color=colour,
                            fontsize=7.5, ha="left", va="center",
                            xytext=(3, dy), textcoords="offset points")
    handles = [plt.Line2D([], [], color=METHOD_COLORS.get(m, INK), lw=1.8, label=m)
               for m in levels]
    handles += [plt.Line2D([], [], color=INK, lw=1.4, ls=styles[k], label=k)
                for k in styles if any(k in d for d in levels.values())]
    ax.legend(handles=handles, fontsize=7.5, frameon=False, labelcolor=INK,
              loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_aspect("equal")
    _style(ax, f"x ({units})", f"z ({units})", title)
    fig.tight_layout()
    return _save(fig, out, name)


# --------------------------------------------------------------------------
# Figure 5 -- the slice pipeline's own profile, annotated
# --------------------------------------------------------------------------
def slice_profile_figure(profile, marks: dict[str, float] | None = None,
                         matlab: dict[str, float] | None = None,
                         out: Path | None = None,
                         name: str = "slice_profile.png", title: str = ""):
    """Perimeter against height for the slice pipeline: sum of loops vs largest loop.

    ``profile`` is the backend's ``*_slices.csv`` as a DataFrame. ``marks`` maps a
    measurement to the height percent it was taken at; ``matlab`` gives the
    reference value for the same measurement, in cm.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    fig.patch.set_alpha(0)
    pct = profile["height_percent"].to_numpy()
    ax.plot(pct, profile["sum_perimeter"].to_numpy() / 10, "-",
            c=METHOD_COLORS["slice"], lw=1.8, label="sum of all loops (reported)")
    ax.plot(pct, profile["max_perimeter"].to_numpy() / 10, "-",
            c=METHOD_COLORS["slice"], lw=1.4, alpha=0.45,
            label="largest single loop")

    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.12 * (hi - lo))     # headroom for the level labels
    for measurement, at in (marks or {}).items():
        ax.axvline(at, color=INK, lw=0.9, ls=":", alpha=0.8)
        ax.annotate(measurement, (at, ax.get_ylim()[1]), fontsize=7.5,
                    color=INK, ha="center", va="top",
                    xytext=(0, -2), textcoords="offset points")
        if matlab and measurement in matlab:
            ax.plot([at], [matlab[measurement]], "o", ms=6,
                    c=METHOD_COLORS["matlab"], zorder=4)
    if matlab:
        ax.plot([], [], "o", ms=6, c=METHOD_COLORS["matlab"], label="Avatar.m value")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK, loc="upper left")
    _style(ax, "height (% of stature)", "perimeter (cm)", title)
    fig.tight_layout()
    return _save(fig, out, name)


def build(obj_path: Path, out: Path, scale: float = 0.1, units: str = "cm"):
    """Render the avatar-side figures for one mesh."""
    v, f = load_obj(obj_path)
    avatar = MatlabAvatar(v, f).run()
    stem = Path(obj_path).stem.replace(" ", "_")
    avatar_slices_on_body(avatar, scale, units, out,
                          f"{stem}_avatar_slices_on_body.png", Path(obj_path).stem)
    avatar_cross_sections(avatar, None, scale, units, out,
                          f"{stem}_avatar_cross_sections.png")
    return avatar


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="an OBJ, or a directory of them")
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--scale-to-cm", type=float, default=1.0, dest="scale")
    ap.add_argument("--units", default=None)
    ap.add_argument("--subjects", nargs="+", default=None,
                    help="only meshes whose stem contains one of these")
    args = ap.parse_args()

    units = args.units or ("cm" if args.scale != 1.0 else "mesh units")
    meshes = ([args.input] if args.input.is_file()
              else sorted(args.input.glob("*.obj")))
    if args.subjects:
        meshes = [m for m in meshes if any(s in m.stem for s in args.subjects)]
    for mesh in meshes:
        print(f"  {mesh.name}")
        build(mesh, args.out, args.scale, units)
    print(f"Figures written to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------------------
# Figure 6 -- the body decomposition and landmarks each measurement rests on
# --------------------------------------------------------------------------
SEGMENT_COLORS = {
    "trunk": "#5E7A8A", "head": "#8A6FBF",
    "left_arm": "#7F77DD", "right_arm": "#3C3489",
    "left_leg": "#1D9E75", "right_leg": "#0F6E56",
}
LANDMARK_ORDER = ["crotch", "l_armpit", "r_armpit", "lShoulder", "rShoulder",
                  "collar", "l_wrist", "r_wrist", "l_hip", "r_hip",
                  "l_ankle", "r_ankle", "nose_tip"]


def avatar_segments_figure(avatar: MatlabAvatar, scale: float = 1.0,
                           units: str = "cm", out: Path | None = None,
                           name: str = "avatar_segments.png", title: str = "",
                           other: dict[str, dict[str, float]] | None = None):
    """How the pipeline cut the body up, and the landmarks every girth hangs off.

    Segment assignment is the step everything downstream inherits: an arm that
    leaks into the torso moves the torso area, the arm girths and the arm length
    together.
    """
    plt = _plt()
    v = avatar.v * scale
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 7.2))
    fig.patch.set_alpha(0)

    ax = axes[0]
    assigned = np.zeros(len(v), dtype=bool)
    for key in ("trunk", "head", "left_arm", "right_arm", "left_leg", "right_leg"):
        idx = avatar.segments.get(key)
        if idx is None or not len(idx):
            continue
        assigned[idx] = True
        ax.scatter(v[idx, 0], v[idx, 2], s=3.5, c=SEGMENT_COLORS[key],
                   linewidths=0, label=f"{key.replace('_', ' ')} ({len(idx)})")
    if (~assigned).any():
        ax.scatter(v[~assigned, 0], v[~assigned, 2], s=2, c=BODY, linewidths=0,
                   label=f"unassigned ({int((~assigned).sum())})")
    ax.set_aspect("equal")
    ax.legend(fontsize=6.8, frameon=False, labelcolor=INK, loc="upper center",
              bbox_to_anchor=(0.5, -0.09), ncol=3, columnspacing=1.0,
              handletextpad=.4)
    _style(ax, f"x ({units})", f"z ({units})", "Segment assignment")

    ax = axes[1]
    ax.scatter(v[:, 0], v[:, 2], s=0.5, c=BODY, linewidths=0, alpha=0.5)
    for landmark in LANDMARK_ORDER:
        point = avatar.landmarks.get(landmark)
        if point is None or not np.all(np.isfinite(point)):
            continue
        px, pz = point[0] * scale, point[2] * scale
        ax.plot([px], [pz], "+", ms=9, mew=1.6, c=METHOD_COLORS["avatar"])
        ax.annotate(landmark, (px, pz), fontsize=6.8, color=METHOD_COLORS["avatar"],
                    xytext=(5, 2), textcoords="offset points")

    # Other methods report landmarks in their own frame, so only height is
    # comparable. They get a lane each, at the correct height, in their colour.
    if other:
        z_lo, z_hi = v[:, 2].min(), v[:, 2].max()
        x_lo, x_hi = v[:, 0].min(), v[:, 0].max()
        span = x_hi - x_lo
        for i, (method, marks) in enumerate(other.items()):
            colour = METHOD_COLORS.get(method, INK)
            x0 = x_lo - (0.10 + 0.13 * i) * span
            for landmark, fraction in marks.items():
                z = z_lo + fraction * (z_hi - z_lo)
                ax.plot([x0, x0 + 0.09 * span], [z, z], "-", c=colour, lw=2.0)
                ax.annotate(landmark, (x0, z), fontsize=6.5, color=colour,
                            ha="right", va="center", xytext=(-2, 0),
                            textcoords="offset points")
        handles = [plt.Line2D([], [], color=METHOD_COLORS["avatar"], marker="+", ls="",
                              mew=1.6, ms=8, label="avatar")]
        handles += [plt.Line2D([], [], color=METHOD_COLORS.get(m, INK), lw=2, label=m)
                    for m in other]
        ax.legend(handles=handles, fontsize=6.8, frameon=False, labelcolor=INK,
                  loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
    ax.set_aspect("equal")
    _style(ax, f"x ({units})", f"z ({units})", "Landmarks")

    if title:
        fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout()
    return _save(fig, out, name)


# --------------------------------------------------------------------------
# Figure 7 -- the same three girths, placed by each method, side by side
# --------------------------------------------------------------------------
def placement_panels(avatar: MatlabAvatar, levels: dict[str, dict[str, float]],
                     values: dict[str, dict[str, float]] | None = None,
                     scale: float = 1.0, units: str = "cm",
                     out: Path | None = None, name: str = "placement.png",
                     title: str = ""):
    """One panel per method, same body, that method's chest/waist/hip drawn on it.

    ``levels`` maps method -> {measurement: fraction of stature in [0, 1]}, which is
    the only frame all three backends can be compared in: each reports heights in
    its own units and origin.

    Side by side rather than overlaid, because the question this answers is
    whether a placement looks anatomically right, and nine lines on one body
    cannot be read that way.
    """
    plt = _plt()
    v = avatar.v * scale
    z_lo, z_hi = v[:, 2].min(), v[:, 2].max()
    order = ["avatar", "segmentation", "slice"]
    shown = [m for m in order if m in levels]

    fig, axes = plt.subplots(1, len(shown), figsize=(3.3 * len(shown), 7.0),
                             sharey=True)
    fig.patch.set_alpha(0)
    axes = np.atleast_1d(axes)
    styles = {"chest": "-", "waist": "--", "hip": ":"}

    for ax, method in zip(axes, shown):
        colour = METHOD_COLORS.get(method, INK)
        ax.scatter(v[:, 0], v[:, 2], s=0.5, c=BODY, linewidths=0, alpha=0.55)
        x_lo, x_hi = v[:, 0].min(), v[:, 0].max()
        pad = 0.08 * (x_hi - x_lo)
        for measurement, fraction in levels[method].items():
            z = z_lo + fraction * (z_hi - z_lo)
            ax.plot([x_lo - pad, x_hi + pad], [z, z], styles.get(measurement, "-"),
                    c=colour, lw=1.7, alpha=.95)
            label = measurement
            reported = (values or {}).get(method, {}).get(measurement)
            if reported is not None:
                label = f"{measurement} {reported:.0f}"
            ax.annotate(label, (x_lo - pad, z), color=colour, fontsize=7.5,
                        ha="left", va="bottom", xytext=(1, 2),
                        textcoords="offset points")
        ax.set_aspect("equal")
        _style(ax, f"x ({units})", f"z ({units})" if method == shown[0] else "", method)
    if title:
        fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout()
    return _save(fig, out, name)


# --------------------------------------------------------------------------
# Figure 8 -- how each backend cuts the body up, side by side
# --------------------------------------------------------------------------
SEG_PART_COLORS = {
    "trunk": "#5E7A8A", "head": "#8A6FBF",
    "left_arm": "#7F77DD", "right_arm": "#3C3489",
    "left_leg": "#1D9E75", "right_leg": "#0F6E56",
}


def segmentation_parts(obj_path):
    """``{part: vertices}`` from the segmentation backend's own decomposition.

    Imports the backend package directly rather than shelling out, because the
    per-subject artifacts it writes record vertex *counts* but not the split.
    Returns ``None`` if the backend cannot be imported or the mesh fails.
    """
    import sys
    root = Path(__file__).resolve().parent / "backends" / "segmentation"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from src.body import Body
    except Exception:
        return None
    try:
        body = Body(str(obj_path))
    except Exception:
        return None
    # The backend keys parts as "left arm"; normalise to match SEG_PART_COLORS.
    return {name.replace(" ", "_"): np.asarray(part.vertices, dtype=float)
            for name, part in body.subregion_meshes.items()
            if part is not None and len(part.vertices)}


def cut_comparison(avatar: MatlabAvatar, obj_path, scale: float = 1.0,
                   units: str = "cm", out: Path | None = None,
                   name: str = "cut_comparison.png", title: str = ""):
    """The same body, decomposed by Avatar.m and by the segmentation backend.

    Each backend orients the mesh itself, so the two panels are independently
    normalised to put the floor at z = 0. Vertex counts are in the legend,
    which is where the two disagree most visibly.
    """
    plt = _plt()
    parts = segmentation_parts(obj_path)
    panels = [("avatar", {k: avatar.v[idx] for k, idx in (
        ("trunk", avatar.segments["trunk"]), ("head", avatar.segments["head"]),
        ("left_arm", avatar.segments["left_arm"]),
        ("right_arm", avatar.segments["right_arm"]),
        ("left_leg", avatar.segments["left_leg"]),
        ("right_leg", avatar.segments["right_leg"])) if len(idx)})]
    if parts:
        panels.append(("segmentation", parts))

    # Both bodies are the same person, so draw them at the same true size: fix
    # each panel's unit from its own z-extent against the known stature. Reading
    # a backend's declared units and trusting them is how the first version of
    # this figure ended up with one body 100x the other.
    stature = float((avatar.v[:, 2].max() - avatar.v[:, 2].min()) * scale)

    fig, axes = plt.subplots(1, len(panels), figsize=(4.9 * len(panels), 7.0))
    fig.patch.set_alpha(0)
    axes = np.atleast_1d(axes)
    for ax, (method, groups) in zip(axes, panels):
        # Each backend has its own frame; normalise so the floor sits at zero.
        z_lo = min(v[:, 2].min() for v in groups.values())
        z_hi = max(v[:, 2].max() for v in groups.values())
        x_mid = np.mean([np.mean([v[:, 0].min(), v[:, 0].max()])
                         for v in groups.values()])
        unit = stature / (z_hi - z_lo)
        for key, verts in groups.items():
            ax.scatter((verts[:, 0] - x_mid) * unit, (verts[:, 2] - z_lo) * unit,
                       s=3.2, c=SEG_PART_COLORS.get(key, BODY), linewidths=0,
                       label=f"{key.replace('_', ' ')} ({len(verts)})")
        ax.set_aspect("equal")
        ax.legend(fontsize=6.6, frameon=False, labelcolor=INK, loc="upper center",
                  bbox_to_anchor=(0.5, -0.09), ncol=3, columnspacing=.9,
                  handletextpad=.35)
        _style(ax, f"x ({units})", f"z ({units})" if method == panels[0][0] else "",
               f"{method} — how it cuts the body")
    if title:
        fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout()
    return _save(fig, out, name)
