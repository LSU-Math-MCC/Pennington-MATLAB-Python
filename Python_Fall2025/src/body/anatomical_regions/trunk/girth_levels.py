import numpy as np
import trimesh

from ....utils.section_geometry import (
    central_section,
    central_section_polygon,
    local_extrema,
    normalized_stack_curve,
    polygon_measurement,
    smooth_curve,
)
from ..anatomical_region import get_geometry_config


def best_extremum(rows, indices, signal, kind, prefer_low=False, prefer_high=False):
    """Pick the most meaningful section-curve peak or valley, balancing prominence with z preference."""
    if not indices:
        return None
    z_values = np.array([row["z"] for row in rows])
    signal_values = np.array([row[signal] for row in rows])
    z_span = max(np.ptp(z_values), 1e-9)
    signal_span = max(np.ptp(signal_values), 1e-9)

    def score(i):
        prominence = abs(signal_values[i] - np.mean([signal_values[i - 1], signal_values[i + 1]])) / signal_span
        z_norm = (z_values[i] - z_values.min()) / z_span
        z_bias = -z_norm if prefer_low else z_norm if prefer_high else 0.0
        return (prominence, z_bias, signal_values[i] if kind == "max" else -signal_values[i])

    return max(indices, key=score)


def nipple_level_from_band(rows, indices, curve):
    """Choose the tape level for chest/bust girth: fullest forward torso section below the axilla."""
    if not indices:
        return None
    z_band = np.array([rows[i]["z"] for i in indices], dtype=float)
    z_min, z_span = z_band.min(), max(np.ptp(z_band), 1e-9)

    def band_norm(signal):
        values = np.array([rows[i][signal] for i in indices], dtype=float)
        span = np.ptp(values)
        return np.zeros(len(values)) if span <= 1e-9 else (values - values.min()) / span

    front = band_norm("front_smooth")
    depth = band_norm("depth_smooth")
    perimeter = band_norm("perimeter_smooth")
    area = band_norm("area_smooth")
    bust_curve = 0.42 * front + 0.24 * depth + 0.20 * perimeter + 0.14 * area

    allowed = set(indices)
    local_maxima = [i for i in local_extrema(curve, "max") if i in allowed]
    candidates = local_maxima or indices
    score_by_index = {i: bust_curve[position] for position, i in enumerate(indices)}

    def score(i):
        z_norm = (rows[i]["z"] - z_min) / z_span
        middle_bias = -0.16 * abs(z_norm - 0.55)
        axilla_penalty = -0.22 * max(z_norm - 0.82, 0.0)
        return score_by_index[i] + middle_bias + axilla_penalty

    return max(candidates, key=score)


def section_stack(mesh: trimesh.Trimesh, z_min: float, z_max: float, samples: int = 96, x_limit: float | None = None):
    """Sample central horizontal body sections and record area/perimeter/front-back shape curves over z."""
    rows = []
    for z in np.linspace(z_min, z_max, samples):
        section = central_section_polygon(mesh, z, x_limit=x_limit)
        if section is None:
            continue
        polygon, vertices_3d = section
        minx, miny, maxx, maxy = polygon.bounds
        if polygon.area <= 0 or len(vertices_3d) < 8:
            continue
        center_y = polygon.centroid.y
        rows.append({
            "z": float(z),
            "polygon": polygon,
            "vertices_3d": vertices_3d,
            "area": float(polygon.area),
            "perimeter": float(polygon.exterior.length),
            "width": float(maxx - minx),
            "depth": float(maxy - miny),
            "front": float(maxy - center_y),
            "back": float(center_y - miny),
        })
    if len(rows) < 5:
        return rows
    for signal in ("area", "perimeter", "width", "depth", "front", "back"):
        smoothed = smooth_curve(np.array([row[signal] for row in rows]))
        for row, value in zip(rows, smoothed):
            row[f"{signal}_smooth"] = float(value)
    return rows


def trunk_girth_levels(mesh: trimesh.Trimesh, trunk_api) -> dict[str, tuple[float, object, np.ndarray]]:
    """
    Select tape-measure z-levels from smoothed central section curves.

    The section stack removes arms for the torso/pelvis curves, then measures coherent
    central polygons over z. Chest is the nipple/bust tape level below the axilla;
    stomach is anterior abdomen fullness; natural waist is the intervening narrowing;
    hip is posterior/depth fullness in the glute-pelvis band.
    """
    from ..arms.arm import Arm

    body_without_arms = trunk_api._body_without_arms(mesh)
    stack_mesh = body_without_arms
    crotch = trunk_api._locate_crotch(body_without_arms)
    left_armpit, right_armpit = trunk_api._locate_armpits(mesh)
    left_shoulder = Arm._locate_shoulder(mesh, "left")
    right_shoulder = Arm._locate_shoulder(mesh, "right")

    vertices = stack_mesh.vertices
    height = np.ptp(vertices[:, 2])
    geometry_config = get_geometry_config(mesh)
    step = max(geometry_config["median_edge_length"], geometry_config["convexity_slice_width"])
    z_floor = crotch[2] + 2 * step
    z_ceiling = min(left_shoulder[2], right_shoulder[2]) - step
    z_ceiling = min(z_ceiling, vertices[:, 2].min() + 0.88 * height)
    z_ceiling = max(z_ceiling, max(left_armpit[2], right_armpit[2]) + 0.06 * height)
    if z_ceiling <= z_floor:
        z_floor = crotch[2] + 0.05 * height
        z_ceiling = vertices[:, 2].min() + 0.82 * height

    rows = section_stack(stack_mesh, z_floor, z_ceiling)
    if len(rows) < 5:
        full_rows = section_stack(mesh, z_floor, z_ceiling)
        if len(full_rows) > len(rows):
            print(
                f"section stack source: full mesh central-loop fallback "
                f"rows={len(rows)}->{len(full_rows)}"
            )
            rows = full_rows
    if len(rows) < 5:
        return {}

    z_values = np.array([row["z"] for row in rows])
    span = max(z_values.max() - z_values.min(), 1e-9)

    def band_indices(low, high):
        return [
            i for i, z in enumerate(z_values)
            if z_values.min() + low * span <= z <= z_values.min() + high * span
        ]

    def extrema_in(indices, signal, kind):
        allowed = set(indices)
        return [i for i in local_extrema(np.array([row[signal] for row in rows]), kind) if i in allowed]

    def best_curve_max(curve, indices):
        allowed = set(indices)
        maxima = [i for i in local_extrema(curve, "max") if i in allowed]
        return max(maxima, key=lambda i: curve[i]) if maxima else max(indices, key=lambda i: curve[i]) if indices else None

    chest_signals = ("front_smooth", "depth_smooth", "perimeter_smooth", "area_smooth")
    chest_curve = normalized_stack_curve(rows, chest_signals)
    stomach_curve = normalized_stack_curve(rows, ("front_smooth", "perimeter_smooth", "area_smooth"))
    hip_curve = normalized_stack_curve(rows, ("back_smooth", "depth_smooth", "perimeter_smooth", "area_smooth"))

    hip_band = band_indices(0.06, 0.36)
    stomach_band = band_indices(0.34, 0.62)
    natural_band = band_indices(0.34, 0.70)
    shoulder_z = min(left_shoulder[2], right_shoulder[2])
    chest_low_z = crotch[2] + 0.60 * (shoulder_z - crotch[2])
    chest_high_z = crotch[2] + 0.78 * (shoulder_z - crotch[2])
    area_floor = 0.25 * np.percentile([row["area_smooth"] for row in rows], 75)
    perimeter_floor = 0.35 * np.percentile([row["perimeter_smooth"] for row in rows], 75)
    chest_band = [
        i for i, row in enumerate(rows)
        if chest_low_z <= row["z"] <= chest_high_z
        and row["area_smooth"] >= area_floor
        and row["perimeter_smooth"] >= perimeter_floor
    ]
    if chest_band:
        print(
            f"chest search window: z=[{rows[chest_band[0]]['z']:.4f}, "
            f"{rows[chest_band[-1]]['z']:.4f}], shoulder_z={shoulder_z:.4f}"
        )
    else:
        print("chest search window: no valid shoulder-window rows, using fallback band")
        chest_band = band_indices(0.60, 0.86)
    axilla_band = band_indices(0.74, 0.98)
    hip_i = best_curve_max(hip_curve, hip_band)

    axilla_i = best_extremum(rows, extrema_in(axilla_band, "area_smooth", "min"), "area_smooth", "min", prefer_high=True)
    if axilla_i is None and axilla_band:
        axilla_i = min(axilla_band, key=lambda i: rows[i]["area_smooth"])
    if hip_i is None:
        return {}

    chest_i = nipple_level_from_band(rows, chest_band, chest_curve)
    chest_source_rows = rows

    if chest_i is None:
        torso_width_rows = [row for row in rows if row["z"] <= chest_low_z and row["area"] > 0]
        chest_x_limit = max(torso_width_rows, key=lambda row: row["z"])["width"] * 0.58 if torso_width_rows else None
        armpit_x_limit = 0.95 * max(abs(left_armpit[0]), abs(right_armpit[0]))
        chest_x_limit = armpit_x_limit if chest_x_limit is None else max(chest_x_limit, armpit_x_limit)
        full_chest_rows = section_stack(mesh, chest_low_z, chest_high_z, samples=64, x_limit=chest_x_limit)
        full_chest_i = None
        if len(full_chest_rows) >= 5:
            full_chest_curve = normalized_stack_curve(full_chest_rows, chest_signals)
            full_chest_i = nipple_level_from_band(full_chest_rows, list(range(len(full_chest_rows))), full_chest_curve)
        if full_chest_i is not None:
            print("chest source: full-mesh nipple window")
            chest_source_rows = full_chest_rows
            chest_i = full_chest_i

    if chest_i is not None:
        chest_z = chest_source_rows[chest_i]["z"]
        stomach_band = [i for i in stomach_band if rows[hip_i]["z"] < rows[i]["z"] < chest_z]
        natural_band = [i for i in natural_band if rows[hip_i]["z"] < rows[i]["z"] < chest_z]

    stomach_i = best_curve_max(stomach_curve, stomach_band)
    natural_i = best_extremum(rows, extrema_in(natural_band, "area_smooth", "min"), "area_smooth", "min")
    if natural_i is None and natural_band:
        natural_i = min(natural_band, key=lambda i: (
            rows[i]["area_smooth"],
            rows[i]["perimeter_smooth"],
            rows[i]["depth_smooth"],
        ))

    levels = {}
    if chest_i is not None:
        row = chest_source_rows[chest_i]
        print(
            f"chest_full_level: z={row['z']:.4f}, area={row['area']:.4f}, "
            f"perimeter={row['perimeter']:.4f}, front={row['front']:.4f}, back={row['back']:.4f}"
        )
        levels["chest_full_level"] = (row["z"], row["polygon"], row["vertices_3d"])

    for name, index in {
        "chest_axilla_level": axilla_i,
        "hip_full_level": hip_i,
        "stomach_waist_level": stomach_i,
        "natural_waist_level": natural_i,
    }.items():
        if index is None:
            continue
        row = rows[index]
        print(
            f"{name}: z={row['z']:.4f}, area={row['area']:.4f}, "
            f"perimeter={row['perimeter']:.4f}, front={row['front']:.4f}, back={row['back']:.4f}"
        )
        levels[name] = (row["z"], row["polygon"], row["vertices_3d"])
    return levels


def level_measurement(mesh: trimesh.Trimesh, name: str, trunk_api):
    """Convert a named anatomical girth level into a perimeter and closed 3D tape path."""
    levels = trunk_api._trunk_girth_levels(mesh)
    if name not in levels:
        return None
    _, polygon, vertices_3d = levels[name]
    return polygon_measurement(polygon, vertices_3d)


def hip_section(mesh: trimesh.Trimesh, trunk_api, crotch_point: np.ndarray | None = None):
    """Find the first broad pelvis section above the crotch by tracking central-section depth maxima."""
    crotch_point = trunk_api._locate_crotch(mesh) if crotch_point is None else crotch_point
    vertices = mesh.vertices
    geometry_config = get_geometry_config(mesh)
    step = max(geometry_config["median_edge_length"], geometry_config["convexity_slice_width"])
    z_values = np.linspace(crotch_point[2] + step, vertices[:, 2].max() - step, 80)
    sections = []
    for z in z_values:
        points = central_section(mesh, z)
        if points is None or len(points) < 8:
            continue
        x_width = np.ptp(points[:, 0])
        if x_width == 0 or abs(points[:, 0].mean()) > x_width:
            continue
        sections.append((z, np.ptp(points[:, 1]), x_width, points))
    if not sections:
        return None
    scores = np.array([section[1] for section in sections])
    if len(scores) >= 5:
        scores = np.convolve(scores, np.ones(5) / 5, mode="same")
    for i in range(1, len(sections) - 1):
        if scores[i - 1] < scores[i] >= scores[i + 1]:
            return sections[i][0], sections[i][3]
    best = max(sections, key=lambda section: section[1])
    return best[0], best[3]
