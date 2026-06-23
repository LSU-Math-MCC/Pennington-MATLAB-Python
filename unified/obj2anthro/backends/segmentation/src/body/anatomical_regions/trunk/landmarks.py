import numpy as np
import trimesh
from scipy.spatial import cKDTree

from ....utils.convexity_search import convexity_search
from ....utils.section_geometry import central_loop_index, central_section, section_loops
from .girth_levels import hip_section
from ..anatomical_region import get_geometry_config


def locate_crotch(mesh: trimesh.Trimesh) -> np.ndarray:
    """Find the saddle point where the two leg openings merge into the lower trunk."""
    new_mesh = mesh.copy()
    kdtree = cKDTree(new_mesh.vertices)
    minimum_z = new_mesh.vertices[np.argmin(new_mesh.vertices, axis=0)[2], 2]
    intersects = new_mesh.ray.intersects_location(
        ray_origins=[np.array([0, 0, minimum_z])],
        ray_directions=[np.array([0, 0, 1])],
    )[0]
    min_viable_point = (
        new_mesh.vertices[np.argmin(new_mesh.vertices[:, 2])]
        if len(intersects) == 0
        else intersects[np.argmin(intersects, axis=0)[2], :]
    )
    viable_point = new_mesh.vertices[kdtree.query(min_viable_point)[1]]
    crotch_point_nearest = convexity_search(
        new_mesh,
        rays=32,
        origin=viable_point,
        slice_width=get_geometry_config(new_mesh)["convexity_slice_width"],
    )
    crotch_point = new_mesh.vertices[kdtree.query(crotch_point_nearest)[1]]
    min_z = new_mesh.vertices[:, 2].min()
    height = np.ptp(new_mesh.vertices[:, 2])
    if (crotch_point[2] - min_z) / height < 0.40:
        maxmin_crotch = maxmin_crotch_point(new_mesh)
        if maxmin_crotch[2] > crotch_point[2]:
            print(f"Crotch maxmin raised z {crotch_point[2]:.4f} -> {maxmin_crotch[2]:.4f}")
            crotch_point = maxmin_crotch
    topology_crotch = topology_crotch_point(new_mesh)
    if topology_crotch is not None and topology_crotch[2] > crotch_point[2]:
        print(f"Crotch topology raised z {crotch_point[2]:.4f} -> {topology_crotch[2]:.4f}")
        crotch_point = topology_crotch
    return crotch_point


def topology_crotch_point(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """Slice upward from the feet and detect the first stable center neck after two leg loops join."""
    vertices = mesh.vertices
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    height = z_max - z_min
    saw_two_legs = False
    merged_sections = []
    for z in np.linspace(z_min + 0.03 * height, z_min + 0.45 * height, 80):
        loops = section_loops(mesh, z)
        side_loops = []
        for loop in loops:
            x = loop[:, 0]
            if np.ptp(x) and (x.max() < 0 or x.min() > 0):
                side_loops.append(loop)
        has_left = any(loop[:, 0].mean() < 0 for loop in side_loops)
        has_right = any(loop[:, 0].mean() > 0 for loop in side_loops)
        if has_left and has_right:
            saw_two_legs = True
            continue
        central = central_section(mesh, z)
        if saw_two_legs and central is not None and central[:, 0].min() < 0 < central[:, 0].max():
            ratio = neck_ratio(central)
            if ratio is not None:
                merged_sections.append((z, ratio, central[np.argmin(np.abs(central[:, 0]))]))
    if not merged_sections:
        return None
    ratios = np.array([ratio for _, ratio, _ in merged_sections])
    low, high = np.percentile(ratios, [20, 80])
    threshold = low + 0.5 * (high - low)
    for z, ratio, point in merged_sections:
        if ratio >= threshold:
            print(f"Crotch topology neck z={z:.4f}, ratio={ratio:.3f}, threshold={threshold:.3f}")
            return mesh.vertices[cKDTree(vertices).query(point)[1]]
    z, ratio, point = max(merged_sections, key=lambda row: row[1])
    print(f"Crotch topology neck fallback z={z:.4f}, ratio={ratio:.3f}")
    return mesh.vertices[cKDTree(vertices).query(point)[1]]


def neck_ratio(points: np.ndarray) -> float | None:
    """Measure how pinched the central x=0 bridge is relative to the side lobes of a section."""
    x = points[:, 0]
    y = points[:, 1]
    width = np.ptp(x)
    if width <= 0:
        return None
    strip = max(0.08 * width, 1e-9)

    def y_span(center):
        ys = y[np.abs(x - center) <= strip]
        return np.ptp(ys) if len(ys) >= 3 else np.nan

    neck = y_span(0.0)
    lobe = np.nanmean([y_span(x.min() + 0.30 * width), y_span(x.max() - 0.30 * width)])
    return None if not np.isfinite(neck) or not np.isfinite(lobe) or lobe <= 0 else float(neck / lobe)


def maxmin_crotch_point(mesh: trimesh.Trimesh) -> np.ndarray:
    """Choose the highest column-wise minimum between the feet as a fallback crotch saddle."""
    from ..legs import Leg

    vertices = mesh.vertices
    left_foot, right_foot = Leg._identify_feet(mesh)
    x_min, x_max = sorted([left_foot[0], right_foot[0]])
    candidates = []
    for left, right in zip(np.linspace(x_min, x_max, 51)[:-1], np.linspace(x_min, x_max, 51)[1:]):
        column = vertices[(vertices[:, 0] > left) & (vertices[:, 0] < right)]
        if len(column):
            candidates.append(column[np.argmin(column[:, 2])])
    return vertices[np.argmin(vertices[:, 2])] if not candidates else np.asarray(candidates)[np.argmax(np.asarray(candidates)[:, 2])]


def locate_armpits(mesh: trimesh.Trimesh, trunk_api) -> tuple:
    """Locate left and right axilla points by tracing up from hips and validating torso-arm separation."""
    print("Called locate_armpits (Trunk)")
    new_mesh = mesh.copy()
    kdtree = cKDTree(new_mesh.vertices)
    left_hip, right_hip = trunk_api._locate_hips(mesh)
    left_armpit, right_armpit = choose_armpit_pair(
        new_mesh,
        kdtree,
        left_hip,
        right_hip,
        armpit_candidates(new_mesh, kdtree, left_hip, "left"),
        armpit_candidates(new_mesh, kdtree, right_hip, "right"),
    )
    print(f"Armpits: left={left_armpit}, right={right_armpit}")
    return left_armpit, right_armpit


def add_armpit_candidate(candidates: list, source: str, point: np.ndarray | None):
    """Store one possible axilla height if a detector produced a real mesh point."""
    if point is not None and len(point) == 3 and np.all(np.isfinite(point)):
        candidates.append({"source": source, "point": point})


def armpit_candidates(mesh: trimesh.Trimesh, kdtree: cKDTree, hip: np.ndarray, side: str) -> list:
    """Collect trace, topology, horizontal-gap, and vertical-gap armpit hypotheses."""
    candidates = []
    add_armpit_candidate(candidates, "trace", trace_hip_to_armpit(mesh, kdtree, hip, side=side))
    for candidate in section_merge_armpit_candidates(mesh, kdtree, hip):
        add_armpit_candidate(candidates, "section-merge", candidate)
    add_armpit_candidate(candidates, "section-gap", section_gap_armpit(mesh, kdtree, hip))
    add_armpit_candidate(candidates, "side-gap", side_gap_armpit(mesh, kdtree, hip))
    if not candidates:
        add_armpit_candidate(candidates, "hip", hip)
    return candidates


def side_relative_height(mesh: trimesh.Trimesh, hip: np.ndarray, armpit: np.ndarray) -> float:
    """Express an armpit z as progress from hip height toward that side's shoulder/arm cap."""
    side = "left" if hip[0] < 0 else "right"
    outward = -1 if side == "left" else 1
    side_vertices = mesh.vertices[(mesh.vertices[:, 0] - hip[0]) * outward > 0]
    side_top = np.percentile(side_vertices[:, 2], 98) if len(side_vertices) else mesh.vertices[:, 2].max()
    return float((armpit[2] - hip[2]) / max(side_top - hip[2], 1e-9))


def scored_armpit_candidates(mesh: trimesh.Trimesh, kdtree: cKDTree, hip: np.ndarray, candidates: list) -> list:
    """Convert detector outputs into central-edge axilla candidates with geometry scores."""
    scored = []
    source_weights = {"section-merge": 2.7, "section-gap": 2.0, "trace": 1.2, "side-gap": 1.4, "hip": -4.0}
    for candidate in candidates:
        point = candidate["point"]
        relative = side_relative_height(mesh, hip, point)
        height_score = 2.2 - 6.0 * abs(relative - 0.72)
        if relative < 0.46:
            height_score -= 5.0 + 20.0 * (0.46 - relative)
        if relative > 0.92:
            height_score -= 3.0 + 10.0 * (relative - 0.92)
        score = source_weights.get(candidate["source"], 0.0) + height_score
        score += arm_support_score(mesh, hip, point)
        score += arm_cut_viability_score(mesh, hip, point)
        scored.append({**candidate, "relative": relative, "score": score})
    return sorted(scored, key=lambda row: row["score"], reverse=True)


def arm_support_score(mesh: trimesh.Trimesh, hip: np.ndarray, point: np.ndarray) -> float:
    """Reward candidates that have arm surface outward/downward and torso surface inward at the same height."""
    geometry_config = get_geometry_config(mesh)
    vertices = mesh.vertices
    height = np.ptp(vertices[:, 2])
    side = "left" if hip[0] < 0 else "right"
    outward = -1 if side == "left" else 1
    z_band = max(0.018 * height, geometry_config["median_edge_length"])
    x_band = max(geometry_config["armpit_lateral_band"], 0.02 * height)
    local = vertices[np.abs(vertices[:, 2] - point[2]) <= z_band]
    if len(local) == 0:
        return -2.0
    outward_span = np.max((local[:, 0] - point[0]) * outward)
    inward_span = np.max((point[0] - local[:, 0]) * outward)
    score = 0.0
    score += min(1.2, outward_span / max(x_band, 1e-9))
    score += min(0.8, inward_span / max(x_band, 1e-9))
    below = vertices[
        ((vertices[:, 0] - point[0]) * outward > x_band)
        & (vertices[:, 2] < point[2])
        & (vertices[:, 2] > point[2] - 0.25 * height)
    ]
    score += 0.8 if len(below) else -0.8
    return score


def arm_cut_viability_score(mesh: trimesh.Trimesh, hip: np.ndarray, point: np.ndarray) -> float:
    """Score whether an x-plane through this armpit would leave a plausible A-pose arm component."""
    vertices = mesh.vertices
    height = np.ptp(vertices[:, 2])
    side = "left" if hip[0] < 0 else "right"
    outward = -1 if side == "left" else 1
    side_vertices = vertices[
        ((vertices[:, 0] - point[0]) * outward > 0)
        & (vertices[:, 2] > hip[2] - 0.08 * height)
    ]
    if len(side_vertices) < 30:
        return -5.0
    z_min, z_max = side_vertices[:, 2].min(), side_vertices[:, 2].max()
    score = 0.0
    score += 1.4 if z_max > point[2] + 0.04 * height else -3.0
    score += 1.4 if z_min < point[2] - 0.18 * height else -2.0
    span = z_max - z_min
    if span < 0.28 * height:
        score -= 4.0
    elif span > 0.78 * height:
        score -= 2.5
    else:
        score += 1.0
    return score


def choose_armpit_pair(
    mesh: trimesh.Trimesh,
    kdtree: cKDTree,
    left_hip: np.ndarray,
    right_hip: np.ndarray,
    left_candidates: list,
    right_candidates: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose left/right axilla candidates jointly so one side cannot win via a shoulder or hip outlier."""
    left_scored = scored_armpit_candidates(mesh, kdtree, left_hip, left_candidates)
    right_scored = scored_armpit_candidates(mesh, kdtree, right_hip, right_candidates)
    height = np.ptp(mesh.vertices[:, 2])
    best_pair = None
    for left in left_scored[:5]:
        for right in right_scored[:5]:
            relative_penalty = 5.0 * abs(left["relative"] - right["relative"])
            z_penalty = 3.0 * abs(left["point"][2] - right["point"][2]) / max(height, 1e-9)
            score = left["score"] + right["score"] - relative_penalty - z_penalty
            if best_pair is None or score > best_pair[0]:
                best_pair = (score, left, right)
    _, left, right = best_pair
    left_point, right_point = rebalance_high_armpit(
        mesh,
        kdtree,
        left_hip,
        right_hip,
        left["point"],
        right["point"],
        left["relative"],
        right["relative"],
    )
    print(
        "Armpit candidates: "
        f"left={left['source']} z={left_point[2]:.4f} rel={side_relative_height(mesh, left_hip, left_point):.3f} score={left['score']:.2f}; "
        f"right={right['source']} z={right_point[2]:.4f} rel={side_relative_height(mesh, right_hip, right_point):.3f} score={right['score']:.2f}"
    )
    return left_point, right_point


def lateral_band_point_at_relative(mesh: trimesh.Trimesh, kdtree: cKDTree, hip: np.ndarray, relative: float) -> np.ndarray:
    """Snap to the lateral hip strip at a target axilla-band height for the current side."""
    geometry_config = get_geometry_config(mesh)
    vertices = mesh.vertices
    side = "left" if hip[0] < 0 else "right"
    outward = -1 if side == "left" else 1
    side_vertices = vertices[(vertices[:, 0] - hip[0]) * outward > 0]
    side_top = np.percentile(side_vertices[:, 2], 98) if len(side_vertices) else vertices[:, 2].max()
    target_z = hip[2] + relative * max(side_top - hip[2], 0.0)
    band_width = geometry_config["armpit_lateral_band"] * 3
    band = vertices[
        (vertices[:, 0] >= hip[0] - band_width / 2)
        & (vertices[:, 0] <= hip[0] + band_width / 2)
        & (np.abs(vertices[:, 2] - target_z) <= max(0.025 * np.ptp(vertices[:, 2]), geometry_config["median_edge_length"]))
    ]
    if len(band):
        point = band[np.argmin(np.abs(band[:, 2] - target_z))]
        return mesh.vertices[kdtree.query(point)[1]]
    return edge_armpit_at_z(mesh, kdtree, hip, target_z)


def rebalance_high_armpit(
    mesh: trimesh.Trimesh,
    kdtree: cKDTree,
    left_hip: np.ndarray,
    right_hip: np.ndarray,
    left_point: np.ndarray,
    right_point: np.ndarray,
    left_relative: float,
    right_relative: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Lower a one-sided shoulder-gap choice when the other side found a normal axilla-band cut."""
    if left_relative > 0.88 and 0.55 <= right_relative <= 0.80:
        target = float(np.clip(right_relative, 0.62, 0.76))
        corrected = lateral_band_point_at_relative(mesh, kdtree, left_hip, target)
        print(f"Armpit rebalance left: high rel {left_relative:.3f} -> {side_relative_height(mesh, left_hip, corrected):.3f}")
        left_point = corrected
    if right_relative > 0.88 and 0.55 <= left_relative <= 0.80:
        target = float(np.clip(left_relative, 0.62, 0.76))
        corrected = lateral_band_point_at_relative(mesh, kdtree, right_hip, target)
        print(f"Armpit rebalance right: high rel {right_relative:.3f} -> {side_relative_height(mesh, right_hip, corrected):.3f}")
        right_point = corrected
    return left_point, right_point


def edge_armpit_at_z(mesh: trimesh.Trimesh, kdtree: cKDTree, hip: np.ndarray, z: float) -> np.ndarray:
    """Place a corrected axilla point on the central torso edge at a bilaterally consistent height."""
    loops = section_loops(mesh, z)
    if not loops:
        return hip
    central = loops[central_loop_index(loops)]
    candidate = central[np.argmin(central[:, 0]) if hip[0] < 0 else np.argmax(central[:, 0])]
    return mesh.vertices[kdtree.query(candidate)[1]]


def section_merge_armpit_candidates(mesh: trimesh.Trimesh, kdtree: cKDTree, hip: np.ndarray) -> list[np.ndarray]:
    """Find heights where a separate arm loop merges into the central torso section loop."""
    geometry_config = get_geometry_config(mesh)
    vertices = mesh.vertices
    side = "left" if hip[0] < 0 else "right"
    outward = -1 if side == "left" else 1
    step = max(geometry_config["median_edge_length"], geometry_config["convexity_slice_width"])
    side_vertices = vertices[(vertices[:, 0] - hip[0]) * outward > 0]
    if len(side_vertices) == 0:
        return []
    z_top = np.percentile(side_vertices[:, 2], 98)
    had_side_loop = False
    previous_edge = None
    transitions = []
    for z in np.linspace(hip[2] + step, z_top, 60):
        loops = section_loops(mesh, z)
        if not loops:
            continue
        central_index = central_loop_index(loops)
        central = loops[central_index]
        edge = central[np.argmin(central[:, 0]) if side == "left" else np.argmax(central[:, 0])]
        side_loop = any(
            i != central_index and np.mean((loop[:, 0] - edge[0]) * outward > 0) > 0.8
            for i, loop in enumerate(loops)
        )
        if had_side_loop and not side_loop and previous_edge is not None:
            candidate = central[np.argmin(np.linalg.norm(central[:, :2] - previous_edge[:2], axis=1))]
            transitions.append(mesh.vertices[kdtree.query(candidate)[1]])
        had_side_loop = side_loop
        previous_edge = edge
    return transitions


def section_gap_armpit(mesh: trimesh.Trimesh, kdtree: cKDTree, hip: np.ndarray) -> np.ndarray | None:
    """Find the upper narrow horizontal gap between the central torso loop and a side arm loop."""
    geometry_config = get_geometry_config(mesh)
    vertices = mesh.vertices
    side = "left" if hip[0] < 0 else "right"
    outward = -1 if side == "left" else 1
    side_vertices = vertices[(vertices[:, 0] - hip[0]) * outward > 0]
    if len(side_vertices) < 10:
        return None
    side_top = np.percentile(side_vertices[:, 2], 96)
    z_low = hip[2] + 0.42 * max(side_top - hip[2], 0.0)
    z_high = hip[2] + 0.84 * max(side_top - hip[2], 0.0)
    if z_high <= z_low:
        return None
    rows = []
    for z in np.linspace(z_low, z_high, 48):
        loops = section_loops(mesh, z)
        if len(loops) < 2:
            continue
        central_index = central_loop_index(loops)
        central = loops[central_index]
        central_edge = central[np.argmin(central[:, 0]) if side == "left" else np.argmax(central[:, 0])]
        side_loop_rows = []
        for i, loop in enumerate(loops):
            if i == central_index:
                continue
            x = loop[:, 0]
            if np.mean((x - central_edge[0]) * outward > 0) < 0.7:
                continue
            inner_x = x.max() if side == "left" else x.min()
            side_loop_rows.append((abs(inner_x - central_edge[0]), loop))
        if side_loop_rows:
            gap, _ = min(side_loop_rows, key=lambda row: row[0])
            rows.append((gap, z, central_edge))
    if not rows:
        return None
    gaps = np.array([row[0] for row in rows])
    tolerance = max(0.25 * np.median(gaps), geometry_config["armpit_lateral_band"])
    _, _, candidate = max([row for row in rows if row[0] <= gaps.min() + tolerance], key=lambda row: row[1])
    return mesh.vertices[kdtree.query(candidate)[1]]


def side_gap_armpit(mesh: trimesh.Trimesh, kdtree: cKDTree, hip: np.ndarray) -> np.ndarray | None:
    """Use a vertical gap in a lateral vertex band as a fallback estimate of the axilla opening."""
    geometry_config = get_geometry_config(mesh)
    vertices = mesh.vertices
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    height = z_max - z_min
    band_width = geometry_config["armpit_lateral_band"] * 3
    side = "left" if hip[0] < 0 else "right"
    outward = -1 if side == "left" else 1
    side_vertices = vertices[(vertices[:, 0] - hip[0]) * outward > 0]
    side_top = np.percentile(side_vertices[:, 2], 98) if len(side_vertices) else z_max - 0.12 * height
    z_lower = hip[2] + 0.50 * max(side_top - hip[2], 0.0)
    band = vertices[
        (vertices[:, 0] >= hip[0] - band_width / 2) &
        (vertices[:, 0] <= hip[0] + band_width / 2) &
        (vertices[:, 2] >= z_lower) &
        (vertices[:, 2] <= side_top)
    ]
    if len(band) < 2:
        return None
    band = band[np.argsort(band[:, 2])]
    candidate = band[np.argmax(np.diff(band[:, 2]))]
    return None if candidate[2] < z_lower else mesh.vertices[kdtree.query(candidate)[1]]


def trace_hip_to_armpit(mesh: trimesh.Trimesh, kdtree: cKDTree, hip_point: np.ndarray, side: str) -> np.ndarray:
    """Follow a narrow vertical strip above the hip until vertex spacing indicates the underarm void."""
    geometry_config = get_geometry_config(mesh)
    lateral_band_width = geometry_config["armpit_lateral_band"]
    lateral_vertices = mesh.vertices[
        (mesh.vertices[:, 0] >= hip_point[0] - lateral_band_width / 2) &
        (mesh.vertices[:, 0] <= hip_point[0] + lateral_band_width / 2) &
        (mesh.vertices[:, 2] >= hip_point[2])
    ]
    if len(lateral_vertices) == 0:
        print(f"Warning: No lateral vertices found for {side} side, using hip point")
        return hip_point
    armpit_candidate = hip_point.copy()
    prev_z = hip_point[2]
    for current_point in lateral_vertices[np.argsort(lateral_vertices[:, 2])][1:]:
        if current_point[2] - prev_z < geometry_config["armpit_z_stop"]:
            armpit_candidate = current_point
            break
        prev_z = current_point[2]
        armpit_candidate = current_point
    try:
        armpit_refined = convexity_search(
            mesh,
            rays=32,
            origin=armpit_candidate,
            slice_width=geometry_config["convexity_slice_width"],
        )
        return mesh.vertices[kdtree.query(armpit_refined)[1]]
    except Exception as e:
        print(f"Warning: Convexity search failed for {side} armpit: {e}")
        return mesh.vertices[kdtree.query(armpit_candidate)[1]]


def locate_hips(mesh: trimesh.Trimesh, trunk_api):
    """Select the leftmost and rightmost points on the best pelvis-width section above the crotch."""
    print("Called locate_hips (Trunk)")
    crotch_point = trunk_api._locate_crotch(mesh)
    section = hip_section(mesh, trunk_api, crotch_point)
    if section is None:
        body_height = np.ptp(mesh.vertices[:, 2])
        slice_z_min = crotch_point[2] + body_height * 0.05
        slice_z_max = crotch_point[2] + body_height * 0.1
        torso_vertices = mesh.vertices[(mesh.vertices[:, 2] >= slice_z_min) & (mesh.vertices[:, 2] <= slice_z_max)]
    else:
        _, torso_vertices = section
    kdtree = cKDTree(mesh.vertices)
    return (
        mesh.vertices[kdtree.query(torso_vertices[np.argmin(torso_vertices[:, 0])])[1]],
        mesh.vertices[kdtree.query(torso_vertices[np.argmax(torso_vertices[:, 0])])[1]],
    )


def locate_collar(mesh: trimesh.Trimesh, trunk_api):
    """Snap the shoulder midpoint onto the front trunk surface to approximate the collar landmark."""
    from ..arms.arm import Arm

    trunk_mesh = trunk_api._get_submesh(mesh)
    trunk_vertices = trunk_mesh.vertices
    midpoint = (Arm._locate_shoulder(mesh, "right") + Arm._locate_shoulder(mesh, "left")) / 2
    front_mesh = trunk_mesh.slice_plane(plane_origin=mesh.centroid, plane_normal=np.array([0, 1, 0]))
    search_vertices = front_mesh.vertices if front_mesh is not None and len(front_mesh.vertices) > 0 else trunk_vertices
    snapped = search_vertices[cKDTree(search_vertices).query(midpoint)[1]].copy()
    snapped[2] += 0.01 * (trunk_vertices[:, 2].max() - trunk_vertices[:, 2].min())
    return snapped
