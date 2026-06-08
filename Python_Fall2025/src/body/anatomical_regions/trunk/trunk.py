from functools import cache

import trimesh
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull, cKDTree
from shapely.geometry import box

from ....utils.convexity_search import convexity_search

from ..anatomical_region import Anatomical_Region, get_geometry_config


class Trunk(Anatomical_Region):
    """
    Trunk (torso) region segmentation, landmark detection, and measurement computation.
    
    The Trunk class handles extraction and analysis of the torso from a full body mesh.
    It identifies critical landmarks (crotch, armpits, hips, collar) and computes important
    measurements (trunk length plus chest, natural-waist, stomach-peak, and hip girths).
    
    Purpose
    -------
    This class solves the problem of automatically measuring the torso region from body scans.
    Without this class, you would need to manually:
    - Separate trunk from head, arms, and legs
    - Locate complex landmarks like armpits (concave regions) and crotch (convex region)
    - Build coherent horizontal torso/pelvis section polygons
    - Compute girths at anatomical z-levels rather than arbitrary surface points
    
    With this class, creating a Trunk instance automatically performs all these operations.
    
    Class Structure and Design Choices
    -----------------------------------
    **Central role in body segmentation**:
    The Trunk class is architecturally central because:
    - It provides landmarks used by other regions (armpits for arms, crotch for legs)
    - Other regions depend on trunk landmarks for their own segmentation
    - It contains the most complex landmark detection algorithms
    
    **Static methods with @cache**:
    See Anatomical_Region docstring for the rationale behind this design pattern.
    Key points specific to Trunk: armpits and crotch are computed once and shared
    with Arm and Leg classes that depend on these landmarks.
    
    **Convexity-based landmark detection**:
    Armpits and crotch are found using convexity_search() which:
    - Casts rays horizontally at different heights
    - Computes convexity of the resulting cross-section
    - Finds points where the body becomes concave (armpits) or convex (crotch)
    
    This is a heuristic approach that works well for standard poses but is sensitive to:
    - Mesh quality (holes or artifacts near landmarks)
    - Pose variation (arms raised, legs together)
    - Body proportions (very thin or heavy individuals)
    
    **Boolean operations for segmentation**:
    The trunk mesh is extracted by:
    1. Start with full body
    2. Remove both legs (mesh_difference)
    3. Remove both arms (mesh_difference)
    4. Remove head (mesh_difference)
    
    This "subtraction" approach is simpler than trying to slice the trunk directly
    but requires good mesh quality for boolean operations to succeed.
    
    **Runtime imports**:
    Imports appear inside methods because:
    - Trunk needs Leg/Arm/Head to remove them
    - Leg/Arm/Head need Trunk for landmarks
    - This creates circular dependencies broken by runtime imports
    
    **Why KD-trees**:
    The class uses scipy's cKDTree for fast nearest-neighbor queries when:
    - Finding vertices on vertical lines (for landmark detection)
    - Computing circumferences (finding vertices at specific heights)
    
    KD-trees make these O(log n) instead of O(n) searches.
    
    Attributes
    ----------
    body_mesh : trimesh.Trimesh
        The full body mesh (cleaned and oriented)
    
    Properties
    ----------
    mesh : trimesh.Trimesh
        The segmented trunk mesh (cached, computed on first access)
    landmarks : dict[str, np.ndarray or tuple]
        Key anatomical points:
        - "crotch": Point where legs meet trunk
        - "armpits": Tuple of (left_armpit, right_armpit) points
        - "hips": Tuple of (left_hip, right_hip) points
        - "collar": Point at base of neck
    measurements : dict[str, float]
        Anthropometric measurements:
        - "trunk length": Crotch to collar distance (in XZ plane only)
        - "chest circumference": Girth at upper-torso/chest-full level
        - "waist circumference": Girth at natural waist narrowing
        - "stomach peak circumference": Girth at the fullest anterior abdomen
        - "hip circumference": Girth at posterior/glute-pelvis fullness
    
    Examples
    --------
    >>> import trimesh
    >>> from body import Body  # doctest: +SKIP
    >>> body = Body("model_files/man.obj")  # doctest: +SKIP
    >>> trunk = body.parts["trunk"]  # doctest: +SKIP
    >>> trunk_length = trunk.measurements["trunk length"]  # doctest: +SKIP
    >>> print(f"Trunk length: {trunk_length:.2f} cm")  # doctest: +SKIP
    Trunk length: 52.8 cm
    
    >>> crotch = trunk.landmarks["crotch"]  # doctest: +SKIP
    >>> left_armpit, right_armpit = trunk.landmarks["armpits"]  # doctest: +SKIP
    >>> print(f"Crotch: {crotch}")  # doctest: +SKIP
    Crotch: [0.0, 0.0, 20.5]
    
    >>> chest_circ = trunk.measurements["chest circumference"]  # doctest: +SKIP
    >>> waist_circ = trunk.measurements["waist circumference"]  # doctest: +SKIP
    >>> hip_circ = trunk.measurements["hip circumference"]  # doctest: +SKIP
    >>> print(f"Chest: {chest_circ:.1f}, Waist: {waist_circ:.1f}, Hip: {hip_circ:.1f}")  # doctest: +SKIP
    Chest: 95.2, Waist: 82.3, Hip: 98.5
    
    Notes
    -----
    - Most complex landmark detection of all body parts
    - Armpit/crotch detection uses convexity_search (can fail on poor meshes)
    - Requires body mesh to be in standard orientation (Z-axis vertical)
    - Assumes standard pose (arms down, legs apart)
    - Girth measurements are z-levels where a tape wraps one coherent central section.
      The selected level is not itself a point landmark on the skin.
    - Chest is searched as upper anterior/full-torso fullness near the shoulder-to-crotch
      chest band, with a full-mesh shoulder-window fallback when arm removal trims the bust.
    - Waist naming is explicit: "waist" is natural waist/narrowing, while "stomach peak"
      is the fullest anterior abdomen.
    - Hip is keyed to posterior/depth fullness in the pelvis/glute interval, not just global
      area or left-right width.
    - All measurements are in the same units as the input mesh
    - Trunk length uses only X and Z coordinates (ignores Y to avoid posture effects)
    
    See Also
    --------
    convexity_search : Algorithm for finding concave/convex regions
    mesh_difference : Boolean operation for removing body parts
    Anatomical_Region : Abstract base class defining the interface
    """

    def __init__(self, body_mesh: trimesh.Trimesh):
        print("Called __init__ (Trunk)")

        self.body_mesh = body_mesh

    @property
    def volume(self):
        print("Called volume (Trunk)")
        return self._trimesh.volume
    
    @property
    def surface_area(self):
        print("Called surface_area (Trunk)")
        return self._trimesh.area

    # Properties of Trunk

    # Vertex Indices (provided by `mesh`)
    @property
    def mesh(self):
        return Trunk._get_submesh(self.body_mesh)

    @staticmethod
    @cache
    def _get_submesh(mesh: trimesh.Trimesh):
        """
        Get trunk mesh by removing legs, arms, and head from body.
        """
        from ..legs import Leg
        from ..arms import Arm
        from ..head import Head
        from ....mesh.boolean_ops import mesh_difference
        
        # Start with full body
        trunk_mesh = mesh.copy()
        
        # Remove both legs
        left_leg_mesh = Leg._get_submesh("left", mesh)
        trunk_mesh = mesh_difference(trunk_mesh, left_leg_mesh)
        
        right_leg_mesh = Leg._get_submesh("right", mesh)
        trunk_mesh = mesh_difference(trunk_mesh, right_leg_mesh)
        
        # Remove both arms
        left_arm_mesh = Arm._get_submesh("left", mesh)
        trunk_mesh = mesh_difference(trunk_mesh, left_arm_mesh)
        
        right_arm_mesh = Arm._get_submesh("right", mesh)
        trunk_mesh = mesh_difference(trunk_mesh, right_arm_mesh)
        
        # Remove head
        head_mesh = Head._get_submesh(mesh)
        trunk_mesh = mesh_difference(trunk_mesh, head_mesh)
        
        return trunk_mesh

    # Landmarks # TODO: Might want to make these properties so the access is simpler, but not required. This goes for all landmarks and measurements in src actually

    @property
    def landmarks(self):
        return {
            "crotch": Trunk._locate_crotch(self.body_mesh),
            "armpits": Trunk._locate_armpits(self.body_mesh),         # TODO: split into left and right
            "hips": Trunk._locate_hips(self.body_mesh),               # TODO: same, split left/right
            "collar": Trunk._locate_collar(self.body_mesh)
        }

    @staticmethod
    @cache
    def _locate_crotch(mesh: trimesh.Trimesh) -> np.ndarray:
        new_mesh = mesh.copy()
        
        kdtree = cKDTree(new_mesh.vertices)
        
        minimum_z = new_mesh.vertices[np.argmin(new_mesh.vertices, axis=0)[2], 2]
        
        ray_origin = np.array([0, 0, minimum_z])
        ray_direction = np.array([0, 0, 1])
        
        intersects = new_mesh.ray.intersects_location(
            ray_origins=[ray_origin],
            ray_directions=[ray_direction]
        )[0]
        
        if len(intersects) == 0:
            min_viable_point = new_mesh.vertices[np.argmin(new_mesh.vertices[:, 2])]
        else:
            min_viable_point = intersects[np.argmin(intersects, axis=0)[2], :]
        viable_point_idx =  kdtree.query(min_viable_point)[1]
        viable_point = new_mesh.vertices[viable_point_idx]
        
        crotch_point_nearest = convexity_search(new_mesh, 
                                        rays=32,
                                        origin=viable_point,
                                        slice_width=get_geometry_config(new_mesh)["convexity_slice_width"])
        
        crotch_point_idx = kdtree.query(crotch_point_nearest)[1]
        crotch_point = new_mesh.vertices[crotch_point_idx]

        min_z = new_mesh.vertices[:, 2].min()
        height = np.ptp(new_mesh.vertices[:, 2])
        if (crotch_point[2] - min_z) / height < 0.40:
            maxmin_crotch = Trunk._maxmin_crotch(new_mesh)
            if maxmin_crotch[2] > crotch_point[2]:
                print(f"Crotch maxmin raised z {crotch_point[2]:.4f} -> {maxmin_crotch[2]:.4f}")
                crotch_point = maxmin_crotch

        topology_crotch = Trunk._topology_crotch(new_mesh)
        if topology_crotch is not None and topology_crotch[2] > crotch_point[2]:
            print(f"Crotch topology raised z {crotch_point[2]:.4f} -> {topology_crotch[2]:.4f}")
            crotch_point = topology_crotch
        
        return crotch_point

    @staticmethod
    def _topology_crotch(mesh: trimesh.Trimesh) -> np.ndarray | None:
        vertices = mesh.vertices
        z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
        height = z_max - z_min
        z_values = np.linspace(z_min + 0.03 * height, z_min + 0.45 * height, 80)
        saw_two_legs = False
        merged_sections = []

        for z in z_values:
            loops = Trunk._section_loops(mesh, z)
            side_loops = []
            for loop in loops:
                x = loop[:, 0]
                if np.ptp(x) == 0:
                    continue
                if x.max() < 0 or x.min() > 0:
                    side_loops.append(loop)

            has_left = any(loop[:, 0].mean() < 0 for loop in side_loops)
            has_right = any(loop[:, 0].mean() > 0 for loop in side_loops)
            if has_left and has_right:
                saw_two_legs = True
                continue

            central = Trunk._central_section(mesh, z)
            if saw_two_legs and central is not None and central[:, 0].min() < 0 < central[:, 0].max():
                ratio = Trunk._neck_ratio(central)
                if ratio is not None:
                    mid = central[np.argmin(np.abs(central[:, 0]))]
                    merged_sections.append((z, ratio, mid))

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

    @staticmethod
    def _neck_ratio(points: np.ndarray) -> float | None:
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
        left_lobe = y_span(x.min() + 0.30 * width)
        right_lobe = y_span(x.max() - 0.30 * width)
        lobe = np.nanmean([left_lobe, right_lobe])

        if not np.isfinite(neck) or not np.isfinite(lobe) or lobe <= 0:
            return None

        return float(neck / lobe)

    @staticmethod
    def _maxmin_crotch(mesh: trimesh.Trimesh) -> np.ndarray:
        from ..legs import Leg

        vertices = mesh.vertices
        left_foot, right_foot = Leg._identify_feet(mesh)
        x_min, x_max = sorted([left_foot[0], right_foot[0]])
        candidates = []

        for left, right in zip(np.linspace(x_min, x_max, 51)[:-1], np.linspace(x_min, x_max, 51)[1:]):
            column = vertices[(vertices[:, 0] > left) & (vertices[:, 0] < right)]
            if len(column):
                candidates.append(column[np.argmin(column[:, 2])])

        if not candidates:
            return vertices[np.argmin(vertices[:, 2])]

        candidates = np.asarray(candidates)
        return candidates[np.argmax(candidates[:, 2])]

    @staticmethod
    def _ordered_loop(points: np.ndarray) -> np.ndarray:
        center = points[:, :2].mean(axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        return points[np.argsort(angles)]

    @staticmethod
    def _section_loops(mesh: trimesh.Trimesh, z: float) -> list[np.ndarray]:
        section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if section is None:
            return []

        loops = [
            np.asarray(loop)
            for loop in getattr(section, "discrete", [])
            if len(loop) >= 3
        ]
        if not loops and len(section.vertices) >= 3:
            loops = [section.vertices]
        return loops

    @staticmethod
    def _central_section(mesh: trimesh.Trimesh, z: float) -> np.ndarray | None:
        loops = Trunk._section_loops(mesh, z)
        if not loops:
            return None

        return loops[Trunk._central_loop_index(loops)]

    @staticmethod
    def _central_loop_index(loops: list[np.ndarray]) -> int:
        def score(points):
            x = points[:, 0]
            x_width = np.ptp(x)
            crosses_midline = x.min() <= 0 <= x.max()
            return (crosses_midline, x_width / (abs(x.mean()) + x_width + 1e-9), x_width)

        return max(range(len(loops)), key=lambda i: score(loops[i]))

    @staticmethod
    def _central_section_polygon(mesh: trimesh.Trimesh, z: float, x_limit: float | None = None):
        section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if section is None:
            return None

        try:
            path_2d, to_3d = section.to_planar()
            polygons = list(path_2d.polygons_full)
        except Exception:
            return None

        if not polygons:
            return None

        def score(polygon):
            minx, _, maxx, _ = polygon.bounds
            crosses_midline = minx <= 0 <= maxx
            return (crosses_midline, polygon.area / (abs(polygon.centroid.x) + 1e-9), polygon.area)

        polygon = max(polygons, key=score)
        if x_limit is not None:
            try:
                _, miny, _, maxy = polygon.bounds
                clipped = polygon.intersection(box(-x_limit, miny - 1.0, x_limit, maxy + 1.0))
                if not clipped.is_empty:
                    pieces = [clipped] if clipped.geom_type == "Polygon" else [
                        geom for geom in clipped.geoms if geom.geom_type == "Polygon"
                    ]
                    if pieces:
                        polygon = max(pieces, key=lambda geom: geom.area)
            except Exception:
                pass

        coords_2d = np.asarray(polygon.exterior.coords[:-1])
        coords_3d = trimesh.transformations.transform_points(
            np.column_stack([coords_2d, np.zeros(len(coords_2d))]),
            to_3d,
        )
        return polygon, coords_3d

    @staticmethod
    def _body_without_arms(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        from ..arms import Arm
        from ....mesh.boolean_ops import mesh_difference

        body_without_arms = mesh.copy()
        left_arm_mesh = Arm._get_submesh("left", mesh)
        body_without_arms = mesh_difference(body_without_arms, left_arm_mesh)
        right_arm_mesh = Arm._get_submesh("right", mesh)
        body_without_arms = mesh_difference(body_without_arms, right_arm_mesh)
        return body_without_arms

    @staticmethod
    def _smooth_curve(values: np.ndarray, window: int = 7) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if len(values) < 3:
            return values
        window = min(window, len(values) if len(values) % 2 else len(values) - 1)
        return uniform_filter1d(values, size=window, mode="nearest") if window >= 3 else values

    @staticmethod
    def _local_extrema(values: np.ndarray, kind: str) -> list[int]:
        curve = np.asarray(values, dtype=float)
        return find_peaks(curve if kind == "max" else -curve)[0].tolist()

    @staticmethod
    def _best_extremum(
        rows: list[dict],
        indices: list[int],
        signal: str,
        kind: str,
        prefer_low: bool = False,
        prefer_high: bool = False,
    ) -> int | None:
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

    @staticmethod
    def _normalized_stack_curve(rows: list[dict], signals: tuple[str, ...]) -> np.ndarray:
        curves = []
        for signal in signals:
            values = np.array([row[signal] for row in rows], dtype=float)
            span = np.ptp(values)
            if span <= 1e-9:
                curves.append(np.zeros(len(values)))
            else:
                curves.append((values - values.min()) / span)
        return np.mean(curves, axis=0)

    @staticmethod
    def _first_stable_max_from_top(
        rows: list[dict],
        indices: list[int],
        curve: np.ndarray,
        min_prominence: float = 0.08,
    ) -> int | None:
        if not indices:
            return None

        allowed = set(indices)
        maxima = [i for i in Trunk._local_extrema(curve, "max") if i in allowed]
        if not maxima:
            return None

        curve_span = max(np.ptp(curve), 1e-9)
        first_i, last_i = min(indices), max(indices)
        for i in sorted(maxima, key=lambda index: rows[index]["z"], reverse=True):
            lo = max(first_i, i - 4)
            hi = min(last_i, i + 4)
            local_floor = np.percentile(curve[lo:hi + 1], 25)
            if (curve[i] - local_floor) / curve_span >= min_prominence:
                return i

        return max(maxima, key=lambda index: rows[index]["z"])

    @staticmethod
    def _upper_rise_level(rows: list[dict], upper_i: int | None, lower_i: int | None, curve: np.ndarray) -> int | None:
        if upper_i is None or lower_i is None or upper_i <= lower_i:
            return lower_i

        upper_value = curve[upper_i]
        lower_value = curve[lower_i]
        if lower_value <= upper_value:
            return lower_i

        threshold = upper_value + 0.25 * (lower_value - upper_value)
        for i in range(upper_i, lower_i - 1, -1):
            if curve[i] >= threshold:
                return i
        return lower_i

    @staticmethod
    def _section_stack(
        mesh: trimesh.Trimesh,
        z_min: float,
        z_max: float,
        samples: int = 96,
        x_limit: float | None = None,
    ) -> list[dict]:
        rows = []
        for z in np.linspace(z_min, z_max, samples):
            section = Trunk._central_section_polygon(mesh, z, x_limit=x_limit)
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
            smoothed = Trunk._smooth_curve(np.array([row[signal] for row in rows]))
            for row, value in zip(rows, smoothed):
                row[f"{signal}_smooth"] = float(value)
        return rows

    @staticmethod
    @cache
    def _trunk_girth_levels(mesh: trimesh.Trimesh) -> dict[str, tuple[float, object, np.ndarray]]:
        """
        Select tape-measure z-levels from smoothed central section curves.

        The section stack removes arms for the torso/pelvis curves, then measures coherent
        central polygons over z. Chest is the upper torso fullness; stomach is anterior
        abdomen fullness; natural waist is the intervening narrowing; hip is posterior/depth
        fullness in the glute-pelvis band.
        """
        body_without_arms = Trunk._body_without_arms(mesh)
        stack_mesh = body_without_arms
        crotch = Trunk._locate_crotch(body_without_arms)
        left_armpit, right_armpit = Trunk._locate_armpits(mesh)
        from ..arms.arm import Arm
        left_shoulder = Arm._locate_shoulder(mesh, "left")
        right_shoulder = Arm._locate_shoulder(mesh, "right")

        vertices = stack_mesh.vertices
        height = np.ptp(vertices[:, 2])
        step = max(get_geometry_config(mesh)["median_edge_length"], get_geometry_config(mesh)["convexity_slice_width"])
        z_floor = crotch[2] + 2 * step
        z_ceiling = min(left_shoulder[2], right_shoulder[2]) - step
        z_ceiling = min(z_ceiling, vertices[:, 2].min() + 0.88 * height)
        z_ceiling = max(z_ceiling, max(left_armpit[2], right_armpit[2]) + 0.06 * height)
        if z_ceiling <= z_floor:
            z_floor = crotch[2] + 0.05 * height
            z_ceiling = vertices[:, 2].min() + 0.82 * height

        rows = Trunk._section_stack(stack_mesh, z_floor, z_ceiling)
        if len(rows) < 5:
            return {}

        z_values = np.array([row["z"] for row in rows])
        span = max(z_values.max() - z_values.min(), 1e-9)

        def band_indices(low: float, high: float) -> list[int]:
            return [
                i for i, z in enumerate(z_values)
                if z_values.min() + low * span <= z <= z_values.min() + high * span
            ]

        def extrema_in(indices: list[int], signal: str, kind: str) -> list[int]:
            allowed = set(indices)
            return [i for i in Trunk._local_extrema(np.array([row[signal] for row in rows]), kind) if i in allowed]

        def best_curve_max(curve: np.ndarray, indices: list[int]) -> int | None:
            allowed = set(indices)
            maxima = [i for i in Trunk._local_extrema(curve, "max") if i in allowed]
            if maxima:
                return max(maxima, key=lambda i: curve[i])
            return max(indices, key=lambda i: curve[i]) if indices else None

        chest_signals = (
            "front_smooth",
            "depth_smooth",
            "perimeter_smooth",
            "area_smooth",
        )
        chest_curve = Trunk._normalized_stack_curve(rows, chest_signals)
        stomach_curve = Trunk._normalized_stack_curve(rows, (
            "front_smooth",
            "perimeter_smooth",
            "area_smooth",
        ))
        hip_curve = Trunk._normalized_stack_curve(rows, (
            "back_smooth",
            "depth_smooth",
            "perimeter_smooth",
            "area_smooth",
        ))

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

        axilla_i = Trunk._best_extremum(
            rows,
            extrema_in(axilla_band, "area_smooth", "min"),
            "area_smooth",
            "min",
            prefer_high=True,
        )
        if axilla_i is None and axilla_band:
            axilla_i = min(axilla_band, key=lambda i: rows[i]["area_smooth"])

        if hip_i is None:
            return {}

        chest_i = Trunk._first_stable_max_from_top(rows, chest_band, chest_curve)
        if chest_i is None and chest_band:
            chest_i = max(chest_band, key=lambda i: chest_curve[i])
        chest_i = Trunk._upper_rise_level(rows, axilla_i, chest_i, chest_curve)
        chest_source_rows = rows

        chest_band_top_z = max((rows[i]["z"] for i in chest_band), default=-np.inf)
        chest_band_truncated = chest_band_top_z < chest_high_z - 2 * step
        chest_on_band_top = bool(
            chest_band
            and chest_i is not None
            and chest_band_truncated
            and rows[chest_i]["z"] >= chest_band_top_z - 2 * step
        )
        if chest_i is None or rows[chest_i]["z"] < chest_low_z or chest_on_band_top:
            torso_width_rows = [row for row in rows if row["z"] <= chest_low_z and row["area"] > 0]
            if torso_width_rows:
                torso_width = max(torso_width_rows, key=lambda row: row["z"])["width"]
                chest_x_limit = 0.58 * torso_width
            else:
                chest_x_limit = None
            armpit_x_limit = 0.95 * max(abs(left_armpit[0]), abs(right_armpit[0]))
            chest_x_limit = armpit_x_limit if chest_x_limit is None else max(chest_x_limit, armpit_x_limit)
            full_chest_rows = Trunk._section_stack(mesh, chest_low_z, chest_high_z, samples=64, x_limit=chest_x_limit)
            full_chest_i = None
            if len(full_chest_rows) >= 5:
                full_chest_curve = Trunk._normalized_stack_curve(full_chest_rows, chest_signals)
                full_chest_i = Trunk._first_stable_max_from_top(
                    full_chest_rows,
                    list(range(len(full_chest_rows))),
                    full_chest_curve,
                    min_prominence=0.05,
                )
                if full_chest_i is None:
                    full_chest_i = int(np.argmax(full_chest_curve))
            if full_chest_i is not None:
                print("chest source: full-mesh shoulder window")
                chest_source_rows = full_chest_rows
                chest_i = full_chest_i

        if chest_i is not None:
            chest_z = chest_source_rows[chest_i]["z"]
            stomach_band = [i for i in stomach_band if rows[hip_i]["z"] < rows[i]["z"] < chest_z]
            natural_band = [i for i in natural_band if rows[hip_i]["z"] < rows[i]["z"] < chest_z]

        stomach_i = best_curve_max(stomach_curve, stomach_band)

        natural_i = Trunk._best_extremum(
            rows,
            extrema_in(natural_band, "area_smooth", "min"),
            "area_smooth",
            "min",
        )
        if natural_i is None and natural_band:
            natural_i = min(natural_band, key=lambda i: (
                rows[i]["area_smooth"],
                rows[i]["perimeter_smooth"],
                rows[i]["depth_smooth"],
            ))

        selections = {
            "chest_axilla_level": axilla_i,
            "hip_full_level": hip_i,
            "stomach_waist_level": stomach_i,
            "natural_waist_level": natural_i,
        }

        levels = {}
        if chest_i is not None:
            row = chest_source_rows[chest_i]
            print(
                f"chest_full_level: z={row['z']:.4f}, area={row['area']:.4f}, "
                f"perimeter={row['perimeter']:.4f}, front={row['front']:.4f}, back={row['back']:.4f}"
            )
            levels["chest_full_level"] = (row["z"], row["polygon"], row["vertices_3d"])

        for name, index in selections.items():
            if index is None:
                continue
            row = rows[index]
            print(
                f"{name}: z={row['z']:.4f}, area={row['area']:.4f}, "
                f"perimeter={row['perimeter']:.4f}, front={row['front']:.4f}, back={row['back']:.4f}"
            )
            levels[name] = (row["z"], row["polygon"], row["vertices_3d"])

        return levels

    @staticmethod
    def _closed_path(vertices_3d: np.ndarray) -> trimesh.path.Path3D:
        indices = np.arange(len(vertices_3d) + 1)
        indices[-1] = 0
        return trimesh.path.Path3D(
            entities=[trimesh.path.entities.Line(indices)],
            vertices=vertices_3d,
        )

    @staticmethod
    def _polygon_measurement(polygon, vertices_3d: np.ndarray) -> tuple[float, trimesh.path.Path3D]:
        return float(polygon.exterior.length), Trunk._closed_path(vertices_3d)

    @staticmethod
    def _loop_measurement(points: np.ndarray, z: float) -> tuple[float, trimesh.path.Path3D]:
        points = np.asarray(points)
        try:
            hull = ConvexHull(points[:, :2])
            points = points[hull.vertices]
            circumference = hull.area
        except Exception:
            points = Trunk._ordered_loop(points)
            loop_2d = points[:, :2]
            circumference = np.linalg.norm(np.diff(np.vstack([loop_2d, loop_2d[0]]), axis=0), axis=1).sum()

        loop_2d = points[:, :2]
        vertices_3d = np.column_stack([loop_2d, np.full(len(loop_2d), z)])
        return float(circumference), Trunk._closed_path(vertices_3d)

    @staticmethod
    def _level_measurement(mesh: trimesh.Trimesh, name: str):
        levels = Trunk._trunk_girth_levels(mesh)
        if name not in levels:
            return None
        _, polygon, vertices_3d = levels[name]
        return Trunk._polygon_measurement(polygon, vertices_3d)

    @staticmethod
    def _slice_measurement(mesh: trimesh.Trimesh, z: float, label: str):
        polygon = Trunk._central_section_polygon(mesh, z)
        if polygon is not None:
            return Trunk._polygon_measurement(*polygon)

        loop = Trunk._central_section(mesh, z)
        if loop is not None:
            return Trunk._loop_measurement(loop, z)

        print(f"Warning: No section found at {label} level")
        return (0.0, trimesh.load_path(np.array([[0, 0, 0]])))

    @staticmethod
    def _hip_section(mesh: trimesh.Trimesh, crotch_point: np.ndarray | None = None) -> tuple[float, np.ndarray] | None:
        crotch_point = Trunk._locate_crotch(mesh) if crotch_point is None else crotch_point
        vertices = mesh.vertices
        geometry_config = get_geometry_config(mesh)
        step = max(geometry_config["median_edge_length"], geometry_config["convexity_slice_width"])
        z_values = np.linspace(crotch_point[2] + step, vertices[:, 2].max() - step, 80)

        sections = []
        for z in z_values:
            points = Trunk._central_section(mesh, z)
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

    @staticmethod
    @cache
    def _locate_armpits(mesh: trimesh.Trimesh) -> tuple:
        """Locate both armpits. Returns tuple of (left_armpit, right_armpit) as np.ndarray."""
        print("Called locate_armpits (Trunk)")

        new_mesh = mesh.copy()
        kdtree = cKDTree(new_mesh.vertices)

        # 1) Locate hips
        left_hip, right_hip = Trunk._locate_hips(mesh)

        # 2-4) Trace from each hip to armpit
        left_armpit_point = Trunk._trace_hip_to_armpit(new_mesh, kdtree, left_hip, side='left')
        right_armpit_point = Trunk._trace_hip_to_armpit(new_mesh, kdtree, right_hip, side='right')

        if not Trunk._is_plausible_armpit(mesh, left_hip, left_armpit_point):
            left_armpit_point = Trunk._profile_armpit(new_mesh, kdtree, left_hip)
        if not Trunk._is_plausible_armpit(mesh, right_hip, right_armpit_point):
            right_armpit_point = Trunk._profile_armpit(new_mesh, kdtree, right_hip)

        print(f"Armpits: left={left_armpit_point}, right={right_armpit_point}")

        return (left_armpit_point, right_armpit_point)

    @staticmethod
    def _is_plausible_armpit(mesh: trimesh.Trimesh, hip: np.ndarray, armpit: np.ndarray) -> bool:
        height = np.ptp(mesh.vertices[:, 2])
        side = "left" if hip[0] < 0 else "right"
        outward = -1 if side == "left" else 1
        side_vertices = mesh.vertices[(mesh.vertices[:, 0] - hip[0]) * outward > 0]
        side_top = np.percentile(side_vertices[:, 2], 98) if len(side_vertices) else mesh.vertices[:, 2].max()
        relative_height = (armpit[2] - hip[2]) / max(side_top - hip[2], 1e-9)
        return armpit[2] > hip[2] + 0.15 * height and relative_height < 0.9

    @staticmethod
    def _profile_armpit(mesh: trimesh.Trimesh, kdtree: cKDTree, hip: np.ndarray) -> np.ndarray:
        geometry_config = get_geometry_config(mesh)
        vertices = mesh.vertices
        side = "left" if hip[0] < 0 else "right"
        outward = -1 if side == "left" else 1
        step = max(geometry_config["median_edge_length"], geometry_config["convexity_slice_width"])
        side_vertices = vertices[(vertices[:, 0] - hip[0]) * outward > 0]

        if len(side_vertices):
            z_top = np.percentile(side_vertices[:, 2], 98)
            z_values = np.linspace(hip[2] + step, z_top, 60)
            had_side_loop = False
            previous_edge = None
            transitions = []

            for z in z_values:
                loops = Trunk._section_loops(mesh, z)
                if len(loops) < 1:
                    continue

                central_index = Trunk._central_loop_index(loops)
                central = loops[central_index]

                edge = central[np.argmin(central[:, 0]) if side == "left" else np.argmax(central[:, 0])]
                side_loop = False
                for i, loop in enumerate(loops):
                    if i == central_index:
                        continue
                    if np.mean((loop[:, 0] - edge[0]) * outward > 0) > 0.8:
                        side_loop = True
                        break

                if had_side_loop and not side_loop and previous_edge is not None:
                    candidate = central[np.argmin(np.linalg.norm(central[:, :2] - previous_edge[:2], axis=1))]
                    transitions.append(mesh.vertices[kdtree.query(candidate)[1]])

                had_side_loop = side_loop
                previous_edge = edge

            if transitions:
                armpit = max(transitions, key=lambda point: point[2])
                hip_to_top = z_top - hip[2]
                if armpit[2] > hip[2] + 0.65 * hip_to_top:
                    print(f"Armpit profile {side}: section-merge z={armpit[2]:.4f}")
                    return armpit
                merge_candidate = armpit
                merge_relative_height = (armpit[2] - hip[2]) / max(hip_to_top, 1e-9)
            else:
                merge_candidate = None
                merge_relative_height = 0.0
        else:
            z_top = vertices[:, 2].max()
            merge_candidate = None
            merge_relative_height = 0.0

        section_gap = Trunk._section_gap_armpit(mesh, kdtree, hip)
        if (
            merge_candidate is not None
            and merge_relative_height > 0.45
            and (section_gap is None or section_gap[2] > merge_candidate[2] + 0.08 * np.ptp(vertices[:, 2]))
        ):
            print(f"Armpit profile {side}: section-merge z={merge_candidate[2]:.4f}")
            return merge_candidate

        if merge_candidate is not None:
            print(f"Armpit profile {side}: rejected low section-merge z={merge_candidate[2]:.4f}")

        if section_gap is not None:
            print(f"Armpit profile {side}: section-gap z={section_gap[2]:.4f}")
            return section_gap

        candidate = Trunk._side_gap_armpit(mesh, kdtree, hip)
        if candidate is not None:
            print(f"Armpit profile {side}: side-gap z={candidate[2]:.4f}")
            return candidate

        return hip

    @staticmethod
    def _section_gap_armpit(mesh: trimesh.Trimesh, kdtree: cKDTree, hip: np.ndarray) -> np.ndarray | None:
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
            loops = Trunk._section_loops(mesh, z)
            if len(loops) < 2:
                continue

            central_index = Trunk._central_loop_index(loops)
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
                gap = abs(inner_x - central_edge[0])
                side_loop_rows.append((gap, loop))

            if not side_loop_rows:
                continue

            gap, _ = min(side_loop_rows, key=lambda row: row[0])
            rows.append((gap, z, central_edge))

        if not rows:
            return None

        gaps = np.array([row[0] for row in rows])
        min_gap = gaps.min()
        tolerance = max(0.25 * np.median(gaps), geometry_config["armpit_lateral_band"])
        close_rows = [row for row in rows if row[0] <= min_gap + tolerance]
        _, _, candidate = min(close_rows, key=lambda row: row[1])
        return mesh.vertices[kdtree.query(candidate)[1]]

    @staticmethod
    def _side_gap_armpit(mesh: trimesh.Trimesh, kdtree: cKDTree, hip: np.ndarray) -> np.ndarray | None:
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
        gaps = np.diff(band[:, 2])
        candidate = band[np.argmax(gaps)]
        if candidate[2] < z_lower:
            return None
        return mesh.vertices[kdtree.query(candidate)[1]]
    
    @staticmethod
    @cache
    def _trace_hip_to_armpit(mesh: trimesh.Trimesh, kdtree: cKDTree, hip_point: np.ndarray, side: str) -> np.ndarray:
        """
        Trace from hip point upward to find armpit.
        
        Algorithm:
        1. Project to xz plane (frontal view)
        2. Trace line up (increasing z) along lateral body
        3. Stop when z starts decreasing (arm bends inward)
        4. Use convexity search to refine armpit location
        
        Args:
            mesh: The body mesh
            kdtree: KDTree of mesh vertices for nearest neighbor queries
            hip_point: Starting point (hip location)
            side: 'left' or 'right'
        
        Returns:
            Armpit vertex as np.ndarray
        """
        # Define lateral band width (region along body side)
        geometry_config = get_geometry_config(mesh)
        lateral_band_width = geometry_config["armpit_lateral_band"]
        
        # Create band centered on hip x-coordinate
        lateral_min_x = hip_point[0] - lateral_band_width / 2
        lateral_max_x = hip_point[0] + lateral_band_width / 2
        
        # Filter vertices in lateral band above hip
        lateral_vertices = mesh.vertices[
            (mesh.vertices[:, 0] >= lateral_min_x) &
            (mesh.vertices[:, 0] <= lateral_max_x) &
            (mesh.vertices[:, 2] >= hip_point[2])
        ]
        
        if len(lateral_vertices) == 0:
            print(f"Warning: No lateral vertices found for {side} side, using hip point")
            return hip_point
        
        # Sort by z-coordinate (ascending) to trace upward
        sorted_indices = np.argsort(lateral_vertices[:, 2])
        sorted_lateral = lateral_vertices[sorted_indices]
        
        # Trace upward until z stops increasing (armpit region)
        armpit_candidate = hip_point.copy()
        prev_z = hip_point[2]
        
        for i in range(1, len(sorted_lateral)):
            current_point = sorted_lateral[i]
            current_z = current_point[2]
            
            # Check if z increase has become very small or negative
            z_increase = current_z - prev_z
            
            # Armpit is where vertical progress stops (arm bends away from torso)
            if z_increase < geometry_config["armpit_z_stop"]:
                armpit_candidate = current_point
                break
            
            prev_z = current_z
            armpit_candidate = current_point
        
        # Refine using convexity search (armpit is concave)
        try:
            armpit_refined = convexity_search(
                mesh,
                rays=32,
                origin=armpit_candidate,
                slice_width=geometry_config["convexity_slice_width"],
            )
            
            # Get actual mesh vertex using KDTree
            armpit_idx = kdtree.query(armpit_refined)[1]
            armpit_vertex = mesh.vertices[armpit_idx]
            
            return armpit_vertex
        except Exception as e:
            print(f"Warning: Convexity search failed for {side} armpit: {e}")
            # Fallback to candidate point
            armpit_idx = kdtree.query(armpit_candidate)[1]
            return mesh.vertices[armpit_idx]

    @staticmethod
    @cache
    def _locate_hips(mesh: trimesh.Trimesh):
        print("Called locate_hips (Trunk)")

        crotch_point = Trunk._locate_crotch(mesh)

        hip_section = Trunk._hip_section(mesh, crotch_point)
        if hip_section is None:
            body_height = np.ptp(mesh.vertices[:, 2])
            slice_z_min = crotch_point[2] + body_height * 0.05
            slice_z_max = crotch_point[2] + body_height * 0.1
            in_range_mask = (mesh.vertices[:, 2] >= slice_z_min) & (mesh.vertices[:, 2] <= slice_z_max)
            torso_vertices = mesh.vertices[in_range_mask]
        else:
            _, torso_vertices = hip_section
        
        # Find farthest left and right points on the torso
        left_hip = torso_vertices[np.argmin(torso_vertices[:, 0])]
        right_hip = torso_vertices[np.argmax(torso_vertices[:, 0])]
        
        # Snap to actual mesh vertices
        kdtree = cKDTree(mesh.vertices)
        left_hip = mesh.vertices[kdtree.query(left_hip)[1]]
        right_hip = mesh.vertices[kdtree.query(right_hip)[1]]
        
        return (left_hip, right_hip)

    @staticmethod
    @cache
    def _locate_collar(mesh: trimesh.Trimesh):
        from ..arms.arm import Arm
        
        trunk_mesh = Trunk._get_submesh(mesh)  # FIX: Call static method
        trunk_vertices = trunk_mesh.vertices   # No need for np.array()
        lshoulder = Arm._locate_shoulder(mesh, "left")
        rshoulder = Arm._locate_shoulder(mesh, "right")
        
        midpoint = (rshoulder + lshoulder) / 2
        
        # Get the centroid of the body mesh
        centroid = mesh.centroid
        
        # Slice the mesh at the centroid with a plane normal to Y-axis
        front_mesh = trunk_mesh.slice_plane(
            plane_origin=centroid,
            plane_normal=np.array([0, 1, 0])
        )
        
        # Use front vertices if available, otherwise fall back to all vertices
        if front_mesh is not None and len(front_mesh.vertices) > 0:
            search_vertices = front_mesh.vertices
        else:
            search_vertices = trunk_vertices
        
        kdtree = cKDTree(search_vertices)
        _, idx = kdtree.query(midpoint)
        snapped = search_vertices[idx].copy()
    
        total_height = trunk_vertices[:, 2].max() - trunk_vertices[:, 2].min()
        snapped[2] += 0.01 * total_height
    
        return snapped

    # Measurements & Drawings

    @property
    def measurements(self):
        """Extract just the measurement values (first element of tuples)."""
        return {
            "crotch height": Trunk._measure_crotch_height(self.body_mesh)[0],
            "hip circumference": Trunk._measure_hip_circumference(self.body_mesh)[0],
            "chest circumference": Trunk._measure_chest_circumference(self.body_mesh)[0],
            "waist circumference": Trunk._measure_waist_circumference(self.body_mesh)[0],
            "stomach peak circumference": Trunk._measure_stomach_peak_circumference(self.body_mesh)[0],
            "trunk length": Trunk._measure_trunk_length(self.body_mesh)[0]
        }

    @property
    def drawings(self):
        """Extract the 3D paths showing where measurements were taken (second element of tuples)."""
        return {
            "crotch height": Trunk._measure_crotch_height(self.body_mesh)[1],
            "hip circumference": Trunk._measure_hip_circumference(self.body_mesh)[1],
            "chest circumference": Trunk._measure_chest_circumference(self.body_mesh)[1],
            "waist circumference": Trunk._measure_waist_circumference(self.body_mesh)[1],
            "stomach peak circumference": Trunk._measure_stomach_peak_circumference(self.body_mesh)[1],
            "trunk length": Trunk._measure_trunk_length(self.body_mesh)[1]
        }

    @staticmethod
    @cache
    def _measure_crotch_height(mesh: trimesh.Trimesh):
        """
        Measure crotch height from ground to crotch point.
        
        Returns
        -------
        tuple[float, trimesh.path.Path3D]
            (height_value, vertical_line_path_in_original_coordinates)
        """
        print("Called measure_crotch_height (Trunk)")
        
        # Get crotch point
        crotch_point = Trunk._locate_crotch(mesh)
        
        # Get z coordinate of crotch
        crotch_z = crotch_point[2]
        
        # Get minimum z coordinate (ground level)
        min_z = np.min(mesh.vertices[:, 2])
        
        # Calculate crotch height
        crotch_height = crotch_z - min_z
        
        # Create vertical line from ground to crotch (at crotch x,y position)
        ground_point = np.array([crotch_point[0], crotch_point[1], min_z])
        vertices = np.array([ground_point, crotch_point])
        entities = [trimesh.path.entities.Line([0, 1])]
        path_3d = trimesh.path.Path3D(entities=entities, vertices=vertices)
        
        return (float(crotch_height), path_3d)

    @staticmethod
    @cache
    def _measure_hip_circumference(mesh: trimesh.Trimesh):
        """
        Measure hip circumference at the posterior/glute-pelvis fullness level.

        This is profile-aware: a large abdomen should not win hip placement just because it
        has the largest total area or perimeter.
        
        Returns
        -------
        tuple[float, trimesh.path.Path3D]
            (circumference_value, path_in_original_coordinates)
        """
        print("Called measure_hip_circumference (Trunk)")

        measurement = Trunk._level_measurement(mesh, "hip_full_level")
        if measurement is not None:
            return measurement

        body_without_arms = Trunk._body_without_arms(mesh)
        hip_section = Trunk._hip_section(body_without_arms)
        if hip_section is not None:
            hip_z, _ = hip_section
            return Trunk._slice_measurement(body_without_arms, hip_z, "hip")

        print("Warning: No section found in hip region")
        empty_path = trimesh.load_path(np.array([[0, 0, 0]]))
        return (0.0, empty_path)

    @staticmethod
    @cache
    def _measure_chest_circumference(mesh: trimesh.Trimesh):
        """
        Measure chest/bust circumference at upper-torso fullness.

        Chest is not the armpit marker. It is the first stable upper chest/full-bust level
        where a horizontal tape would wrap one central torso section.
        
        Returns
        -------
        tuple[float, trimesh.path.Path3D]
            (circumference_value, path_in_original_coordinates)
        """
        print("Called measure_chest_circumference (Trunk)")

        measurement = Trunk._level_measurement(mesh, "chest_full_level")
        if measurement is not None:
            return measurement

        torso_mesh = Trunk._get_submesh(mesh)
        left_armpit, right_armpit = Trunk._locate_armpits(mesh)
        chest_z = np.median([left_armpit[2], right_armpit[2]])
        print(f"Chest level fallback z={chest_z:.4f}")
        return Trunk._slice_measurement(torso_mesh, chest_z, "chest")

    @staticmethod
    @cache
    def _measure_waist_circumference(mesh: trimesh.Trimesh):
        """
        Measure canonical/natural waist circumference.

        This is the narrowing between chest and hip/stomach, not the fullest belly. Use
        _measure_stomach_peak_circumference for the abdomen peak.
        
        Returns
        -------
        tuple[float, trimesh.path.Path3D]
            (circumference_value, path_in_original_coordinates)
        """
        print("Called measure_waist_circumference (Trunk)")

        measurement = Trunk._level_measurement(mesh, "natural_waist_level")
        if measurement is not None:
            return measurement

        torso_mesh = Trunk._get_submesh(mesh)
        left_armpit, right_armpit = Trunk._locate_armpits(mesh)
        left_hip, right_hip = Trunk._locate_hips(mesh)
        waist_z = np.mean([np.median([left_armpit[2], right_armpit[2]]), np.median([left_hip[2], right_hip[2]])])
        return Trunk._slice_measurement(torso_mesh, waist_z, "waist")

    @staticmethod
    @cache
    def _measure_stomach_peak_circumference(mesh: trimesh.Trimesh):
        """
        Measure the fullest anterior abdomen/stomach section between hips and chest.

        Returns
        -------
        tuple[float, trimesh.path.Path3D]
            (circumference_value, path_in original coordinates)
        """
        print("Called measure_stomach_peak_circumference (Trunk)")

        return Trunk._level_measurement(mesh, "stomach_waist_level") or Trunk._measure_waist_circumference(mesh)

    @staticmethod
    @cache
    def _measure_trunk_length(mesh: trimesh.Trimesh):
        """
        Calculate trunk length as the Euclidean distance between the 
        crotch and collar landmarks, projected onto the (x, z) plane.

        Returns
        -------
        tuple[float, trimesh.path.Path3D]
            (length_value, line_segment_path_in_original_coordinates)
        """
        print("Called _measure_trunk_length (Trunk)")

        # Step 1. Get crotch and collar coordinates
        crotch = Trunk._locate_crotch(mesh)
        collar = Trunk._locate_collar(mesh)

        # Step 2. Validate both are numpy arrays
        if not isinstance(crotch, np.ndarray) or not isinstance(collar, np.ndarray):
            raise TypeError("Crotch or collar point not found or invalid (expected np.ndarray).")

        if crotch.shape != (3,) or collar.shape != (3,):
            raise ValueError(f"Unexpected point shape. Got crotch={crotch.shape}, collar={collar.shape}")

        # Step 3. Compute the differences in x and z (ignore y)
        dx = crotch[0] - collar[0]
        dz = crotch[2] - collar[2]

        # Step 4. Compute Euclidean distance in x–z plane
        trunk_length = np.sqrt(dx**2 + dz**2)

        # Step 5. Debug print for verification
        print(f"Crotch point: {crotch}")
        print(f"Collar point: {collar}")
        print(f"Computed trunk length (||crotch - collar||_(x,z)) = {trunk_length:.3f}")

        # Step 6. Create Path3D line segment from collar to crotch
        vertices = np.array([collar, crotch])
        entities = [trimesh.path.entities.Line([0, 1])]
        path_3d = trimesh.path.Path3D(entities=entities, vertices=vertices)

        # Step 7. Ensure return type is float (not np.float64)
        return (float(trunk_length), path_3d)
