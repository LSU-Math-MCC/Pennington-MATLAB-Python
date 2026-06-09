from functools import cache
import numpy as np
import trimesh
from ..anatomical_region import Anatomical_Region, LEFT_OR_RIGHT, get_geometry_config
from ....utils.section_geometry import empty_measurement, line_path


def _loop_length(points: np.ndarray) -> float:
    points = np.asarray(points)
    if len(points) < 2:
        return 0.0
    loop = points[:-1] if np.allclose(points[0], points[-1]) else points
    return float(np.linalg.norm(np.diff(np.vstack([loop, loop[0]]), axis=0), axis=1).sum())


def _arm_slice_loop(path_2d) -> tuple[np.ndarray, float] | None:
    loops = [np.asarray(loop) for loop in getattr(path_2d, "discrete", []) if len(loop) >= 3]
    if not loops:
        from scipy.spatial import ConvexHull

        vertices = np.asarray(getattr(path_2d, "vertices", []))[:, :2]
        if len(vertices) >= 3:
            try:
                loops.append(vertices[ConvexHull(vertices).vertices])
            except Exception:
                loops.append(vertices)
    if not loops:
        return None
    measured = [(loop[:-1] if np.allclose(loop[0], loop[-1]) else loop, _loop_length(loop)) for loop in loops]
    lengths = np.array([length for _, length in measured])
    threshold = max(1e-9, 0.15 * lengths.max())
    candidates = [(loop, length) for loop, length in measured if length >= threshold]
    return min(candidates or measured, key=lambda row: row[1])


def _arm_loop_path(loop_2d: np.ndarray, z: float, inverse_transform: np.ndarray) -> trimesh.path.Path3D:
    loop_2d = np.asarray(loop_2d)[:, :2]
    vertices_3d_aligned = np.column_stack([loop_2d[:, 0], loop_2d[:, 1], np.full(len(loop_2d), z)])
    vertices_3d_original = trimesh.transform_points(vertices_3d_aligned, inverse_transform)
    indices = np.arange(len(vertices_3d_original) + 1)
    indices[-1] = 0
    return trimesh.path.Path3D(
        entities=[trimesh.path.entities.Line(indices)],
        vertices=vertices_3d_original,
    )


def _aligned_arm_mesh(mesh: trimesh.Trimesh, side: LEFT_OR_RIGHT) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Return a copy of the arm aligned to z plus the inverse transform back to body space."""
    from ....mesh.mesh import Mesh

    arm_mesh_copy = Arm._get_submesh(side, mesh).copy()
    transform_matrix = Mesh.align_mesh_to_z_axis(arm_mesh_copy)
    return arm_mesh_copy, np.linalg.inv(transform_matrix)


def _arm_section_measurement(
    arm_mesh: trimesh.Trimesh,
    z: float,
    inverse_transform: np.ndarray,
) -> tuple[float, trimesh.path.Path3D]:
    """Measure one aligned horizontal arm section and map its loop back to body space."""
    slice_2d = arm_mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if slice_2d is None:
        return empty_measurement()
    loop_data = _arm_slice_loop(slice_2d)
    if loop_data is None:
        return empty_measurement()
    loop_2d, perimeter = loop_data
    return float(perimeter), _arm_loop_path(loop_2d, z, inverse_transform)


def _largest_fraction_loop(arm_mesh: trimesh.Trimesh, fractions: np.ndarray) -> tuple[float, float, np.ndarray] | None:
    """Find the largest clean loop across fractional arm heights."""
    z_min = arm_mesh.vertices[:, 2].min()
    arm_height = arm_mesh.vertices[:, 2].max() - z_min
    best = None
    for fraction in fractions:
        z = z_min + arm_height * fraction
        slice_2d = arm_mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if slice_2d is None:
            continue
        loop_data = _arm_slice_loop(slice_2d)
        if loop_data is None:
            continue
        loop_2d, perimeter = loop_data
        if best is None or perimeter > best[1]:
            best = (z, perimeter, loop_2d)
    return best


def _stable_wrist_loop(arm_mesh: trimesh.Trimesh, slice_step: float) -> tuple[np.ndarray, float, float] | None:
    """Pick the smallest stable lower-arm section while ignoring tiny finger/open-boundary slivers."""
    z_min = arm_mesh.vertices[:, 2].min()
    z_max = arm_mesh.vertices[:, 2].max()
    arm_height = z_max - z_min
    z_heights = np.arange(z_min + arm_height * 0.12, z_min + arm_height * 0.36, slice_step)
    if len(z_heights) == 0:
        z_heights = np.array([z_min + arm_height * 0.24])
    sections = []
    for z in z_heights:
        slice_2d = arm_mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if slice_2d is None:
            continue
        loop_data = _arm_slice_loop(slice_2d)
        if loop_data is None:
            continue
        loop_2d, perimeter = loop_data
        sections.append((loop_2d, float(perimeter), z))
    if not sections:
        return None
    perimeters = np.array([perimeter for _, perimeter, _ in sections])
    floor = max(0.55 * np.median(perimeters), 0.35 * np.percentile(perimeters, 80))
    stable = [section for section in sections if section[1] >= floor]
    return min(stable or sections, key=lambda section: section[1])


class Arm(Anatomical_Region):
    """
    Arm region segmentation, landmark detection, and measurement computation.
    
    The Arm class handles extraction and analysis of left or right arm from a full body mesh.
    It identifies the arm portion by slicing at the armpit, locates key anatomical landmarks
    (shoulder, wrist), and computes measurements (arm length, circumferences at different points).
    
    Purpose
    -------
    This class solves the problem of automatically measuring arm dimensions from body scans.
    Without this class, you would need to manually:
    - Identify where the arm separates from the trunk (armpit location)
    - Slice the mesh to extract just the arm
    - Find anatomical landmarks (shoulder, elbow, wrist)
    - Compute lengths and circumferences
    
    With this class, creating an Arm instance automatically performs all these operations.
    
    Class Structure and Design Choices
    -----------------------------------
    **Static methods with @cache**:
    See Anatomical_Region docstring for the rationale behind this design pattern.
    
    **Runtime imports**:
    Imports like `from ..trunk import Trunk` appear inside methods instead of at the top
    because of circular dependencies:
    - Arm needs Trunk (to locate armpits)
    - Trunk needs Arm (to remove arms from trunk mesh)
    
    Importing at runtime (inside methods) breaks the circular dependency at import time.
    The actual dependency is fine because these methods run after both classes are fully defined.
    
    **Mesh slicing approach**:
    The arm is extracted by:
    1. Finding the armpit point (using Trunk._locate_armpits)
    2. Slicing the body mesh with a plane at the armpit
    3. Keeping only the portion on the arm side AND picking the connected piece
       with the highest vertex to ensure none of the leg or body is included with the arm
    
    This approach assumes:
    - Arms are angled down and away (A-pose)
    - Mesh is in standard orientation (Z-axis vertical from feet to head, X-axis left-right, Y-axis front-back)
    - Armpit detection succeeds (requires clean mesh)
    
    **Why separate left/right parameter**:
    Rather than having LeftArm and RightArm classes, we use one Arm class with a parameter
    because:
    - Algorithms are identical for both sides
    - Reduces code duplication
    - Easier to maintain (fix bug once, applies to both)
    - Can iterate over both arms: `for side in ['left', 'right']: Arm(mesh, side)`
    
    Attributes
    ----------
    body_mesh : trimesh.Trimesh
        The full body mesh (cleaned and oriented)
    side : LEFT_OR_RIGHT ('left' or 'right')
        Which arm this instance represents
    
    Properties
    ----------
    mesh : trimesh.Trimesh
        The segmented arm mesh (cached, computed on first access)
    landmarks : dict[str, np.ndarray]
        Key anatomical points:
        - "shoulder": Glenohumeral joint approximation
        - "wrist": Distal end of arm
        - "highest point of arm": Top of shoulder region
    measurements : dict[str, float]
        Anthropometric measurements:
        - "arm length": Shoulder to wrist distance
        - "wrist girth": Circumference at wrist
        - "forearm girth": Circumference at mid-forearm
        - "bicep girth": Circumference at mid-upper-arm
    
    Examples
    --------
    >>> import trimesh
    >>> from body import Body  # doctest: +SKIP
    >>> body = Body("model_files/man.obj")  # doctest: +SKIP
    >>> left_arm = body.parts["left arm"]  # doctest: +SKIP
    >>> arm_length = left_arm.measurements["left arm length"]  # doctest: +SKIP
    >>> print(f"Arm length: {arm_length:.2f} cm")  # doctest: +SKIP
    Arm length: 62.4 cm
    
    >>> shoulder = left_arm.landmarks["shoulder"]  # doctest: +SKIP
    >>> wrist = left_arm.landmarks["wrist"]  # doctest: +SKIP
    >>> print(f"Shoulder: {shoulder}, Wrist: {wrist}")  # doctest: +SKIP
    Shoulder: [-15.2, 0.0, 52.1], Wrist: [-15.5, 0.2, -10.3]
    
    Notes
    -----
    - Requires body mesh to be in standard orientation (Z-axis vertical from feet to head, X-axis left-right, Y-axis front-back)
    - Assumes body is in A-pose (standard anthropometric pose)
    - Armpit detection may fail on poor quality meshes
    - All measurements are in the same units as the input mesh
    
    See Also
    --------
    Trunk : Provides armpit landmark used for arm segmentation
    Anatomical_Region : Abstract base class defining the interface
    """

    def __init__(self, body_mesh: trimesh.Trimesh, left_or_right: LEFT_OR_RIGHT):
        print("Called __init__ (Arm)")

        self.side = left_or_right
        self.body_mesh = body_mesh

    # Properties of Arm

    @property
    def volume(self):
        print("Called volume (Arm)")
        return self._trimesh.volume
    
    @property
    def surface_area(self):
        print("Called surface_area (Arm)")
        return self._trimesh.area
    
    # Vertex Indices (provided by `mesh`)
    @property
    def mesh(self):
        return Arm._get_submesh(self.side, self.body_mesh)

    @staticmethod
    @cache
    def _get_submesh(side: LEFT_OR_RIGHT, mesh: trimesh.Trimesh):
        """Get vertices for left or right arm using mesh splitting approach"""
        
        # 1. Get armpit for this arm
        from ..trunk import Trunk
        left_armpit, right_armpit = Trunk._locate_armpits(mesh)
        armpit = left_armpit if side == 'left' else right_armpit
        
        
        # 2. Slice mesh by plane at armpit
        plane_normal = np.array([1, 0, 0])  # X-axis normal (YZ plane)
        if side == 'left':
            plane_normal = -plane_normal  # Flip normal for right side
            
        # Slice and get the correct side
        sliced_mesh = mesh.slice_plane(
            plane_origin=armpit,
            plane_normal=plane_normal
        )
        
        # 3. Split the sliced mesh into disconnected parts
        # if sliced_mesh is None:
        #     print(f"Error: No mesh to split for {side} side!")
        #     return np.array([])
            
        # Clean mesh before splitting
        sliced_mesh.remove_unreferenced_vertices()
        sliced_mesh.fill_holes()
        
        parts = sliced_mesh.split(only_watertight=False)
        print(f"Split mesh into {len(parts)} parts")
                
        # if not parts:
        #     print(f"Error: No parts found after splitting {side} side!")
        #     return np.array([])
        
        # 4. Find the real arm, not tiny cap fragments or a sliced torso side.
        height = np.ptp(mesh.vertices[:, 2])
        crotch = Trunk._locate_crotch(mesh)
        min_vertices = max(20, int(0.01 * len(mesh.vertices)))
        outward = -1 if side == "left" else 1
        candidates = []
        fallback_candidates = []

        def geodesic_arm_candidate():
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import dijkstra

            vertices = sliced_mesh.vertices
            edges = sliced_mesh.edges_unique
            if len(vertices) == 0 or len(edges) == 0:
                return None
            edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
            graph = coo_matrix(
                (
                    np.r_[edge_lengths, edge_lengths],
                    (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
                ),
                shape=(len(vertices), len(vertices)),
            ).tocsr()
            seed = int(np.argmin(np.linalg.norm(vertices - armpit, axis=1)))
            distances = dijkstra(graph, indices=seed, limit=0.35 * height)
            vertex_mask = np.isfinite(distances)
            eligible = np.flatnonzero(vertex_mask & (vertices[:, 2] >= armpit[2] - 0.42 * height))
            if len(eligible) >= 2:
                distal = eligible[np.argmax((vertices[eligible, 0] - armpit[0]) * outward)]
                shoulder = int(np.argmax(vertices[:, 2]))
                start = vertices[shoulder, [0, 2]]
                end = vertices[distal, [0, 2]]
                axis = end - start
                axis_length_sq = max(float(np.dot(axis, axis)), 1e-9)
                points = vertices[:, [0, 2]]
                t = np.clip(((points - start) @ axis) / axis_length_sq, 0.0, 1.0)
                closest = start + t[:, None] * axis
                distance_to_axis = np.linalg.norm(points - closest, axis=1)
                side_width = max(np.ptp(vertices[vertex_mask, 0]), 1e-9)
                tube_radius = max(0.26 * side_width, 0.035 * height)
                vertex_mask &= distance_to_axis <= tube_radius
            face_indices = np.flatnonzero(vertex_mask[sliced_mesh.faces].all(axis=1))
            if len(face_indices) == 0:
                return None
            candidate = sliced_mesh.submesh([face_indices], append=True, repair=False)
            candidate.remove_unreferenced_vertices()
            pieces = candidate.split(only_watertight=False)
            if len(pieces):
                candidate = max(pieces, key=lambda part: len(part.vertices))
            return candidate if len(candidate.vertices) >= min_vertices else None

        def reaches_underarm(candidate):
            return candidate is not None and candidate.vertices[:, 2].max() >= armpit[2] + 0.02 * height

        for part in parts:
            vertices = part.vertices
            if len(vertices) < min_vertices:
                continue
            z_min = vertices[:, 2].min()
            z_max = vertices[:, 2].max()
            if np.ptp(vertices[:, 2]) > 0.6 * height:
                continue
            if z_min < crotch[2] - 0.10 * height:
                continue
            fallback_candidates.append(part)
            if z_max < armpit[2] - 0.05 * height:
                continue
            if z_min > armpit[2] - 0.10 * height:
                continue
            candidates.append(part)

        if not candidates:
            geodesic_candidate = geodesic_arm_candidate()
            if reaches_underarm(geodesic_candidate):
                candidates.append(geodesic_candidate)

        parts_to_score = candidates or fallback_candidates or parts
        arm_mesh = max(
            parts_to_score,
            key=lambda part: (
                np.mean((part.vertices[:, 0] - armpit[0]) * outward > 0),
                np.ptp(part.vertices[:, 2]),
                len(part.vertices),
            )
        )

        if (
            np.ptp(arm_mesh.vertices[:, 2]) > 0.6 * height
            or arm_mesh.vertices[:, 2].min() < crotch[2] - 0.10 * height
        ):
            geodesic_candidate = geodesic_arm_candidate()
            if reaches_underarm(geodesic_candidate):
                print(f"{side}_arm_mesh: geodesic recovery replaced connected side component")
                arm_mesh = geodesic_candidate

        geodesic_candidate = geodesic_arm_candidate()
        if reaches_underarm(geodesic_candidate):
            arm_depth = np.ptp(arm_mesh.vertices[:, 1])
            geodesic_depth = np.ptp(geodesic_candidate.vertices[:, 1])
            if (
                geodesic_depth < 0.88 * arm_depth
                and np.ptp(geodesic_candidate.vertices[:, 2]) > 0.70 * np.ptp(arm_mesh.vertices[:, 2])
                and len(geodesic_candidate.vertices) > 0.35 * len(arm_mesh.vertices)
            ):
                print(f"{side}_arm_mesh: geodesic tube removed torso-side slab")
                arm_mesh = geodesic_candidate

        print(
            f"{side}_arm_mesh: vertices={len(arm_mesh.vertices)}, "
            f"z=[{arm_mesh.vertices[:, 2].min():.4f}, {arm_mesh.vertices[:, 2].max():.4f}], "
            f"armpit={armpit}"
        )
        
        return arm_mesh

    # Landmarks # TODO: Might want to make these properties so the access is simpler, but not required. This goes for all landmarks and measurements in src actually

    @property
    def landmarks(self):
        return {
            f"highest point of arm": Arm._locate_highest_point_of_arm(self.body_mesh, self.side),
            f"shoulder": Arm._locate_shoulder(self.body_mesh, self.side),
            f"wrist": Arm._locate_wrist(self.body_mesh, self.side)
        }

    @staticmethod
    @cache
    def _locate_highest_point_of_arm(mesh: trimesh.Trimesh, side: LEFT_OR_RIGHT):
        """
        Pseudo:
        Get arm mesh
        Get highest z
        return that
        """
        print("Called locate_highest_point_of_arm (Arm)")
        
        # Get arm mesh for this side
        arm_mesh = Arm._get_submesh(side, mesh)
        
        # Find vertex with highest z coordinate
        highest_idx = np.argmax(arm_mesh.vertices[:, 2])
        highest_point = arm_mesh.vertices[highest_idx]
        
        return highest_point

    @staticmethod
    @cache
    def _locate_shoulder(mesh: trimesh.Trimesh, side: str):
        """
        Shoulder = highest point on the segmented arm.
        """
        print("Called locate_shoulder (Arm)")
        
        arm_mesh = Arm._get_submesh(side, mesh)
        return arm_mesh.vertices[np.argmax(arm_mesh.vertices[:, 2])]

    @staticmethod
    @cache
    def _locate_wrist(mesh: trimesh.Trimesh, side: LEFT_OR_RIGHT):
        """
        Locate the wrist centroid using minimum perimeter detection.
        
        Algorithm:
        1. Get arm mesh and orient it with shoulder up, wrist/fingers down
        2. For each horizontal slice in the wrist search region (10-30% of arm height):
           a. Compute the 2D cross-section
           b. Calculate the perimeter
        3. The wrist is the centroid of the slice with minimum perimeter
        4. Map centroid position back to original coordinate system using nearest vertex as anchor
        
        Returns the actual centroid position (not snapped to mesh vertices).
        """
        print("Called locate_wrist (Arm)")

        arm_mesh_copy, inverse_transform = _aligned_arm_mesh(mesh, side)
        slice_step = get_geometry_config(mesh)["arm_slice_step"]
        wrist_section = _stable_wrist_loop(arm_mesh_copy, slice_step)
        if wrist_section is None:
            z_min = arm_mesh_copy.vertices[:, 2].min()
            z_max = arm_mesh_copy.vertices[:, 2].max()
            wrist_centroid_aligned = np.array([0, 0, z_min + 0.24 * (z_max - z_min)])
        else:
            loop_2d, _, z = wrist_section
            wrist_centroid_aligned = np.array([loop_2d[:, 0].mean(), loop_2d[:, 1].mean(), z])
        
        # Map the centroid position back to original coordinate system
        # Convert to homogeneous coordinates
        wrist_homogeneous = np.append(wrist_centroid_aligned, 1.0)
        
        wrist_centroid_original = (inverse_transform @ wrist_homogeneous)[:3]
        
        return wrist_centroid_original
    

    # Measurements & Drawings

    @property
    def measurements(self):
        """Extract just the measurement values (first element of tuples)."""
        return {
            "wrist girth": Arm._measure_wrist_girth(self.body_mesh, self.side)[0],
            "arm length": Arm._measure_arm_length(self.body_mesh, self.side)[0],
            "forearm girth": Arm._measure_forearm_girth(self.body_mesh, self.side)[0],
            "bicep girth": Arm._measure_bicep_girth(self.body_mesh, self.side)[0]
        }

    @property
    def drawings(self):
        """Extract the 3D paths showing where measurements were taken (second element of tuples)."""
        return {
            "wrist girth": Arm._measure_wrist_girth(self.body_mesh, self.side)[1],
            "arm length": Arm._measure_arm_length(self.body_mesh, self.side)[1],
            "forearm girth": Arm._measure_forearm_girth(self.body_mesh, self.side)[1],
            "bicep girth": Arm._measure_bicep_girth(self.body_mesh, self.side)[1]
        }

    @staticmethod
    @cache
    def _measure_wrist_girth(mesh: trimesh.Trimesh, side: LEFT_OR_RIGHT):
        """
        Measure wrist girth by finding minimum perimeter in wrist region.
        
        Algorithm:
        1. Get arm mesh and orient it with shoulder up, wrist/fingers down
        2. For each horizontal slice in the wrist search region (10-30% of arm height):
           a. Compute the 2D cross-section
           b. Calculate the perimeter
        3. Return the minimum perimeter (the wrist girth) and a 3D path showing the cross-section
        
        Returns
        -------
        tuple[float, trimesh.path.Path3D]
            (girth_value, path_in_original_coordinates)
        """
        print("Called measure_wrist_girth (Arm)")

        arm_mesh_copy, inverse_transform = _aligned_arm_mesh(mesh, side)
        slice_step = get_geometry_config(mesh)["arm_slice_step"]
        wrist_section = _stable_wrist_loop(arm_mesh_copy, slice_step)
        if wrist_section is None:
            return empty_measurement()
        best_loop, min_perimeter, best_z = wrist_section
        path_3d = _arm_loop_path(best_loop, best_z, inverse_transform)
        
        return (float(min_perimeter), path_3d)

    @staticmethod
    @cache
    def _measure_arm_length(mesh: trimesh.Trimesh, side: LEFT_OR_RIGHT):
        """
        Measure arm length as distance from shoulder/armpit midpoint to wrist.
        
        Calculates the 2D Euclidean distance (in the x-z plane) between the wrist landmark 
        and the midpoint of the shoulder and armpit landmarks.
        
        Returns
        -------
        tuple[float, trimesh.path.Path3D]
            (length_value, line_segment_path_in_original_coordinates)
        """
        print("Called measure_arm_length (Arm)")
        
        # Get shoulder landmark
        shoulder = Arm._locate_shoulder(mesh, side)
        
        # Get armpit landmark
        from ..trunk import Trunk
        left_armpit, right_armpit = Trunk._locate_armpits(mesh)
        armpit = left_armpit if side == 'left' else right_armpit
        
        # Get wrist landmark
        wrist = Arm._locate_wrist(mesh, side)
        
        # Calculate midpoint of shoulder and armpit
        midpoint = (shoulder + armpit) / 2.0
        
        # Calculate 2D distance in x-z plane
        # ||wrist - midpoint||_(x,z) means using only x and z components
        diff = wrist - midpoint
        distance = np.sqrt(diff[0]**2 + diff[2]**2)
        
        return (float(distance), line_path([midpoint, wrist]))

    @staticmethod
    @cache
    def _measure_forearm_girth(mesh: trimesh.Trimesh, side: LEFT_OR_RIGHT):
        """
        Measure forearm girth at 50% up from fingertip to shoulder.
        
        Returns
        -------
        tuple[float, trimesh.path.Path3D]
            (girth_value, path_in_original_coordinates)
        """
        print("Called measure_forearm_girth (Arm)")

        arm_mesh_copy, inverse_transform = _aligned_arm_mesh(mesh, side)
        z_min = arm_mesh_copy.vertices[:, 2].min()
        z_max = arm_mesh_copy.vertices[:, 2].max()
        arm_height = z_max - z_min
        z_slice = z_min + arm_height * 0.5
        return _arm_section_measurement(arm_mesh_copy, z_slice, inverse_transform)

    @staticmethod
    @cache
    def _measure_bicep_girth(mesh: trimesh.Trimesh, side: LEFT_OR_RIGHT):
        """
        Measure bicep girth at 75% up from fingertip to shoulder.
        
        Returns
        -------
        tuple[float, trimesh.path.Path3D]
            (girth_value, path_in_original_coordinates)
        """
        print("Called measure_bicep_girth (Arm)")

        arm_mesh_copy, inverse_transform = _aligned_arm_mesh(mesh, side)
        best = _largest_fraction_loop(arm_mesh_copy, np.linspace(0.45, 0.62, 12))
        if best is None:
            return empty_measurement()
        z_slice, perimeter, loop_2d = best
        path_3d = _arm_loop_path(loop_2d, z_slice, inverse_transform)

        return (float(perimeter), path_3d)
