from functools import cache

import trimesh
import numpy as np
from scipy.spatial import cKDTree

from ....utils.convexity_search import convexity_search

from ..anatomical_region import Anatomical_Region, get_geometry_config


class Trunk(Anatomical_Region):
    """
    Trunk (torso) region segmentation, landmark detection, and measurement computation.
    
    The Trunk class handles extraction and analysis of the torso from a full body mesh.
    It identifies critical landmarks (crotch, armpits, hips, collar) and computes important
    measurements (trunk length, chest/waist/hip circumferences).
    
    Purpose
    -------
    This class solves the problem of automatically measuring the torso region from body scans.
    Without this class, you would need to manually:
    - Separate trunk from head, arms, and legs
    - Locate complex landmarks like armpits (concave regions) and crotch (convex region)
    - Find hip and collar points for measurements
    - Compute circumferences at different heights
    
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
        - "chest circumference": Girth at chest level
        - "waist circumference": Girth at narrowest point
        - "hip circumference": Girth at hip level
    
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
        
        min_viable_point = intersects[np.argmin(intersects, axis=0)[2], :]
        viable_point_idx =  kdtree.query(min_viable_point)[1]
        viable_point = new_mesh.vertices[viable_point_idx]
        
        crotch_point_nearest = convexity_search(new_mesh, 
                                        rays=32,
                                        origin=viable_point,
                                        slice_width=get_geometry_config(new_mesh)["convexity_slice_width"])
        
        crotch_point_idx = kdtree.query(crotch_point_nearest)[1]
        crotch_point = new_mesh.vertices[crotch_point_idx]
        
        return crotch_point

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

        return (left_armpit_point, right_armpit_point)
    
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
        
        # Define z-range around crotch
        body_height = mesh.vertices[:, 2].max() - mesh.vertices[:, 2].min()
        slice_z_min = crotch_point[2] + body_height * 0.05
        slice_z_max = crotch_point[2] + body_height * 0.1
        
        # Get vertices in z-range
        in_range_mask = (mesh.vertices[:, 2] >= slice_z_min) & (mesh.vertices[:, 2] <= slice_z_max)
        in_range_indices = np.where(in_range_mask)[0]
        
        # Build submesh from these vertices
        # Get faces that have all vertices in range
        faces_in_range = []
        for i, face in enumerate(mesh.faces):
            if all(in_range_mask[v] for v in face):
                faces_in_range.append(i)
        
        # Create submesh
        submesh = mesh.submesh([faces_in_range], append=True)

        # Split into disconnected components
        parts = submesh.split(only_watertight=False)
        print(f"Split into {len(parts)} parts")
        
        # Find the central part (torso) - the one with vertices closest to x=0
        x_distances = [np.abs(part.vertices[:, 0]).mean() for part in parts]
        torso_part = parts[np.argmin(x_distances)]
        
        # Find farthest left and right points on the torso
        left_hip = torso_part.vertices[np.argmin(torso_part.vertices[:, 0])]
        right_hip = torso_part.vertices[np.argmax(torso_part.vertices[:, 0])]
        
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
        Measure hip circumference at minimum perimeter in bottom 10% of trunk.
        Uses body mesh without arms (includes legs for proper hip measurement).
        
        Returns
        -------
        tuple[float, trimesh.path.Path3D]
            (circumference_value, path_in_original_coordinates)
        """
        print("Called measure_hip_circumference (Trunk)")
        
        from ..arms import Arm
        from ....mesh.boolean_ops import mesh_difference
        
        # Remove arms from body mesh (keep trunk + legs for proper hip measurement)
        body_without_arms = mesh.copy()
        left_arm_mesh = Arm._get_submesh("left", mesh)
        body_without_arms = mesh_difference(body_without_arms, left_arm_mesh)
        right_arm_mesh = Arm._get_submesh("right", mesh)
        body_without_arms = mesh_difference(body_without_arms, right_arm_mesh)
        
        # Get trunk mesh to determine trunk height
        trunk_mesh = Trunk._get_submesh(mesh)
        trunk_z_min = np.min(trunk_mesh.vertices[:, 2])
        trunk_z_max = np.max(trunk_mesh.vertices[:, 2])
        trunk_height = trunk_z_max - trunk_z_min
        
        # Define hip region (bottom 10% of trunk)
        hip_region_z_max = trunk_z_min + 0.1 * trunk_height
        
        # Sample slices in hip region to find minimum perimeter
        num_samples = 20
        z_values = np.linspace(trunk_z_min, hip_region_z_max, num_samples)
        
        min_perimeter = float('inf')
        best_section = None
        best_z = trunk_z_min
        
        plane_normal = np.array([0, 0, 1])  # Horizontal plane
        
        for z in z_values:
            plane_origin = np.array([0, 0, z])
            section = body_without_arms.section(plane_normal=plane_normal, plane_origin=plane_origin)
            
            if section is not None and section.length < min_perimeter:
                min_perimeter = section.length
                best_section = section
                best_z = z
        
        if best_section is None:
            print("Warning: No section found in hip region")
            empty_path = trimesh.load_path(np.array([[0, 0, 0]]))
            return (0.0, empty_path)
        
        # Get properly ordered vertices from the Path2D
        vertices_2d_ordered = best_section.vertices
        if len(best_section.entities) > 0:
            ordered_indices = []
            for entity in best_section.entities:
                ordered_indices.extend(entity.points)
            seen = set()
            ordered_indices = [i for i in ordered_indices if not (i in seen or seen.add(i))]
            vertices_2d_ordered = vertices_2d_ordered[ordered_indices]
        
        # Convert to 3D path
        vertices_3d = np.column_stack([
            vertices_2d_ordered[:, 0],
            vertices_2d_ordered[:, 1],
            np.full(len(vertices_2d_ordered), best_z)
        ])
        
        # Create closed loop path
        indices = np.arange(len(vertices_3d) + 1)
        indices[-1] = 0
        entities = [trimesh.path.entities.Line(indices)]
        path_3d = trimesh.path.Path3D(entities=entities, vertices=vertices_3d)
        
        return (float(min_perimeter), path_3d)

    @staticmethod
    @cache
    def _measure_chest_circumference(mesh: trimesh.Trimesh):
        """
        Measure chest circumference at armpit level.
        
        Returns
        -------
        tuple[float, trimesh.path.Path3D]
            (circumference_value, path_in_original_coordinates)
        """
        print("Called measure_chest_circumference (Trunk)")
        
        # Get torso mesh (same as trunk mesh)
        torso_mesh = Trunk._get_submesh(mesh)
        
        # Get armpit locations
        left_armpit, right_armpit = Trunk._locate_armpits(mesh)
        
        # Calculate median z of armpits
        armpit_z_median = np.median([left_armpit[2], right_armpit[2]])
        
        # Take slice at chest level
        plane_normal = np.array([0, 0, 1])  # Horizontal plane
        plane_origin = np.array([0, 0, armpit_z_median])
        
        section = torso_mesh.section(plane_normal=plane_normal, plane_origin=plane_origin)
        
        if section is None:
            print("Warning: No section found at chest level")
            empty_path = trimesh.load_path(np.array([[0, 0, 0]]))
            return (0.0, empty_path)
        
        # Get properly ordered vertices from the Path2D
        vertices_2d_ordered = section.vertices
        if len(section.entities) > 0:
            ordered_indices = []
            for entity in section.entities:
                ordered_indices.extend(entity.points)
            seen = set()
            ordered_indices = [i for i in ordered_indices if not (i in seen or seen.add(i))]
            vertices_2d_ordered = vertices_2d_ordered[ordered_indices]
        
        # Convert to 3D path
        vertices_3d = np.column_stack([
            vertices_2d_ordered[:, 0],
            vertices_2d_ordered[:, 1],
            np.full(len(vertices_2d_ordered), armpit_z_median)
        ])
        
        # Create closed loop path
        indices = np.arange(len(vertices_3d) + 1)
        indices[-1] = 0
        entities = [trimesh.path.entities.Line(indices)]
        path_3d = trimesh.path.Path3D(entities=entities, vertices=vertices_3d)
        
        return (float(section.length), path_3d)

    @staticmethod
    @cache
    def _measure_waist_circumference(mesh: trimesh.Trimesh):
        """
        Measure waist circumference between armpit and hip.
        
        Returns
        -------
        tuple[float, trimesh.path.Path3D]
            (circumference_value, path_in_original_coordinates)
        """
        print("Called measure_waist_circumference (Trunk)")
        
        # Get torso mesh (same as trunk mesh)
        torso_mesh = Trunk._get_submesh(mesh)
        
        # Get armpit and hip locations
        left_armpit, right_armpit = Trunk._locate_armpits(mesh)
        left_hip, right_hip = Trunk._locate_hips(mesh)
        
        # Calculate mean z of right armpit and right hip (as per pseudocode)
        waist_z_mean = np.mean([right_armpit[2], right_hip[2]])
        
        # Take slice at waist level
        plane_normal = np.array([0, 0, 1])  # Horizontal plane
        plane_origin = np.array([0, 0, waist_z_mean])
        
        section = torso_mesh.section(plane_normal=plane_normal, plane_origin=plane_origin)
        
        if section is None:
            print("Warning: No section found at waist level")
            empty_path = trimesh.load_path(np.array([[0, 0, 0]]))
            return (0.0, empty_path)
        
        # Get properly ordered vertices from the Path2D
        vertices_2d_ordered = section.vertices
        if len(section.entities) > 0:
            ordered_indices = []
            for entity in section.entities:
                ordered_indices.extend(entity.points)
            seen = set()
            ordered_indices = [i for i in ordered_indices if not (i in seen or seen.add(i))]
            vertices_2d_ordered = vertices_2d_ordered[ordered_indices]
        
        # Convert to 3D path
        vertices_3d = np.column_stack([
            vertices_2d_ordered[:, 0],
            vertices_2d_ordered[:, 1],
            np.full(len(vertices_2d_ordered), waist_z_mean)
        ])
        
        # Create closed loop path
        indices = np.arange(len(vertices_3d) + 1)
        indices[-1] = 0
        entities = [trimesh.path.entities.Line(indices)]
        path_3d = trimesh.path.Path3D(entities=entities, vertices=vertices_3d)
        
        return (float(section.length), path_3d)

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
