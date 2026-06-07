from functools import cache
import numpy as np
import trimesh
from ..anatomical_region import Anatomical_Region, LEFT_OR_RIGHT, get_geometry_config


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
        
        # 4. Find the part with the highest vertex
        z_maxes = [part.vertices[:,2].max() for part in parts]
        arm_index = np.argmax(z_maxes)
        arm_mesh = parts[arm_index]
        
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
        pseudo:
        Shoulder = extreme X point among the top ~10% highest vertices.
        Right shoulder = max X.
        Left shoulder = min X.
        """
        print("Called locate_shoulder (Arm)")
        
        vertices = np.asarray(mesh.vertices)
        z_cut = np.percentile(vertices[:, 2], 90)
        upper = vertices[vertices[:, 2] >= z_cut]

        if side == "right":
            shoulder = upper[np.argmax(upper[:, 0])]
        else:
            shoulder = upper[np.argmin(upper[:, 0])]

        return shoulder

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
        
        # Get arm mesh for this side
        arm_mesh = Arm._get_submesh(side, mesh)
        
        # Orient arm upright to z-axis
        from ....mesh.mesh import Mesh
        arm_mesh_copy = arm_mesh.copy()
        
        # Align arm to z-axis and get transformation matrix
        transform_matrix = Mesh.align_mesh_to_z_axis(arm_mesh_copy)
        
        # Get z bounds of the arm
        z_min = arm_mesh_copy.vertices[:, 2].min()
        z_max = arm_mesh_copy.vertices[:, 2].max()
        arm_height = z_max - z_min
        
        # Search for wrist in the lower portion of the arm
        search_z_min = z_min + arm_height * 0.1
        search_z_max = z_min + arm_height * 0.3
        
        # Create slice heights
        slice_step = get_geometry_config(mesh)["arm_slice_step"]
        z_heights = np.arange(search_z_min, search_z_max, slice_step)
        
        if len(z_heights) == 0:
            z_heights = np.array([(search_z_min + search_z_max) / 2])
        
        min_perimeter = float('inf')
        wrist_centroid_aligned = None
        fallback_centroid = None
        
        # Find the slice with minimum perimeter
        for z in z_heights:
            slice_2d = arm_mesh_copy.section(
                plane_origin=[0, 0, z],
                plane_normal=[0, 0, 1]
            )
            
            if slice_2d is not None:
                # Keep track of first valid slice as fallback
                if fallback_centroid is None:
                    fallback_centroid = np.array([slice_2d.centroid[0], slice_2d.centroid[1], z])
                
                # Use raw perimeter (simpler, no convex hull needed)
                perimeter = slice_2d.length
                
                if perimeter < min_perimeter:
                    min_perimeter = perimeter
                    # Get centroid of the 2D slice (in aligned coordinate system)
                    wrist_centroid_aligned = np.array([slice_2d.centroid[0], slice_2d.centroid[1], z])
        
        # Use fallback if no valid perimeter was found
        if wrist_centroid_aligned is None:
            wrist_centroid_aligned = fallback_centroid
        
        # If still no valid centroid, use a point in the middle of search range
        if wrist_centroid_aligned is None:
            wrist_centroid_aligned = np.array([0, 0, (search_z_min + search_z_max) / 2])
        
        # Map the centroid position back to original coordinate system
        # Convert to homogeneous coordinates
        wrist_homogeneous = np.append(wrist_centroid_aligned, 1.0)
        
        # Apply inverse transformation to map back to original space
        inverse_transform = np.linalg.inv(transform_matrix)
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
        
        # Get arm mesh for this side
        arm_mesh = Arm._get_submesh(side, mesh)
        
        # Orient arm upright to z-axis
        from ....mesh.mesh import Mesh
        arm_mesh_copy = arm_mesh.copy()
        transform_matrix = Mesh.align_mesh_to_z_axis(arm_mesh_copy)
        
        # Get z bounds of the arm
        z_min = arm_mesh_copy.vertices[:, 2].min()
        z_max = arm_mesh_copy.vertices[:, 2].max()
        arm_height = z_max - z_min
        
        # Search for wrist in the lower portion of the arm (same range as landmark detection)
        search_z_min = z_min + arm_height * 0.1
        search_z_max = z_min + arm_height * 0.3
        
        # Create slice heights
        slice_step = get_geometry_config(mesh)["arm_slice_step"]
        z_heights = np.arange(search_z_min, search_z_max, slice_step)
        
        if len(z_heights) == 0:
            z_heights = np.array([(search_z_min + search_z_max) / 2])
        
        min_perimeter = float('inf')
        best_slice = None
        best_z = None
        
        # Find the slice with minimum perimeter
        for z in z_heights:
            slice_2d = arm_mesh_copy.section(
                plane_origin=[0, 0, z],
                plane_normal=[0, 0, 1]
            )
            
            if slice_2d is not None:
                perimeter = slice_2d.length
                
                if perimeter < min_perimeter:
                    min_perimeter = perimeter
                    best_slice = slice_2d
                    best_z = z
        
        # If no valid perimeter found, return 0 and empty path
        if min_perimeter == float('inf'):
            empty_path = trimesh.load_path(np.array([[0, 0, 0]]))
            return (0.0, empty_path)
        
        # The 2D slice is a Path2D with properly ordered vertices
        # We need to convert it to 3D while preserving the vertex order from the entities
        vertices_2d_ordered = best_slice.vertices
        
        # Get the proper ordering from the Path2D entities
        if len(best_slice.entities) > 0:
            # Extract vertex indices in the correct order from the path entities
            ordered_indices = []
            for entity in best_slice.entities:
                ordered_indices.extend(entity.points)
            # Remove duplicates while preserving order
            seen = set()
            ordered_indices = [i for i in ordered_indices if not (i in seen or seen.add(i))]
            vertices_2d_ordered = vertices_2d_ordered[ordered_indices]
        
        # Convert to 3D in aligned coordinates
        vertices_3d_aligned = np.column_stack([
            vertices_2d_ordered[:, 0],  # x
            vertices_2d_ordered[:, 1],  # y
            np.full(len(vertices_2d_ordered), best_z)  # z
        ])
        
        # Transform back to original coordinates
        inverse_transform = np.linalg.inv(transform_matrix)
        vertices_3d_original = trimesh.transform_points(vertices_3d_aligned, inverse_transform)
        
        # Create closed loop path
        indices = np.arange(len(vertices_3d_original) + 1)
        indices[-1] = 0  # Close the loop
        entities = [trimesh.path.entities.Line(indices)]
        path_3d = trimesh.path.Path3D(entities=entities, vertices=vertices_3d_original)
        
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
        
        # Create Path3D line segment from midpoint to wrist
        vertices = np.array([midpoint, wrist])
        entities = [trimesh.path.entities.Line([0, 1])]
        path_3d = trimesh.path.Path3D(entities=entities, vertices=vertices)
        
        return (float(distance), path_3d)

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
        
        # Get arm mesh for this side
        arm_mesh = Arm._get_submesh(side, mesh)
        
        # Orient arm upright to z-axis
        from ....mesh.mesh import Mesh
        arm_mesh_copy = arm_mesh.copy()
        transform_matrix = Mesh.align_mesh_to_z_axis(arm_mesh_copy)
        
        # Get z bounds of the arm
        z_min = arm_mesh_copy.vertices[:, 2].min()
        z_max = arm_mesh_copy.vertices[:, 2].max()
        arm_height = z_max - z_min
        
        # Slice at 25% up from finger tip (z_min) towards shoulder (z_max)
        z_slice = z_min + arm_height * 0.5
        
        slice_2d = arm_mesh_copy.section(
            plane_origin=[0, 0, z_slice],
            plane_normal=[0, 0, 1]
        )
        
        if slice_2d is None:
            empty_path = trimesh.load_path(np.array([[0, 0, 0]]))
            return (0.0, empty_path)
        
        # Get properly ordered vertices from the Path2D
        vertices_2d_ordered = slice_2d.vertices
        if len(slice_2d.entities) > 0:
            ordered_indices = []
            for entity in slice_2d.entities:
                ordered_indices.extend(entity.points)
            seen = set()
            ordered_indices = [i for i in ordered_indices if not (i in seen or seen.add(i))]
            vertices_2d_ordered = vertices_2d_ordered[ordered_indices]
        
        # Convert to 3D in aligned coordinates
        vertices_3d_aligned = np.column_stack([
            vertices_2d_ordered[:, 0],
            vertices_2d_ordered[:, 1],
            np.full(len(vertices_2d_ordered), z_slice)
        ])
        
        # Transform back to original coordinates
        inverse_transform = np.linalg.inv(transform_matrix)
        vertices_3d_original = trimesh.transform_points(vertices_3d_aligned, inverse_transform)
        
        # Create closed loop path
        indices = np.arange(len(vertices_3d_original) + 1)
        indices[-1] = 0
        entities = [trimesh.path.entities.Line(indices)]
        path_3d = trimesh.path.Path3D(entities=entities, vertices=vertices_3d_original)
        
        return (float(slice_2d.length), path_3d)

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
        
        # Get arm mesh for this side
        arm_mesh = Arm._get_submesh(side, mesh)
        
        # Orient arm upright to z-axis
        from ....mesh.mesh import Mesh
        arm_mesh_copy = arm_mesh.copy()
        transform_matrix = Mesh.align_mesh_to_z_axis(arm_mesh_copy)
        
        # Get z bounds of the arm
        z_min = arm_mesh_copy.vertices[:, 2].min()
        z_max = arm_mesh_copy.vertices[:, 2].max()
        arm_height = z_max - z_min
        
        # Slice at 75% up from finger tip (z_min) towards shoulder (z_max)
        z_slice = z_min + arm_height * 0.75
        
        slice_2d = arm_mesh_copy.section(
            plane_origin=[0, 0, z_slice],
            plane_normal=[0, 0, 1]
        )
        
        if slice_2d is None:
            empty_path = trimesh.load_path(np.array([[0, 0, 0]]))
            return (0.0, empty_path)
        
        # Get properly ordered vertices from the Path2D
        vertices_2d_ordered = slice_2d.vertices
        if len(slice_2d.entities) > 0:
            ordered_indices = []
            for entity in slice_2d.entities:
                ordered_indices.extend(entity.points)
            seen = set()
            ordered_indices = [i for i in ordered_indices if not (i in seen or seen.add(i))]
            vertices_2d_ordered = vertices_2d_ordered[ordered_indices]
        
        # Convert to 3D in aligned coordinates
        vertices_3d_aligned = np.column_stack([
            vertices_2d_ordered[:, 0],
            vertices_2d_ordered[:, 1],
            np.full(len(vertices_2d_ordered), z_slice)
        ])
        
        # Transform back to original coordinates
        inverse_transform = np.linalg.inv(transform_matrix)
        vertices_3d_original = trimesh.transform_points(vertices_3d_aligned, inverse_transform)
        
        # Create closed loop path
        indices = np.arange(len(vertices_3d_original) + 1)
        indices[-1] = 0
        entities = [trimesh.path.entities.Line(indices)]
        path_3d = trimesh.path.Path3D(entities=entities, vertices=vertices_3d_original)
        
        return (float(slice_2d.length), path_3d)
