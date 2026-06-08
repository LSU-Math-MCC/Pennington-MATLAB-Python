from functools import cache
from typing import Tuple

import trimesh
import numpy as np
from scipy.spatial import cKDTree

from ....mesh.boolean_ops import mesh_difference

from ..anatomical_region import Anatomical_Region, LEFT_OR_RIGHT, get_geometry_config

class Leg(Anatomical_Region):
    """
    Leg region segmentation, landmark detection, and measurement computation.
    
    The Leg class handles extraction and analysis of left or right leg from a full body mesh.
    It identifies the leg by removing other body parts and slicing at the hip/crotch,
    locates key landmarks (ankle, foot), and computes measurements (leg length, circumferences).
    
    Purpose
    -------
    This class solves the problem of automatically measuring leg dimensions from body scans.
    Without this class, you would need to manually:
    - Separate legs from trunk and each other
    - Remove arms that might interfere with segmentation
    - Find anatomical landmarks (hip, knee, ankle, foot)
    - Compute lengths and circumferences
    
    With this class, creating a Leg instance automatically performs all these operations.
    
    Class Structure and Design Choices
    -----------------------------------
    **Complex segmentation process**:
    Leg extraction is more complex than arm extraction because:
    1. Must first remove arms (they can interfere with leg detection)
    2. Must separate left and right legs at the crotch
    
    The multi-step process uses mesh boolean operations extensively.
    
    **Static methods with @cache**:
    See Anatomical_Region docstring for the rationale behind this design pattern.
    
    **Runtime imports**:
    Imports appear inside methods because of circular dependencies:
    - Leg needs Trunk (for crotch/hip landmarks)
    - Leg needs Arm (to remove arms before leg extraction)
    - Trunk needs Leg (to remove legs from trunk)
    
    **Boolean difference approach**:
    The leg extraction uses mesh_difference() to:
    1. Remove left arm: body_without_left = mesh_difference(body, left_arm)
    2. Remove right arm: body_without_arms = mesh_difference(body_without_left, right_arm)
    3. Split at crotch to separate legs
    
    This is simpler than trying to slice the mesh directly but requires clean meshes.
    
    **Ankle and foot detection**:
    The leg class uses multiple strategies to find these landmarks:
    - Foot: Lowest point of the leg mesh
    - Ankle: Point at specific height above foot (based on leg length)
    
    These are heuristics that work for standard poses but may fail for unusual positions.
    
    Attributes
    ----------
    body_mesh : trimesh.Trimesh
        The full body mesh (cleaned and oriented)
    side : LEFT_OR_RIGHT ('left' or 'right')
        Which leg this instance represents
    
    Properties
    ----------
    mesh : trimesh.Trimesh
        The segmented leg mesh (cached, computed on first access)
    landmarks : dict[str, np.ndarray]
        Key anatomical points:
        - "foot": Lowest point of leg (sole)
        - "ankle": Point above foot at ankle height
    measurements : dict[str, float]
        Anthropometric measurements:
        - "{side} leg length": Hip to ankle distance
        - "{side} ankle girth": Circumference at ankle
        - "{side} calf girth": Circumference at mid-calf
        - "{side} thigh girth": Circumference at mid-thigh
    
    Examples
    --------
    >>> import trimesh
    >>> from body import Body  # doctest: +SKIP
    >>> body = Body("model_files/man.obj")  # doctest: +SKIP
    >>> left_leg = body.parts["left leg"]  # doctest: +SKIP
    >>> leg_length = left_leg.measurements["left leg length"]  # doctest: +SKIP
    >>> print(f"Leg length: {leg_length:.2f} cm")  # doctest: +SKIP
    Leg length: 89.3 cm
    
    >>> ankle = left_leg.landmarks["ankle"]  # doctest: +SKIP
    >>> foot = left_leg.landmarks["foot"]  # doctest: +SKIP
    >>> print(f"Ankle: {ankle}, Foot: {foot}")  # doctest: +SKIP
    Ankle: [-8.2, 0.1, 8.5], Foot: [-8.5, 0.0, 0.0]
    
    Notes
    -----
    - Requires body mesh to be in standard orientation (Z-axis vertical)
    - Most complex segmentation process of all body parts
    - Sensitive to mesh quality (boolean operations can fail on bad meshes)
    - Assumes legs are slightly apart (standard anthropometric pose)
    - All measurements are in the same units as the input mesh
    
    See Also
    --------
    Trunk : Provides crotch and hip landmarks used for leg segmentation
    Arm : Must be extracted first to avoid interference
    mesh_difference : Boolean operation used for leg isolation
    Anatomical_Region : Abstract base class defining the interface
    """

    def __init__(self, body_mesh: trimesh.Trimesh, left_or_right: LEFT_OR_RIGHT):
        print("Called __init__ (Leg)")

        self.side = left_or_right
        self.body_mesh = body_mesh

    @property
    def volume(self):
        print("Called volume (Leg)")
        return self.mesh.volume
    
    @property
    def surface_area(self):
        print("Called surface_area (Leg)")
        return self.mesh.area

    # Properties of Leg

    # Vertex Indices (provided by `mesh`)
    @property
    def mesh(self):
        return Leg._get_submesh(self.side, self.body_mesh)
    
    @staticmethod
    @cache
    def _get_submesh(side: LEFT_OR_RIGHT, mesh: trimesh.Trimesh):
        from ..trunk import Trunk
        from ..arms import Arm
        
        
        # 1. Remove pesky arms
        left_arm_mesh = Arm._get_submesh("left", mesh)
        right_arm_mesh = Arm._get_submesh("right", mesh)

        body_without_arm = mesh_difference(mesh, left_arm_mesh)
        body_without_arms = mesh_difference(body_without_arm, right_arm_mesh)
        mesh = body_without_arms

        # 2. Get hip for this leg, from trunk
        left_hip, right_hip = Trunk._locate_hips(mesh)
        hip = left_hip if side == 'left' else right_hip

        # 3. Slice mesh by plane connecting crotch and hip
        # Identify the plane first
        if side == 'left':
            front_back_vector = np.array([0.0, 1.0, 0.0])
            left_right_vector = np.array([-1.0, 0.0, 0.0])
        else:
            front_back_vector = np.array([0.0, -1.0, 0.0])
            left_right_vector = np.array([1.0, 0.0, 0.0])
        
        crotch = Trunk._locate_crotch(mesh)
        v = np.asarray(hip) - np.asarray(crotch)
        n = np.cross(v, front_back_vector)

        #Then slice and get the correct side
        sliced_mesh = mesh.slice_plane(
            plane_origin=crotch,
            plane_normal=n
        )

        # Take part only one side
        one_leg_sliced_mesh = sliced_mesh.slice_plane(
            plane_origin=crotch,
            plane_normal=left_right_vector
        )

        # Clean mesh before splitting
        one_leg_sliced_mesh.remove_unreferenced_vertices()
        one_leg_sliced_mesh.fill_holes()

        return one_leg_sliced_mesh

    # Landmarks # TODO: Might want to make these properties so the access is simpler, but not required. This goes for all landmarks and measurements in src actually

    @property
    def landmarks(self):
        return {
            "foot": Leg._locate_foot(self.body_mesh, self.side),
            "ankle": Leg._locate_ankle(self.body_mesh, self.side)
        }

    @staticmethod
    @cache
    def _locate_foot(mesh: trimesh.Trimesh, left_or_right: LEFT_OR_RIGHT):
        print("Called locate_foot (Leg)")

        index = 0 if left_or_right == "left" else 1
        return Leg._identify_feet(mesh)[index]


    @staticmethod
    @cache
    def _locate_ankle(mesh: trimesh.Trimesh, side: LEFT_OR_RIGHT):
        """
        The centroid of the horizontal leg cross-section S that minimizes the perimeter of the 2D boundary of S, 
        found by searching slices in the distal 1/8th of the lower leg.
        
        Psuedo:
        * trimesh multislice up 1/4 leg in like 1 inch steps
        * get perimeter of each slice
        * take slice that was min
        * centroid of slice is ankle point
        """

        print("Called locate_ankle (Leg)")

        # Get leg mesh for this side
        leg_mesh = Leg._get_submesh(side, mesh)
        
        # Orient leg upright to z-axis for consistent slicing
        from ....mesh.mesh import Mesh
        leg_mesh_copy = leg_mesh.copy()
        
        # Align the copy (leg_mesh remains in original coordinate system)
        Mesh.align_mesh_to_z_axis(leg_mesh_copy)
        
        # Get z bounds of the leg
        z_min = leg_mesh_copy.vertices[:, 2].min()
        z_max = leg_mesh_copy.vertices[:, 2].max()
        leg_height = z_max - z_min
        
        # Search in the distal 1/8th (lower 12.5%) of the leg
        search_z_min = z_min + leg_height * 0.05
        search_z_max = z_min + leg_height * 0.125
        
        # Create slice heights approximately 1 inch (0.0254 meters) apart
        slice_step = get_geometry_config(mesh)["leg_foot_slice_step"]
        z_heights = np.arange(search_z_min, search_z_max, slice_step)
        
        if len(z_heights) == 0:
            z_heights = np.array([search_z_min + leg_height * 0.0625])
        
        min_perimeter = float('inf')
        ankle_centroid_aligned = None
        
        # Find the slice with minimum perimeter
        for z in z_heights:
            slice_2d = leg_mesh_copy.section(
                plane_origin=[0, 0, z],
                plane_normal=[0, 0, 1]
            )
            
            if slice_2d is not None:
                # Get perimeter of the 2D cross-section
                perimeter = slice_2d.length
                if perimeter < min_perimeter:
                    min_perimeter = perimeter
                    # Get centroid of the 2D slice (in aligned coordinate system)
                    ankle_centroid_aligned = np.array([slice_2d.centroid[0], slice_2d.centroid[1], z])

        # Handle case where no valid cross-sections were found
        if ankle_centroid_aligned is None:
            return None

        # Map the ankle centroid back to the original coordinate system
        # Find closest vertex to the centroid in aligned mesh as a reference point
        tree_aligned = cKDTree(leg_mesh_copy.vertices)
        _, ref_idx = tree_aligned.query(ankle_centroid_aligned)
        
        # Compute offset from reference vertex to centroid in aligned space
        offset = ankle_centroid_aligned - leg_mesh_copy.vertices[ref_idx]
        
        # Apply the same offset to the corresponding vertex in original mesh
        ankle_centroid_original = leg_mesh.vertices[ref_idx] + offset

        return ankle_centroid_original

    # Measurements & Drawings

    @property
    def measurements(self):
        """Extract just the measurement values (first element of tuples)."""
        return {
            "leg length": Leg._measure_leg_length(self.body_mesh, self.side)[0],
            "ankle girth": Leg._measure_ankle_girth(self.body_mesh, self.side)[0],
            "calf girth": Leg._measure_calf_girth(self.body_mesh, self.side)[0],
            "thigh girth": Leg._measure_thigh_girth(self.body_mesh, self.side)[0]
        }

    @property
    def drawings(self):
        """Extract the 3D paths showing where measurements were taken (second element of tuples)."""
        return {
            "leg length": Leg._measure_leg_length(self.body_mesh, self.side)[1],
            "ankle girth": Leg._measure_ankle_girth(self.body_mesh, self.side)[1],
            "calf girth": Leg._measure_calf_girth(self.body_mesh, self.side)[1],
            "thigh girth": Leg._measure_thigh_girth(self.body_mesh, self.side)[1]
        }

    @staticmethod
    @cache
    def _measure_leg_length(mesh: trimesh.Trimesh, side: LEFT_OR_RIGHT):
        """
        Measure leg length from hip to foot.
        
        Calculates leg length based on the vertical distance from the hip landmark to the lowest point on the corresponding foot, 
        combined with the horizontal distance from the body's centerline to the foot.
        
        Returns
        -------
        tuple[float, trimesh.path.Path3D]
            (length_value, line_segment_path_in_original_coordinates)
        """

        print("Called measure_leg_length (Leg)")

        from ..trunk import Trunk
        
        # Get hip landmark for this side
        left_hip, right_hip = Trunk._locate_hips(mesh)
        hip = left_hip if side == 'left' else right_hip
        
        # Get foot landmark for this side
        foot = Leg._locate_foot(mesh, side)
        
        # Calculate leg length using the formula:
        # sqrt( (hip_z - foot_z)^2 + (foot_x - centerline_x)^2 )
        # Centerline is at x=0
        hip_z = hip[2]
        foot_z = foot[2]
        foot_x = foot[0]
        centerline_x = 0.0
        
        leg_length = np.sqrt((hip_z - foot_z)**2 + (foot_x - centerline_x)**2)
        
        # Create Path3D line segment from hip to foot
        vertices = np.array([hip, foot])
        entities = [trimesh.path.entities.Line([0, 1])]
        path_3d = trimesh.path.Path3D(entities=entities, vertices=vertices)
        
        return (float(leg_length), path_3d)

    @staticmethod
    @cache
    def _measure_ankle_girth(mesh: trimesh.Trimesh, side: LEFT_OR_RIGHT):
        """
        Measure ankle girth by finding minimum perimeter in lower 1/4 of leg.
        
        Returns
        -------
        tuple[float, trimesh.path.Path3D]
            (girth_value, path_in_original_coordinates)
        """

        print("Called measure_ankle_girth (Leg)")

        # Get leg mesh for this side
        leg_mesh = Leg._get_submesh(side, mesh)
        
        # Orient leg upright to z-axis for consistent slicing
        from ....mesh.mesh import Mesh
        leg_mesh_copy = leg_mesh.copy()
        transform_matrix = Mesh.align_mesh_to_z_axis(leg_mesh_copy)
        
        # Get z bounds of the leg
        z_min = leg_mesh_copy.vertices[:, 2].min()
        z_max = leg_mesh_copy.vertices[:, 2].max()
        leg_height = z_max - z_min
        
        # Search in the lower 1/4 of the leg
        search_z_min = z_min + leg_height * 0.05
        search_z_max = z_min + leg_height * 0.25
        
        # Create slice heights approximately 1 inch apart
        slice_step = get_geometry_config(mesh)["leg_slice_step"]
        z_heights = np.arange(search_z_min, search_z_max, slice_step)
                
        min_perimeter = float('inf')
        best_slice = None
        best_z = None
        
        # Find the slice with minimum perimeter
        for z in z_heights:
            slice_2d = leg_mesh_copy.section(
                plane_origin=[0, 0, z],
                plane_normal=[0, 0, 1]
            )
            
            if slice_2d is not None:
                # Get perimeter of the 2D cross-section
                perimeter = slice_2d.length
                if perimeter < min_perimeter:
                    min_perimeter = perimeter
                    best_slice = slice_2d
                    best_z = z
        
        # If no valid slice found, return empty
        if min_perimeter == float('inf'):
            empty_path = trimesh.load_path(np.array([[0, 0, 0]]))
            return (0.0, empty_path)
        
        # Get properly ordered vertices from the Path2D
        vertices_2d_ordered = best_slice.vertices
        if len(best_slice.entities) > 0:
            ordered_indices = []
            for entity in best_slice.entities:
                ordered_indices.extend(entity.points)
            seen = set()
            ordered_indices = [i for i in ordered_indices if not (i in seen or seen.add(i))]
            vertices_2d_ordered = vertices_2d_ordered[ordered_indices]
        
        # Convert to 3D in aligned coordinates
        vertices_3d_aligned = np.column_stack([
            vertices_2d_ordered[:, 0],
            vertices_2d_ordered[:, 1],
            np.full(len(vertices_2d_ordered), best_z)
        ])
        
        # Transform back to original coordinates
        inverse_transform = np.linalg.inv(transform_matrix)
        vertices_3d_original = trimesh.transform_points(vertices_3d_aligned, inverse_transform)
        
        # Create closed loop path
        indices = np.arange(len(vertices_3d_original) + 1)
        indices[-1] = 0
        entities = [trimesh.path.entities.Line(indices)]
        path_3d = trimesh.path.Path3D(entities=entities, vertices=vertices_3d_original)
        
        return (float(min_perimeter), path_3d)

    @staticmethod
    @cache
    def _measure_calf_girth(mesh: trimesh.Trimesh, side: LEFT_OR_RIGHT):
        """
        Measure calf girth as maximum perimeter between 15% and 50% of leg height.
        
        Returns
        -------
        tuple[float, trimesh.path.Path3D]
            (girth_value, path_in_original_coordinates)
        """
        print("Called measure_calf_girth (Leg)")

        # Get leg mesh for this side
        leg_mesh = Leg._get_submesh(side, mesh)
        
        # Orient leg upright to z-axis
        from ....mesh.mesh import Mesh
        leg_mesh_copy = leg_mesh.copy()
        transform_matrix = Mesh.align_mesh_to_z_axis(leg_mesh_copy)
        
        # Get z bounds of the leg
        z_min = leg_mesh_copy.vertices[:, 2].min()
        z_max = leg_mesh_copy.vertices[:, 2].max()
        leg_height = z_max - z_min
        
        # Search between 15% and 50% of leg height
        search_z_min = z_min + leg_height * 0.15
        search_z_max = z_min + leg_height * 0.50
        
        # Create slice heights approximately 1 inch apart
        slice_step = get_geometry_config(mesh)["leg_slice_step"]
        z_heights = np.arange(search_z_min, search_z_max, slice_step)
                
        max_perimeter = 0.0
        best_slice = None
        best_z = None
        
        # Find the slice with maximum perimeter
        for z in z_heights:
            slice_2d = leg_mesh_copy.section(
                plane_origin=[0, 0, z],
                plane_normal=[0, 0, 1]
            )
            
            if slice_2d is not None:
                # Get perimeter of the 2D cross-section
                perimeter = slice_2d.length
                if perimeter > max_perimeter:
                    max_perimeter = perimeter
                    best_slice = slice_2d
                    best_z = z
        
        # If no valid slice found, return empty
        if max_perimeter == 0.0:
            empty_path = trimesh.load_path(np.array([[0, 0, 0]]))
            return (0.0, empty_path)
        
        # Get properly ordered vertices from the Path2D
        vertices_2d_ordered = best_slice.vertices
        if len(best_slice.entities) > 0:
            ordered_indices = []
            for entity in best_slice.entities:
                ordered_indices.extend(entity.points)
            seen = set()
            ordered_indices = [i for i in ordered_indices if not (i in seen or seen.add(i))]
            vertices_2d_ordered = vertices_2d_ordered[ordered_indices]
        
        # Convert to 3D in aligned coordinates
        vertices_3d_aligned = np.column_stack([
            vertices_2d_ordered[:, 0],
            vertices_2d_ordered[:, 1],
            np.full(len(vertices_2d_ordered), best_z)
        ])
        
        # Transform back to original coordinates
        inverse_transform = np.linalg.inv(transform_matrix)
        vertices_3d_original = trimesh.transform_points(vertices_3d_aligned, inverse_transform)
        
        # Create closed loop path
        indices = np.arange(len(vertices_3d_original) + 1)
        indices[-1] = 0
        entities = [trimesh.path.entities.Line(indices)]
        path_3d = trimesh.path.Path3D(entities=entities, vertices=vertices_3d_original)
        
        return (float(max_perimeter), path_3d)

    @staticmethod
    @cache
    def _measure_thigh_girth(mesh: trimesh.Trimesh, side: LEFT_OR_RIGHT):
        """
        Measure thigh girth at 75% of distance from ankle to hip.
        
        Returns
        -------
        tuple[float, trimesh.path.Path3D]
            (girth_value, path_in_original_coordinates)
        """
        print("Called measure_thigh_girth (Leg)")

        from ..trunk import Trunk
        
        # Get hip landmark for this side
        left_hip, right_hip = Trunk._locate_hips(mesh)
        hip = left_hip if side == 'left' else right_hip
        
        # Get ankle landmark for this side
        ankle = Leg._locate_ankle(mesh, side)
        
        # Calculate the z position for thigh measurement
        ankle_z = ankle[2]
        hip_z = hip[2]
        thigh_z = ankle_z + 0.75 * (hip_z - ankle_z)
        
        # Get the perimeter of cross-section at this z height
        # Note: thigh measurement is already in original coordinates (no alignment needed)
        leg_mesh = Leg._get_submesh(side, mesh)
        slice_2d = leg_mesh.section(
            plane_origin=[0, 0, thigh_z],
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
        
        # Convert to 3D by adding z coordinate
        vertices_3d = np.column_stack([
            vertices_2d_ordered[:, 0],
            vertices_2d_ordered[:, 1],
            np.full(len(vertices_2d_ordered), thigh_z)
        ])
        
        # Create closed loop path (already in original coordinates)
        indices = np.arange(len(vertices_3d) + 1)
        indices[-1] = 0
        entities = [trimesh.path.entities.Line(indices)]
        path_3d = trimesh.path.Path3D(entities=entities, vertices=vertices_3d)
        
        return (float(slice_2d.length), path_3d)

    # Helper static methods

    @staticmethod
    @cache
    def _identify_feet(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
        # This method assumes the mesh is oriented
        new_mesh = mesh.copy()
        
        kdtree = cKDTree(new_mesh.vertices)
        
        lower_half_vertices = new_mesh.vertices[new_mesh.vertices[:, 2] < 0, :]
        left_side = lower_half_vertices[lower_half_vertices[:, 0] < 0, :]
        left_foot = left_side[np.argmin(left_side, axis=0)[2], :]
        
        right_foot_estimate = left_foot.copy()
        right_foot_estimate[0] *= -1
        
        right_foot = new_mesh.vertices[kdtree.query(right_foot_estimate)[1], :]
        
        return left_foot, right_foot
    
