from functools import cache

import trimesh

from ..anatomical_region import Anatomical_Region


class Head(Anatomical_Region):
    """
    Head region segmentation, landmark detection, and measurement computation.
    
    The Head class handles extraction and analysis of the head from a full body mesh.
    It identifies the head by slicing above the shoulders and locates key facial
    landmarks (tip of nose) for measurements (head height from collar to scalp).
    
    Purpose
    -------
    This class solves the problem of automatically isolating and measuring the head
    region from body scans. Without this class, you would need to manually:
    - Determine where the neck/shoulders end and head begins
    - Slice the mesh to extract just the head
    - Locate facial landmarks
    - Compute head dimensions
    
    With this class, creating a Head instance automatically performs all these operations.
    
    Class Structure and Design Choices
    -----------------------------------
    **Simpler than other regions**:
    The Head class is architecturally simpler than Arm or Leg because:
    - No left/right distinction needed (there's only one head)
    - Fewer landmarks (just nose tip currently)
    - Simpler segmentation (single horizontal plane cut)
    
    This makes Head a good starting point for understanding the codebase.
    
    **Static method with @cache**:
    See Anatomical_Region docstring for the rationale behind this design pattern.
    
    **Runtime imports**:
    `from ..arms import Arm` appears inside methods because:
    - Head depends on Arm (to find shoulders for slicing plane)
    - Arm may depend on Head (for full body segmentation)
    - Runtime imports break circular dependency at import time
    
    **Shoulder-based slicing**:
    The head is extracted by:
    1. Finding both shoulder landmarks (from Arm class)
    2. Taking the maximum Z-coordinate (higher shoulder)
    3. Slicing above this plane to keep only the head
    
    This approach assumes:
    - Shoulders are the highest non-head points
    - Body is upright (Z-axis vertical)
    - Shoulder detection succeeds
    
    Attributes
    ----------
    body_mesh : trimesh.Trimesh
        The full body mesh (cleaned and oriented)
    
    Properties
    ----------
    mesh : trimesh.Trimesh
        The segmented head mesh (cached, computed on first access)
    landmarks : dict[str, np.ndarray]
        Key anatomical points:
        - "tip of nose": Most forward-protruding point of face
        Note: More landmarks (eyes, ears) could be added in future
    measurements : dict[str, float]
        Anthropometric measurements:
        - "collar to scalp length": Vertical distance from neck to top of head
        Note: More measurements (head circumference) could be added in future
    
    Examples
    --------
    >>> import trimesh
    >>> from body import Body  # doctest: +SKIP
    >>> body = Body("model_files/man.obj")  # doctest: +SKIP
    >>> head = body.parts["head"]  # doctest: +SKIP
    >>> head_height = head.measurements["collar to scalp length"]  # doctest: +SKIP
    >>> print(f"Head height: {head_height:.2f} cm")  # doctest: +SKIP
    Head height: 23.5 cm
    
    >>> nose_tip = head.landmarks["tip of nose"]  # doctest: +SKIP
    >>> print(f"Nose tip location: {nose_tip}")  # doctest: +SKIP
    Nose tip location: [0.2, 8.5, 62.3]
    
    Notes
    -----
    - Requires body mesh to be in standard orientation (Z-axis vertical)
    - Shoulder detection quality affects head segmentation accuracy
    - Currently limited landmarks and measurements (expandable in future)
    - All measurements are in the same units as the input mesh
    
    See Also
    --------
    Arm : Provides shoulder landmarks used for head segmentation
    Trunk : Provides collar landmark used for head measurements
    Anatomical_Region : Abstract base class defining the interface
    """

    def __init__(self, body_mesh: trimesh.Trimesh):
        print("Called __init__ (Head)")

        self.body_mesh = body_mesh
    
    @property
    def volume(self):
        print("Called volume (Head)")
        return self.mesh.volume
    
    @property
    def surface_area(self):
        print("Called surface_area (Head)")
        return self.mesh.area

    # Properties of Head

    # Vertex Indices (provided by `mesh`)
    @property
    def mesh(self):
        return Head._get_submesh(self.body_mesh)

    @staticmethod
    @cache
    def _get_submesh(mesh: trimesh.Trimesh):
        """
        Extract the head mesh by slicing above the shoulders.
        Uses the highest points of the arms (shoulders) as the cutting plane.
        """
        from ..arms import Arm
        
        # Get shoulder landmarks from both arms
        left_shoulder = Arm._locate_shoulder(mesh, "left")
        right_shoulder = Arm._locate_shoulder(mesh, "right")
        
        # Find the higher of the two shoulders
        max_shoulder_z = max(left_shoulder[2], right_shoulder[2])
        
        # Slice mesh to keep everything above the shoulders
        # Plane at shoulder height, normal pointing down (to keep upper part)
        head_mesh = mesh.slice_plane(
            plane_origin=[0, 0, max_shoulder_z],
            plane_normal=[0, 0, 1]
        )
        
        # Clean the mesh
        head_mesh.remove_unreferenced_vertices()
        head_mesh.fill_holes()
        
        return head_mesh

    # Landmarks

    @property
    def landmarks(self):
        return {
            "tip of nose": Head._locate_tip_of_nose(self.body_mesh)
        }

    @staticmethod
    @cache
    def _locate_tip_of_nose(mesh: trimesh.Trimesh):
        """
        The vertex v corresponding to argmax y from the subset of head vertices (above shoulders) where z is between 30% and 60% of the vertical distance from shoulder-height to scalp-height (max z).
        
        pseudo:
        get max of highest point of arms
        get vertices between 30% to 60% above shoulders and between top of head
        get vertex which has largest y
        return tip of nose
        """
        from ..arms import Arm
        import numpy as np
        
        print("Called locate_tip_of_nose (Head)")
        
        # Step 1: Get shoulder landmarks from both arms
        left_shoulder = Arm._locate_shoulder(mesh, "left")
        right_shoulder = Arm._locate_shoulder(mesh, "right")
        
        # Step 2: Find the higher of the two shoulders (max z)
        shoulder_height = max(left_shoulder[2], right_shoulder[2])
        
        # Step 3: Get the highest point (scalp) of the head
        vertices = np.asarray(mesh.vertices)
        scalp_height = vertices[:, 2].max()
        
        # Step 4: Calculate the vertical distance from shoulder to scalp
        vertical_distance = scalp_height - shoulder_height
        
        # Step 5: Define the z-range for nose detection (30% to 60% above shoulders)
        z_min = shoulder_height + 0.30 * vertical_distance
        z_max = shoulder_height + 0.60 * vertical_distance
        
        # Step 6: Filter vertices in the z-range
        in_range_mask = (vertices[:, 2] >= z_min) & (vertices[:, 2] <= z_max)
        vertices_in_range = vertices[in_range_mask]
        
        # Step 7: Find the vertex with the largest  y (most forward point)
        tip_of_nose_idx = np.argmax(vertices_in_range[:, 1])
        tip_of_nose = vertices_in_range[tip_of_nose_idx]
        
        return tip_of_nose

    # Measurements & Drawings

    @property
    def measurements(self):
        """Extract just the measurement values (first element of tuples)."""
        return {
            "collar to scalp length": Head._measure_collar_to_scalp_length(self.body_mesh)[0],
            "head circumference": Head._measure_head_circumference(self.body_mesh)[0],
        }

    @property
    def drawings(self):
        """Extract the 3D paths showing where measurements were taken (second element of tuples)."""
        return {
            "collar to scalp length": Head._measure_collar_to_scalp_length(self.body_mesh)[1],
            "head circumference": Head._measure_head_circumference(self.body_mesh)[1],
        }

    @staticmethod
    @cache
    def _measure_head_circumference(mesh: trimesh.Trimesh):
        """Measure the largest horizontal head section above the neck."""
        from ....utils.section_geometry import empty_measurement, slice_measurement
        import numpy as np

        print("Called measure_head_circumference (Head)")
        head_mesh = Head._get_submesh(mesh)
        z_min, z_max = head_mesh.bounds[:, 2]
        z_values = np.linspace(z_min + 0.25 * (z_max - z_min), z_min + 0.75 * (z_max - z_min), 20)
        measurements = [slice_measurement(head_mesh, z, "head") for z in z_values]
        measurements = [measurement for measurement in measurements if measurement[0] > 0]
        return max(measurements, key=lambda measurement: measurement[0]) if measurements else empty_measurement()

    @staticmethod
    @cache
    def _measure_collar_to_scalp_length(mesh: trimesh.Trimesh):
        """
        Measure vertical extent of head from collar to highest point.
        
        Returns
        -------
        tuple[float, trimesh.path.Path3D]\n            (length_value, line_segment_path_in_original_coordinates)
        """
        from ..trunk import Trunk
        import numpy as np
        
        print("Called measure_collar_to_scalp_length (Head)")
        
        # Step 1: Get collar landmark from trunk
        collar = Trunk._locate_collar(mesh)
        
        # Step 2: Get the highest point (scalp) of the head
        vertices = np.asarray(mesh.vertices)
        scalp_idx = np.argmax(vertices[:, 2])
        scalp = vertices[scalp_idx]
        
        # Step 4: Compute the differences in x and z (ignore y)
        dx = scalp[0] - collar[0]
        dz = scalp[2] - collar[2]
        
        # Step 5: Compute Euclidean distance in x–z plane
        collar_to_scalp_length = np.sqrt(dx**2 + dz**2)
        
        # # Debug print for verification
        # print(f"Collar point: {collar}")
        # print(f"Scalp point: {scalp}")
        # print(f"Computed collar to scalp length (||scalp - collar||_(x,z)) = {collar_to_scalp_length:.3f}")
        
        # Create Path3D line segment from collar to scalp
        path_vertices = np.array([collar, scalp])
        entities = [trimesh.path.entities.Line([0, 1])]
        path_3d = trimesh.path.Path3D(entities=entities, vertices=path_vertices)
        
        # Step 6: Ensure return type is float (not np.float64)
        return (float(collar_to_scalp_length), path_3d)

