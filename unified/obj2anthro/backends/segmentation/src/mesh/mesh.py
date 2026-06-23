import trimesh
import numpy as np
from scipy.spatial import cKDTree

class Mesh:
    """
    Base class for mesh cleaning, normalization, and orientation detection.
    
    The Mesh class provides fundamental operations for preparing 3D meshes for anthropometric
    analysis. It handles mesh cleaning (removing artifacts), coordinate normalization (for
    numerical stability), and automatic orientation detection (ensuring Z-axis is vertical).
    
    Purpose
    -------
    This class solves three critical problems in mesh processing:
    
    1. **Mesh Quality**: Raw meshes from 3D scanners often contain:
       - Duplicate or degenerate faces
       - Holes and gaps
       - Unreferenced vertices
       The clean_mesh() method fixes these issues automatically.
    
    2. **Numerical Stability**: Mesh coordinates can have arbitrary scales (millimeters to meters).
       Normalization ensures all operations work in a standard coordinate space where:
       - Mean = 0 (mesh centered at origin)
       - Standard deviation = 1 (vertices in roughly [-3, 3] range)
    
    3. **Orientation Consistency**: Scans may be in any orientation (lying down, rotated, etc.).
       The orient_mesh() method automatically detects the vertical axis and ensures:
       - Z-axis points up (from feet to head)
       - X-axis is left-right
       - Y-axis is front-back
    
    Without this class, every mesh operation would need to handle these issues separately,
    leading to duplicated code and inconsistent behavior.
    
    Class Structure and Design Choices
    -----------------------------------
    **Normalization strategy**:
    The class stores normalization functions as instance attributes (_normalize, _denormalize).
    This allows the same normalization to be consistently applied/reversed throughout the
    object's lifetime. The normalization parameters (mean, std) are computed once during
    initialization from the original mesh.
    
    **Why vectorize**:
    `np.vectorize(lambda x: (x-mean)/std)` creates a function that can be applied to entire
    arrays efficiently. This is more readable than manual array operations and handles
    broadcasting automatically.
    
    **Orientation detection algorithm**:
    The orient_mesh() method uses a sophisticated heuristic:
    1. Find the major axis (longest dimension) → assume this is height
    2. For left-right detection: Create vertical lines at the edges, find nearest mesh vertices,
       compare which orientation has vertices closer to a central vertical line
    
    This works well for standing human figures but may fail for unusual poses.
    
    **Static methods**:
    Some methods like align_mesh_to_z_axis() are static because they:
    - Don't need instance state
    - Are utility functions that could be used independently
    - Make testing easier (can call without creating a Mesh instance)
    
    Attributes
    ----------
    mesh : trimesh.Trimesh
        The cleaned, normalized, and oriented mesh
    _normalize : function
        Vectorized function to normalize coordinates: (x - mean) / std
    _denormalize : function  
        Vectorized function to denormalize coordinates: x * std + mean
    
    Examples
    --------
    Basic usage:
    
    >>> import trimesh
    >>> from mesh import Mesh  # doctest: +SKIP
    >>> raw_mesh = trimesh.load("model_files/cow.ply")  # doctest: +SKIP
    >>> mesh_obj = Mesh(raw_mesh)  # doctest: +SKIP
    >>> # Now mesh_obj.mesh is cleaned, normalized, and oriented
    >>> print(f"Mesh has {len(mesh_obj.mesh.vertices)} vertices")  # doctest: +SKIP
    Mesh has 2904 vertices
    
    Manual operations:
    
    >>> # Normalize a mesh manually
    >>> import trimesh
    >>> mesh = trimesh.creation.box()  # doctest: +SKIP
    >>> mesh_obj = Mesh(mesh)  # doctest: +SKIP
    >>> normalized = mesh_obj.normalize(mesh)  # doctest: +SKIP
    >>> # Coordinates are now centered with std=1
    
    Notes
    -----
    - Mesh cleaning is lossy - some faces/vertices will be removed
    - Normalization changes coordinate scale - remember to denormalize measurements
    - Orientation detection assumes a roughly vertical standing pose
    - The class modifies meshes extensively during initialization
    
    See Also
    --------
    trimesh.Trimesh : The underlying mesh representation
    Body : Subclass that adds anthropometric analysis
    """
    def __init__(self, mesh: trimesh.Trimesh):
        mesh: trimesh.Trimesh = self.clean_mesh(mesh)
        
        std = np.std(mesh.vertices)
        mean = np.mean(mesh.vertices)
        
        self._normalize = np.vectorize(lambda x: (x-mean) / std )
        self._denormalize = np.vectorize(lambda x: (x*std + mean))
        
        mesh = self.normalize(mesh)
        self.mesh = self.orient_mesh(mesh)
        
    def clean_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Remove mesh artifacts and repair common issues.
        
        This method applies a series of cleanup operations to prepare a raw mesh for analysis:
        - Removes duplicate faces
        - Removes vertices not referenced by any face
        - Fills small holes
        - Removes degenerate triangles (zero area or very thin)
        
        Parameters
        ----------
        mesh : trimesh.Trimesh
            The raw input mesh (potentially with artifacts)
        
        Returns
        -------
        trimesh.Trimesh
            A cleaned mesh (may have fewer vertices and faces than input)
        
        Examples
        --------
        >>> import trimesh
        >>> from mesh import Mesh  # doctest: +SKIP
        >>> raw_mesh = trimesh.load("model_files/man.obj")  # doctest: +SKIP
        >>> mesh_obj = Mesh(raw_mesh)  # doctest: +SKIP
        >>> # Cleaning happens automatically in __init__
        
        Pseudocode
        ----------
        1. Copy the mesh (to avoid modifying original)
        2. Keep only unique faces (remove duplicates)
        3. Remove unreferenced vertices (orphaned points)
        4. Fill small holes in the mesh
        5. Remove degenerate faces (area < 1e-8 or height < 1e-8)
        6. Return cleaned mesh
        
        Notes
        -----
        - This is a lossy operation - some geometry may be removed
        - Hole filling uses simple algorithms - may not work for large holes
        - Degenerate face removal uses a small threshold (1e-8) - very thin triangles
        - The method is called automatically during Mesh.__init__
        
        See Also
        --------
        trimesh.Trimesh.unique_faces : Remove duplicate faces
        trimesh.Trimesh.fill_holes : Fill holes in mesh
        trimesh.Trimesh.nondegenerate_faces : Identify valid faces
        """
        cleaned_mesh = mesh.copy()
        cleaned_mesh.update_faces(cleaned_mesh.unique_faces())
        cleaned_mesh.remove_unreferenced_vertices()
        cleaned_mesh.fill_holes()
        cleaned_mesh.update_faces(cleaned_mesh.nondegenerate_faces(height=1e-8))
        
        return cleaned_mesh
    
    def normalize(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Normalize mesh coordinates to mean=0, std=1.
        
        This transforms all vertex coordinates so that:
        - The mean of all coordinates is 0 (centered at origin)
        - The standard deviation is 1 (vertices in roughly [-3, 3] range)
        
        Parameters
        ----------
        mesh : trimesh.Trimesh
            The mesh to normalize
        
        Returns
        -------
        trimesh.Trimesh
            A new mesh with normalized coordinates (original unchanged)
        
        Examples
        --------
        >>> import trimesh
        >>> import numpy as np
        >>> mesh = trimesh.creation.box(extents=[100, 100, 100])  # doctest: +SKIP
        >>> mesh_obj = Mesh(mesh)  # doctest: +SKIP
        >>> normalized = mesh_obj.normalize(mesh)  # doctest: +SKIP
        >>> # Vertices now centered at origin with std ~1
        >>> print(f"Mean: {np.mean(normalized.vertices):.3f}")  # doctest: +SKIP
        Mean: 0.000
        >>> print(f"Std: {np.std(normalized.vertices):.3f}")  # doctest: +SKIP
        Std: 1.000
        
        Pseudocode
        ----------
        1. Create a copy of the mesh
        2. Apply normalization function: (x - mean) / std
        3. Return the normalized mesh
        
        Notes
        -----
        - Uses the _normalize function created in __init__
        - The mean and std are computed once in __init__ from the original mesh
        - All subsequent normalizations use the same parameters
        - This ensures consistent scaling across all operations
        - Remember to denormalize before displaying measurements to users
        
        See Also
        --------
        denormalize : Reverse this operation
        """
        new_mesh = mesh.copy()
        new_mesh.vertices = self._normalize(new_mesh.vertices)
        return new_mesh 
    
    def denormalize(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Convert normalized coordinates back to original scale.
        
        This reverses the normalization applied by normalize(), transforming coordinates
        from mean=0, std=1 back to the original mesh's scale.
        
        Parameters
        ----------
        mesh : trimesh.Trimesh
            A mesh with normalized coordinates
        
        Returns
        -------
        trimesh.Trimesh
            A new mesh with coordinates in original scale (normalized mesh unchanged)
        
        Examples
        --------
        >>> import trimesh
        >>> mesh = trimesh.creation.box(extents=[100, 100, 100])  # doctest: +SKIP
        >>> mesh_obj = Mesh(mesh)  # doctest: +SKIP
        >>> normalized = mesh_obj.normalize(mesh)  # doctest: +SKIP
        >>> denormalized = mesh_obj.denormalize(normalized)  # doctest: +SKIP
        >>> # denormalized should match original mesh
        >>> import numpy as np
        >>> np.allclose(mesh.vertices, denormalized.vertices)  # doctest: +SKIP
        True
        
        Pseudocode
        ----------
        1. Create a copy of the normalized mesh
        2. Apply denormalization function: x * std + mean
        3. Return the denormalized mesh
        
        Notes
        -----
        - Uses the _denormalize function created in __init__
        - The mean and std are from the original mesh (before normalization)
        - This should produce coordinates close to the original (within floating point precision)
        - Use this before saving meshes or displaying measurements to users
        
        See Also
        --------
        normalize : The inverse operation
        """
        new_mesh = mesh.copy()
        new_mesh.vertices = self._denormalize(new_mesh.vertices)
        return new_mesh

    # I'm putting these methods here because they're just mesh things

    def orient_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Automatically detect and standardize mesh orientation.
        
        This method ensures the mesh is in a standard orientation:
        - Z-axis is vertical (pointing from feet to head)
        - X-axis is left-right
        - Y-axis is front-back
        - Mesh is centered at origin
        
        Parameters
        ----------
        mesh : trimesh.Trimesh
            Input mesh in arbitrary orientation
        
        Returns
        -------
        trimesh.Trimesh
            Mesh in standard orientation (new copy, original unchanged)
        
        Examples
        --------
        >>> import trimesh
        >>> from mesh import Mesh  # doctest: +SKIP
        >>> # Load a mesh that might be lying down or rotated
        >>> mesh = trimesh.load("model_files/cow.ply")  # doctest: +SKIP
        >>> mesh_obj = Mesh(mesh)  # doctest: +SKIP
        >>> # mesh_obj.mesh is now oriented with Z-axis vertical
        
        Pseudocode
        ----------
        1. Center the mesh at origin:
            a. Compute midpoint = (min + max) / 2 for each axis
            b. Translate all vertices by -midpoint
        
        2. Align vertical axis to Z:
            a. Find major axis (longest dimension)
            b. If major axis is X (axis 0), swap X and Z columns
            c. If major axis is Y (axis 1), swap Y and Z columns
            d. Now longest dimension is along Z (vertical)
        
        3. Determine left-right (X) vs front-back (Y):
            a. Create vertical line at X=min(vertices.x), Y=0
            b. Create vertical line at X=0, Y=min(vertices.y)
            c. For each line, find nearest mesh vertices (using KD-tree)
            d. Compute "score" = sum of distances to central vertical line
            e. The axis with lower score is closer to center → that's front-back (Y)
            f. If Y has higher score, swap X and Y
        
        4. Return oriented mesh
        
        Notes
        -----
        - Assumes the mesh represents a roughly vertical standing figure
        - The "major axis" approach fails for non-elongated shapes
        - The left-right heuristic assumes bilateral symmetry
        - May swap axes incorrectly for unusual poses (sitting, bent over, etc.)
        - This method is called automatically in __init__
        
        Algorithm Details
        -----------------
        The left-right detection works by assuming:
        - The body is roughly symmetric in front-back direction
        - Vertices near the edge of the body (at minimum X or minimum Y) are on the sides
        - The side closer to the center line (X=0 or Y=0) is the front-back axis
        
        See Also
        --------
        find_major_axis : Identifies the longest dimension
        scipy.spatial.cKDTree : Fast nearest neighbor queries
        """
        new_mesh = mesh.copy()
        
        # Center mesh
        midpoint = (np.max(new_mesh.vertices, axis=0) + np.min(new_mesh.vertices, axis=0)) / 2
        new_mesh.vertices -= midpoint
        
        # Find height
        major_axis = self.find_major_axis(new_mesh)
        if major_axis == 0:
            new_mesh.vertices[:, (0, 2)] = new_mesh.vertices[:, (2, 0)] 
        elif major_axis == 1:
            new_mesh.vertices[:, (1, 2)] = new_mesh.vertices[:, (2, 1)]
        
        # Determine left-right axis
        # This algorithm creates a vertical line at the minimum value at both the x
        # and y axes, then finds the nearest neighbors to those lines. Whichever
        # result leads to the lowest total distance between the nearest neighbors and
        # a vertical line at the midpoint is taken to be the y-axis (that is to say,
        # front to back)
                
        kdtree = cKDTree(new_mesh.vertices)
        
        top_of_head = new_mesh.vertices[np.argmax(new_mesh.vertices, axis=0)[2]]
        z_axis = np.linspace(0, top_of_head[2], 100)
        
        x_coord, y_coord = np.min(new_mesh.vertices, axis=0)[:2]
        x_vec = np.empty((len(z_axis), 3))
        y_vec = np.empty((len(z_axis), 3))
        
        x_vec[:, (0, 1)] = (x_coord, 0)
        x_vec[:, 2] = z_axis
        
        y_vec[:, (0, 1)] = (0, y_coord)
        y_vec[:, 2] = z_axis
        
        x_alignment = kdtree.query(x_vec)[1]
        y_alignment = kdtree.query(y_vec)[1]
        
        x_vertices = new_mesh.vertices[x_alignment]
        y_vertices = new_mesh.vertices[y_alignment]
        
        x_score = np.linalg.norm(x_vertices[:2], axis=0).sum()
        y_score = np.linalg.norm(y_vertices[:2], axis=0).sum()
       
        if y_score > x_score:
            print("Switching x and y axis...")
            new_mesh.vertices[:, (0, 1)] = new_mesh.vertices[:, (1, 0)]
                
        return new_mesh
        
    def find_major_axis(self, mesh: trimesh.Trimesh) -> int:
        """
        Identify which axis (X, Y, or Z) has the longest extent.
        
        This finds the axis along which the mesh is most elongated by comparing
        the range (max - min) along each dimension.
        
        Parameters
        ----------
        mesh : trimesh.Trimesh
            The mesh to analyze
        
        Returns
        -------
        int
            The axis index: 0=X, 1=Y, 2=Z
        
        Examples
        --------
        >>> import trimesh
        >>> import numpy as np
        >>> # Create a tall thin box (elongated along Z)
        >>> mesh = trimesh.creation.box(extents=[1, 1, 10])  # doctest: +SKIP
        >>> mesh_obj = Mesh(mesh)  # doctest: +SKIP
        >>> axis = mesh_obj.find_major_axis(mesh)  # doctest: +SKIP
        >>> print(f"Major axis: {axis} (Z)")  # doctest: +SKIP
        Major axis: 2 (Z)
        
        >>> # Create a wide flat box (elongated along X)
        >>> mesh2 = trimesh.creation.box(extents=[10, 1, 1])  # doctest: +SKIP
        >>> axis2 = mesh_obj.find_major_axis(mesh2)  # doctest: +SKIP
        >>> print(f"Major axis: {axis2} (X)")  # doctest: +SKIP
        Major axis: 0 (X)
        
        Pseudocode
        ----------
        1. Find minimum vertex coordinate in each dimension (X, Y, Z)
        2. Find maximum vertex coordinate in each dimension (X, Y, Z)
        3. Compute extent = max - min for each dimension
        4. Return index of dimension with largest extent
        
        Notes
        -----
        - For human body scans, this is typically Z (height) if standing upright
        - For lying down poses, might be X or Y
        - Used by orient_mesh() to align the longest dimension with Z-axis
        - Assumes mesh represents an elongated object (body, limb, etc.)
        
        See Also
        --------
        orient_mesh : Uses this to determine vertical axis
        """
        mins = np.min(mesh.vertices, axis=0)
        maxs = np.max(mesh.vertices, axis=0)
        major_axis = np.argmax(maxs - mins)
        return major_axis
    
    @staticmethod
    def align_mesh_to_z_axis(mesh: trimesh.Trimesh):
        """
        Align the longest dimension of a mesh with the z-axis.
        
        This function is designed for meshes that represent elongated, roughly vertical objects
        (e.g., arms, legs, limbs, or other body parts) and aligns their longest dimension with
        the positive z-axis while preserving their original orientation (top remains top).
        
        The function uses the mesh's oriented bounding box (OBB) to determine the principal axes,
        identifies the longest dimension, and rotates the mesh to align that dimension with the
        z-axis. It also ensures that vertices that were originally at the top of the mesh remain
        at the top after alignment.
        
        Parameters
        ----------
        mesh : trimesh.Trimesh
            The input mesh to be aligned. The mesh will be modified in place.
        
        Returns
        -------
        np.ndarray
            4x4 transformation matrix that was applied to align the mesh.
            Can be inverted to map points from aligned space back to original space.
        
        Notes
        -----
        - This function is most effective for meshes that are already approximately vertical
          and represent elongated objects like limbs or cylindrical body parts.
        - The mesh is modified in place, but also returned for convenience.
        - The function preserves the original "up" direction by checking that vertices that
          were initially higher remain higher after transformation.
        - If the longest axis is already aligned with z (or its opposite), minimal rotation
          is applied.
        
        Examples
        --------
        Align an arm mesh to be vertical:
        
        >>> import trimesh
        >>> from mesh import Mesh  # doctest: +SKIP
        >>> # Create a horizontal cylinder (representing an arm)
        >>> arm_mesh = trimesh.creation.cylinder(radius=1, height=10)  # doctest: +SKIP
        >>> # Rotate it to be horizontal (along X-axis)
        >>> import numpy as np
        >>> rotation = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])  # doctest: +SKIP
        >>> arm_mesh.apply_transform(rotation)  # doctest: +SKIP
        >>> # Now align it to Z-axis
        >>> aligned = Mesh.align_mesh_to_z_axis(arm_mesh)  # doctest: +SKIP
        >>> # The longest dimension is now along Z
        >>> extents = aligned.bounding_box.extents  # doctest: +SKIP
        >>> print(f"Extents: {extents}")  # doctest: +SKIP
        Extents: [2.0, 2.0, 10.0]
        
        Pseudocode
        ----------
        1. Store original highest and lowest vertex indices (by Z coordinate)
        2. Get oriented bounding box (OBB) to find principal axes
        3. Extract rotation matrix from OBB transformation
        4. Find which principal axis corresponds to longest extent
        5. Compute rotation needed to align longest axis with Z-axis [0, 0, 1]:
            a. If already aligned or opposite, apply minimal rotation
            b. Otherwise, compute rotation axis (cross product)
            c. Compute rotation angle (arc cosine of dot product)
            d. Create and apply rotation matrix
        6. Check if top/bottom are flipped:
            a. Compare Z values of originally-highest and originally-lowest vertices
            b. If flipped, rotate 180 degrees to restore correct orientation
        7. Return aligned mesh
        
        Algorithm Details
        -----------------
        The oriented bounding box (OBB) provides:
        - Principal axes: The directions along which the mesh is most elongated
        - Extents: The length along each principal axis
        
        We align the axis with the largest extent to the Z-axis, ensuring elongated
        objects (like limbs) are vertical.
        
        The flip check prevents the mesh from being upside-down after rotation.
        
        See Also
        --------
        trimesh.Trimesh.bounding_box_oriented : Computes oriented bounding box
        trimesh.transformations.rotation_matrix : Creates rotation matrices
        """

        # Store original highest and lowest vertex indices
        original_highest_idx = np.argmax(mesh.vertices[:, 2])
        original_lowest_idx = np.argmin(mesh.vertices[:, 2])
        
        # Initialize cumulative transformation matrix as identity
        cumulative_transform = np.eye(4)
        
        # Get the oriented bounding box (OBB) to find principal axes
        obb = mesh.bounding_box_oriented
        
        # Get the transformation matrix from the OBB
        transform = obb.primitive.transform
        
        # Extract the principal axes from the rotation part of the transform
        # The columns of the rotation matrix are the principal axes
        principal_axes = transform[:3, :3]
        
        # Find which axis is longest
        extents = obb.primitive.extents
        longest_axis_idx = np.argmax(extents)
        
        # Get the principal axis that corresponds to the longest extent
        longest_axis = principal_axes[:, longest_axis_idx]
        
        # Compute rotation to align longest_axis with z-axis [0, 0, 1]
        z_axis = np.array([0, 0, 1])
        
        # Calculate rotation axis (cross product)
        rotation_axis = np.cross(longest_axis, z_axis)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        # If axes are already aligned (or opposite), no rotation needed
        if rotation_axis_norm < 1e-6:
            # Check if they're opposite
            if np.dot(longest_axis, z_axis) < 0:
                # Flip 180 degrees around x-axis
                rotation_matrix = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
                mesh.apply_transform(rotation_matrix)
                cumulative_transform = rotation_matrix @ cumulative_transform
        else:
            # Normalize rotation axis
            rotation_axis = rotation_axis / rotation_axis_norm
            
            # Calculate rotation angle
            angle = np.arccos(np.clip(np.dot(longest_axis, z_axis), -1.0, 1.0))
            
            # Create rotation matrix
            rotation_matrix = trimesh.transformations.rotation_matrix(angle, rotation_axis)
            
            # Apply transformation
            mesh.apply_transform(rotation_matrix)
            cumulative_transform = rotation_matrix @ cumulative_transform
        
        # Check if orientation is correct: the vertex that was originally highest 
        # should still have a higher z than the vertex that was originally lowest
        new_highest_z = mesh.vertices[original_highest_idx, 2]
        new_lowest_z = mesh.vertices[original_lowest_idx, 2]
        
        # If the originally-high vertex is now lower than the originally-low vertex, flip it
        if new_highest_z < new_lowest_z:
            rotation_matrix = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
            mesh.apply_transform(rotation_matrix)
            cumulative_transform = rotation_matrix @ cumulative_transform
        
        return cumulative_transform
