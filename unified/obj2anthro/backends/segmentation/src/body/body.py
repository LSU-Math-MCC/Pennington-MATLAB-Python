import trimesh

from ..mesh import Mesh
from .anatomical_regions.anatomical_region import (
    ANATOMICAL_REGION,
    DEFAULT_GEOMETRY_CONFIG,
    Anatomical_Region,
    build_geometry_config,
)
from .anatomical_regions import Arm, Head, Leg, Trunk

class Body(Mesh):
    """
    Main class for processing 3D human body scans and extracting anthropometric measurements.
    
    The Body class is the primary entry point for analyzing body scan meshes. It loads a mesh file,
    cleans and orients it, segments it into anatomical regions (head, trunk, arms, legs), and
    computes landmarks and measurements for each region.
    
    Purpose
    -------
    This class solves the problem of extracting standardized body measurements from 3D scans.
    Without this class, you would need to manually:
    - Clean mesh artifacts (duplicate faces, holes, degenerate triangles)
    - Orient the mesh to a standard coordinate system
    - Segment the mesh into body parts
    - Locate anatomical landmarks (shoulders, hips, armpits, etc.)
    - Compute measurements (lengths, circumferences, volumes)
    
    With this class, all of these steps happen automatically when you create a Body instance.
    
    Class Structure and Design Choices
    -----------------------------------
    **Inheritance from Mesh**:
    The Body class inherits from Mesh to gain mesh cleaning and orientation capabilities.
    This design allows Body to focus on anatomical segmentation while reusing generic mesh
    operations from the base class.
    
    **Cached Properties**:
    Each anatomical region (Arm, Leg, etc.) uses @cache decorators extensively. This means
    expensive operations like mesh segmentation and landmark detection are computed once and
    reused. This is critical because:
    - Mesh slicing operations are computationally expensive
    - Landmark detection involves ray casting and convexity analysis
    - Different measurements may need the same intermediate results
    
    **Separation of Concerns**:
    The Body class delegates anatomical knowledge to specialized classes (Arm, Head, Leg, Trunk).
    Each region knows how to:
    - Extract its portion from the full body mesh
    - Find its specific landmarks
    - Compute its measurements
    
    This architecture allows you to modify one region without affecting others.
    
    **Why dictionaries for landmarks/measurements**:
    Using dictionaries with string keys (e.g., body.landmarks["trunk"]["crotch"]) instead of
    object attributes (e.g., body.trunk.crotch) provides:
    - Uniform interface across all body parts
    - Easy iteration over all measurements
    - Simple export to JSON or other formats
    
    Attributes
    ----------
    mesh : trimesh.Trimesh
        The cleaned, oriented, and normalized full body mesh
    parts : dict[str, Anatomical_Region]
        Dictionary mapping region names to their instances:
        "head", "trunk", "left arm", "right arm", "left leg", "right leg"
    subregion_meshes : dict[str, trimesh.Trimesh]
        Dictionary mapping region names to their segmented meshes
    landmarks : dict[str, dict]
        Nested dictionary of anatomical landmarks by body part
        Example: landmarks["trunk"]["crotch"] returns a 3D point
    measurements : dict[str, dict]
        Nested dictionary of body measurements by body part
        Example: measurements["left arm"]["left arm length"] returns a float
    
    Examples
    --------
    Basic usage:
    
    >>> from body import Body  # doctest: +SKIP
    >>> body = Body("model_files/man.obj")  # doctest: +SKIP
    >>> # Access measurements
    >>> arm_length = body.measurements["left arm"]["left arm length"]  # doctest: +SKIP
    >>> print(f"Left arm: {arm_length:.2f} units")  # doctest: +SKIP
    Left arm: 45.23 units
    
    >>> # Access landmarks
    >>> crotch_point = body.landmarks["trunk"]["crotch"]  # doctest: +SKIP
    >>> print(f"Crotch position: {crotch_point}")  # doctest: +SKIP
    Crotch position: [0.0, -2.5, 15.3]
    
    >>> # Access body part meshes
    >>> trunk_mesh = body.subregion_meshes["trunk"]  # doctest: +SKIP
    >>> print(f"Trunk volume: {trunk_mesh.volume:.2f}")  # doctest: +SKIP
    Trunk volume: 123.45
    
    >>> # Iterate over all measurements
    >>> for part_name, measurements in body.measurements.items():  # doctest: +SKIP
    ...     print(f"{part_name}:")
    ...     for measure_name, value in measurements.items():
    ...         print(f"  {measure_name}: {value:.2f}")
    
    Notes
    -----
    - Input mesh should be a standing human figure in A-pose
    - Mesh quality significantly affects results - clean, watertight meshes work best
    - The class assumes an A pose, where arms are diagonally pointed toward bottom left and right, and legs are slightly apart
    - All coordinates are normalized internally (mean=0, std=1) for numerical stability
    - Measurements are in the same units as the input mesh
    
    See Also
    --------
    Mesh : Base class providing mesh cleaning and orientation
    Anatomical_Region : Abstract base class for body part implementations
    """

    def __init__(self, mesh_file, units="dm"):
        """
        Load a mesh file and initialize a fully analyzed human body model.
        
        This constructor performs the complete processing pipeline:
        1. Loads the mesh from the specified file
        2. Cleans and normalizes the mesh (via parent Mesh class)
        3. Orients the mesh to standard coordinates (Z-up, X-left/right, Y-front/back)
        4. Segments the mesh into anatomical regions (head, trunk, arms, legs)
        5. Computes landmarks and measurements for each region
        
        Parameters
        ----------
        mesh_file : str or path-like
            Path to the 3D mesh file. Supported formats include OBJ, PLY, STL,
            and any other format supported by trimesh.load_mesh().
        
        Examples
        --------
        >>> from body import Body  # doctest: +SKIP
        >>> body = Body("model_files/man.obj")  # doctest: +SKIP
        >>> print(body.parts.keys())  # doctest: +SKIP
        dict_keys(['head', 'trunk', 'left arm', 'right arm', 'left leg', 'right leg'])
        
        Notes
        -----
        - The mesh file should contain a standing human figure in approximately upright pose
        - Processing time depends on mesh complexity (typically 1-10 seconds for ~10k vertices)
        - After initialization, all body parts, landmarks, and measurements are immediately
          accessible via the instance attributes
        
        See Also
        --------
        Mesh.__init__ : Parent class constructor handling mesh cleaning and normalization
        """
        if isinstance(mesh_file, trimesh.Trimesh):
            mesh = mesh_file.copy()
            self.geometry_config = mesh.metadata.get("geometry_config", DEFAULT_GEOMETRY_CONFIG)
        else:
            mesh = trimesh.load_mesh(mesh_file)
            self.geometry_config = build_geometry_config(mesh, units)
            mesh.vertices *= self.geometry_config["unit_to_internal"]
            mesh.metadata["geometry_config"] = self.geometry_config

        super().__init__(mesh)
        self.mesh = self.orient_mesh(mesh)
        self.mesh.metadata["geometry_config"] = self.geometry_config
        
        # Body parts

        self.parts: dict[ANATOMICAL_REGION, Anatomical_Region] = {
            "head": Head(self.mesh),
            "trunk": Trunk(self.mesh),
            "left arm": Arm(self.mesh, 'left'),
            "right arm": Arm(self.mesh, 'right'),
            "left leg": Leg(self.mesh, "left"),
            "right leg": Leg(self.mesh, "right"),
        }

        # Landmarks & measurements

        self.subregion_meshes = { key: bp.mesh for key, bp in self.parts.items() }
        self.landmarks = { key: bp.landmarks for key, bp in self.parts.items() }
        self.measurements = { key: bp.measurements for key, bp in self.parts.items() }
        self.drawings = { key: bp.drawings for key, bp in self.parts.items() }
        
    @property 
    def units(self):
        """
        The units of measurement for this body mesh.
        
        This property provides a stable interface to the underlying mesh's unit system,
        following the principle that Body is separated from Mesh mechanics. While the
        Mesh class handles the underlying implementation details, Body exposes units
        through this unchanging interface.
        
        Returns
        -------
        str or None
            The unit system of the mesh (e.g., 'meters', 'millimeters'), or None
            if units are not defined in the source file.
        
        Notes
        -----
        This property, along with `volume` and `surface_area`, exemplifies the
        separation between Body (anatomical concerns) and Mesh (geometric mechanics).
        These properties provide a stable, unchanging policy interface regardless of
        how the underlying Mesh implementation may evolve.
        """
        return self.mesh.units

    @property
    def volume(self):
        """
        The total volume of the body mesh.
        
        This property provides a stable interface to the underlying mesh's volume
        calculation, following the principle that Body is separated from Mesh mechanics.
        While the Mesh class handles the underlying computational details, Body exposes
        volume through this unchanging interface.
        
        Returns
        -------
        float
            The volume enclosed by the mesh, in cubic units matching the mesh's
            coordinate system.
        
        Notes
        -----
        This property, along with `units` and `surface_area`, exemplifies the
        separation between Body (anatomical concerns) and Mesh (geometric mechanics).
        These properties provide a stable, unchanging policy interface regardless of
        how the underlying Mesh implementation may evolve.
        
        For accurate results, the mesh should be watertight (closed surface).
        """
        return -self.mesh.volume
    
    @property
    def surface_area(self):
        """
        The total surface area of the body mesh.
        
        This property provides a stable interface to the underlying mesh's area
        calculation, following the principle that Body is separated from Mesh mechanics.
        While the Mesh class handles the underlying computational details, Body exposes
        surface area through this unchanging interface.
        
        Returns
        -------
        float
            The total surface area of all mesh faces, in square units matching
            the mesh's coordinate system.
        
        Notes
        -----
        This property, along with `units` and `volume`, exemplifies the
        separation between Body (anatomical concerns) and Mesh (geometric mechanics).
        These properties provide a stable, unchanging policy interface regardless of
        how the underlying Mesh implementation may evolve.
        """
        return self.mesh.area
