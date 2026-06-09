from functools import cache

import trimesh

from .girth_levels import hip_section, level_measurement, trunk_girth_levels
from .landmarks import (
    locate_armpits as locate_armpits_impl,
    locate_collar as locate_collar_impl,
    locate_crotch as locate_crotch_impl,
    locate_hips as locate_hips_impl,
)
from .measurements import (
    body_without_arms as body_without_arms_impl,
    get_submesh as get_submesh_impl,
    measure_chest_circumference as measure_chest_circumference_impl,
    measure_crotch_height as measure_crotch_height_impl,
    measure_hip_circumference as measure_hip_circumference_impl,
    measure_stomach_peak_circumference as measure_stomach_peak_circumference_impl,
    measure_trunk_length as measure_trunk_length_impl,
    measure_waist_circumference as measure_waist_circumference_impl,
)

from ..anatomical_region import Anatomical_Region


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
        - "chest circumference": Girth at nipple/bust tape level
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
    - Chest is searched as the nipple/bust tape level below the axilla, with a
      full-mesh fallback when arm removal trims the bust.
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

    # Landmarks # TODO: Might want to make these properties so the access is simpler, but not required. This goes for all landmarks and measurements in src actually

    @property
    def landmarks(self):
        return {
            "crotch": Trunk._locate_crotch(self.body_mesh),
            "armpits": Trunk._locate_armpits(self.body_mesh),         # TODO: split into left and right
            "hips": Trunk._locate_hips(self.body_mesh),               # TODO: same, split left/right
            "collar": Trunk._locate_collar(self.body_mesh)
        }

    # Measurements & Drawings

    def _measurement_results(self):
        return {
            "crotch height": Trunk._measure_crotch_height(self.body_mesh),
            "hip circumference": Trunk._measure_hip_circumference(self.body_mesh),
            "chest circumference": Trunk._measure_chest_circumference(self.body_mesh),
            "waist circumference": Trunk._measure_waist_circumference(self.body_mesh),
            "stomach peak circumference": Trunk._measure_stomach_peak_circumference(self.body_mesh),
            "trunk length": Trunk._measure_trunk_length(self.body_mesh),
        }

    @property
    def measurements(self):
        """Extract just the measurement values (first element of tuples)."""
        return {name: result[0] for name, result in self._measurement_results().items()}

    @property
    def drawings(self):
        """Extract the 3D paths showing where measurements were taken (second element of tuples)."""
        return {name: result[1] for name, result in self._measurement_results().items()}


def _install_trunk_impl():
    cached = lambda fn: staticmethod(cache(fn))
    Trunk._get_submesh = cached(get_submesh_impl)
    Trunk._locate_crotch = cached(locate_crotch_impl)
    Trunk._body_without_arms = staticmethod(body_without_arms_impl)
    Trunk._trunk_girth_levels = cached(lambda mesh: trunk_girth_levels(mesh, Trunk))
    Trunk._level_measurement = staticmethod(lambda mesh, name: level_measurement(mesh, name, Trunk))
    Trunk._hip_section = staticmethod(lambda mesh, crotch_point=None: hip_section(mesh, Trunk, crotch_point))
    Trunk._locate_armpits = cached(lambda mesh: locate_armpits_impl(mesh, Trunk))
    Trunk._locate_hips = cached(lambda mesh: locate_hips_impl(mesh, Trunk))
    Trunk._locate_collar = cached(lambda mesh: locate_collar_impl(mesh, Trunk))
    Trunk._measure_crotch_height = cached(lambda mesh: measure_crotch_height_impl(mesh, Trunk))
    Trunk._measure_hip_circumference = cached(lambda mesh: measure_hip_circumference_impl(mesh, Trunk))
    Trunk._measure_chest_circumference = cached(lambda mesh: measure_chest_circumference_impl(mesh, Trunk))
    Trunk._measure_waist_circumference = cached(lambda mesh: measure_waist_circumference_impl(mesh, Trunk))
    Trunk._measure_stomach_peak_circumference = cached(lambda mesh: measure_stomach_peak_circumference_impl(mesh, Trunk))
    Trunk._measure_trunk_length = cached(lambda mesh: measure_trunk_length_impl(mesh, Trunk))


_install_trunk_impl()
