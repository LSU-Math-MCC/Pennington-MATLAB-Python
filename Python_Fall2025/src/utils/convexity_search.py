import numpy as np
import trimesh
from shapely.geometry import Polygon

def convexity_search(mesh: trimesh.Trimesh, 
                    rays: int,
                    origin=np.array([0, 0, 0]),
                    max_iter=20,
                    slice_width=.005,
                    convexity_threshold=.05):
    """
    Find a landmark point by searching for the most convex horizontal cross-section.
    
    This function searches upward from a starting point, analyzing horizontal slices
    of the mesh to find where the body is most convex (rounded) or least concave (indented).
    It's used to locate landmarks like the crotch (most convex point between legs) or
    armpits (transition from concave to convex).
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to search through (should be oriented with Z-axis vertical)
    rays : int
        Number of rays to cast radially at each height (more rays = finer resolution)
        Typical value: 50-100
    origin : np.ndarray, default=[0, 0, 0]
        Starting point for the search (X, Y, Z coordinates)
        Search proceeds upward from origin[2]
    max_iter : int, default=20
        Maximum number of horizontal slices to analyze
    slice_width : float, default=0.005
        Vertical distance between consecutive slices
        Total search height = max_iter * slice_width
    convexity_threshold : float, default=0.05
        Threshold for "convex enough" (0.05 means 95% convex)
        If a slice is >= (1 - threshold) convex, return immediately
    
    Returns
    -------
    np.ndarray
        The 3D coordinates [x, y, z] of the landmark point (center of most convex slice)
    
    Examples
    --------
    Find the crotch (most convex point between legs):
    
    >>> import trimesh
    >>> import numpy as np
    >>> from convexity_search import convexity_search  # doctest: +SKIP
    >>> mesh = trimesh.load("model_files/man.obj")  # doctest: +SKIP
    >>> # Start search from bottom of mesh, moving upward
    >>> bottom_z = mesh.bounds[0, 2]  # doctest: +SKIP
    >>> crotch = convexity_search(  # doctest: +SKIP
    ...     mesh,
    ...     rays=50,
    ...     origin=np.array([0, 0, bottom_z]),
    ...     max_iter=30,
    ...     slice_width=0.01
    ... )
    >>> print(f"Crotch at: {crotch}")  # doctest: +SKIP
    Crotch at: [0.0, 0.0, 20.5]
    
    Pseudocode
    ----------
    1. Initialize ray directions (radially outward in XY plane)
    2. For each height from origin[2] to origin[2] + max_iter * slice_width:
        a. Cast rays horizontally through the mesh at this height
        b. Find intersection points with mesh surface
        c. Compute centroid of intersection points
        d. Group intersections by ray direction
        e. Keep only outermost point per ray
        f. Order points into a loop around the centroid
        g. Compute convexity = area / convex_hull_area
        h. If convexity >= threshold, return centroid immediately
    3. If no slice meets threshold, return centroid of most convex slice
    
    Notes
    -----
    - Convexity score of 1.0 means perfectly convex (circular cross-section)
    - Convexity score < 1.0 indicates concavity (indentations)
    - The function searches upward, so set origin to start below the landmark
    - Ray count affects precision: more rays = smoother cross-section but slower
    - Slice width affects sensitivity: smaller = finer search but more iterations
    
    Algorithm Details
    -----------------
    The convexity score is computed as:
        convexity = polygon_area / convex_hull_area
    
    Where:
    - polygon_area: Area of the actual cross-section at this height
    - convex_hull_area: Area of the smallest convex polygon containing the cross-section
    
    If the cross-section is already convex, these areas are equal (convexity = 1.0).
    If there are concave regions (like armpits), polygon_area < convex_hull_area.
    
    See Also
    --------
    assign_points_to_rays : Groups intersection points by ray direction
    pick_outermost_points : Selects boundary points for convexity calculation
    compute_convexity : Computes the convexity score
    """
    
    z = origin[2]
    
    thetas = np.linspace(0, 2*np.pi, rays)
    ray_directions = np.empty((rays, 3))
    ray_directions[:, 0] = np.cos(thetas)
    ray_directions[:, 1] = np.sin(thetas)
    
    ray_origins = np.empty((rays, 3))
    ray_origins[:, :] = origin
    
    convexity_scores = []
    for iteration in range(max_iter):
        ray_directions[:, 2] = iteration * slice_width + z
        ray_origins[:, 2] = iteration * slice_width + z
        
        intersections = mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions
        )[0]
        
        center = np.mean(intersections, axis=0)
        assigned = assign_points_to_rays(intersections, ray_directions, center)
        outer_section = pick_outermost_points(assigned, center)
        loops = order_loop_from_points(outer_section)
        
        convexity_score = compute_convexity(loops)
        if convexity_score >= 1 - convexity_threshold:
            return center
        convexity_scores.append((convexity_score, center))
    
    convexity_scores.sort(key=lambda x: x[0], reverse=True)  # sort by score, not (score, ndarray)
    return convexity_scores[0][1]
        
def assign_points_to_rays(points, directions, centroid):
    """
    Assign each intersection point to the ray direction it aligns with most.
    
    When rays are cast through a mesh, multiple intersection points may align with
    the same ray (if the ray passes through multiple parts of the mesh). This function
    groups all intersection points by which ray direction they're closest to (in angle).
    
    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
        Intersection points from ray casting
    directions : np.ndarray, shape (M, 3)
        Ray directions (unit vectors)
    centroid : np.ndarray, shape (3,)
        Center point to measure angles from
    
    Returns
    -------
    list of lists
        Length M list where assigned[i] contains all points aligned with directions[i]
        Each inner list contains 3D coordinates of points
    
    Examples
    --------
    >>> import numpy as np
    >>> # Three points and two ray directions
    >>> points = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0]])  # doctest: +SKIP
    >>> directions = np.array([[1, 0, 0], [0, 1, 0]])  # doctest: +SKIP
    >>> centroid = np.array([0, 0, 0])  # doctest: +SKIP
    >>> assigned = assign_points_to_rays(points, directions, centroid)  # doctest: +SKIP
    >>> # assigned[0] contains points near direction [1,0,0]
    >>> # assigned[1] contains points near direction [0,1,0]
    
    Pseudocode
    ----------
    1. Create empty list for each ray direction
    2. For each intersection point:
        a. Compute vector from centroid to point
        b. Normalize to unit vector
        c. Compute dot product with all ray directions
        d. Find ray with maximum dot product (smallest angle)
        e. Add point to that ray's list
    3. Return list of lists
    
    Notes
    -----
    - Uses dot product to measure alignment (dot = 1 means parallel)
    - Points equidistant from two rays go to whichever has larger dot product
    - Some ray directions may have no assigned points (empty lists)
    """
    assigned = [[] for _ in range(len(directions))]
    vectors = points - centroid  # shift to geometric center
    
    for p, v in zip(points, vectors):
        unit_v = v / np.linalg.norm(v)
        dots = directions @ unit_v
        idx = np.argmax(dots)  # most aligned ray
        assigned[idx].append(p)

    return assigned

def pick_outermost_points(assigned, centroid):
    """
    Select the farthest point from centroid for each ray direction.
    
    When multiple points align with the same ray, we keep only the outermost one
    (farthest from the centroid) because this defines the boundary of the cross-section.
    
    Parameters
    ----------
    assigned : list of lists
        Points grouped by ray direction (from assign_points_to_rays)
    centroid : np.ndarray, shape (3,)
        Center point to measure distances from
    
    Returns
    -------
    np.ndarray, shape (K, 3)
        One point per ray direction (only rays that had hits)
        K <= number of ray directions
    
    Examples
    --------
    >>> import numpy as np
    >>> # Two rays, first has 2 points, second has 1
    >>> assigned = [  # doctest: +SKIP
    ...     [np.array([1, 0, 0]), np.array([2, 0, 0])],  # doctest: +SKIP
    ...     [np.array([0, 1.5, 0])]  # doctest: +SKIP
    ... ]
    >>> centroid = np.array([0, 0, 0])  # doctest: +SKIP
    >>> outer = pick_outermost_points(assigned, centroid)  # doctest: +SKIP
    >>> # Result: [[2, 0, 0], [0, 1.5, 0]]  # doctest: +SKIP
    
    Pseudocode
    ----------
    1. For each ray's list of points:
        a. If no points, skip this ray
        b. Compute distance from centroid to each point
        c. Keep the point with maximum distance
    2. Return array of outermost points
    
    Notes
    -----
    - Rays with no intersections are skipped (not included in output)
    - Output may have fewer points than input rays
    - Used to extract the boundary of a cross-section
    """
    result = []
    for hits in assigned:
        if not hits:
            continue
        hits = np.array(hits)
        dists = np.linalg.norm(hits - centroid, axis=1)
        result.append(hits[np.argmax(dists)])
    return np.array(result)

def order_loop_from_points(points):
    """
    Order points into a loop by angle around their centroid (in XY plane).
    
    Given a set of boundary points, arrange them in counter-clockwise order
    around their center. This creates a closed loop suitable for convexity calculation.
    
    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
        Boundary points to order
    
    Returns
    -------
    np.ndarray, shape (N, 3)
        Same points in counter-clockwise angular order
    
    Examples
    --------
    >>> import numpy as np
    >>> # Four points in random order
    >>> points = np.array([  # doctest: +SKIP
    ...     [1, 0, 5],   # 0 degrees
    ...     [0, 1, 5],   # 90 degrees
    ...     [-1, 0, 5],  # 180 degrees
    ...     [0, -1, 5]   # 270 degrees
    ... ])
    >>> ordered = order_loop_from_points(points)  # doctest: +SKIP
    >>> # Result: points ordered by increasing angle
    
    Pseudocode
    ----------
    1. Compute centroid of points (in XY plane)
    2. For each point:
        a. Compute vector from centroid to point (in XY)
        b. Compute angle using atan2(y, x)
    3. Sort points by angle
    4. Return sorted points
    
    Notes
    -----
    - Uses only XY coordinates for angle calculation (ignores Z)
    - Angles are measured counter-clockwise from positive X-axis
    - Z coordinates are preserved but don't affect ordering
    - Result forms a closed loop when connected in sequence
    """
    centroid = points[:, :2].mean(axis=0)
    rel = points[:, :2] - centroid
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    order = np.argsort(angles)
    return points[order]

def compute_convexity(loop):
    """
    Compute the convexity score of a closed loop of points.
    
    Convexity is the ratio of the loop's area to its convex hull's area.
    A score of 1.0 means the loop is already convex. Lower scores indicate concavity.
    
    Parameters
    ----------
    loop : np.ndarray, shape (N, 3)
        Points forming a closed loop (in angular order)
    
    Returns
    -------
    float
        Convexity score in range [0, 1]
        1.0 = perfectly convex
        < 1.0 = has concave regions
        0 = invalid/degenerate polygon
    
    Examples
    --------
    >>> import numpy as np
    >>> # Perfect circle (convex)
    >>> angles = np.linspace(0, 2*np.pi, 50)  # doctest: +SKIP
    >>> circle = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(50)])  # doctest: +SKIP
    >>> score = compute_convexity(circle)  # doctest: +SKIP
    >>> print(f"Circle convexity: {score:.3f}")  # doctest: +SKIP
    Circle convexity: 1.000
    
    >>> # Star shape (concave)
    >>> # Would have convexity < 1.0
    
    Pseudocode
    ----------
    1. Project loop to 2D (use XY coordinates only)
    2. Create Shapely polygon from 2D points
    3. If polygon is invalid or has zero area, return 0
    4. Compute area of polygon
    5. Compute area of polygon's convex hull
    6. Return area / hull_area
    
    Notes
    -----
    - Uses Shapely library for robust polygon area calculation
    - Ignores Z coordinate (only uses XY plane)
    - Returns 0 for degenerate cases (self-intersecting, zero area)
    - Convex hull is the smallest convex polygon containing all points
    
    Mathematical Definition
    -----------------------
    convexity = area(polygon) / area(convex_hull(polygon))
    
    Where:
    - area(polygon) = actual area enclosed by the boundary
    - area(convex_hull) = area if we "filled in" all concave regions
    
    See Also
    --------
    shapely.geometry.Polygon : Used for area computation
    """
    if len(loop) < 3: return 0.0  # a ring needs >=3 vertices (a triangle); fewer can't form a polygon
    loop_2d = loop[:, :2]
    poly = Polygon(loop_2d)

    if not poly.is_valid or poly.area == 0:
        return 0.0  # consistent float so caller's `score >= ...` never sees a tuple

    area = poly.area
    hull = poly.convex_hull
    hull_area = hull.area
    convexity = area / hull_area

    return convexity