"""Faithful Python ports of the low-level primitives in MATLAB ``Avatar.m``.

Every function here mirrors a specific MATLAB routine.  Where the MATLAB code
contains a quirk (or an outright bug), the quirk is reproduced and flagged with
a ``MATLAB QUIRK`` comment, because the goal of this module is bit-comparable
agreement with the reference implementation -- not "better" geometry.

MATLAB is 1-indexed and these ports are 0-indexed; index arithmetic has been
translated accordingly and the translation is noted wherever it is not obvious.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

try:  # scipy >= 1.8 exposes QhullError here
    from scipy.spatial import QhullError
except ImportError:  # older scipy
    from scipy.spatial.qhull import QhullError  # type: ignore[no-redef]


# --------------------------------------------------------------------------
# rotate_person (Avatar.m local function)
# --------------------------------------------------------------------------
def rotate_person(v1: np.ndarray, v3: np.ndarray, alpha: float):
    """Rotate the (v1, v3) plane by ``alpha`` radians, counter-clockwise.

    MATLAB: R = [cos -sin; sin cos];  person_rotated = (R * [v1 v3]')'
    """
    v1 = np.asarray(v1, dtype=float)
    v3 = np.asarray(v3, dtype=float)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return ca * v1 - sa * v3, sa * v1 + ca * v3


# --------------------------------------------------------------------------
# getVOnLine (Avatar.m method)
# --------------------------------------------------------------------------
def get_v_on_line(
    v_slice: np.ndarray,
    faces: np.ndarray,
    z_values,
    keep_idx: np.ndarray,
    v_return: np.ndarray,
):
    """Vertices belonging to faces that straddle each plane ``z = z_values[i]``.

    This is *not* an exact mesh cross-section.  MATLAB collects the vertices of
    every face crossing the plane -- points above and below it alike -- and never
    interpolates.  Downstream circumferences are therefore hull perimeters over a
    thin *band* of vertices.  Reproducing this is essential for matching values.

    MATLAB QUIRK: ``vOnLine`` is always indexed out of ``self.v`` (the unrotated
    vertex array) even when a rotated array was passed in as ``v``.  Callers that
    need rotated coordinates re-index the rotated array with the returned
    indices.  ``v_slice`` therefore drives the plane test and ``v_return`` drives
    the returned points.

    Returns ``(points_list, index_list)``; if a scalar ``z_values`` was given,
    returns the single ``(points, indices)`` pair directly, as MATLAB does.
    """
    v_slice = np.asarray(v_slice, dtype=float)
    v_return = np.asarray(v_return, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)

    scalar_input = np.isscalar(z_values) or np.asarray(z_values).ndim == 0
    z_list = np.atleast_1d(np.asarray(z_values, dtype=float))

    keep_idx = np.unique(np.asarray(keep_idx, dtype=np.int64))

    # z values of all faces, shape (n_faces, 3)
    zf = v_slice[faces, 2]
    z0, z1, z2 = zf[:, 0], zf[:, 1], zf[:, 2]

    points_out, index_out = [], []
    for zv in z_list:
        on_line = (
            ((z0 >= zv) & ((z1 <= zv) | (z2 <= zv)))
            | ((z1 >= zv) & ((z0 <= zv) | (z2 <= zv)))
            | ((z2 >= zv) & ((z0 <= zv) | (z1 <= zv)))
        )
        vidx = np.unique(faces[on_line].reshape(-1))
        # MATLAB: intersect(keepIdx, vIdxOnLine) -- sorted intersection
        vidx = np.intersect1d(keep_idx, vidx, assume_unique=True)
        index_out.append(vidx)
        points_out.append(v_return[vidx] if vidx.size else np.zeros((0, 3)))

    if scalar_input:
        return points_out[0], index_out[0]
    return points_out, index_out


# --------------------------------------------------------------------------
# getCircumference (Avatar.m local function)
# --------------------------------------------------------------------------
def get_circumference(x: np.ndarray, y: np.ndarray):
    """Perimeter of the 2-D convex hull of the (x, y) point set.

    MATLAB calls ``boundary(x, y, 0)``.  A shrink factor of 0 is *defined* by
    MATLAB to be the convex hull, so ``convhull`` is an exact equivalent.

    Note this means every "circumference" in the pipeline is a convex-hull
    perimeter, which slightly overestimates any concave cross-section (e.g. the
    waist).  That is the reference behaviour.

    Returns ``(circumference, hull_indices)`` where ``hull_indices`` is the
    closed loop (first index repeated at the end), matching MATLAB's boundary.

    The loop is rotated to begin at the hull point with the lowest index in the
    input arrays, which is where MATLAB's ``boundary`` starts its trace.  scipy
    starts somewhere else.  The perimeter does not care, but callers that
    average the closed loop do: the closing point is a *duplicate*, so the
    starting vertex is counted twice and the centroid shifts with it.
    ``getWrist`` centroids its hull loop and feeds the result to ``getArmLength``,
    so this alone accounts for the arm-length drift against the reference.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size < 3:
        return 0.0, np.arange(x.size, dtype=np.int64)

    pts = np.column_stack([x, y])
    try:
        hull = ConvexHull(pts)
    except (QhullError, ValueError):
        return 0.0, np.arange(x.size, dtype=np.int64)

    b = np.asarray(hull.vertices, dtype=np.int64)
    start = int(np.argmin(b))
    b = np.concatenate([b[start:], b[:start]])
    b_closed = np.concatenate([b, b[:1]])
    loop = pts[b_closed]
    perimeter = float(np.sum(np.linalg.norm(loop[:-1] - loop[1:], axis=1)))
    return perimeter, b_closed


# --------------------------------------------------------------------------
# find_minmax (Avatar.m local function)
# --------------------------------------------------------------------------
def find_minmax(v1: np.ndarray, v3: np.ndarray, left: float, right: float, size: int):
    """Bin along v1, take min(v3) per bin, return the bin whose min is largest.

    This is the "maximum of the minima" trick: sweeping vertical planes between
    the two feet, the lowest surface point is a foot over each leg but rises to
    the crotch in between.
    """
    v1 = np.asarray(v1, dtype=float).reshape(-1)
    v3 = np.asarray(v3, dtype=float).reshape(-1)
    x_pts = np.linspace(left, right, size)

    n = len(x_pts) - 1
    mn_v3 = np.full(n, -np.inf)
    corr_v1 = np.zeros(n)
    for i in range(n):
        lo, hi = x_pts[i], x_pts[i + 1]
        # MATLAB uses strict inequalities on both sides.
        sel = (lo < v1) & (v1 < hi)
        if not np.any(sel):
            mn_v3[i] = -np.inf
            continue
        portion_v3 = v3[sel]
        portion_v1 = v1[sel]
        j = int(np.argmin(portion_v3))
        mn_v3[i] = portion_v3[j]
        corr_v1[i] = portion_v1[j]

    mx = int(np.argmax(mn_v3))
    return float(corr_v1[mx]), float(mn_v3[mx])


# --------------------------------------------------------------------------
# sosmooth3 (Avatar.m local function)
# --------------------------------------------------------------------------
def sosmooth3(x: np.ndarray, N: int) -> np.ndarray:
    """Edge-compensated boxcar smoother.  N must be odd.

    MATLAB QUIRK: the trailing pad in the reconstruction loop uses the constant
    ``output(2*(n-1))`` on every iteration rather than a mirrored index, so the
    tail of the signal is padded with a repeated value.  Reproduced verbatim.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    h = np.ones(N)
    out = np.convolve(h, x)  # 'full', length len(x)+N-1
    L = len(out)
    a = np.concatenate([
        np.arange(1, N + 1, dtype=float),
        np.full(max(L - 2 * N, 0), float(N)),
        np.arange(N, 0, -1, dtype=float),
    ])
    if len(a) != L:  # short signals: fall back to a plain normalisation
        a = np.full(L, float(N))
    out = out / a

    n = (N - 1) // 2
    # MATLAB: cut = output(2n+1 : end-2n)   (1-based, inclusive)
    cut = out[2 * n: L - 2 * n]
    for i in range(1, n + 1):
        head = out[2 * (n - i + 1) - 1 - 1]      # output(2*(n-i+1)-1), 1-based
        tail = out[2 * (n - 1) - 1]              # MATLAB QUIRK: constant index
        cut = np.concatenate([[head], cut, [tail]])
    return cut


# --------------------------------------------------------------------------
# getFaces (Avatar.m local function)
# --------------------------------------------------------------------------
def get_faces(faces: np.ndarray, v_index: np.ndarray):
    """Faces having ANY vertex in ``v_index``.

    MATLAB QUIRK: the docstring in Avatar.m claims "all 3 vertices", but the
    implementation ORs the three columns, so a single shared vertex is enough.
    Segment surface areas therefore include a ring of boundary faces and the
    segment areas sum to more than the whole-body area.  Reproduced as-is,
    since the reference totals depend on it.
    """
    faces = np.asarray(faces, dtype=np.int64)
    v_index = np.asarray(v_index, dtype=np.int64).reshape(-1)
    member = np.isin(faces, v_index)
    mask = member[:, 0] | member[:, 1] | member[:, 2]
    return faces[mask], mask


# --------------------------------------------------------------------------
# Areas / volumes (normAll, crossAll, SignedVolumeOfTriangle)
# --------------------------------------------------------------------------
def triangle_area_sum(v: np.ndarray, faces: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return 0.0
    v1, v2, v3 = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    return float(np.sum(np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)) / 2.0)


def signed_volume(v: np.ndarray, faces: np.ndarray) -> float:
    """MATLAB SignedVolumeOfTriangle summed over all faces."""
    v = np.asarray(v, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return 0.0
    v1, v2, v3 = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", v1, np.cross(v2, v3)) / 6.0))


# --------------------------------------------------------------------------
# fixOrientation (Avatar.m method)
# --------------------------------------------------------------------------
def fix_orientation(v: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Rotate the avatar into the Styku position using 90/180-degree turns only.

    Crucially this is NOT a PCA alignment -- it only permutes and flips axes so
    that the longest extent becomes z, the head ends up at +z and the feet point
    towards -y.  Because the rotations are exact multiples of 90 degrees, the
    resulting height equals one of the original bounding-box extents exactly.
    """
    new_v = np.array(v, dtype=float, copy=True)

    dist = np.array([
        new_v[:, 0].max() - new_v[:, 0].min(),
        new_v[:, 1].max() - new_v[:, 1].min(),
        new_v[:, 2].max() - new_v[:, 2].min(),
    ])
    # MATLAB sort() returns ascending order as 1-based axis labels
    order = tuple(int(i) + 1 for i in np.argsort(dist, kind="stable"))

    if order == (1, 2, 3):
        new_v[:, 0], new_v[:, 1] = rotate_person(new_v[:, 0], new_v[:, 1], -np.pi / 2)
    elif order == (1, 3, 2):
        new_v[:, 0], new_v[:, 2] = rotate_person(new_v[:, 0], new_v[:, 2], -np.pi / 2)
        new_v[:, 1], new_v[:, 2] = rotate_person(new_v[:, 1], new_v[:, 2], -np.pi / 2)
    elif order == (2, 3, 1):
        new_v[:, 0], new_v[:, 2] = rotate_person(new_v[:, 0], new_v[:, 2], np.pi / 2)
    elif order == (3, 1, 2):
        new_v[:, 1], new_v[:, 2] = rotate_person(new_v[:, 1], new_v[:, 2], -np.pi / 2)
    elif order == (3, 2, 1):
        new_v[:, 1], new_v[:, 2] = rotate_person(new_v[:, 1], new_v[:, 2], -np.pi / 2)
        new_v[:, 0], new_v[:, 2] = rotate_person(new_v[:, 0], new_v[:, 2], -np.pi / 2)

    # Flip if the feet ended up on top: the shoulders span more x than the feet.
    M, m = new_v[:, 2].max(), new_v[:, 2].min()
    top90 = new_v[:, 2] > (M - m) * 0.9 + m
    bottom10 = new_v[:, 2] < (M - m) * 0.1 + m
    dist_top = new_v[top90, 0].max() - new_v[top90, 0].min()
    dist_bottom = new_v[bottom10, 0].max() - new_v[bottom10, 0].min()
    if dist_top > dist_bottom:
        new_v[:, 1], new_v[:, 2] = rotate_person(new_v[:, 1], new_v[:, 2], np.pi)
        M, m = new_v[:, 2].max(), new_v[:, 2].min()
        bottom10 = new_v[:, 2] < (M - m) * 0.1 + m

    # Feet must point towards -y.
    pts, _ = get_v_on_line(
        new_v, faces, (M - m) * 0.1 + m, np.arange(len(new_v)), new_v
    )
    center_y = np.mean([pts[:, 1].max(), pts[:, 1].min()])
    if abs(center_y - new_v[bottom10, 1].max()) > abs(center_y - new_v[bottom10, 1].min()):
        new_v[:, 0], new_v[:, 1] = rotate_person(new_v[:, 0], new_v[:, 1], np.pi)

    return new_v


# --------------------------------------------------------------------------
# Constrained flood fill shared by armSearch / trunkSearch
# --------------------------------------------------------------------------
def constrained_flood_fill(
    faces: np.ndarray,
    seed_faces: np.ndarray,
    keep_mask_fn,
) -> np.ndarray:
    """Grow through the face graph, keeping only vertices passing ``keep_mask_fn``.

    This is the mechanism behind MATLAB's armSearch and trunkSearch.  Growth
    halts at the geometric constraint boundary, which is what stops an arm search
    from leaking into the torso.  It is deliberately NOT a plain connected
    component: the constraint is applied at every wavefront.
    """
    faces = np.asarray(faces, dtype=np.int64)
    visited_faces = np.asarray(seed_faces, dtype=np.int64).copy()
    new_faces = visited_faces.copy()
    collected: list[np.ndarray] = []

    seen = np.zeros(len(faces), dtype=bool)
    seen[visited_faces] = True

    while new_faces.size:
        new_v_idx = np.unique(faces[new_faces].reshape(-1))
        new_v_idx = new_v_idx[keep_mask_fn(new_v_idx)]
        if new_v_idx.size:
            collected.append(new_v_idx)
            member = np.isin(faces, new_v_idx)
            cand = np.flatnonzero(member[:, 0] | member[:, 1] | member[:, 2])
            new_faces = cand[~seen[cand]]
            seen[new_faces] = True
        else:
            new_faces = np.empty(0, dtype=np.int64)

    if not collected:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(collected))


# --------------------------------------------------------------------------
# checkFaceOrientation / fixFaceOrientation2 (Avatar.m local functions)
# --------------------------------------------------------------------------
def _rows_in(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MATLAB ``ismember(a, b, 'rows')`` for two integer Nx2 arrays."""
    if a.size == 0:
        return np.zeros(len(a), dtype=bool)
    if b.size == 0:
        return np.zeros(len(a), dtype=bool)
    known = set(map(tuple, np.asarray(b, dtype=np.int64).tolist()))
    return np.fromiter(
        (tuple(row) in known for row in np.asarray(a, dtype=np.int64).tolist()),
        dtype=bool, count=len(a),
    )


def check_face_orientation(faces: np.ndarray) -> bool:
    """True when every directed edge has its reverse present.

    Avatar.m's own comment notes this is also false when the mesh has holes,
    so an open scan always sends ``fixFaceOrientation2`` down the full path.
    """
    faces = np.asarray(faces, dtype=np.int64)
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return not np.any(~_rows_in(e[:, [1, 0]], e))


def fix_face_orientation2(faces: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Port of ``fixFaceOrientation2``: make face windings mutually consistent.

    ``Avatar.m`` runs this on ``self.f`` immediately before the volume pass
    whenever ``Vol_SA`` is on.  Surface area does not care about winding, but
    signed volume does -- a scan carrying the same triangle twice with opposite
    winding has the two contributions cancel unless this runs first.

    Part 1 casts a ray along the normal of every 200th face and flips that face
    when the intersection count is odd; Part 2 floods the corrected winding out
    across shared edges.

    MATLAB QUIRK: ``e1/e2/e3`` and ``E1/E2`` are captured from the *original*
    ``f`` and never refreshed, so Part 2 matches against pre-flip edges and
    Part 1's Moeller-Trumbore edges go stale as it flips.  Both are reproduced.
    """
    f = np.array(faces, dtype=np.int64, copy=True)
    v = np.asarray(v, dtype=float)
    n = len(f)
    if n == 0:
        return f

    fixed = np.ones(n, dtype=bool)
    e1, e2, e3 = f[:, [0, 1]].copy(), f[:, [1, 2]].copy(), f[:, [2, 0]].copy()

    # Part 0 -- nothing to do when the mesh is already consistent and closed.
    if check_face_orientation(f):
        return f

    # Part 1 -- ray/face intersection parity on a 0.5% sample.
    p, epsilon = 0.005, 1e-10
    count = int(np.floor(n * p + 0.5))  # MATLAB round(): half away from zero
    if count < 1:
        return f
    pre = np.floor(np.linspace(1, n, count) + 0.5).astype(np.int64) - 1  # 1-based -> 0

    n1 = -v[f[pre, 1]] + v[f[pre, 0]]
    n2 = -v[f[pre, 2]] + v[f[pre, 0]]
    normal = np.cross(n1, n2)
    normal = normal / np.linalg.norm(normal, axis=1)[:, None]
    c = (v[f[pre, 0]] + v[f[pre, 1]] + v[f[pre, 2]]) / 3.0

    E1 = v[f[:, 1]] - v[f[:, 0]]
    E2 = v[f[:, 2]] - v[f[:, 0]]

    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(len(normal)):
            D = normal[i]
            T = c[i] - v[f[:, 0]]
            P = np.cross(np.broadcast_to(D, E2.shape), E2)
            Q = np.cross(T, E1)
            PE = np.einsum("ij,ij->i", P, E1)
            t = np.einsum("ij,ij->i", Q, E2) / PE
            a = np.einsum("ij,ij->i", P, T) / PE
            b = Q @ D / PE
            hits = int(np.count_nonzero((a + b < 1) & (a > 0) & (b > 0) & (t > epsilon)))
            if hits % 2:
                f[pre[i]] = f[pre[i]][[0, 2, 1]]
    fixed[pre] = False

    # Part 2 -- propagate the corrected winding across shared edges.
    new_faces = f[pre]
    while new_faces.size:
        edges = np.vstack([new_faces[:, [1, 0]],
                           new_faces[:, [2, 1]],
                           new_faces[:, [0, 2]]])
        keep = (_rows_in(e1, edges) | _rows_in(e2, edges) | _rows_in(e3, edges)) & fixed
        fixed[keep] = False

        redges = edges[:, [1, 0]]
        flip = (_rows_in(e1, redges) | _rows_in(e2, redges) | _rows_in(e3, redges)) & fixed
        f[flip] = f[flip][:, [1, 0, 2]]
        fixed[flip] = False

        new_faces = f[keep | flip]
    return f
