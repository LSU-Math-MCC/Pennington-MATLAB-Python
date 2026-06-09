import numpy as np
import trimesh
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, box


def smooth_curve(values: np.ndarray, window: int = 7) -> np.ndarray:
    """Dampen section-to-section noise so anatomical levels follow body shape, not mesh tessellation."""
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return values
    window = min(window, len(values) if len(values) % 2 else len(values) - 1)
    return uniform_filter1d(values, size=window, mode="nearest") if window >= 3 else values


def local_extrema(values: np.ndarray, kind: str) -> list[int]:
    """Return indices where a sampled geometric signal forms local peaks or valleys along z."""
    curve = np.asarray(values, dtype=float)
    return find_peaks(curve if kind == "max" else -curve)[0].tolist()


def normalized_stack_curve(rows: list[dict], signals: tuple[str, ...]) -> np.ndarray:
    """Blend several section metrics into one unitless body-shape curve for level selection."""
    curves = []
    for signal in signals:
        values = np.array([row[signal] for row in rows], dtype=float)
        span = np.ptp(values)
        curves.append(np.zeros(len(values)) if span <= 1e-9 else (values - values.min()) / span)
    return np.mean(curves, axis=0)


def ordered_loop(points: np.ndarray) -> np.ndarray:
    """Order unordered section points around their xy centroid to approximate a closed perimeter."""
    center = points[:, :2].mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


def section_loops(mesh: trimesh.Trimesh, z: float) -> list[np.ndarray]:
    """Intersect the mesh with a horizontal plane and return every closed or usable section loop."""
    section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if section is None:
        return []
    loops = [np.asarray(loop) for loop in getattr(section, "discrete", []) if len(loop) >= 3]
    return loops or ([section.vertices] if len(section.vertices) >= 3 else [])


def central_loop_index(loops: list[np.ndarray]) -> int:
    """Choose the section loop most likely to be the torso by preferring loops spanning x=0."""
    def score(points):
        x = points[:, 0]
        x_width = np.ptp(x)
        return (x.min() <= 0 <= x.max(), x_width / (abs(x.mean()) + x_width + 1e-9), x_width)

    return max(range(len(loops)), key=lambda i: score(loops[i]))


def central_section(mesh: trimesh.Trimesh, z: float) -> np.ndarray | None:
    """Return the torso-centered horizontal section loop at a given height."""
    loops = section_loops(mesh, z)
    return None if not loops else loops[central_loop_index(loops)]


def central_section_polygon(mesh: trimesh.Trimesh, z: float, x_limit: float | None = None):
    """Project a horizontal section to 2D, choose the central polygon, and map its boundary back to 3D."""
    def clipped_central(polygon):
        if x_limit is None:
            return polygon
        try:
            _, miny, _, maxy = polygon.bounds
            clipped = polygon.intersection(box(-x_limit, miny - 1.0, x_limit, maxy + 1.0))
            pieces = [clipped] if clipped.geom_type == "Polygon" else [
                geom for geom in clipped.geoms if geom.geom_type == "Polygon"
            ]
            return max(pieces, key=lambda geom: geom.area) if not clipped.is_empty and pieces else polygon
        except Exception:
            return polygon

    def from_raw_loop():
        points = central_section(mesh, z)
        if points is None or len(points) < 3:
            return None
        coords = np.asarray(points[:, :2], dtype=float)
        polygon = Polygon(coords)
        if polygon.area <= 0 or not polygon.is_valid:
            coords = ordered_loop(points)[:, :2]
            polygon = Polygon(coords)
        if polygon.area <= 0 or not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            return None
        if polygon.geom_type != "Polygon":
            pieces = [geom for geom in polygon.geoms if geom.geom_type == "Polygon"]
            if not pieces:
                return None
            polygon = max(pieces, key=lambda geom: geom.area)
        polygon = clipped_central(polygon)
        coords_2d = np.asarray(polygon.exterior.coords[:-1])
        coords_3d = np.column_stack([coords_2d, np.full(len(coords_2d), z)])
        return polygon, coords_3d

    section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if section is None:
        return None
    try:
        path_2d, to_3d = section.to_planar()
        polygons = list(path_2d.polygons_full)
    except Exception:
        return from_raw_loop()
    if not polygons:
        return from_raw_loop()

    def score(polygon):
        minx, _, maxx, _ = polygon.bounds
        return (minx <= 0 <= maxx, polygon.area / (abs(polygon.centroid.x) + 1e-9), polygon.area)

    polygon = max(polygons, key=score)
    polygon = clipped_central(polygon)

    coords_2d = np.asarray(polygon.exterior.coords[:-1])
    coords_3d = trimesh.transformations.transform_points(
        np.column_stack([coords_2d, np.zeros(len(coords_2d))]),
        to_3d,
    )
    return polygon, coords_3d


def closed_path(vertices_3d: np.ndarray) -> trimesh.path.Path3D:
    """Build a closed 3D polyline matching a tape path around a section boundary."""
    indices = np.arange(len(vertices_3d) + 1)
    indices[-1] = 0
    return trimesh.path.Path3D(entities=[trimesh.path.entities.Line(indices)], vertices=vertices_3d)


def line_path(vertices_3d: np.ndarray) -> trimesh.path.Path3D:
    """Build a straight 3D measurement path between two landmark points."""
    return trimesh.path.Path3D(
        entities=[trimesh.path.entities.Line([0, 1])],
        vertices=np.asarray(vertices_3d),
    )


def empty_measurement() -> tuple[float, trimesh.path.Path3D]:
    """Return a zero-length placeholder when no valid geometric section can be measured."""
    return (0.0, trimesh.load_path(np.array([[0, 0, 0]])))


def polygon_measurement(polygon, vertices_3d: np.ndarray) -> tuple[float, trimesh.path.Path3D]:
    """Measure perimeter directly from a clean 2D section polygon while preserving its 3D boundary."""
    return float(polygon.exterior.length), closed_path(vertices_3d)


def loop_measurement(points: np.ndarray, z: float) -> tuple[float, trimesh.path.Path3D]:
    """Estimate a closed horizontal circumference from raw loop points, using the xy hull when needed."""
    points = np.asarray(points)
    try:
        hull = ConvexHull(points[:, :2])
        points = points[hull.vertices]
        circumference = hull.area
    except Exception:
        points = ordered_loop(points)
        loop_2d = points[:, :2]
        circumference = np.linalg.norm(np.diff(np.vstack([loop_2d, loop_2d[0]]), axis=0), axis=1).sum()
    vertices_3d = np.column_stack([points[:, :2], np.full(len(points), z)])
    return float(circumference), closed_path(vertices_3d)


def slice_measurement(mesh: trimesh.Trimesh, z: float, label: str):
    """Measure the central horizontal slice at z, falling back from polygon geometry to raw loops."""
    polygon = central_section_polygon(mesh, z)
    if polygon is not None:
        return polygon_measurement(*polygon)
    loop = central_section(mesh, z)
    if loop is not None:
        return loop_measurement(loop, z)
    print(f"Warning: No section found at {label} level")
    return empty_measurement()
