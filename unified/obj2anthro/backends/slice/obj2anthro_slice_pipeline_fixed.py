"""
Full OBJ to PCA alignment to slicing to 42 biomarkers.

This pipeline:
1. Loads OBJ files.
2. PCA-aligns each body mesh.
3. Slices the aligned mesh from bottom to top.
4. Saves slice CSV files.
5. Saves aligned OBJ files.
6. Saves slice-profile PNG files.
7. Saves biomarker-method PNG files.
8. Extracts 42 Pennington-style slicing-derived biomarker proxies.
9. Saves final biomarker CSV.
10. Saves one contact-sheet image of all biomarker-method PNGs.

Run:

python -m unified.obj2anthro.backends.slice.slice \
  --input data/obj \
  --all \
  --recursive \
  --n-slices 200
"""




import argparse
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh


BIOMARKER_COLUMNS = [
    "Height (cm)",
    "Abdomen Circumference",
    "Ankle Circumference Left",
    "Arm Length Left",
    "Arm Volume Left",
    "Bicep Circumference Left",
    "Calf Circumference Left",
    "Chest",
    "Collar Circumference",
    "Forearm Circumference Left",
    "Head Circumference",
    "Hip Circumference",
    "Horizontal Waist",
    "Inseam Left",
    "Leg Volume Left",
    "MidThigh Circumference Left",
    "Narrow Waist",
    "Outside Leg Length Left",
    "Seat Circumference",
    "Surface Area Arm Left",
    "Surface Area Leg Left",
    "Surface Area Torso",
    "Surface Area Total",
    "Thigh Circumference Left",
    "Torso Volume",
    "Upper Arm Circumference Left",
    "Volume",
    "Waist Circumference",
    "Ankle Circumference Right",
    "Arm Length Right",
    "Arm Volume Right",
    "Bicep Circumference Right",
    "Calf Circumference Right",
    "Forearm Circumference Right",
    "Inseam Right",
    "Leg Volume Right",
    "MidThigh Circumference Right",
    "Outside Leg Length Right",
    "Surface Area Arm Right",
    "Surface Area Leg Right",
    "Thigh Circumference Right",
    "Upper Arm Circumference Right",
]


# =============================================================================
# General helpers
# =============================================================================

def safe_stem(path: Path) -> str:
    stem = Path(path).stem
    for ch in [" ", "/", "\\", ":", ";", ",", "(", ")", "[", "]"]:
        stem = stem.replace(ch, "_")
    while "__" in stem:
        stem = stem.replace("__", "_")
    return stem.strip("_")


def collect_obj_files(input_path: Path, recursive: bool = False):
    input_path = Path(input_path)

    if input_path.is_file():
        if input_path.suffix.lower() != ".obj":
            raise ValueError(f"Input file is not an OBJ: {input_path}")
        return [input_path]

    if input_path.is_dir():
        if recursive:
            return sorted(
                [
                    p for p in input_path.rglob("*")
                    if p.is_file() and p.suffix.lower() == ".obj"
                ]
            )

        return sorted(
            [
                p for p in input_path.iterdir()
                if p.is_file() and p.suffix.lower() == ".obj"
            ]
        )

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def trapz_integral(y, x):
    """
    Compatible with both older and newer NumPy versions.
    """
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


# =============================================================================
# Mesh loading and alignment
# =============================================================================

def load_mesh(obj_file: Path) -> trimesh.Trimesh:
    obj_file = Path(obj_file)

    if not obj_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {obj_file}")

    loaded = trimesh.load(obj_file, process=True)

    if isinstance(loaded, trimesh.Scene):
        geometries = [
            g for g in loaded.geometry.values()
            if isinstance(g, trimesh.Trimesh) and len(g.vertices) > 0
        ]

        if not geometries:
            raise ValueError(f"No valid mesh geometry found in scene: {obj_file}")

        mesh = trimesh.util.concatenate(geometries)

    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded

    else:
        raise ValueError(f"Could not load valid mesh from: {obj_file}")

    if len(mesh.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {obj_file}")

    # Keep largest connected component if multiple components exist.
    try:
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            mesh = max(parts, key=lambda m: len(m.vertices))
    except Exception:
        pass

    mesh.remove_unreferenced_vertices()
    return mesh


def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return np.eye(3)

    a = a / norm_a
    b = b / norm_b

    cross = np.cross(a, b)
    dot = np.dot(a, b)

    if np.isclose(dot, 1.0):
        return np.eye(3)

    if np.isclose(dot, -1.0):
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])

        v = np.cross(a, axis)
        v = v / np.linalg.norm(v)

        vx = np.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ])

        return np.eye(3) + 2.0 * vx @ vx

    vx = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])

    return np.eye(3) + vx + vx @ vx * (
        (1.0 - dot) / (np.linalg.norm(cross) ** 2)
    )


def pca_align_to_z(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    PCA-align mesh so the longest body direction becomes the Z-axis.

    Then fix PCA sign ambiguity so feet are at bottom and head is at top.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)

    center = vertices.mean(axis=0)
    centered = vertices - center

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    main_axis = eigvecs[:, np.argmax(eigvals)]

    z_axis = np.array([0.0, 0.0, 1.0])
    R = rotation_matrix_from_vectors(main_axis, z_axis)

    aligned_vertices = centered @ R.T

    aligned = mesh.copy()
    aligned.vertices = aligned_vertices

    # Move bottom to z=0.
    aligned.vertices[:, 2] -= aligned.vertices[:, 2].min()

    # Fix upside-down ambiguity.
    v = np.asarray(aligned.vertices)
    z_min = v[:, 2].min()
    z_max = v[:, 2].max()
    height = z_max - z_min

    if height > 0:
        bottom_region = v[v[:, 2] <= z_min + 0.12 * height]
        top_region = v[v[:, 2] >= z_max - 0.12 * height]

        def region_size(region):
            if len(region) == 0:
                return np.nan
            width = region[:, 0].max() - region[:, 0].min()
            depth = region[:, 1].max() - region[:, 1].min()
            return np.nanmean([width, depth])

        bottom_size = region_size(bottom_region)
        top_size = region_size(top_region)

        # If bottom is smaller than top, body is probably upside down.
        if np.isfinite(bottom_size) and np.isfinite(top_size):
            if bottom_size < top_size:
                aligned.vertices[:, 2] = z_max - aligned.vertices[:, 2]
                aligned.vertices[:, 2] -= aligned.vertices[:, 2].min()

    return aligned


# =============================================================================
# Slicing
# =============================================================================

def closed_loop_perimeter(points: np.ndarray) -> float:
    points = np.asarray(points)

    if len(points) < 3:
        return np.nan

    closed = np.vstack([points, points[0]])
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)

    return float(np.sum(segment_lengths))


def polygon_area(points: np.ndarray) -> float:
    points = np.asarray(points)

    if len(points) < 3:
        return np.nan

    x = points[:, 0]
    y = points[:, 1]

    return float(0.5 * abs(
        np.dot(x, np.roll(y, -1)) -
        np.dot(y, np.roll(x, -1))
    ))


def horizontal_slice_loops_3d(mesh: trimesh.Trimesh, z: float):
    """
    Return ordered 3D slice loops/polylines.

    trimesh.section(...) returns a Path3D object.  For circumference and
    area calculations, section.discrete is safer than manually indexing
    section.entities because it gives ordered polylines.
    """
    section = mesh.section(
        plane_origin=[0.0, 0.0, float(z)],
        plane_normal=[0.0, 0.0, 1.0],
    )

    if section is None:
        return []

    loops = []

    try:
        for pts in section.discrete:
            pts = np.asarray(pts, dtype=float)
            if len(pts) >= 3:
                loops.append(pts)
    except Exception:
        # Fallback for unusual trimesh Path3D objects where .discrete fails.
        vertices = np.asarray(section.vertices)
        for entity in section.entities:
            try:
                pts = vertices[np.asarray(entity.points)]
                if len(pts) >= 3:
                    loops.append(pts)
            except Exception:
                continue

    return loops


def loop_measurements_from_3d(pts3d: np.ndarray):
    """
    Measure a horizontal slice loop using its x-y projection.
    """
    pts3d = np.asarray(pts3d)

    if len(pts3d) < 3:
        return {
            "perimeter": np.nan,
            "area": np.nan,
            "width": np.nan,
            "depth": np.nan,
            "center_x": np.nan,
            "center_y": np.nan,
        }

    pts2d = pts3d[:, :2]

    return {
        "perimeter": closed_loop_perimeter(pts2d),
        "area": polygon_area(pts2d),
        "width": float(pts2d[:, 0].max() - pts2d[:, 0].min()),
        "depth": float(pts2d[:, 1].max() - pts2d[:, 1].min()),
        "center_x": float(np.nanmean(pts2d[:, 0])),
        "center_y": float(np.nanmean(pts2d[:, 1])),
    }


def create_slice_dataframe(aligned: trimesh.Trimesh, obj_file: Path, n_slices: int):
    z_min, z_max = aligned.bounds[:, 2]
    height = float(z_max - z_min)

    if height <= 0:
        raise ValueError(f"Invalid mesh height: {height}")

    z_values = np.linspace(
        z_min + 0.01 * height,
        z_max - 0.01 * height,
        n_slices,
    )

    rows = []

    for i, z in enumerate(z_values):
        loops3d = horizontal_slice_loops_3d(aligned, z)
        height_percent = 100.0 * (z - z_min) / height

        if not loops3d:
            rows.append({
                "source_file": str(obj_file),
                "slice_index": i,
                "z": z,
                "height_percent": height_percent,
                "num_loops": 0,
                "max_perimeter": np.nan,
                "sum_perimeter": np.nan,
                "max_area": np.nan,
                "sum_area": np.nan,
                "width": np.nan,
                "depth": np.nan,
                "left_max_perimeter": np.nan,
                "right_max_perimeter": np.nan,
                "left_sum_area": np.nan,
                "right_sum_area": np.nan,
            })
            continue

        stats = [loop_measurements_from_3d(loop) for loop in loops3d]

        perimeters = [s["perimeter"] for s in stats]
        areas = [s["area"] for s in stats]
        all_points = np.vstack(loops3d)

        left_perimeters = []
        right_perimeters = []
        left_areas = []
        right_areas = []

        for s in stats:
            if pd.isna(s["center_x"]):
                continue

            if s["center_x"] < 0:
                left_perimeters.append(s["perimeter"])
                left_areas.append(s["area"])
            else:
                right_perimeters.append(s["perimeter"])
                right_areas.append(s["area"])

        rows.append({
            "source_file": str(obj_file),
            "slice_index": i,
            "z": z,
            "height_percent": height_percent,
            "num_loops": len(loops3d),
            "max_perimeter": float(np.nanmax(perimeters)),
            "sum_perimeter": float(np.nansum(perimeters)),
            "max_area": float(np.nanmax(areas)),
            "sum_area": float(np.nansum(areas)),
            "width": float(all_points[:, 0].max() - all_points[:, 0].min()),
            "depth": float(all_points[:, 1].max() - all_points[:, 1].min()),
            "left_max_perimeter": float(np.nanmax(left_perimeters)) if left_perimeters else np.nan,
            "right_max_perimeter": float(np.nanmax(right_perimeters)) if right_perimeters else np.nan,
            "left_sum_area": float(np.nansum(left_areas)) if left_areas else np.nan,
            "right_sum_area": float(np.nansum(right_areas)) if right_areas else np.nan,
        })

    return pd.DataFrame(rows), height


def detect_crotch_from_loop_count(df: pd.DataFrame):
    """
    Simple crotch proxy:
    lower slices often have 2 leg loops;
    at crotch/pelvis, loops merge.
    """
    valid = df[df["num_loops"] > 0].copy()
    valid = valid[
        (valid["height_percent"] >= 5) &
        (valid["height_percent"] <= 65)
    ].copy()

    if len(valid) == 0:
        return np.nan, np.nan

    multi = valid[valid["num_loops"] >= 2].copy()

    if len(multi) == 0:
        return np.nan, np.nan

    idx = multi["height_percent"].idxmax()

    return float(valid.loc[idx, "z"]), float(valid.loc[idx, "height_percent"])


# =============================================================================
# Static images
# =============================================================================

def save_slice_profile_image(df: pd.DataFrame, image_path: Path, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(10, 14), dpi=150, sharex=True)

    axes[0].plot(df["height_percent"], df["sum_perimeter"])
    axes[0].set_ylabel("Sum Perimeter")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(df["height_percent"], df["sum_area"])
    axes[1].set_ylabel("Sum Area")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(df["height_percent"], df["width"])
    axes[2].set_ylabel("Width")
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(df["height_percent"], df["depth"])
    axes[3].set_ylabel("Depth")
    axes[3].grid(True, alpha=0.25)

    axes[4].plot(df["height_percent"], df["num_loops"])
    axes[4].set_ylabel("Loops")
    axes[4].set_xlabel("Body Height Percent")
    axes[4].grid(True, alpha=0.25)

    crotch_z, crotch_hp = detect_crotch_from_loop_count(df)
    if pd.notna(crotch_hp):
        for ax in axes:
            ax.axvline(crotch_hp, linestyle="--", linewidth=1)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)


def get_biomarker_slice_targets(df: pd.DataFrame):
    """
    Select important biomarker slices.
    Labels are used for the interactive HTML legend/hover only.
    """
    valid = df[df["num_loops"] > 0].copy()

    def local_band(local_df, low, high):
        return local_df[
            (local_df["height_percent"] >= low) &
            (local_df["height_percent"] <= high)
        ].copy()

    def pick_max(label, low, high, col):
        b = local_band(valid, low, high)
        b = b.dropna(subset=[col, "z", "height_percent"])
        if len(b) == 0:
            return None

        idx = b[col].idxmax()

        return {
            "label": label,
            "z": float(b.loc[idx, "z"]),
            "height_percent": float(b.loc[idx, "height_percent"]),
            "rule": f"max {col} in {low}-{high}%"
        }

    def pick_min(label, low, high, col):
        b = local_band(valid, low, high)
        b = b.dropna(subset=[col, "z", "height_percent"])
        if len(b) == 0:
            return None

        idx = b[col].idxmin()

        return {
            "label": label,
            "z": float(b.loc[idx, "z"]),
            "height_percent": float(b.loc[idx, "height_percent"]),
            "rule": f"min {col} in {low}-{high}%"
        }

    targets = [
        pick_min("Ankle", 2, 12, "max_perimeter"),
        pick_max("Calf", 10, 32, "max_perimeter"),
        pick_max("Thigh", 30, 47, "max_perimeter"),
        pick_max("Seat/Hip", 35, 56, "sum_perimeter"),
        pick_min("Waist", 45, 65, "sum_perimeter"),
        pick_max("Abdomen", 48, 68, "sum_perimeter"),
        pick_max("Chest", 65, 82, "sum_perimeter"),
        pick_min("Collar/Neck", 80, 92, "sum_perimeter"),
        pick_max("Head", 90, 99, "sum_perimeter"),
    ]

    return [t for t in targets if t is not None]


def get_extreme_points_3d(points_3d: np.ndarray):
    """
    Red dots:
    - min x and max x = width endpoints
    - min y and max y = depth endpoints
    """
    pts = np.asarray(points_3d)

    if len(pts) == 0:
        return np.empty((0, 3))

    idx_min_x = np.argmin(pts[:, 0])
    idx_max_x = np.argmax(pts[:, 0])
    idx_min_y = np.argmin(pts[:, 1])
    idx_max_y = np.argmax(pts[:, 1])

    selected = pts[[idx_min_x, idx_max_x, idx_min_y, idx_max_y]]

    rounded = np.round(selected, decimals=6)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)

    return selected[sorted(unique_idx)]


def save_biomarker_method_image(
    aligned_mesh: trimesh.Trimesh,
    slice_df: pd.DataFrame,
    image_path: Path,
    title: str = "",
    max_points: int = 9000,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)

    vertices = np.asarray(aligned_mesh.vertices)

    if len(vertices) > max_points:
        step = max(1, len(vertices) // max_points)
        vertices_plot = vertices[::step]
    else:
        vertices_plot = vertices

    targets = get_biomarker_slice_targets(slice_df)

    fig = plt.figure(figsize=(18, 9), dpi=160)

    ax_front = fig.add_subplot(1, 3, 1)
    ax_side = fig.add_subplot(1, 3, 2)
    ax_top = fig.add_subplot(1, 3, 3)

    ax_front.scatter(vertices_plot[:, 0], vertices_plot[:, 2], s=0.4, alpha=0.35)
    ax_front.set_title("Front")
    ax_front.set_xlabel("x")
    ax_front.set_ylabel("z")
    ax_front.set_aspect("equal", adjustable="box")
    ax_front.grid(True, alpha=0.2)

    ax_side.scatter(vertices_plot[:, 1], vertices_plot[:, 2], s=0.4, alpha=0.35)
    ax_side.set_title("Side")
    ax_side.set_xlabel("y")
    ax_side.set_ylabel("z")
    ax_side.set_aspect("equal", adjustable="box")
    ax_side.grid(True, alpha=0.2)

    ax_top.scatter(vertices_plot[:, 0], vertices_plot[:, 1], s=0.4, alpha=0.25)
    ax_top.set_title("Top")
    ax_top.set_xlabel("x")
    ax_top.set_ylabel("y")
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.grid(True, alpha=0.2)

    for target in targets:
        z = target["z"]
        loops3d = horizontal_slice_loops_3d(aligned_mesh, z)

        for pts in loops3d:
            pts = np.asarray(pts)

            if len(pts) < 2:
                continue

            red_pts = get_extreme_points_3d(pts)

            ax_front.plot(pts[:, 0], pts[:, 2], color="black", linewidth=1.2)
            ax_side.plot(pts[:, 1], pts[:, 2], color="black", linewidth=1.2)
            ax_top.plot(pts[:, 0], pts[:, 1], color="black", linewidth=1.2)

            if len(red_pts) > 0:
                ax_front.scatter(red_pts[:, 0], red_pts[:, 2], s=18, color="red", zorder=5)
                ax_side.scatter(red_pts[:, 1], red_pts[:, 2], s=18, color="red", zorder=5)
                ax_top.scatter(red_pts[:, 0], red_pts[:, 1], s=18, color="red", zorder=5)

    crotch_z, crotch_hp = detect_crotch_from_loop_count(slice_df)
    if pd.notna(crotch_z):
        ax_front.axhline(crotch_z, linestyle="--", linewidth=1)
        ax_side.axhline(crotch_z, linestyle="--", linewidth=1)

    fig.suptitle(
        f"{title}\nBlack loops = selected cross-sections, red dots = width/depth endpoint points",
        fontsize=12
    )

    fig.text(
        0.5,
        0.02,
        "Dashed line = crotch proxy when detected. Measurements are slicing-derived biomarker proxies.",
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)


def set_3d_axes_equal(ax, vertices):
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)

    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)


def save_3d_view_images(
    aligned_mesh: trimesh.Trimesh,
    image_base_path: Path,
    title: str = "",
    max_points: int = 12000,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_base_path = Path(image_base_path)
    image_base_path.parent.mkdir(parents=True, exist_ok=True)

    vertices = np.asarray(aligned_mesh.vertices)

    if len(vertices) > max_points:
        step = max(1, len(vertices) // max_points)
        vertices_plot = vertices[::step]
    else:
        vertices_plot = vertices

    views = {
        "main": (18, -70),
        "front": (15, -90),
        "side": (15, 0),
        "back": (15, 90),
        "top_tilt": (60, -70),
    }

    saved_paths = {}

    for view_name, (elev, azim) in views.items():
        fig = plt.figure(figsize=(9, 10), dpi=160)
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            vertices_plot[:, 0],
            vertices_plot[:, 1],
            vertices_plot[:, 2],
            s=0.4,
            alpha=0.45
        )

        ax.set_title(f"{title}\n3D PCA-aligned mesh view: {view_name}", fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        set_3d_axes_equal(ax, vertices)
        ax.view_init(elev=elev, azim=azim)

        if view_name == "main":
            out_path = image_base_path
        else:
            out_path = image_base_path.with_name(
                image_base_path.stem.replace("_3d_view", f"_3d_view_{view_name}") +
                image_base_path.suffix
            )

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

        saved_paths[view_name] = str(out_path)

    return saved_paths


# =============================================================================
# Interactive 3D HTML
# =============================================================================

def save_interactive_3d_html(
    aligned_mesh: trimesh.Trimesh,
    slice_df: pd.DataFrame,
    html_path: Path,
    title: str = "",
    max_points: int = 25000,
):
    """
    Save interactive 3D HTML with:
    - gray body mesh or point cloud
    - black selected biomarker slice loops
    - red endpoint dots
    - orange dashed crotch-proxy slice if detected

    Open the HTML in browser:
    - left mouse drag = rotate
    - scroll = zoom
    - right mouse drag = pan
    """
    import plotly.graph_objects as go

    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    vertices = np.asarray(aligned_mesh.vertices)
    faces = np.asarray(aligned_mesh.faces)

    data = []

    # Body mesh / point cloud
    if len(vertices) > max_points:
        step = max(1, len(vertices) // max_points)
        vertices_plot = vertices[::step]

        data.append(
            go.Scatter3d(
                x=vertices_plot[:, 0],
                y=vertices_plot[:, 1],
                z=vertices_plot[:, 2],
                mode="markers",
                marker=dict(
                    size=1.2,
                    opacity=0.35,
                    color="lightgray",
                ),
                name="Body mesh points",
            )
        )

    else:
        data.append(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.45,
                color="lightgray",
                name="Body mesh",
                flatshading=True,
            )
        )

    # Biomarker selected slices
    targets = get_biomarker_slice_targets(slice_df)

    all_red_points = []

    for target in targets:
        z = target["z"]
        label = target.get("label", "Selected slice")
        rule = target.get("rule", "")

        loops3d = horizontal_slice_loops_3d(aligned_mesh, z)

        first_loop_for_label = True

        for pts in loops3d:
            pts = np.asarray(pts)

            if len(pts) < 2:
                continue

            pts_closed = np.vstack([pts, pts[0]])

            data.append(
                go.Scatter3d(
                    x=pts_closed[:, 0],
                    y=pts_closed[:, 1],
                    z=pts_closed[:, 2],
                    mode="lines",
                    line=dict(
                        color="black",
                        width=5,
                    ),
                    name=f"{label} slice",
                    text=[f"{label}: {rule}"] * len(pts_closed),
                    hoverinfo="text",
                    showlegend=first_loop_for_label,
                )
            )

            first_loop_for_label = False

            red_pts = get_extreme_points_3d(pts)

            if len(red_pts) > 0:
                all_red_points.append(red_pts)

    # Red endpoint dots combined into one trace
    if all_red_points:
        red_all = np.vstack(all_red_points)

        data.append(
            go.Scatter3d(
                x=red_all[:, 0],
                y=red_all[:, 1],
                z=red_all[:, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color="red",
                    opacity=0.95,
                ),
                name="Width/depth endpoint dots",
                text=["Width/depth endpoint"] * len(red_all),
                hoverinfo="text",
            )
        )

    # Crotch proxy as orange dashed slice
    crotch_z, crotch_hp = detect_crotch_from_loop_count(slice_df)

    if pd.notna(crotch_z):
        loops3d = horizontal_slice_loops_3d(aligned_mesh, crotch_z)

        first_crotch_loop = True

        for pts in loops3d:
            pts = np.asarray(pts)

            if len(pts) < 2:
                continue

            pts_closed = np.vstack([pts, pts[0]])

            data.append(
                go.Scatter3d(
                    x=pts_closed[:, 0],
                    y=pts_closed[:, 1],
                    z=pts_closed[:, 2],
                    mode="lines",
                    line=dict(
                        color="orange",
                        width=4,
                        dash="dash",
                    ),
                    name="Crotch proxy",
                    text=[f"Crotch proxy: {crotch_hp:.2f}% height"] * len(pts_closed),
                    hoverinfo="text",
                    showlegend=first_crotch_loop,
                )
            )

            first_crotch_loop = False

    fig = go.Figure(data=data)

    fig.update_layout(
        title=(
            f"{title}<br>"
            "Interactive 3D PCA-aligned mesh with selected biomarker slices"
        ),
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        ),
        legend=dict(
            x=0.02,
            y=0.98,
        ),
        margin=dict(l=0, r=0, b=0, t=75),
    )

    fig.write_html(
        str(html_path),
        include_plotlyjs="cdn",
        full_html=True,
    )

    return str(html_path)


# =============================================================================
# Contact sheet
# =============================================================================

def save_contact_sheet(
    image_dir: Path,
    output_file: Path,
    pattern: str,
    thumb_width: int = 520,
    cols: int = 3,
    padding: int = 30,
):
    from PIL import Image

    image_dir = Path(image_dir)
    output_file = Path(output_file)

    files = sorted(image_dir.glob(pattern))

    if not files:
        print(f"No images found for contact sheet pattern {pattern} in {image_dir}")
        return None

    thumbs = []

    for f in files:
        try:
            img = Image.open(f).convert("RGB")
            w, h = img.size

            scale = thumb_width / float(w)
            thumb_height = int(h * scale)

            img = img.resize((thumb_width, thumb_height), Image.LANCZOS)
            thumbs.append(img)

        except Exception as e:
            print(f"Skipping image {f}: {e}")

    if not thumbs:
        print("No valid images loaded for contact sheet.")
        return None

    rows = math.ceil(len(thumbs) / cols)
    max_thumb_height = max(img.height for img in thumbs)

    sheet_width = cols * thumb_width + (cols + 1) * padding
    sheet_height = rows * max_thumb_height + (rows + 1) * padding

    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")

    for idx, img in enumerate(thumbs):
        row = idx // cols
        col = idx % cols

        x = padding + col * (thumb_width + padding)
        y = padding + row * (max_thumb_height + padding)

        sheet.paste(img, (x, y))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_file, quality=95)

    print(f"Saved contact sheet: {output_file}")
    print(f"Images combined: {len(thumbs)}")
    print(f"Grid: {rows} rows x {cols} columns")

    return output_file


# =============================================================================
# Biomarker extraction
# =============================================================================

def safe_num(series):
    return pd.to_numeric(series, errors="coerce")


def clean_slice_df(df):
    numeric_cols = [
        "slice_index",
        "z",
        "height_percent",
        "num_loops",
        "max_perimeter",
        "sum_perimeter",
        "max_area",
        "sum_area",
        "width",
        "depth",
        "left_max_perimeter",
        "right_max_perimeter",
        "left_sum_area",
        "right_sum_area",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = safe_num(df[col])

    return df.sort_values("height_percent").reset_index(drop=True)


def band(df, low, high):
    return df[
        (df["height_percent"] >= low) &
        (df["height_percent"] <= high)
    ].copy()


def safe_max(series):
    series = safe_num(series).dropna()
    return float(series.max()) if len(series) else np.nan


def safe_min(series):
    series = safe_num(series).dropna()
    return float(series.min()) if len(series) else np.nan


def safe_mean(series):
    series = safe_num(series).dropna()
    return float(series.mean()) if len(series) else np.nan


def fallback(value, fallback_value):
    if pd.notna(value):
        return value
    return fallback_value


def integrate_volume(df, area_col="sum_area"):
    if df.empty:
        return np.nan

    if area_col not in df.columns:
        return np.nan

    temp = df[["z", area_col]].dropna()

    if len(temp) < 2:
        return np.nan

    z = temp["z"].to_numpy()
    area = temp[area_col].to_numpy()

    return float(trapz_integral(area, z))


def integrate_surface_proxy(df):
    if df.empty:
        return np.nan

    temp = df[["z", "sum_perimeter"]].dropna()

    if len(temp) < 2:
        return np.nan

    z = temp["z"].to_numpy()
    perimeter = temp["sum_perimeter"].to_numpy()

    return float(trapz_integral(perimeter, z))


def scale_to_cm_values(row, scale):
    length_like = [
        "Height (cm)",
        "Abdomen Circumference",
        "Ankle Circumference Left",
        "Arm Length Left",
        "Bicep Circumference Left",
        "Calf Circumference Left",
        "Chest",
        "Collar Circumference",
        "Forearm Circumference Left",
        "Head Circumference",
        "Hip Circumference",
        "Horizontal Waist",
        "Inseam Left",
        "MidThigh Circumference Left",
        "Narrow Waist",
        "Outside Leg Length Left",
        "Seat Circumference",
        "Thigh Circumference Left",
        "Upper Arm Circumference Left",
        "Waist Circumference",
        "Ankle Circumference Right",
        "Arm Length Right",
        "Bicep Circumference Right",
        "Calf Circumference Right",
        "Forearm Circumference Right",
        "Inseam Right",
        "MidThigh Circumference Right",
        "Outside Leg Length Right",
        "Thigh Circumference Right",
        "Upper Arm Circumference Right",
    ]

    area_like = [
        "Surface Area Arm Left",
        "Surface Area Leg Left",
        "Surface Area Torso",
        "Surface Area Total",
        "Surface Area Arm Right",
        "Surface Area Leg Right",
    ]

    volume_like = [
        "Arm Volume Left",
        "Leg Volume Left",
        "Torso Volume",
        "Volume",
        "Arm Volume Right",
        "Leg Volume Right",
    ]

    for col in length_like:
        if col in row and pd.notna(row[col]):
            row[col] = row[col] * scale

    for col in area_like:
        if col in row and pd.notna(row[col]):
            row[col] = row[col] * (scale ** 2)

    for col in volume_like:
        if col in row and pd.notna(row[col]):
            row[col] = row[col] * (scale ** 3)

    return row


def extract_42_biomarkers_from_df(
    df: pd.DataFrame,
    subject_id: str,
    source_file: str,
    height_scale_to_cm: float,
):
    df = clean_slice_df(df)
    valid = df[df["num_loops"] > 0].copy()

    row = {
        "subject_id": subject_id,
        "source_file": source_file,
    }

    if len(valid) == 0:
        for col in BIOMARKER_COLUMNS:
            row[col] = np.nan
        return row

    z_min = safe_min(valid["z"])
    z_max = safe_max(valid["z"])
    height = z_max - z_min if pd.notna(z_min) and pd.notna(z_max) else np.nan

    crotch_z, crotch_hp = detect_crotch_from_loop_count(valid)

    foot_ankle = band(valid, 2, 12)
    lower_leg = band(valid, 10, 32)
    thigh = band(valid, 30, 47)
    lower_trunk_hip = band(valid, 38, 56)
    seat_band = band(valid, 35, 50)
    abdomen = band(valid, 48, 68)
    waist = band(valid, 45, 65)
    trunk = band(valid, 45, 82)
    chest = band(valid, 65, 82)
    collar_neck = band(valid, 80, 92)
    head = band(valid, 90, 99)

    abdomen_circ = safe_max(abdomen["sum_perimeter"])
    chest_circ = safe_max(chest["sum_perimeter"])
    collar_circ = safe_min(collar_neck["sum_perimeter"])
    head_circ = safe_max(head["sum_perimeter"])
    hip_circ = safe_max(lower_trunk_hip["sum_perimeter"])
    horizontal_waist = safe_mean(waist["sum_perimeter"])
    narrow_waist = safe_min(waist["sum_perimeter"])
    waist_circ = narrow_waist
    seat_circ = safe_max(seat_band["sum_perimeter"])

    ankle_single = safe_min(foot_ankle["max_perimeter"])
    calf_single = safe_max(lower_leg["max_perimeter"])
    thigh_single = safe_max(thigh["max_perimeter"])
    mid_thigh_single = safe_mean(band(valid, 34, 42)["max_perimeter"])

    ankle_left = fallback(safe_min(foot_ankle["left_max_perimeter"]), ankle_single)
    ankle_right = fallback(safe_min(foot_ankle["right_max_perimeter"]), ankle_single)

    calf_left = fallback(safe_max(lower_leg["left_max_perimeter"]), calf_single)
    calf_right = fallback(safe_max(lower_leg["right_max_perimeter"]), calf_single)

    thigh_left = fallback(safe_max(thigh["left_max_perimeter"]), thigh_single)
    thigh_right = fallback(safe_max(thigh["right_max_perimeter"]), thigh_single)

    mid_thigh_left = fallback(
        safe_mean(band(valid, 34, 42)["left_max_perimeter"]),
        mid_thigh_single
    )

    mid_thigh_right = fallback(
        safe_mean(band(valid, 34, 42)["right_max_perimeter"]),
        mid_thigh_single
    )

    forearm_proxy = safe_min(band(valid, 45, 65)["max_perimeter"])
    bicep_proxy = safe_max(band(valid, 55, 75)["max_perimeter"])
    upper_arm_proxy = bicep_proxy

    if pd.notna(crotch_z) and pd.notna(z_min):
        inseam_proxy = crotch_z - z_min
    else:
        inseam_proxy = 0.45 * height if pd.notna(height) else np.nan

    outside_leg_proxy = 0.53 * height if pd.notna(height) else np.nan
    arm_length_proxy = 0.30 * height if pd.notna(height) else np.nan

    total_volume = integrate_volume(valid)
    torso_volume = integrate_volume(trunk)

    leg_region = band(valid, 2, 47)
    leg_volume_total = integrate_volume(leg_region)

    left_leg_volume = integrate_volume(leg_region, area_col="left_sum_area")
    right_leg_volume = integrate_volume(leg_region, area_col="right_sum_area")

    if pd.isna(left_leg_volume) or left_leg_volume == 0:
        left_leg_volume = leg_volume_total / 2.0 if pd.notna(leg_volume_total) else np.nan

    if pd.isna(right_leg_volume) or right_leg_volume == 0:
        right_leg_volume = leg_volume_total / 2.0 if pd.notna(leg_volume_total) else np.nan

    upper_body_region = band(valid, 45, 82)
    upper_body_volume = integrate_volume(upper_body_region)
    arm_volume_total = upper_body_volume * 0.18 if pd.notna(upper_body_volume) else np.nan
    arm_volume_each = arm_volume_total / 2.0 if pd.notna(arm_volume_total) else np.nan

    surface_total = integrate_surface_proxy(valid)
    surface_torso = integrate_surface_proxy(trunk)
    surface_leg_total = integrate_surface_proxy(leg_region)
    surface_leg_each = surface_leg_total / 2.0 if pd.notna(surface_leg_total) else np.nan

    upper_body_surface = integrate_surface_proxy(upper_body_region)
    surface_arm_total = upper_body_surface * 0.18 if pd.notna(upper_body_surface) else np.nan
    surface_arm_each = surface_arm_total / 2.0 if pd.notna(surface_arm_total) else np.nan

    row.update({
        "Height (cm)": height,
        "Abdomen Circumference": abdomen_circ,

        "Ankle Circumference Left": ankle_left,
        "Arm Length Left": arm_length_proxy,
        "Arm Volume Left": arm_volume_each,
        "Bicep Circumference Left": bicep_proxy,
        "Calf Circumference Left": calf_left,

        "Chest": chest_circ,
        "Collar Circumference": collar_circ,
        "Forearm Circumference Left": forearm_proxy,
        "Head Circumference": head_circ,
        "Hip Circumference": hip_circ,
        "Horizontal Waist": horizontal_waist,

        "Inseam Left": inseam_proxy,
        "Leg Volume Left": left_leg_volume,
        "MidThigh Circumference Left": mid_thigh_left,
        "Narrow Waist": narrow_waist,
        "Outside Leg Length Left": outside_leg_proxy,
        "Seat Circumference": seat_circ,

        "Surface Area Arm Left": surface_arm_each,
        "Surface Area Leg Left": surface_leg_each,
        "Surface Area Torso": surface_torso,
        "Surface Area Total": surface_total,

        "Thigh Circumference Left": thigh_left,
        "Torso Volume": torso_volume,
        "Upper Arm Circumference Left": upper_arm_proxy,
        "Volume": total_volume,
        "Waist Circumference": waist_circ,

        "Ankle Circumference Right": ankle_right,
        "Arm Length Right": arm_length_proxy,
        "Arm Volume Right": arm_volume_each,
        "Bicep Circumference Right": bicep_proxy,
        "Calf Circumference Right": calf_right,
        "Forearm Circumference Right": forearm_proxy,
        "Inseam Right": inseam_proxy,
        "Leg Volume Right": right_leg_volume,
        "MidThigh Circumference Right": mid_thigh_right,
        "Outside Leg Length Right": outside_leg_proxy,
        "Surface Area Arm Right": surface_arm_each,
        "Surface Area Leg Right": surface_leg_each,
        "Thigh Circumference Right": thigh_right,
        "Upper Arm Circumference Right": upper_arm_proxy,
    })

    row = scale_to_cm_values(row, height_scale_to_cm)

    ordered = {
        "subject_id": row["subject_id"],
        "source_file": row["source_file"],
    }

    for col in BIOMARKER_COLUMNS:
        ordered[col] = row.get(col, np.nan)

    return ordered


# =============================================================================
# One OBJ processing
# =============================================================================

def process_one_obj(
    obj_file: Path,
    output_dir: Path,
    n_slices: int,
    height_scale_to_cm: float,
    save_images: bool,
    save_aligned_obj: bool,
):
    obj_file = Path(obj_file)
    output_dir = Path(output_dir)

    name = safe_stem(obj_file)

    slices_dir = output_dir / "slices"
    aligned_dir = output_dir / "aligned_obj"
    images_dir = output_dir / "images"

    slices_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(obj_file)
    aligned = pca_align_to_z(mesh)

    df, height = create_slice_dataframe(aligned, obj_file, n_slices=n_slices)

    slice_csv = slices_dir / f"{name}_slices.csv"
    df.to_csv(slice_csv, index=False)

    aligned_obj = ""
    if save_aligned_obj:
        aligned_obj_path = aligned_dir / f"{name}_aligned.obj"
        aligned.export(aligned_obj_path)
        aligned_obj = str(aligned_obj_path)

    slice_profile_image = ""
    biomarker_method_image = ""
    three_d_view_image = ""
    three_d_view_front = ""
    three_d_view_side = ""
    three_d_view_back = ""
    three_d_view_top_tilt = ""
    interactive_3d_html = ""

    if save_images:
        slice_profile_path = images_dir / f"{name}_slice_profile.png"
        save_slice_profile_image(df, slice_profile_path, title=obj_file.name)
        slice_profile_image = str(slice_profile_path)

        method_image_path = images_dir / f"{name}_biomarker_method.png"
        save_biomarker_method_image(
            aligned_mesh=aligned,
            slice_df=df,
            image_path=method_image_path,
            title=obj_file.name,
        )
        biomarker_method_image = str(method_image_path)

        three_d_base_path = images_dir / f"{name}_3d_view.png"
        view_paths = save_3d_view_images(
            aligned_mesh=aligned,
            image_base_path=three_d_base_path,
            title=obj_file.name,
        )

        three_d_view_image = view_paths.get("main", "")
        three_d_view_front = view_paths.get("front", "")
        three_d_view_side = view_paths.get("side", "")
        three_d_view_back = view_paths.get("back", "")
        three_d_view_top_tilt = view_paths.get("top_tilt", "")

        interactive_html_path = images_dir / f"{name}_interactive_3d.html"
        interactive_3d_html = save_interactive_3d_html(
            aligned_mesh=aligned,
            slice_df=df,
            html_path=interactive_html_path,
            title=obj_file.name,
        )

    biomarker_row = extract_42_biomarkers_from_df(
        df=df,
        subject_id=name,
        source_file=str(obj_file),
        height_scale_to_cm=height_scale_to_cm,
    )

    valid = df["num_loops"] > 0
    crotch_z, crotch_hp = detect_crotch_from_loop_count(df)

    summary = {
        "source_file": str(obj_file),
        "subject_id": name,
        "slice_csv": str(slice_csv),
        "aligned_obj": aligned_obj,
        "slice_profile_image": slice_profile_image,
        "biomarker_method_image": biomarker_method_image,
        "three_d_view_image": three_d_view_image,
        "three_d_view_front": three_d_view_front,
        "three_d_view_side": three_d_view_side,
        "three_d_view_back": three_d_view_back,
        "three_d_view_top_tilt": three_d_view_top_tilt,
        "interactive_3d_html": interactive_3d_html,
        "num_vertices_original": int(len(mesh.vertices)),
        "num_faces_original": int(len(mesh.faces)),
        "num_vertices_aligned": int(len(aligned.vertices)),
        "num_faces_aligned": int(len(aligned.faces)),
        "height_mesh_units": height,
        "crotch_z_mesh_units": crotch_z,
        "crotch_height_percent": crotch_hp,
        "n_slices_requested": int(n_slices),
        "n_slices_with_loops": int(valid.sum()),
        "percent_slices_with_loops": float(100.0 * valid.mean()),
        "mean_num_loops": float(df["num_loops"].mean()),
        "max_sum_perimeter": float(df["sum_perimeter"].max(skipna=True)),
        "max_sum_area": float(df["sum_area"].max(skipna=True)),
    }

    return summary, biomarker_row


def save_biomarker_heatmap(df: pd.DataFrame, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = Path(output_dir) / "biomarker_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    biomarker_df = df[BIOMARKER_COLUMNS].copy()
    biomarker_df = biomarker_df.apply(pd.to_numeric, errors="coerce")

    norm_df = (biomarker_df - biomarker_df.mean()) / biomarker_df.std()
    norm_df = norm_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    plt.figure(figsize=(18, max(6, 0.25 * len(norm_df))))
    plt.imshow(norm_df, aspect="auto")
    plt.colorbar(label="Standardized value")
    plt.yticks(range(len(df)), df["subject_id"], fontsize=6)
    plt.xticks(range(len(BIOMARKER_COLUMNS)), BIOMARKER_COLUMNS, rotation=90, fontsize=7)
    plt.title("42 Slice-Derived Biomarker Heatmap")
    plt.tight_layout()

    path = plot_dir / "slice_42_biomarker_heatmap.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()

    return path


# =============================================================================
# Full batch pipeline
# =============================================================================

def run_pipeline(
    input_path: Path,
    output_dir: Path,
    n_slices: int,
    recursive: bool,
    height_scale_to_cm: float,
    save_images: bool,
    save_aligned_obj: bool,
):
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    obj_files = collect_obj_files(input_path, recursive=recursive)

    if not obj_files:
        raise FileNotFoundError(f"No OBJ files found in: {input_path}")

    print("=" * 90)
    print("FULL OBJ TO SLICING TO 42 BIOMARKERS TO INTERACTIVE 3D PIPELINE")
    print("=" * 90)
    print(f"Input path: {input_path}")
    print(f"Number of OBJ files: {len(obj_files)}")
    print(f"Output directory: {output_dir}")
    print(f"Number of slices per OBJ: {n_slices}")
    print(f"Height scale to cm: {height_scale_to_cm}")
    print("=" * 90)

    summaries = []
    biomarker_rows = []
    failures = []

    for idx, obj_file in enumerate(obj_files, start=1):
        print(f"\n[{idx}/{len(obj_files)}] Processing:")
        print(f"  {obj_file}")

        try:
            summary, biomarker_row = process_one_obj(
                obj_file=obj_file,
                output_dir=output_dir,
                n_slices=n_slices,
                height_scale_to_cm=height_scale_to_cm,
                save_images=save_images,
                save_aligned_obj=save_aligned_obj,
            )

            summaries.append(summary)
            biomarker_rows.append(biomarker_row)

            print(f"  Slice CSV: {summary['slice_csv']}")
            if summary["aligned_obj"]:
                print(f"  Aligned OBJ: {summary['aligned_obj']}")
            if summary["slice_profile_image"]:
                print(f"  Slice profile image: {summary['slice_profile_image']}")
            if summary["biomarker_method_image"]:
                print(f"  Biomarker method image: {summary['biomarker_method_image']}")
            if summary["three_d_view_image"]:
                print(f"  3D view image: {summary['three_d_view_image']}")
            if summary["interactive_3d_html"]:
                print(f"  Interactive 3D HTML: {summary['interactive_3d_html']}")

            print(f"  Crotch height proxy: {summary['crotch_height_percent']}")
            print(f"  Slices with loops: {summary['n_slices_with_loops']}/{n_slices}")
            print("  Status: PASS")

        except Exception as e:
            print("  Status: FAILED")
            print(f"  Error: {e}")

            failures.append({
                "source_file": str(obj_file),
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    summary_csv = output_dir / "full_pipeline_summary.csv"
    failure_csv = output_dir / "full_pipeline_failures.csv"
    biomarker_csv = output_dir / "slice_42_biomarkers.csv"

    if summaries:
        pd.DataFrame(summaries).to_csv(summary_csv, index=False)
        print(f"\nSaved summary CSV: {summary_csv}")

    if biomarker_rows:
        biomarker_df = pd.DataFrame(biomarker_rows)
        expected_cols = ["subject_id", "source_file"] + BIOMARKER_COLUMNS
        biomarker_df = biomarker_df.reindex(columns=expected_cols)
        biomarker_df.to_csv(biomarker_csv, index=False)

        print(f"Saved biomarker CSV: {biomarker_csv}")
        print(f"Biomarker CSV shape: {biomarker_df.shape}")
        print(f"Biomarker columns: {len(BIOMARKER_COLUMNS)}")

        if save_images:
            heatmap_path = save_biomarker_heatmap(biomarker_df, output_dir)
            print(f"Saved biomarker heatmap: {heatmap_path}")

            contact_sheet_method = save_contact_sheet(
                image_dir=output_dir / "images",
                output_file=output_dir / "all_biomarker_method_contact_sheet.png",
                pattern="*_biomarker_method.png",
                thumb_width=520,
                cols=3,
                padding=30,
            )

            if contact_sheet_method is not None:
                print(f"Saved biomarker contact sheet: {contact_sheet_method}")

            contact_sheet_3d = save_contact_sheet(
                image_dir=output_dir / "images",
                output_file=output_dir / "all_3d_view_contact_sheet.png",
                pattern="*_3d_view.png",
                thumb_width=360,
                cols=4,
                padding=25,
            )

            if contact_sheet_3d is not None:
                print(f"Saved 3D view contact sheet: {contact_sheet_3d}")

    if failures:
        pd.DataFrame(failures).to_csv(failure_csv, index=False)
        print(f"Saved failures CSV: {failure_csv}")

    print("\n" + "=" * 90)
    print("PIPELINE COMPLETE")
    print("=" * 90)
    print(f"Successful files: {len(summaries)}")
    print(f"Failed files: {len(failures)}")
    print(f"Final biomarker file: {biomarker_csv}")
    print(f"Method contact sheet: {output_dir / 'all_biomarker_method_contact_sheet.png'}")
    print(f"3D contact sheet: {output_dir / 'all_3d_view_contact_sheet.png'}")

    return summaries, biomarker_rows, failures


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Full OBJ to slicing profiles to 42 biomarkers pipeline."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input OBJ file or folder containing OBJ files.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Treat input as folder and process all OBJ files inside.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If input is a folder, search recursively for OBJ files.",
    )

    parser.add_argument(
        "--n-slices",
        type=int,
        default=200,
        help="Number of slices along body height.",
    )

    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output folder.",
    )

    parser.add_argument(
        "--height-scale-to-cm",
        type=float,
        default=1.0,
        help=(
            "Scale factor to convert mesh units to cm. "
            "Use 1 if OBJ units are already cm. "
            "Use 10 if OBJ units are decimeters. "
            "Use 100 if OBJ units are meters."
        ),
    )

    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Do not save PNG and HTML images.",
    )

    parser.add_argument(
        "--no-aligned-obj",
        action="store_true",
        help="Do not save aligned OBJ files.",
    )

    args = parser.parse_args(argv)

    input_path = Path(args.input)

    if args.all and not input_path.is_dir():
        raise ValueError("--all was passed, but --input is not a folder.")

    run_pipeline(
        input_path=input_path,
        output_dir=Path(args.output_dir),
        n_slices=args.n_slices,
        recursive=args.recursive,
        height_scale_to_cm=args.height_scale_to_cm,
        save_images=not args.no_images,
        save_aligned_obj=not args.no_aligned_obj,
    )


if __name__ == "__main__":
    # In a normal terminal run, argparse should behave normally.
    # In Jupyter/Colab, executing the whole file as a cell gives no --input
    # argument and can produce a confusing IPython traceback.  In that case,
    # do not auto-run; call main([...]) manually as shown below.
    import sys

    running_in_notebook = "ipykernel" in sys.modules

    if running_in_notebook and "--input" not in sys.argv:
        print(
            "Notebook detected. The pipeline was loaded but not run because "
            "--input was not provided.\n\n"
            "Run it like this in a notebook cell:\n"
            "main([\n"
            "    '--input', '/content/data/obj',\n"
            "    '--recursive',\n"
            "    '--n-slices', '200',\n"
            "    '--output-dir', '/content/results',\n"
            "])\n\n"
            "Or from a shell:\n"
            "python slice.py --input /content/data/obj --recursive --n-slices 200 --output-dir /content/results"
        )
    else:
        main()
