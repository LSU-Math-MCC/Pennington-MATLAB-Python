

"""
FULL OBJ → PCA ALIGNMENT → SLICING → 42 BIOMARKERS → PNG METHOD IMAGES → contact sheet

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

python3 -m slice \
  --input /Python_Fall2025/model_files/OBJ \
  --all \
  --recursive \
  --n-slices 200
"""

import argparse
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
    """
    Collect OBJ files from one file or folder.

    It supports:
    - lowercase .obj
    - uppercase .OBJ
    - nested folders when --recursive is used
    """
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

    # Keep largest connected component.
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

    # Fix upside-down direction.
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


def horizontal_slice_loops(mesh: trimesh.Trimesh, z: float):
    """
    Return 2D slice loops for measurement.
    """
    section = mesh.section(
        plane_origin=[0.0, 0.0, float(z)],
        plane_normal=[0.0, 0.0, 1.0],
    )

    if section is None:
        return []

    try:
        path2d, _ = section.to_planar()
    except Exception:
        try:
            path2d, _ = section.to_2D()
        except Exception:
            return []

    loops = []

    for entity in path2d.entities:
        try:
            pts = path2d.vertices[np.asarray(entity.points)]
            if len(pts) >= 3:
                loops.append(pts)
        except Exception:
            continue

    return loops


def horizontal_slice_loops_3d(mesh: trimesh.Trimesh, z: float):
    """
    Return 3D slice loops for visualization.
    """
    section = mesh.section(
        plane_origin=[0.0, 0.0, float(z)],
        plane_normal=[0.0, 0.0, 1.0],
    )

    if section is None:
        return []

    loops = []
    vertices = np.asarray(section.vertices)

    for entity in section.entities:
        try:
            pts = vertices[np.asarray(entity.points)]
            if len(pts) >= 2:
                loops.append(pts)
        except Exception:
            continue

    return loops


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
        loops = horizontal_slice_loops(aligned, z)
        height_percent = 100.0 * (z - z_min) / height

        if not loops:
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
            })
            continue

        perimeters = [closed_loop_perimeter(loop) for loop in loops]
        areas = [polygon_area(loop) for loop in loops]
        all_points = np.vstack(loops)

        rows.append({
            "source_file": str(obj_file),
            "slice_index": i,
            "z": z,
            "height_percent": height_percent,
            "num_loops": len(loops),
            "max_perimeter": float(np.nanmax(perimeters)),
            "sum_perimeter": float(np.nansum(perimeters)),
            "max_area": float(np.nanmax(areas)),
            "sum_area": float(np.nansum(areas)),
            "width": float(all_points[:, 0].max() - all_points[:, 0].min()),
            "depth": float(all_points[:, 1].max() - all_points[:, 1].min()),
        })

    return pd.DataFrame(rows), height


# =============================================================================
# Images
# =============================================================================

def save_slice_profile_image(df: pd.DataFrame, image_path: Path, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), dpi=150, sharex=True)

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
    axes[3].set_xlabel("Body Height Percent")
    axes[3].grid(True, alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)


def get_biomarker_slice_targets(df: pd.DataFrame):
    """
    Choose selected slices for method visualization and biomarker proxies.

    The labels are only used internally.
    They are not drawn on the image.
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
    Get red-dot points from one 3D slice loop.

    These points represent:
    - min x and max x: width endpoints
    - min y and max y: depth endpoints
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
    """
    Save PNG explaining slicing-derived biomarkers.

    Image style:
    - No biomarker names such as head, waist, chest.
    - Black lines are selected actual slice loops.
    - Red dots are point-to-point extreme locations from each slice.
    """
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

    # Body point cloud
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

    # Selected biomarker slice loops
    for target in targets:
        z = target["z"]
        loops3d = horizontal_slice_loops_3d(aligned_mesh, z)

        for pts in loops3d:
            pts = np.asarray(pts)

            if len(pts) < 2:
                continue

            red_pts = get_extreme_points_3d(pts)

            # Black slice loop/segment in front view.
            ax_front.plot(
                pts[:, 0],
                pts[:, 2],
                color="black",
                linewidth=1.2
            )

            # Black slice loop/segment in side view.
            ax_side.plot(
                pts[:, 1],
                pts[:, 2],
                color="black",
                linewidth=1.2
            )

            # Black true cross-section loop in top view.
            ax_top.plot(
                pts[:, 0],
                pts[:, 1],
                color="black",
                linewidth=1.2
            )

            # Red dots: front view uses x-z.
            if len(red_pts) > 0:
                ax_front.scatter(
                    red_pts[:, 0],
                    red_pts[:, 2],
                    s=18,
                    color="red",
                    zorder=5
                )

                # Red dots: side view uses y-z.
                ax_side.scatter(
                    red_pts[:, 1],
                    red_pts[:, 2],
                    s=18,
                    color="red",
                    zorder=5
                )

                # Red dots: top view uses x-y.
                ax_top.scatter(
                    red_pts[:, 0],
                    red_pts[:, 1],
                    s=18,
                    color="red",
                    zorder=5
                )

    fig.suptitle(
        f"{title}\nBlack loops = selected cross-sections, red dots = width/depth endpoint points",
        fontsize=12
    )

    note = (
        "No anatomical labels shown. Red dots mark extreme points used for point-to-point width/depth on selected slices."
    )

    fig.text(0.5, 0.02, note, ha="center", fontsize=8)

    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Contact sheet / collage
# =============================================================================

def save_contact_sheet(
    image_dir: Path,
    output_file: Path,
    pattern: str = "*_biomarker_method.png",
    thumb_width: int = 520,
    cols: int = 3,
    padding: int = 30,
):
    """
    Combine all biomarker-method PNG images into one big collage/contact sheet.

    No file names are written on the collage.
    """
    from PIL import Image
    import math

    image_dir = Path(image_dir)
    output_file = Path(output_file)

    files = sorted(image_dir.glob(pattern))

    if not files:
        print(f"No images found for contact sheet in {image_dir}")
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


def integrate_volume(df, area_col="sum_area"):
    if df.empty:
        return np.nan

    temp = df[["z", area_col]].dropna()

    if len(temp) < 2:
        return np.nan

    z = temp["z"].to_numpy()
    area = temp[area_col].to_numpy()

    return float(np.trapezoid(area, z))


def integrate_surface_proxy(df):
    if df.empty:
        return np.nan

    temp = df[["z", "sum_perimeter"]].dropna()

    if len(temp) < 2:
        return np.nan

    z = temp["z"].to_numpy()
    perimeter = temp["sum_perimeter"].to_numpy()

    return float(np.trapezoid(perimeter, z))


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
    ankle_single = safe_min(foot_ankle["max_perimeter"])
    calf_single = safe_max(lower_leg["max_perimeter"])
    chest_circ = safe_max(chest["sum_perimeter"])
    collar_circ = safe_min(collar_neck["sum_perimeter"])
    head_circ = safe_max(head["sum_perimeter"])
    hip_circ = safe_max(lower_trunk_hip["sum_perimeter"])
    horizontal_waist = safe_mean(waist["sum_perimeter"])
    narrow_waist = safe_min(waist["sum_perimeter"])
    waist_circ = narrow_waist
    seat_circ = safe_max(seat_band["sum_perimeter"])
    thigh_single = safe_max(thigh["max_perimeter"])
    mid_thigh_single = safe_mean(band(valid, 34, 42)["max_perimeter"])

    forearm_proxy = safe_min(band(valid, 45, 65)["max_perimeter"])
    bicep_proxy = safe_max(band(valid, 55, 75)["max_perimeter"])
    upper_arm_proxy = bicep_proxy

    arm_length_proxy = 0.30 * height if pd.notna(height) else np.nan
    inseam_proxy = 0.45 * height if pd.notna(height) else np.nan
    outside_leg_proxy = 0.53 * height if pd.notna(height) else np.nan

    total_volume = integrate_volume(valid)
    torso_volume = integrate_volume(trunk)

    leg_region = band(valid, 2, 47)
    leg_volume_total = integrate_volume(leg_region)

    upper_body_region = band(valid, 45, 82)
    upper_body_volume = integrate_volume(upper_body_region)
    arm_volume_total = upper_body_volume * 0.18 if pd.notna(upper_body_volume) else np.nan

    leg_volume_each = leg_volume_total / 2.0 if pd.notna(leg_volume_total) else np.nan
    arm_volume_each = arm_volume_total / 2.0 if pd.notna(arm_volume_total) else np.nan

    surface_total = integrate_surface_proxy(valid)
    surface_torso = integrate_surface_proxy(trunk)
    surface_leg_total = integrate_surface_proxy(leg_region)

    upper_body_surface = integrate_surface_proxy(upper_body_region)
    surface_arm_total = upper_body_surface * 0.18 if pd.notna(upper_body_surface) else np.nan

    surface_leg_each = surface_leg_total / 2.0 if pd.notna(surface_leg_total) else np.nan
    surface_arm_each = surface_arm_total / 2.0 if pd.notna(surface_arm_total) else np.nan

    row.update({
        "Height (cm)": height,
        "Abdomen Circumference": abdomen_circ,

        "Ankle Circumference Left": ankle_single,
        "Arm Length Left": arm_length_proxy,
        "Arm Volume Left": arm_volume_each,
        "Bicep Circumference Left": bicep_proxy,
        "Calf Circumference Left": calf_single,

        "Chest": chest_circ,
        "Collar Circumference": collar_circ,
        "Forearm Circumference Left": forearm_proxy,
        "Head Circumference": head_circ,
        "Hip Circumference": hip_circ,
        "Horizontal Waist": horizontal_waist,

        "Inseam Left": inseam_proxy,
        "Leg Volume Left": leg_volume_each,
        "MidThigh Circumference Left": mid_thigh_single,
        "Narrow Waist": narrow_waist,
        "Outside Leg Length Left": outside_leg_proxy,
        "Seat Circumference": seat_circ,

        "Surface Area Arm Left": surface_arm_each,
        "Surface Area Leg Left": surface_leg_each,
        "Surface Area Torso": surface_torso,
        "Surface Area Total": surface_total,

        "Thigh Circumference Left": thigh_single,
        "Torso Volume": torso_volume,
        "Upper Arm Circumference Left": upper_arm_proxy,
        "Volume": total_volume,
        "Waist Circumference": waist_circ,

        "Ankle Circumference Right": ankle_single,
        "Arm Length Right": arm_length_proxy,
        "Arm Volume Right": arm_volume_each,
        "Bicep Circumference Right": bicep_proxy,
        "Calf Circumference Right": calf_single,
        "Forearm Circumference Right": forearm_proxy,
        "Inseam Right": inseam_proxy,
        "Leg Volume Right": leg_volume_each,
        "MidThigh Circumference Right": mid_thigh_single,
        "Outside Leg Length Right": outside_leg_proxy,
        "Surface Area Arm Right": surface_arm_each,
        "Surface Area Leg Right": surface_leg_each,
        "Thigh Circumference Right": thigh_single,
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

    biomarker_row = extract_42_biomarkers_from_df(
        df=df,
        subject_id=name,
        source_file=str(obj_file),
        height_scale_to_cm=height_scale_to_cm,
    )

    valid = df["num_loops"] > 0

    summary = {
        "source_file": str(obj_file),
        "subject_id": name,
        "slice_csv": str(slice_csv),
        "aligned_obj": aligned_obj,
        "slice_profile_image": slice_profile_image,
        "biomarker_method_image": biomarker_method_image,
        "num_vertices_original": int(len(mesh.vertices)),
        "num_faces_original": int(len(mesh.faces)),
        "num_vertices_aligned": int(len(aligned.vertices)),
        "num_faces_aligned": int(len(aligned.faces)),
        "height_mesh_units": height,
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
    print("FULL OBJ → SLICING → 42 BIOMARKERS PIPELINE")
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

            contact_sheet_path = save_contact_sheet(
                image_dir=output_dir / "images",
                output_file=output_dir / "all_biomarker_method_contact_sheet.png",
                pattern="*_biomarker_method.png",
                thumb_width=520,
                cols=3,
                padding=30,
            )

            if contact_sheet_path is not None:
                print(f"Saved biomarker contact sheet: {contact_sheet_path}")

    if failures:
        pd.DataFrame(failures).to_csv(failure_csv, index=False)
        print(f"Saved failures CSV: {failure_csv}")

    print("\n" + "=" * 90)
    print("PIPELINE COMPLETE")
    print("=" * 90)
    print(f"Successful files: {len(summaries)}")
    print(f"Failed files: {len(failures)}")
    print(f"Final biomarker file: {biomarker_csv}")
    print(f"Contact sheet: {output_dir / 'all_biomarker_method_contact_sheet.png'}")

    return summaries, biomarker_rows, failures


def main():
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
        help="Do not save PNG images.",
    )

    parser.add_argument(
        "--no-aligned-obj",
        action="store_true",
        help="Do not save aligned OBJ files.",
    )

    args = parser.parse_args()

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
    main()