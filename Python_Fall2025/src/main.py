"""
Body Measurement Visualization Demo

This script demonstrates the body measurement system by:
1. Loading a 3D body mesh
2. Extracting and visualizing body part meshes (head, trunk, arms, legs)
3. Displaying anatomical landmarks as colored spheres
4. Printing all measurements to console
"""

import argparse
import contextlib
import sys
from pathlib import Path

import trimesh
from .body import Body
from .body.anatomical_regions.anatomical_region import (
    BASELINE_GEOMETRY_CONSTANTS,
    BASELINE_UNITS,
    INTERNAL_UNIT,
    UNIT_TO_CM,
    to_cm,
)

MESH_SUFFIXES = {".obj"}

# ============================================================================
# Color Palette - Professional Blue/Teal Scheme
# ============================================================================
COLORS = {
    # Body part meshes (semi-transparent)
    'head': [50, 180, 180, 180],      # Teal
    'trunk': [70, 120, 170, 180],      # Medium blue
    'left_arm': [50, 180, 180, 180],   # Teal
    'right_arm': [50, 180, 180, 180],  # Teal 
    'left_leg': [50, 180, 180, 180],   # Teal
    'right_leg': [50, 180, 180, 180],  # Teal 
    
    # Landmarks (opaque, bright accents)
    'primary': [255, 100, 100, 255],   # Red - primary landmarks
    'secondary': [255, 180, 50, 255],  # Orange - secondary landmarks
    'tertiary': [100, 255, 150, 255],  # Green - tertiary landmarks
}

LANDMARK_RADIUS = 0.4  # Standard radius for landmark spheres


# ============================================================================
# Helper Functions
# ============================================================================
def create_landmark_sphere(position, color, radius=LANDMARK_RADIUS):
    """Create a colored sphere at the given position to mark a landmark."""
    sphere = trimesh.creation.icosphere(radius=radius)
    sphere.apply_translation(position)
    sphere.visual.vertex_colors = color
    return sphere


def add_body_part_mesh(scene, body, part_name, color):
    """Add a body part mesh to the scene with specified color."""
    mesh = body.subregion_meshes[part_name]
    mesh.visual.vertex_colors = color
    scene.add_geometry(mesh)


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def iter_points(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from iter_points(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_points(item)
    else:
        try:
            if len(value) == 3:
                yield value
        except TypeError:
            pass


def save_diagnostic_image(body, image_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    projections = [
        ("Front", 0, 2, "x", "z"),
        ("Side", 1, 2, "y", "z"),
        ("Top", 0, 1, "x", "y"),
    ]
    mesh_colors = {
        "head": "#39a9a9",
        "trunk": "#5d82a8",
        "left arm": "#39a9a9",
        "right arm": "#39a9a9",
        "left leg": "#39a9a9",
        "right leg": "#39a9a9",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 7), dpi=160)
    all_vertices = np.asarray(body.mesh.vertices)

    for ax, (title, a, b, xlabel, ylabel) in zip(axes, projections):
        for part_name, mesh in body.subregion_meshes.items():
            vertices = np.asarray(mesh.vertices)
            if len(vertices) > 2500:
                vertices = vertices[:: max(1, len(vertices) // 2500)]
            ax.scatter(vertices[:, a], vertices[:, b], s=0.35, c=mesh_colors[part_name], alpha=0.35)

        for part_drawings in body.drawings.values():
            for path in part_drawings.values():
                vertices = np.asarray(path.vertices)
                for entity in path.entities:
                    pts = vertices[np.asarray(entity.points)]
                    ax.plot(pts[:, a], pts[:, b], c="black", lw=1.2)

        for part_landmarks in body.landmarks.values():
            for point in iter_points(part_landmarks):
                point = np.asarray(point)
                ax.scatter(point[a], point[b], s=24, c="#d94b4b", edgecolors="none")

        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(all_vertices[:, a].min(), all_vertices[:, a].max())
        ax.set_ylim(all_vertices[:, b].min(), all_vertices[:, b].max())
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(labelsize=7, length=2, pad=1)
        ax.grid(True, linewidth=0.3, alpha=0.25)

    fig.tight_layout()
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def auto_output_path(mesh_file, subdir, suffix):
    try:
        rel_mesh = Path(mesh_file).with_suffix(suffix)
        if rel_mesh.is_absolute():
            rel_mesh = rel_mesh.relative_to(Path.cwd())
        return Path("output") / subdir / rel_mesh
    except ValueError:
        return Path("output") / subdir / f"{Path(mesh_file).stem}{suffix}"


def auto_diary_path(mesh_file):
    return auto_output_path(mesh_file, "logs", ".txt")


def auto_image_path(mesh_file):
    return auto_output_path(mesh_file, "images", ".png")


def iter_mesh_files(path):
    path = Path(path)
    if path.is_dir():
        return sorted(
            file for file in path.rglob("*")
            if file.is_file() and file.suffix.lower() in MESH_SUFFIXES
        )
    return [path]


def infer_units(mesh_file, units):
    if units != "auto":
        return units

    return "dm" if Path(mesh_file).name == "man.obj" else "mm"


def resolve_output_path(mesh_file, option, subdir, suffix, batch):
    if option == "auto":
        return auto_output_path(mesh_file, subdir, suffix)

    path = Path(option)
    if batch:
        return path / auto_output_path(mesh_file, subdir, suffix).relative_to(Path("output") / subdir)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Run the body measurement visualization demo.")
    parser.add_argument("mesh_file", nargs="?", default="model_files/man.obj")
    parser.add_argument("--diary", default="auto", help="'auto' for output/logs/... or a path for captured stdout")
    parser.add_argument("--show", action="store_true", help="Open the interactive 3D visualization")
    parser.add_argument(
        "--save-image",
        nargs="?",
        const="auto",
        default=None,
        help="Save a headless PNG render. Pass a path or use without a value for output/images/...",
    )
    parser.add_argument(
        "--units",
        choices=("auto", *UNIT_TO_CM.keys()),
        default="auto",
        help="Coordinate units used by the input mesh. 'auto' uses dm for man.obj and mm otherwise.",
    )
    return parser.parse_args()


# ============================================================================
# Main Visualization
# ============================================================================
def run_demo(mesh_file, *, units=BASELINE_UNITS, show=False, save_image=None, **_):
    print("\n" + "=" * 60)
    print("  BODY MEASUREMENT SYSTEM - VISUALIZATION DEMO")
    print("=" * 60)
    print("\nLoading body model...\n")
    
    units = infer_units(mesh_file, units)
    body = Body(mesh_file, units=units)
    geometry_config = body.geometry_config

    print(f"Geometry units: input={units}, internal={INTERNAL_UNIT}, output=cm")
    print(f"Mesh vertices: {len(body.mesh.vertices)}")
    print(f"Mesh height: {to_cm(body.mesh.extents.max(), geometry_config):.3f} cm")
    print(f"Density scale: {geometry_config['density_scale']:.3f}")
    print("Geometry constants:")
    for key in sorted(BASELINE_GEOMETRY_CONSTANTS):
        print(f"  {key}: {geometry_config[key]:.6f} {INTERNAL_UNIT}")
    print()

    scene = trimesh.Scene()
    
    # ========================================================================
    # Add Body Part Meshes
    # ========================================================================
    add_body_part_mesh(scene, body, "head", COLORS['head'])
    add_body_part_mesh(scene, body, "trunk", COLORS['trunk'])
    add_body_part_mesh(scene, body, "left arm", COLORS['left_arm'])
    add_body_part_mesh(scene, body, "right arm", COLORS['right_arm'])
    add_body_part_mesh(scene, body, "left leg", COLORS['left_leg'])
    add_body_part_mesh(scene, body, "right leg", COLORS['right_leg'])
    
    # ========================================================================
    # Add Head Landmarks
    # ========================================================================
    nose_tip = body.landmarks["head"]["tip of nose"]
    scene.add_geometry(create_landmark_sphere(nose_tip, COLORS['primary']))
    
    # ========================================================================
    # Add Trunk Landmarks
    # ========================================================================
    collar = body.landmarks["trunk"]["collar"]
    scene.add_geometry(create_landmark_sphere(collar, COLORS['primary']))
    
    crotch = body.landmarks["trunk"]["crotch"]
    scene.add_geometry(create_landmark_sphere(crotch, COLORS['primary']))
    
    left_armpit, right_armpit = body.landmarks["trunk"]["armpits"]
    scene.add_geometry(create_landmark_sphere(left_armpit, COLORS['secondary']))
    scene.add_geometry(create_landmark_sphere(right_armpit, COLORS['secondary']))
    
    left_hip, right_hip = body.landmarks["trunk"]["hips"]
    scene.add_geometry(create_landmark_sphere(left_hip, COLORS['secondary']))
    scene.add_geometry(create_landmark_sphere(right_hip, COLORS['secondary']))
    
    # ========================================================================
    # Add Arm Landmarks
    # ========================================================================
    # Left arm
    left_shoulder = body.landmarks["left arm"]["shoulder"]
    scene.add_geometry(create_landmark_sphere(left_shoulder, COLORS['primary']))
    
    left_wrist = body.landmarks["left arm"]["wrist"]
    scene.add_geometry(create_landmark_sphere(left_wrist, COLORS['secondary']))
    
    left_highest = body.landmarks["left arm"]["highest point of arm"]
    scene.add_geometry(create_landmark_sphere(left_highest, COLORS['tertiary'], 0.3))
    
    # Right arm
    right_shoulder = body.landmarks["right arm"]["shoulder"]
    scene.add_geometry(create_landmark_sphere(right_shoulder, COLORS['primary']))
    
    right_wrist = body.landmarks["right arm"]["wrist"]
    scene.add_geometry(create_landmark_sphere(right_wrist, COLORS['secondary']))
    
    right_highest = body.landmarks["right arm"]["highest point of arm"]
    scene.add_geometry(create_landmark_sphere(right_highest, COLORS['tertiary'], 0.3))
    
    # ========================================================================
    # Add Leg Landmarks
    # ========================================================================
    # Left leg
    left_foot = body.landmarks["left leg"]["foot"]
    scene.add_geometry(create_landmark_sphere(left_foot, COLORS['primary']))
    
    left_ankle = body.landmarks["left leg"]["ankle"]
    scene.add_geometry(create_landmark_sphere(left_ankle, COLORS['secondary']))
    
    # Right leg
    right_foot = body.landmarks["right leg"]["foot"]
    scene.add_geometry(create_landmark_sphere(right_foot, COLORS['primary']))
    
    right_ankle = body.landmarks["right leg"]["ankle"]
    scene.add_geometry(create_landmark_sphere(right_ankle, COLORS['secondary']))
    
    # ========================================================================
    # Add Measurement Drawings (Paths)
    # ========================================================================
    # Render all measurement paths as black lines to show what's being measured
    for part_name, part_drawings in body.drawings.items():
        for measurement_name, path in part_drawings.items():
            # Configure path appearance as black lines
            if hasattr(path, 'visual'):
                path.visual.vertex_colors = [0, 0, 0, 255]  # Black
            scene.add_geometry(path)
    
    print_section_header("HEAD MEASUREMENTS")
    print(f"  Collar to Scalp Length: {to_cm(body.measurements['head']['collar to scalp length'], geometry_config):.2f} cm")

    print_section_header("TRUNK MEASUREMENTS")
    print(f"  Trunk Length: {to_cm(body.measurements['trunk']['trunk length'], geometry_config):.2f} cm")
    print(f"  Crotch Height: {to_cm(body.measurements['trunk']['crotch height'], geometry_config):.2f} cm")
    print(f"  Chest Circumference: {to_cm(body.measurements['trunk']['chest circumference'], geometry_config):.2f} cm")
    print(f"  Waist Circumference: {to_cm(body.measurements['trunk']['waist circumference'], geometry_config):.2f} cm")
    print(f"  Stomach Peak Circumference: {to_cm(body.measurements['trunk']['stomach peak circumference'], geometry_config):.2f} cm")
    print(f"  Hip Circumference: {to_cm(body.measurements['trunk']['hip circumference'], geometry_config):.2f} cm")

    print_section_header("ARM MEASUREMENTS")
    print("\n  LEFT ARM:")
    print(f"    Length: {to_cm(body.measurements['left arm']['arm length'], geometry_config):.2f} cm")
    print(f"    Wrist Girth: {to_cm(body.measurements['left arm']['wrist girth'], geometry_config):.2f} cm")
    print(f"    Forearm Girth: {to_cm(body.measurements['left arm']['forearm girth'], geometry_config):.2f} cm")
    print(f"    Bicep Girth: {to_cm(body.measurements['left arm']['bicep girth'], geometry_config):.2f} cm")

    print("\n  RIGHT ARM:")
    print(f"    Length: {to_cm(body.measurements['right arm']['arm length'], geometry_config):.2f} cm")
    print(f"    Wrist Girth: {to_cm(body.measurements['right arm']['wrist girth'], geometry_config):.2f} cm")
    print(f"    Forearm Girth: {to_cm(body.measurements['right arm']['forearm girth'], geometry_config):.2f} cm")
    print(f"    Bicep Girth: {to_cm(body.measurements['right arm']['bicep girth'], geometry_config):.2f} cm")

    print_section_header("LEG MEASUREMENTS")
    print("\n  LEFT LEG:")
    print(f"    Length: {to_cm(body.measurements['left leg']['leg length'], geometry_config):.2f} cm")
    print(f"    Ankle Girth: {to_cm(body.measurements['left leg']['ankle girth'], geometry_config):.2f} cm")
    print(f"    Calf Girth: {to_cm(body.measurements['left leg']['calf girth'], geometry_config):.2f} cm")
    print(f"    Thigh Girth: {to_cm(body.measurements['left leg']['thigh girth'], geometry_config):.2f} cm")

    print("\n  RIGHT LEG:")
    print(f"    Length: {to_cm(body.measurements['right leg']['leg length'], geometry_config):.2f} cm")
    print(f"    Ankle Girth: {to_cm(body.measurements['right leg']['ankle girth'], geometry_config):.2f} cm")
    print(f"    Calf Girth: {to_cm(body.measurements['right leg']['calf girth'], geometry_config):.2f} cm")
    print(f"    Thigh Girth: {to_cm(body.measurements['right leg']['thigh girth'], geometry_config):.2f} cm")

    if save_image:
        image_path = auto_image_path(mesh_file) if save_image == "auto" else Path(save_image)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        save_diagnostic_image(body, image_path)
        print(f"\nSaved visualization image: {image_path}")
    
    if show:
        print("\n" + "=" * 60)
        print("  Displaying 3D visualization...")
        print("=" * 60 + "\n")
        scene.show()
    else:
        print("\nVisualization skipped. Pass --show to open the 3D viewer.")


if __name__ == "__main__":
    args = parse_args()
    options = vars(args)
    mesh_files = iter_mesh_files(args.mesh_file)
    batch = len(mesh_files) > 1 or Path(args.mesh_file).is_dir()
    failures = []

    for mesh_file in mesh_files:
        run_options = dict(options, mesh_file=mesh_file)
        if options["save_image"]:
            run_options["save_image"] = resolve_output_path(mesh_file, options["save_image"], "images", ".png", batch)

        diary_path = resolve_output_path(mesh_file, options["diary"], "logs", ".txt", batch)
        diary_path.parent.mkdir(parents=True, exist_ok=True)
        with diary_path.open("w", encoding="utf-8") as diary, contextlib.redirect_stdout(Tee(sys.stdout, diary)):
            try:
                if batch:
                    print(f"Batch item: {mesh_file}")
                run_demo(**run_options)
            except Exception as exc:
                failures.append((mesh_file, exc))
                print(f"\nERROR processing {mesh_file}: {exc}")

    if failures:
        print("\nBatch failures:")
        for mesh_file, exc in failures:
            print(f"  {mesh_file}: {exc}")
        sys.exit(1)
