"""
Body Measurement Visualization Demo

This script demonstrates the body measurement system by:
1. Loading a 3D body mesh
2. Extracting and visualizing body part meshes (head, trunk, arms, legs)
3. Displaying anatomical landmarks as colored spheres
4. Printing all measurements to console
"""

import trimesh
from .body import Body


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


# ============================================================================
# Main Visualization
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  BODY MEASUREMENT SYSTEM - VISUALIZATION DEMO")
    print("=" * 60)
    print("\nLoading body model...\n")
    
    # Load body model
    body = Body("model_files/man.obj")
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
    
    # ========================================================================
    # Print All Measurements
    # ========================================================================
    
    # HEAD MEASUREMENTS
    print_section_header("HEAD MEASUREMENTS")
    collar_to_scalp = body.measurements["head"]["collar to scalp length"]
    print(f"  Collar to Scalp Length: {collar_to_scalp:.2f} cm")
    
    # TRUNK MEASUREMENTS
    print_section_header("TRUNK MEASUREMENTS")
    trunk_length = body.measurements["trunk"]["trunk length"]
    print(f"  Trunk Length: {trunk_length:.2f} cm")
    
    crotch_height = body.measurements["trunk"]["crotch height"]
    print(f"  Crotch Height: {crotch_height:.2f} cm")
    
    chest_circ = body.measurements["trunk"]["chest circumference"]
    print(f"  Chest Circumference: {chest_circ:.2f} cm")
    
    waist_circ = body.measurements["trunk"]["waist circumference"]
    print(f"  Waist Circumference: {waist_circ:.2f} cm")
    
    hip_circ = body.measurements["trunk"]["hip circumference"]
    print(f"  Hip Circumference: {hip_circ:.2f} cm")
    
    # ARM MEASUREMENTS
    print_section_header("ARM MEASUREMENTS")
    print("\n  LEFT ARM:")
    print(f"    Length: {body.measurements['left arm']['arm length']:.2f} cm")
    print(f"    Wrist Girth: {body.measurements['left arm']['wrist girth']:.2f} cm")
    print(f"    Forearm Girth: {body.measurements['left arm']['forearm girth']:.2f} cm")
    print(f"    Bicep Girth: {body.measurements['left arm']['bicep girth']:.2f} cm")
    
    print("\n  RIGHT ARM:")
    print(f"    Length: {body.measurements['right arm']['arm length']:.2f} cm")
    print(f"    Wrist Girth: {body.measurements['right arm']['wrist girth']:.2f} cm")
    print(f"    Forearm Girth: {body.measurements['right arm']['forearm girth']:.2f} cm")
    print(f"    Bicep Girth: {body.measurements['right arm']['bicep girth']:.2f} cm")
    
    # LEG MEASUREMENTS
    print_section_header("LEG MEASUREMENTS")
    print("\n  LEFT LEG:")
    print(f"    Length: {body.measurements['left leg']['leg length']:.2f} cm")
    print(f"    Ankle Girth: {body.measurements['left leg']['ankle girth']:.2f} cm")
    print(f"    Calf Girth: {body.measurements['left leg']['calf girth']:.2f} cm")
    print(f"    Thigh Girth: {body.measurements['left leg']['thigh girth']:.2f} cm")
    
    print("\n  RIGHT LEG:")
    print(f"    Length: {body.measurements['right leg']['leg length']:.2f} cm")
    print(f"    Ankle Girth: {body.measurements['right leg']['ankle girth']:.2f} cm")
    print(f"    Calf Girth: {body.measurements['right leg']['calf girth']:.2f} cm")
    print(f"    Thigh Girth: {body.measurements['right leg']['thigh girth']:.2f} cm")
    
    print("\n" + "=" * 60)
    print("  Displaying 3D visualization...")
    print("=" * 60 + "\n")


           # ========================================================================
    # Save Measurements to CSV
    # ========================================================================
    rows = []

    for part_name, measurements in body.measurements.items():
        for measurement_name, value in measurements.items():
            rows.append({
                "body_part": part_name,
                "measurement": measurement_name,
                "value_cm": value
            })

    df = pd.DataFrame(rows)
    df.to_csv("body_measurements.csv", index=False)

    print("\nSaved measurements to body_measurements.csv")
    
    # Show the 3D scene
    scene.show()
    
