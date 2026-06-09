import numpy as np
import trimesh

from ....utils.section_geometry import empty_measurement, line_path, slice_measurement


def get_submesh(mesh: trimesh.Trimesh):
    """Boolean-subtract limbs and head to leave the trunk volume used for torso-only geometry."""
    from ..legs import Leg
    from ..arms import Arm
    from ..head import Head
    from ....mesh.boolean_ops import mesh_difference

    trunk_mesh = mesh.copy()
    left_leg_mesh = Leg._get_submesh("left", mesh)
    trunk_mesh = mesh_difference(trunk_mesh, left_leg_mesh)
    right_leg_mesh = Leg._get_submesh("right", mesh)
    trunk_mesh = mesh_difference(trunk_mesh, right_leg_mesh)
    left_arm_mesh = Arm._get_submesh("left", mesh)
    trunk_mesh = mesh_difference(trunk_mesh, left_arm_mesh)
    right_arm_mesh = Arm._get_submesh("right", mesh)
    trunk_mesh = mesh_difference(trunk_mesh, right_arm_mesh)
    head_mesh = Head._get_submesh(mesh)
    trunk_mesh = mesh_difference(trunk_mesh, head_mesh)
    return trunk_mesh


def body_without_arms(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Remove arm volumes so horizontal girth slices follow torso/pelvis contours instead of sleeves."""
    from ..arms import Arm
    from ....mesh.boolean_ops import mesh_difference

    body_without_arms = mesh.copy()
    left_arm_mesh = Arm._get_submesh("left", mesh)
    body_without_arms = mesh_difference(body_without_arms, left_arm_mesh)
    right_arm_mesh = Arm._get_submesh("right", mesh)
    body_without_arms = mesh_difference(body_without_arms, right_arm_mesh)
    return body_without_arms


def measure_crotch_height(mesh: trimesh.Trimesh, trunk_api):
    """Measure vertical height from the ground plane up to the crotch saddle landmark."""
    print("Called measure_crotch_height (Trunk)")
    crotch_point = trunk_api._locate_crotch(mesh)
    min_z = np.min(mesh.vertices[:, 2])
    ground_point = np.array([crotch_point[0], crotch_point[1], min_z])
    return float(crotch_point[2] - min_z), line_path([ground_point, crotch_point])


def measure_hip_circumference(mesh: trimesh.Trimesh, trunk_api):
    """Measure the closed tape path around the fullest pelvis section."""
    print("Called measure_hip_circumference (Trunk)")
    measurement = trunk_api._level_measurement(mesh, "hip_full_level")
    if measurement is not None:
        return measurement
    no_arms = trunk_api._body_without_arms(mesh)
    section = trunk_api._hip_section(no_arms)
    if section is not None:
        hip_z, _ = section
        return slice_measurement(no_arms, hip_z, "hip")
    print("Warning: No section found in hip region")
    return empty_measurement()


def measure_chest_circumference(mesh: trimesh.Trimesh, trunk_api):
    """Measure the closed tape path around the nipple/bust level, as with a measuring tape."""
    print("Called measure_chest_circumference (Trunk)")
    measurement = trunk_api._level_measurement(mesh, "chest_full_level")
    if measurement is not None:
        return measurement
    torso_mesh = trunk_api._get_submesh(mesh)
    left_armpit, right_armpit = trunk_api._locate_armpits(mesh)
    chest_z = np.median([left_armpit[2], right_armpit[2]])
    print(f"Chest level fallback z={chest_z:.4f}")
    return slice_measurement(torso_mesh, chest_z, "chest")


def measure_waist_circumference(mesh: trimesh.Trimesh, trunk_api):
    """Measure the closed tape path around the natural waist narrowing between hip and chest."""
    print("Called measure_waist_circumference (Trunk)")
    measurement = trunk_api._level_measurement(mesh, "natural_waist_level")
    if measurement is not None:
        return measurement
    torso_mesh = trunk_api._get_submesh(mesh)
    left_armpit, right_armpit = trunk_api._locate_armpits(mesh)
    left_hip, right_hip = trunk_api._locate_hips(mesh)
    waist_z = np.mean([np.median([left_armpit[2], right_armpit[2]]), np.median([left_hip[2], right_hip[2]])])
    return slice_measurement(torso_mesh, waist_z, "waist")


def measure_stomach_peak_circumference(mesh: trimesh.Trimesh, trunk_api):
    """Measure the abdomen's local fullness level, falling back to waist when no peak is isolated."""
    print("Called measure_stomach_peak_circumference (Trunk)")
    return trunk_api._level_measurement(mesh, "stomach_waist_level") or trunk_api._measure_waist_circumference(mesh)


def measure_trunk_length(mesh: trimesh.Trimesh, trunk_api):
    """Measure the sagittal trunk span from collar landmark to crotch landmark in the x-z plane."""
    print("Called _measure_trunk_length (Trunk)")
    crotch = trunk_api._locate_crotch(mesh)
    collar = trunk_api._locate_collar(mesh)
    if not isinstance(crotch, np.ndarray) or not isinstance(collar, np.ndarray):
        raise TypeError("Crotch or collar point not found or invalid (expected np.ndarray).")
    if crotch.shape != (3,) or collar.shape != (3,):
        raise ValueError(f"Unexpected point shape. Got crotch={crotch.shape}, collar={collar.shape}")
    trunk_length = np.sqrt((crotch[0] - collar[0]) ** 2 + (crotch[2] - collar[2]) ** 2)
    print(f"Crotch point: {crotch}")
    print(f"Collar point: {collar}")
    print(f"Computed trunk length (||crotch - collar||_(x,z)) = {trunk_length:.3f}")
    return float(trunk_length), line_path([collar, crotch])
