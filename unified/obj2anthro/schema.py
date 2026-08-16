from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


METADATA_COLUMNS = [
    "subject_id",
    "source_file",
    "pipeline",
    "pipeline_version",
    "status",
    "error",
    "runtime_seconds",
]

SLICE_FIELD_MAP = {
    "Height (cm)": "height_cm",
    "Abdomen Circumference": "abdomen_circumference_cm",
    "Ankle Circumference Left": "ankle_circumference_left_cm",
    "Arm Length Left": "arm_length_left_cm",
    "Arm Volume Left": "arm_volume_left_cm3",
    "Bicep Circumference Left": "bicep_circumference_left_cm",
    "Calf Circumference Left": "calf_circumference_left_cm",
    "Chest": "chest_circumference_cm",
    "Collar Circumference": "collar_circumference_cm",
    "Forearm Circumference Left": "forearm_circumference_left_cm",
    "Head Circumference": "head_circumference_cm",
    "Hip Circumference": "hip_circumference_cm",
    "Horizontal Waist": "horizontal_waist_circumference_cm",
    "Inseam Left": "inseam_left_cm",
    "Leg Volume Left": "leg_volume_left_cm3",
    "MidThigh Circumference Left": "mid_thigh_circumference_left_cm",
    "Narrow Waist": "narrow_waist_circumference_cm",
    "Outside Leg Length Left": "outside_leg_length_left_cm",
    "Seat Circumference": "seat_circumference_cm",
    "Surface Area Arm Left": "surface_area_arm_left_cm2",
    "Surface Area Leg Left": "surface_area_leg_left_cm2",
    "Surface Area Torso": "surface_area_torso_cm2",
    "Surface Area Total": "surface_area_total_cm2",
    "Thigh Circumference Left": "thigh_circumference_left_cm",
    "Torso Volume": "torso_volume_cm3",
    "Upper Arm Circumference Left": "upper_arm_circumference_left_cm",
    "Volume": "volume_cm3",
    "Waist Circumference": "waist_circumference_cm",
    "Ankle Circumference Right": "ankle_circumference_right_cm",
    "Arm Length Right": "arm_length_right_cm",
    "Arm Volume Right": "arm_volume_right_cm3",
    "Bicep Circumference Right": "bicep_circumference_right_cm",
    "Calf Circumference Right": "calf_circumference_right_cm",
    "Forearm Circumference Right": "forearm_circumference_right_cm",
    "Inseam Right": "inseam_right_cm",
    "Leg Volume Right": "leg_volume_right_cm3",
    "MidThigh Circumference Right": "mid_thigh_circumference_right_cm",
    "Outside Leg Length Right": "outside_leg_length_right_cm",
    "Surface Area Arm Right": "surface_area_arm_right_cm2",
    "Surface Area Leg Right": "surface_area_leg_right_cm2",
    "Thigh Circumference Right": "thigh_circumference_right_cm",
    "Upper Arm Circumference Right": "upper_arm_circumference_right_cm",
}

SEGMENTATION_FIELD_MAP = {
    ("meta", "height"): "height_cm",
    ("meta", "volume"): "volume_cm3",
    ("meta", "surface_area"): "surface_area_total_cm2",
    ("head", "collar to scalp length"): "collar_to_scalp_length_cm",
    ("head", "head circumference"): "head_circumference_cm",
    ("head", "volume"): "head_volume_cm3",
    ("head", "surface_area"): "surface_area_head_cm2",
    ("trunk", "trunk length"): "trunk_length_cm",
    ("trunk", "crotch height"): ("inseam_left_cm", "inseam_right_cm"),
    ("trunk", "chest circumference"): "chest_circumference_cm",
    ("trunk", "waist circumference"): "waist_circumference_cm",
    ("trunk", "horizontal waist circumference"): "horizontal_waist_circumference_cm",
    ("trunk", "narrow waist circumference"): "narrow_waist_circumference_cm",
    ("trunk", "stomach peak circumference"): "abdomen_circumference_cm",
    ("trunk", "hip circumference"): "hip_circumference_cm",
    ("trunk", "seat circumference"): "seat_circumference_cm",
    ("trunk", "collar circumference"): "collar_circumference_cm",
    ("trunk", "volume"): "torso_volume_cm3",
    ("trunk", "surface_area"): "surface_area_torso_cm2",
    ("left arm", "arm length"): "arm_length_left_cm",
    ("left arm", "wrist girth"): "wrist_circumference_left_cm",
    ("left arm", "forearm girth"): "forearm_circumference_left_cm",
    ("left arm", "bicep girth"): "bicep_circumference_left_cm",
    ("left arm", "upper arm girth"): "upper_arm_circumference_left_cm",
    ("left arm", "volume"): "arm_volume_left_cm3",
    ("left arm", "surface_area"): "surface_area_arm_left_cm2",
    ("right arm", "arm length"): "arm_length_right_cm",
    ("right arm", "wrist girth"): "wrist_circumference_right_cm",
    ("right arm", "forearm girth"): "forearm_circumference_right_cm",
    ("right arm", "bicep girth"): "bicep_circumference_right_cm",
    ("right arm", "upper arm girth"): "upper_arm_circumference_right_cm",
    ("right arm", "volume"): "arm_volume_right_cm3",
    ("right arm", "surface_area"): "surface_area_arm_right_cm2",
    ("left leg", "leg length"): "outside_leg_length_left_cm",
    ("left leg", "ankle girth"): "ankle_circumference_left_cm",
    ("left leg", "calf girth"): "calf_circumference_left_cm",
    ("left leg", "mid thigh girth"): "mid_thigh_circumference_left_cm",
    ("left leg", "thigh girth"): "thigh_circumference_left_cm",
    ("left leg", "volume"): "leg_volume_left_cm3",
    ("left leg", "surface_area"): "surface_area_leg_left_cm2",
    ("right leg", "leg length"): "outside_leg_length_right_cm",
    ("right leg", "ankle girth"): "ankle_circumference_right_cm",
    ("right leg", "calf girth"): "calf_circumference_right_cm",
    ("right leg", "mid thigh girth"): "mid_thigh_circumference_right_cm",
    ("right leg", "thigh girth"): "thigh_circumference_right_cm",
    ("right leg", "volume"): "leg_volume_right_cm3",
    ("right leg", "surface_area"): "surface_area_leg_right_cm2",
}

AVATAR_FIELD_MAP = {
    "height": "height_cm",
    "chestGirth": "chest_circumference_cm",
    "waistGirth": "waist_circumference_cm",
    "hipGirth": "hip_circumference_cm",
    "rThighGirth": "thigh_circumference_right_cm",
    "lThighGirth": "thigh_circumference_left_cm",
    "rCalfGirth": "calf_circumference_right_cm",
    "lCalfGirth": "calf_circumference_left_cm",
    "rAnkleGirth": "ankle_circumference_right_cm",
    "lAnkleGirth": "ankle_circumference_left_cm",
    "rBicepGirth": "bicep_circumference_right_cm",
    "lBicepGirth": "bicep_circumference_left_cm",
    "rForearmGirth": "forearm_circumference_right_cm",
    "lForearmGirth": "forearm_circumference_left_cm",
    "rWristGirth": "wrist_circumference_right_cm",
    "lWristGirth": "wrist_circumference_left_cm",
    "rArmLength": "arm_length_right_cm",
    "lArmLength": "arm_length_left_cm",
    "rLegLength": "outside_leg_length_right_cm",
    "lLegLength": "outside_leg_length_left_cm",
    "trunkLength": "trunk_length_cm",
    "crotchHeight": ("inseam_left_cm", "inseam_right_cm"),
    "collarScalpLength": "collar_to_scalp_length_cm",
    "SA_total": "surface_area_total_cm2",
    "SA_head": "surface_area_head_cm2",
    "SA_trunk": "surface_area_torso_cm2",
    "SA_rArm": "surface_area_arm_right_cm2",
    "SA_lArm": "surface_area_arm_left_cm2",
    "SA_rleg": "surface_area_leg_right_cm2",
    "SA_lleg": "surface_area_leg_left_cm2",
    "VOL_total": "volume_cm3",
    "VOL_head": "head_volume_cm3",
    "VOL_trunk": "torso_volume_cm3",
    "VOL_rArm": "arm_volume_right_cm3",
    "VOL_lArm": "arm_volume_left_cm3",
    "VOL_rleg": "leg_volume_right_cm3",
    "VOL_lleg": "leg_volume_left_cm3",
    # Avatar.m bugs corrected; the unsuffixed columns above stay MATLAB-faithful.
    "rAnkleGirth_fixed": "ankle_circumference_right_corrected_cm",
    "lAnkleGirth_fixed": "ankle_circumference_left_corrected_cm",
    "trunkLength_fixed": "trunk_length_corrected_cm",
    "collarScalpLength_fixed": "collar_to_scalp_length_corrected_cm",
}


def _mapped_columns(field_map: Mapping[object, object]) -> list[str]:
    return [
        value
        for mapped in field_map.values()
        for value in (mapped if isinstance(mapped, tuple) else (mapped,))
    ]


MEASUREMENT_COLUMNS = sorted(
    {
        *SLICE_FIELD_MAP.values(),
        *_mapped_columns(SEGMENTATION_FIELD_MAP),
        *_mapped_columns(AVATAR_FIELD_MAP),
    }
)

CANONICAL_COLUMNS = [*METADATA_COLUMNS, *MEASUREMENT_COLUMNS]


def empty_measurements() -> dict[str, float]:
    return {column: np.nan for column in MEASUREMENT_COLUMNS}


def normalize_measurements(values: Mapping[str, object]) -> dict[str, object]:
    row = empty_measurements()
    row.update({key: value for key, value in values.items() if key in row})
    return row


def complete_frame(rows: list[Mapping[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows).reindex(columns=CANONICAL_COLUMNS)
