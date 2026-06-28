from __future__ import annotations

import contextlib
import importlib
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .schema import SEGMENTATION_FIELD_MAP, SLICE_FIELD_MAP


REPO_ROOT = Path(__file__).resolve().parents[2]
FALL2025_ROOT = Path(__file__).resolve().parent / "backends" / "segmentation"
FALL2025_SRC = FALL2025_ROOT / "src"
SLICE_PATH = Path(__file__).resolve().parent / "backends" / "slice" / "slice.py"
UNIT_TO_CM = {"mm": 0.1, "cm": 1.0, "dm": 10.0, "m": 100.0}


@dataclass(frozen=True)
class PipelineOptions:
    units: str = "cm"
    output_dir: Path | None = None
    recursive: bool = True
    n_slices: int = 200
    save_images: bool = True
    save_aligned_obj: bool = True
    height_scale_to_cm: float | None = None
    show: bool = False


class Backend(Protocol):
    name: str
    version: str

    def run(self, obj_file: Path, options: PipelineOptions) -> dict[str, object]:
        ...


def subject_id(path: Path) -> str:
    stem = path.stem
    for char in [" ", "/", "\\", ":", ";", ",", "(", ")", "[", "]"]:
        stem = stem.replace(char, "_")
    while "__" in stem:
        stem = stem.replace("__", "_")
    return stem.strip("_")


def artifact_id(path: Path) -> str:
    try:
        raw = path.resolve().relative_to(REPO_ROOT.resolve()).with_suffix("").as_posix()
    except ValueError:
        raw = path.with_suffix("").as_posix()
    for char in [" ", "/", "\\", ":", ";", ",", "(", ")", "[", "]"]:
        raw = raw.replace(char, "_")
    while "__" in raw:
        raw = raw.replace("__", "_")
    return raw.strip("_")


def infer_units_for_obj(obj_file: Path, units: str) -> str:
    if units != "auto":
        return units
    return "dm" if Path(obj_file).name == "man.obj" else "mm"


def _height_scale_to_cm(obj_file: Path, units: str, explicit: float | None) -> float:
    if explicit is not None:
        return explicit
    return UNIT_TO_CM.get(infer_units_for_obj(obj_file, units), 1.0)


def _canonicalize(raw: dict[object, object], field_map: dict[object, object]) -> dict[str, object]:
    row: dict[str, object] = {}
    for source, target in field_map.items():
        value = raw.get(source, np.nan)
        targets = target if isinstance(target, tuple) else (target,)
        for column in targets:
            row[column] = value
    return row


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _safe_scaled_property(obj, name: str, scale: float):
    try:
        return _as_float(getattr(obj, name)) * scale
    except Exception:
        return np.nan


def _safe_scaled_volume(obj, name: str, scale: float):
    value = _safe_scaled_property(obj, name, scale)
    return abs(value) if not np.isnan(value) else value


def _ensure_fall2025_package():
    package_name = "_fall2025_src"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(FALL2025_SRC)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    return package_name


def _load_slice_module():
    module_name = "_slice2026_impl"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, SLICE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load slice backend from {SLICE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _pushd(path: Path):
    import os

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class SegmentationBackend:
    name = "segmentation"
    version = "segmentation"

    def run(self, obj_file: Path, options: PipelineOptions) -> dict[str, object]:
        log_file = None
        if options.output_dir:
            log_dir = options.output_dir / self.name / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = (log_dir / f"{artifact_id(obj_file)}.txt").open("w", encoding="utf-8")

        with contextlib.ExitStack() as stack:
            if log_file is not None:
                stack.enter_context(log_file)
                stack.enter_context(contextlib.redirect_stdout(log_file))

            package = _ensure_fall2025_package()
            body_module = importlib.import_module(f"{package}.body")
            main_module = importlib.import_module(f"{package}.main")
            region_module = importlib.import_module(f"{package}.body.anatomical_regions.anatomical_region")

            units = options.units
            if units == "auto":
                units = main_module.infer_units(obj_file, units)

            body = body_module.Body(obj_file, units=units)
            geometry_config = body.geometry_config
            length_scale = geometry_config["internal_to_cm"]
            area_scale = length_scale ** 2
            volume_scale = length_scale ** 3
            to_cm = region_module.to_cm

            raw: dict[object, object] = {
                ("meta", "height"): to_cm(body.mesh.extents.max(), geometry_config),
                ("meta", "volume"): _safe_scaled_volume(body, "volume", volume_scale),
                ("meta", "surface_area"): _safe_scaled_property(body, "surface_area", area_scale),
            }
            for part_name, measurements in body.measurements.items():
                for measurement_name, value in measurements.items():
                    raw[(part_name, measurement_name)] = to_cm(value, geometry_config)
            for part_name, part in body.parts.items():
                raw[(part_name, "volume")] = _safe_scaled_volume(part, "volume", volume_scale)
                raw[(part_name, "surface_area")] = _safe_scaled_property(part, "surface_area", area_scale)

            if options.save_images and options.output_dir:
                image_dir = options.output_dir / self.name / "images"
                image_dir.mkdir(parents=True, exist_ok=True)
                main_module.save_diagnostic_image(body, image_dir / f"{artifact_id(obj_file)}.png")

            if options.show:
                body.mesh.show()

            return _canonicalize(raw, SEGMENTATION_FIELD_MAP)


class SliceBackend:
    name = "slice"
    version = "slice"

    def run(self, obj_file: Path, options: PipelineOptions) -> dict[str, object]:
        module = _load_slice_module()
        output_dir = (options.output_dir or Path("results")) / self.name / artifact_id(obj_file)
        summary, biomarker_row = module.process_one_obj(
            obj_file=obj_file,
            output_dir=output_dir,
            n_slices=options.n_slices,
            height_scale_to_cm=_height_scale_to_cm(obj_file, options.units, options.height_scale_to_cm),
            save_images=options.save_images,
            save_aligned_obj=options.save_aligned_obj,
        )
        raw = {key: biomarker_row.get(key, np.nan) for key in SLICE_FIELD_MAP}
        raw["subject_id"] = summary.get("subject_id", subject_id(obj_file))
        return _canonicalize(raw, SLICE_FIELD_MAP)


BACKENDS: dict[str, Backend] = {
    "segmentation": SegmentationBackend(),
    "slice": SliceBackend(),
}
