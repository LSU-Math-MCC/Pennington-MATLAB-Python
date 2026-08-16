from __future__ import annotations

import contextlib
import atexit
import csv
import importlib
import importlib.util
import io
import json
import os
import shutil
import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .schema import AVATAR_FIELD_MAP, SEGMENTATION_FIELD_MAP, SLICE_FIELD_MAP


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


AVATAR_AREA_KEYS = frozenset(
    {"SA_total", "SA_head", "SA_trunk", "SA_rArm", "SA_lArm", "SA_rleg", "SA_lleg", "SA_legs"}
)
AVATAR_VOLUME_KEYS = frozenset(
    {"VOL_total", "VOL_head", "VOL_trunk", "VOL_rArm", "VOL_lArm", "VOL_rleg", "VOL_lleg"}
)


def _avatar_scale(key: str, length_scale: float) -> float:
    if key in AVATAR_AREA_KEYS:
        return length_scale ** 2
    if key in AVATAR_VOLUME_KEYS:
        return length_scale ** 3
    return length_scale


def _avatar_warnings(measurements: dict[str, float], segments) -> list[str]:
    """Cheap sanity checks so a silently wrong mesh does not hide in a batch."""
    warnings: list[str] = []

    left_leg = len(segments["left_leg"])
    right_leg = len(segments["right_leg"])
    if left_leg and right_leg and max(left_leg, right_leg) > 3 * min(left_leg, right_leg):
        warnings.append(f"leg segmentation lopsided ({left_leg} vs {right_leg})")

    height = measurements.get("height", 0.0)
    if height:
        ratio = measurements.get("crotchHeight", 0.0) / height
        if not 0.40 <= ratio <= 0.52:
            warnings.append(f"crotch at {ratio * 100:.0f}% of height (expect ~45)")

    for label, left, right in (
        ("thigh", "lThighGirth", "rThighGirth"),
        ("calf", "lCalfGirth", "rCalfGirth"),
        ("arm length", "lArmLength", "rArmLength"),
    ):
        a, b = measurements.get(left, np.nan), measurements.get(right, np.nan)
        if np.isnan(a) or np.isnan(b) or min(a, b) <= 0:
            continue
        if max(a, b) > 1.5 * min(a, b):
            warnings.append(f"{label} asymmetry ({a:.0f} vs {b:.0f})")

    return warnings


class AvatarBackend:
    """MATLAB-faithful port of ``Avatar.m`` (steps = 3, SA = 'on')."""

    name = "avatar"
    version = "avatar"

    def run(self, obj_file: Path, options: PipelineOptions) -> dict[str, object]:
        from .backends.avatar.avatar_conversion import MatlabAvatar, load_obj

        length_scale = _height_scale_to_cm(obj_file, options.units, options.height_scale_to_cm)

        vertices, faces = load_obj(obj_file)
        avatar = MatlabAvatar(vertices, faces).run()

        raw = {
            key: value * _avatar_scale(key, length_scale)
            for key, value in avatar.measurements.items()
        }
        warnings = _avatar_warnings(avatar.measurements, avatar.segments)

        if options.output_dir is not None:
            self._write_artifacts(obj_file, options, avatar, raw, length_scale, warnings)

        row = _canonicalize(raw, AVATAR_FIELD_MAP)
        row["subject_id"] = subject_id(obj_file)
        return row

    def _write_artifacts(self, obj_file, options, avatar, raw, length_scale, warnings):
        import csv
        import json

        artifact_dir = options.output_dir / self.name / artifact_id(obj_file)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        with (artifact_dir / "measurements.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["name", "value", "units"])
            for key, value in raw.items():
                if key in AVATAR_AREA_KEYS:
                    units = "cm^2"
                elif key in AVATAR_VOLUME_KEYS:
                    units = "cm^3"
                else:
                    units = "cm"
                writer.writerow([key, value, units])

        with (artifact_dir / "landmarks.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["name", "x", "y", "z"])
            for key, point in avatar.landmarks.items():
                writer.writerow([key, *(float(c) * length_scale for c in point)])

        with (artifact_dir / "segments.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["segment", "n_vertices"])
            for key, indices in avatar.segments.items():
                writer.writerow([key, len(indices)])

        summary = {
            "source_file": str(obj_file),
            "n_vertices": int(len(avatar.v)),
            "n_faces": int(len(avatar.f)),
            "units": "cm",
            "scale_applied": length_scale,
            "warnings": warnings,
            "measurements": raw,
            "segments": {key: len(value) for key, value in avatar.segments.items()},
            "notes": [
                "Values reproduce MATLAB Avatar.m (steps=3) exactly.",
                "Girths are convex-hull perimeters over a band of vertices near each"
                " plane, not exact mesh cross-sections.",
                "Columns ending in _fixed correct known Avatar.m bugs; the unsuffixed"
                " columns reproduce MATLAB behaviour.",
            ],
        }
        (artifact_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )


def _matlab_scalar(value) -> float:
    """Copy one scalar returned inside a MATLAB struct into Python."""
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return np.nan
    if array.size != 1:
        return np.nan
    return float(array.reshape(-1)[0])


class MatlabBackend:
    """Run the original MATLAB ``Avatar`` implementation through one Engine.

    The engine is lazy and belongs to the backend instance, not to one OBJ.
    ``obj2anthro.pipeline`` keeps the instance in ``BACKENDS`` while it loops
    over the input folder, so MATLAB starts once and each subsequent OBJ is a
    normal function call in the same MATLAB workspace.
    """

    def __init__(self, steps=(3,), name="matlab"):
        self.name = name
        self.steps = tuple(int(step) for step in steps)
        self.version = f"Avatar.m steps={','.join(str(step) for step in self.steps)} Vol_SA=on"
        self._engine = None
        self._matlab_module = None
        self._engine_module = None
        self._lock = threading.RLock()
        atexit.register(self.close)

    @property
    def engine_started(self) -> bool:
        return self._engine is not None

    @staticmethod
    def _code_dir() -> Path:
        configured = os.environ.get("PENNINGTON_MATLAB_DIR")
        if configured:
            path = Path(configured).expanduser().resolve()
            if (path / "Avatar.m").exists():
                return path
            raise FileNotFoundError(f"PENNINGTON_MATLAB_DIR has no Avatar.m: {path}")

        candidates = (
            REPO_ROOT / "MATLAB Code" / "Pennington",
            Path(__file__).resolve().parent / "backends" / "segmentation" / "matlab" / "original_code",
        )
        for path in candidates:
            if (path / "Avatar.m").exists():
                return path
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Could not find Avatar.m; searched: {searched}")

    @staticmethod
    def _matlab_executable_hint() -> str:
        configured = os.environ.get("MATLAB_EXECUTABLE")
        if configured:
            return configured
        found = shutil.which("matlab")
        if found:
            return found
        return "matlab"

    def _import_engine(self):
        if self._engine_module is not None:
            return self._matlab_module, self._engine_module
        try:
            import matlab
            engine_module = importlib.import_module("matlab.engine")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "MATLAB Engine for Python is unavailable in this interpreter. "
                "Use a supported Python 3.9-3.11 environment and install the "
                "Engine from the MATLAB installation's extern/engines/python "
                f"directory (MATLAB executable hint: {self._matlab_executable_hint()})."
            ) from exc
        self._matlab_module = matlab
        self._engine_module = engine_module
        return self._matlab_module, self._engine_module

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine, self._matlab_module

        matlab_module, engine_module = self._import_engine()
        code_dir = self._code_dir()
        helper_dir = Path(__file__).resolve().parent / "backends" / "matlab"
        helper = helper_dir / "py2mat_avatar_measure.m"
        if not helper.exists():
            raise FileNotFoundError(f"MATLAB adapter is missing: {helper}")

        engine = engine_module.start_matlab("-nodesktop -nosplash")
        try:
            # Add the adapter after the legacy source.  The adapter has no
            # duplicate Avatar name, and this keeps the source path explicit.
            engine.addpath(str(code_dir), nargout=0)
            engine.addpath(str(helper_dir), nargout=0)
            engine.eval("clear py2mat_avatar py2mat_result;", nargout=0)
        except Exception:
            try:
                engine.quit()
            except Exception:  # noqa: BLE001
                pass
            raise

        self._engine = engine
        try:
            self.version = (
                f"Avatar.m steps={','.join(str(step) for step in self.steps)} "
                f"Vol_SA=on MATLAB {engine.version()}"
            )
        except Exception:  # noqa: BLE001
            pass
        return engine, matlab_module

    def run(self, obj_file: Path, options: PipelineOptions) -> dict[str, object]:
        with self._lock:
            engine, matlab_module = self._ensure_engine()
            stdout = io.StringIO()
            stderr = io.StringIO()
            matlab_input = obj_file
            try:
                result = engine.py2mat_avatar_measure(
                    str(matlab_input.resolve()),
                    matlab_module.double([[float(step) for step in self.steps]]),
                    nargout=1,
                    stdout=stdout,
                    stderr=stderr,
                )
            finally:
                # The adapter keeps Avatar local to one call.  Clear any
                # accidental workspace values without terminating the kernel.
                try:
                    engine.eval("clear py2mat_avatar py2mat_result;", nargout=0)
                except Exception:  # noqa: BLE001
                    pass

            raw = {key: _matlab_scalar(result.get(key)) for key in AVATAR_FIELD_MAP}
            length_scale = _height_scale_to_cm(obj_file, options.units, options.height_scale_to_cm)
            scaled = {
                key: value * _avatar_scale(key, length_scale)
                if np.isfinite(value)
                else value
                for key, value in raw.items()
            }
            if options.output_dir is not None:
                self._write_artifacts(
                    obj_file,
                    options,
                    scaled,
                    length_scale,
                    stdout.getvalue(),
                    stderr.getvalue(),
                )

            row = _canonicalize(scaled, AVATAR_FIELD_MAP)
            row["subject_id"] = subject_id(obj_file)
            return row

    def _write_artifacts(
        self,
        obj_file: Path,
        options: PipelineOptions,
        measurements: dict[str, float],
        length_scale: float,
        stdout: str,
        stderr: str,
    ) -> None:
        artifact_dir = options.output_dir / self.name / artifact_id(obj_file)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        with (artifact_dir / "measurements.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["name", "value", "units"])
            for key, value in measurements.items():
                if key in AVATAR_AREA_KEYS:
                    units = "cm^2"
                elif key in AVATAR_VOLUME_KEYS:
                    units = "cm^3"
                else:
                    units = "cm"
                writer.writerow([key, value, units])

        (artifact_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
        (artifact_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
        summary = {
            "backend": self.name,
            "version": self.version,
            "source_file": str(obj_file),
            "steps": list(self.steps),
            "scale_applied_to_cm": length_scale,
            "measurements": measurements,
            "notes": [
                "Avatar.m runs in one persistent MATLAB Engine for the batch.",
                "The MATLAB-faithful fields retain known legacy behavior; *_fixed fields are explicit corrected companions.",
            ],
        }
        (artifact_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
        )

    def close(self) -> None:
        with self._lock:
            engine, self._engine = self._engine, None
            if engine is not None:
                try:
                    engine.quit()
                except Exception:  # noqa: BLE001
                    pass


BACKENDS: dict[str, Backend] = {
    "segmentation": SegmentationBackend(),
    "slice": SliceBackend(),
    "avatar": AvatarBackend(),
    "matlab": MatlabBackend(),
    "matlab-full": MatlabBackend(steps=(1, 2, 3), name="matlab-full"),
}
