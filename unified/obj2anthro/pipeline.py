from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .backend_registry import BACKENDS, PipelineOptions, subject_id
from .schema import CANONICAL_COLUMNS, complete_frame, empty_measurements, normalize_measurements


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_TEST_SET_DIR = (
    Path(__file__).resolve().parent
    / "backends"
    / "segmentation"
    / "model_files"
    / "OBJ"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def canonical_run_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_full_anthro"


def is_generated_raw_path(path: Path) -> bool:
    parts = path.parts
    if any(part == ".pipeline_raw" or part.startswith("unified_pipeline_results_") for part in parts):
        return True
    return any(left == "results" and right == "raw" for left, right in zip(parts, parts[1:]))


def discover_obj_files(input_path: str | Path, recursive: bool = True) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() != ".obj":
            raise ValueError(f"Input file is not an OBJ: {path}")
        return [path]
    if path.is_dir():
        pattern = "**/*.obj" if recursive else "*.obj"
        return sorted(
            item
            for item in path.glob(pattern)
            if item.is_file() and not is_generated_raw_path(item)
        )
    raise FileNotFoundError(f"Input path does not exist: {path}")


def canonical_test_set_files() -> list[Path]:
    return discover_obj_files(CANONICAL_TEST_SET_DIR, recursive=False)


def selected_backends(backend: str):
    if backend == "auto":
        backend = "all"
    if backend == "all":
        return [BACKENDS["segmentation"], BACKENDS["slice"]]
    if backend not in BACKENDS:
        raise ValueError(f"Unknown method {backend!r}; expected auto, all, segmentation, or slice")
    return [BACKENDS[backend]]


def failure_row(obj_file: Path, backend, elapsed: float, error: Exception) -> dict[str, object]:
    return {
        "subject_id": subject_id(obj_file),
        "source_file": str(obj_file),
        "pipeline": backend.name,
        "pipeline_version": backend.version,
        "status": "failed",
        "error": f"{type(error).__name__}: {error}",
        "runtime_seconds": elapsed,
        **empty_measurements(),
    }


def success_row(obj_file: Path, backend, elapsed: float, measurements: dict[str, object]) -> dict[str, object]:
    return {
        "subject_id": subject_id(obj_file),
        "source_file": str(obj_file),
        "pipeline": backend.name,
        "pipeline_version": backend.version,
        "status": "success",
        "error": "",
        "runtime_seconds": elapsed,
        **normalize_measurements(measurements),
    }


def run_pipeline(
    input_path: str | Path = CANONICAL_TEST_SET_DIR,
    backend: str = "all",
    recursive: bool = True,
    units: str = "cm",
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
    run_id: str | None = None,
    n_slices: int = 200,
    save_images: bool = True,
    save_aligned_obj: bool = True,
    height_scale_to_cm: float | None = None,
    show: bool = False,
) -> pd.DataFrame:
    out_dir = Path(output_dir) if output_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_id or canonical_run_id()
    raw_dir = None
    if out_dir is not None:
        raw_dir = out_dir / "raw" if run_name == "results" else out_dir / "raw" / run_name
        raw_dir.mkdir(parents=True, exist_ok=True)

    obj_files = discover_obj_files(input_path, recursive=recursive)
    options = PipelineOptions(
        units=units,
        output_dir=raw_dir,
        recursive=recursive,
        n_slices=n_slices,
        save_images=save_images,
        save_aligned_obj=save_aligned_obj,
        height_scale_to_cm=height_scale_to_cm,
        show=show,
    )

    rows = []
    for obj_file in obj_files:
        for active_backend in selected_backends(backend):
            start = time.perf_counter()
            try:
                measurements = active_backend.run(obj_file, options)
            except Exception as exc:
                elapsed = time.perf_counter() - start
                rows.append(failure_row(obj_file, active_backend, elapsed, exc))
            else:
                elapsed = time.perf_counter() - start
                rows.append(success_row(obj_file, active_backend, elapsed, measurements))

    df = complete_frame(rows)
    if out_dir is not None:
        output_csv = out_dir / "results.csv" if run_name == "results" else out_dir / f"{run_name}.csv"
        df.to_csv(output_csv, index=False)
        df.attrs["output_csv"] = str(output_csv)
    if raw_dir is not None:
        df.attrs["raw_output_dir"] = str(raw_dir)
    df.attrs["run_id"] = run_name
    return df.reindex(columns=CANONICAL_COLUMNS)
