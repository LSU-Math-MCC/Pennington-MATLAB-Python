from __future__ import annotations

import math
from pathlib import Path

import pytest

import unified.pipeline as pipeline
from unified.schema import CANONICAL_COLUMNS, FALL2025_FIELD_MAP, SLICE_FIELD_MAP


class FakeBackend:
    version = "test"

    def __init__(self, name, values=None, error=None):
        self.name = name
        self.values = values or {}
        self.error = error

    def run(self, obj_file, options):
        if self.error:
            raise self.error
        return self.values


def make_obj(tmp_path: Path, name="body.obj") -> Path:
    path = tmp_path / name
    path.write_text("# obj\n", encoding="utf-8")
    return path


def test_schema_completion_and_unsupported_nan(tmp_path, monkeypatch):
    obj = make_obj(tmp_path)
    monkeypatch.setattr(
        pipeline,
        "BACKENDS",
        {"fake": FakeBackend("fake", {"chest_circumference_cm": 90.0})},
    )
    df = pipeline.run_pipeline(obj, backend="fake", output_dir=None)

    assert list(df.columns) == CANONICAL_COLUMNS
    assert df.loc[0, "chest_circumference_cm"] == 90.0
    assert math.isnan(df.loc[0, "waist_circumference_cm"])


def test_all_mode_emits_one_row_per_backend(tmp_path, monkeypatch):
    obj = make_obj(tmp_path)
    monkeypatch.setattr(
        pipeline,
        "BACKENDS",
        {
            "fall2025": FakeBackend("fall2025"),
            "slice": FakeBackend("slice"),
        },
    )
    df = pipeline.run_pipeline(obj, backend="all", output_dir=None)

    assert df["pipeline"].tolist() == ["fall2025", "slice"]
    assert len(df) == 2


def test_runtime_and_failure_row(tmp_path, monkeypatch):
    obj = make_obj(tmp_path)
    monkeypatch.setattr(
        pipeline,
        "BACKENDS",
        {"fake": FakeBackend("fake", error=RuntimeError("boom"))},
    )
    df = pipeline.run_pipeline(obj, backend="fake", output_dir=None)

    assert df.loc[0, "status"] == "failed"
    assert "RuntimeError: boom" == df.loc[0, "error"]
    assert df.loc[0, "runtime_seconds"] >= 0
    assert math.isnan(df.loc[0, "chest_circumference_cm"])


def test_backend_selection(tmp_path, monkeypatch):
    obj = make_obj(tmp_path)
    monkeypatch.setattr(
        pipeline,
        "BACKENDS",
        {
            "fall2025": FakeBackend("fall2025", {"height_cm": 170}),
            "slice": FakeBackend("slice", {"height_cm": 171}),
        },
    )
    df = pipeline.run_pipeline(obj, backend="slice", output_dir=None)

    assert df["pipeline"].tolist() == ["slice"]
    assert df.loc[0, "height_cm"] == 171


def test_representative_field_mappings():
    assert SLICE_FIELD_MAP["Chest"] == "chest_circumference_cm"
    assert SLICE_FIELD_MAP["Surface Area Total"] == "surface_area_total_cm2"
    assert FALL2025_FIELD_MAP[("trunk", "waist circumference")] == "waist_circumference_cm"
    assert FALL2025_FIELD_MAP[("left arm", "arm length")] == "arm_length_left_cm"
    assert FALL2025_FIELD_MAP[("trunk", "crotch height")] == ("inseam_left_cm", "inseam_right_cm")


def test_csv_saved(tmp_path, monkeypatch):
    obj = make_obj(tmp_path)
    output_dir = tmp_path / "results"
    monkeypatch.setattr(
        pipeline,
        "BACKENDS",
        {"fake": FakeBackend("fake", {"volume_cm3": 3.0})},
    )
    pipeline.run_pipeline(obj, backend="fake", output_dir=output_dir)

    csvs = list(output_dir.glob("*_full_anthro.csv"))
    assert len(csvs) == 1


def test_raw_outputs_use_results_raw_timestamped_folder(tmp_path, monkeypatch):
    obj = make_obj(tmp_path)
    output_dir = tmp_path / "results"
    seen_output_dirs = []

    class RecordingBackend(FakeBackend):
        def run(self, obj_file, options):
            seen_output_dirs.append(options.output_dir)
            return {}

    monkeypatch.setattr(
        pipeline,
        "BACKENDS",
        {"fake": RecordingBackend("fake")},
    )
    pipeline.run_pipeline(
        obj,
        backend="fake",
        output_dir=output_dir,
        run_id="20260621T150000Z_full_anthro",
    )

    assert seen_output_dirs == [output_dir / "raw" / "20260621T150000Z_full_anthro"]
    assert (output_dir / "20260621T150000Z_full_anthro.csv").exists()


def test_discovery_skips_generated_raw_outputs(tmp_path):
    source = make_obj(tmp_path, "source.obj")
    raw = tmp_path / "results" / "raw" / "20260621T150000Z_full_anthro" / "slice.obj"
    raw.parent.mkdir(parents=True)
    raw.write_text("# generated\n", encoding="utf-8")

    assert pipeline.discover_obj_files(tmp_path, recursive=True) == [source]


def test_default_input_is_one_canonical_test_set():
    assert pipeline.CANONICAL_TEST_SET_DIR.parts[-3:] == ("Python_Fall2025", "model_files", "OBJ")
