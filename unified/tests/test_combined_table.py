"""Every run folder must end up with one combined measurement table."""

from __future__ import annotations

import pandas as pd
import pytest

from unified.combine import (
    COMBINED_COLUMNS,
    COMBINED_FILENAME,
    build_combined_table,
    summarize_combined,
    write_combined_table,
)
from unified.obj2anthro.schema import CANONICAL_COLUMNS, complete_frame


def make_frame(subject: str, method: str, runtime: float, height: float | None):
    row = {
        "subject_id": subject,
        "source_file": f"/data/{subject}.obj",
        "pipeline": method,
        "pipeline_version": method,
        "status": "success" if height is not None else "failed",
        "error": "" if height is not None else "boom",
        "runtime_seconds": runtime,
    }
    if height is not None:
        row["height_cm"] = height
    return complete_frame([row])


def test_columns_lead_with_provenance_then_timing():
    assert COMBINED_COLUMNS[:5] == [
        "run_id",
        "subject_id",
        "source_file",
        "source_method",
        "anthro_method",
    ]
    assert "runtime_seconds" in COMBINED_COLUMNS
    # Every canonical measurement survives into the combined table.
    measurements = [c for c in CANONICAL_COLUMNS if c.endswith(("_cm", "_cm2", "_cm3"))]
    for column in measurements:
        assert column in COMBINED_COLUMNS


def test_one_row_per_subject_and_method():
    parts = [
        {"frame": make_frame("s1", "avatar", 0.3, 154.0), "run_id": "r", "source_method": "direct"},
        {"frame": make_frame("s1", "slice", 6.0, 151.0), "run_id": "r", "source_method": "direct"},
        {"frame": make_frame("s2", "avatar", 0.4, 168.0), "run_id": "r", "source_method": "direct"},
    ]
    table = build_combined_table(parts)

    assert len(table) == 3
    assert list(table["subject_id"]) == ["s1", "s1", "s2"]
    assert list(table["anthro_method"]) == ["avatar", "slice", "avatar"]
    assert list(table["run_id"]) == ["r", "r", "r"]
    assert table["runtime_seconds"].tolist() == [0.3, 6.0, 0.4]
    assert table["height_cm"].tolist() == [154.0, 151.0, 168.0]


def test_failed_branches_keep_their_row_and_reason():
    parts = [
        {"frame": make_frame("s1", "matlab", 0.01, None), "run_id": "r", "source_method": "direct"},
    ]
    table = build_combined_table(parts)
    assert list(table["status"]) == ["failed"]
    assert list(table["error"]) == ["boom"]
    assert pd.isna(table["height_cm"]).all()


def test_missing_or_unreadable_parts_are_skipped_not_fatal(tmp_path):
    missing = tmp_path / "nope.csv"
    broken = tmp_path / "broken.csv"
    broken.write_bytes(b"\x00\x01\x02")

    parts = [
        {"frame": missing, "run_id": "r", "source_method": "direct"},
        {"frame": make_frame("s1", "avatar", 0.3, 154.0), "run_id": "r", "source_method": "direct"},
    ]
    table = build_combined_table(parts)
    assert len(table) == 1


def test_write_produces_the_file_even_with_no_parts(tmp_path):
    path = write_combined_table(tmp_path, [])
    assert path.name == COMBINED_FILENAME
    assert path.is_file()
    assert list(pd.read_csv(path).columns) == COMBINED_COLUMNS


def test_round_trips_through_csv(tmp_path):
    parts = [
        {
            "frame": make_frame("s1", "avatar", 0.3, 154.0),
            "run_id": "r",
            "source_method": "direct",
            "branch_dir": str(tmp_path / "branch"),
        }
    ]
    path = write_combined_table(tmp_path, parts)
    table = pd.read_csv(path)
    assert table.loc[0, "branch_dir"] == str(tmp_path / "branch")
    assert table.loc[0, "source_method"] == "direct"


def test_summary_reports_each_method_once():
    parts = [
        {"frame": make_frame("s1", "avatar", 0.3, 154.0), "run_id": "r", "source_method": "direct"},
        {"frame": make_frame("s1", "slice", 6.0, 151.0), "run_id": "r", "source_method": "direct"},
    ]
    summary = summarize_combined(build_combined_table(parts))
    assert summary.count("\n") == 1
    assert "avatar" in summary and "slice" in summary


def test_merge_prefers_the_later_run_for_a_repeated_method(tmp_path):
    """A backend re-run in a later table replaces its earlier rows."""
    from unified.combine import merge_tables

    early = tmp_path / "early"
    late = tmp_path / "late"
    write_combined_table(early, [
        {"frame": make_frame("s1", "avatar", 0.3, 154.0), "run_id": "a", "source_method": "direct"},
        {"frame": make_frame("s1", "matlab", 0.01, None), "run_id": "a", "source_method": "direct"},
    ])
    write_combined_table(late, [
        {"frame": make_frame("s1", "matlab", 3.5, 155.0), "run_id": "b", "source_method": "direct"},
    ])

    merged = merge_tables([early, late])
    assert len(merged) == 2
    matlab = merged[merged["anthro_method"] == "matlab"].iloc[0]
    assert matlab["status"] == "success"
    assert matlab["height_cm"] == 155.0
    assert matlab["run_id"] == "b"
    # The method only the early run carried survives untouched.
    assert set(merged["anthro_method"]) == {"avatar", "matlab"}


def test_merge_keeps_a_success_over_a_failure_regardless_of_order(tmp_path):
    from unified.combine import merge_tables

    good = tmp_path / "good"
    bad = tmp_path / "bad"
    write_combined_table(good, [
        {"frame": make_frame("s1", "matlab", 3.5, 155.0), "run_id": "good", "source_method": "direct"},
    ])
    write_combined_table(bad, [
        {"frame": make_frame("s1", "matlab", 0.01, None), "run_id": "bad", "source_method": "direct"},
    ])

    merged = merge_tables([good, bad])
    assert len(merged) == 1
    assert merged.iloc[0]["status"] == "success"
    assert merged.iloc[0]["height_cm"] == 155.0
