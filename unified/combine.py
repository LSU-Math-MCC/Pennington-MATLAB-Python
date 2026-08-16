"""One combined measurement table per run.

Every run folder gets a ``combined_measurements.csv``: one row per
(subject, anthropometry method), carrying the method that produced it, how long
that method took, and every canonical measurement column side by side. It is
written unconditionally so a run folder is always self-describing, whether the
run covered one OBJ and one method or a whole directory under ``--method auto``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from .obj2anthro.schema import CANONICAL_COLUMNS, MEASUREMENT_COLUMNS


COMBINED_FILENAME = "combined_measurements.csv"

# Provenance first, then timing, then the measurements themselves.
LEADING_COLUMNS = [
    "run_id",
    "subject_id",
    "source_file",
    "source_method",
    "anthro_method",
    "pipeline_version",
    "status",
    "error",
    "runtime_seconds",
    "branch_dir",
]

COMBINED_COLUMNS = [*LEADING_COLUMNS, *MEASUREMENT_COLUMNS]


def _normalize(frame: pd.DataFrame, run_id: str, source_method: str, branch_dir: str) -> pd.DataFrame:
    """Reshape one backend result frame into combined-table columns."""
    out = frame.reindex(columns=CANONICAL_COLUMNS).copy()
    out = out.rename(columns={"pipeline": "anthro_method"})
    out.insert(0, "run_id", run_id)
    out["source_method"] = source_method
    out["branch_dir"] = branch_dir
    return out.reindex(columns=COMBINED_COLUMNS)


def build_combined_table(parts: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Build the combined table from ``{frame, source_method, branch_dir}`` parts.

    ``frame`` may be a DataFrame or a path to a backend ``results.csv``. Parts
    that cannot be read are skipped rather than failing the run -- the combined
    table is a reporting artifact and must never be the reason a run errors out.
    """
    frames: list[pd.DataFrame] = []
    for part in parts:
        source = part.get("frame")
        if source is None:
            continue
        if not isinstance(source, pd.DataFrame):
            path = Path(str(source))
            if not path.is_file():
                continue
            try:
                source = pd.read_csv(path)
            except Exception:  # noqa: BLE001 - a bad branch CSV must not sink the run
                continue
        if len(source) == 0:
            continue
        frames.append(
            _normalize(
                source,
                str(part.get("run_id", "")),
                str(part.get("source_method", "")),
                str(part.get("branch_dir", "")),
            )
        )

    if not frames:
        return pd.DataFrame(columns=COMBINED_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(
        ["subject_id", "anthro_method"], kind="stable"
    ).reset_index(drop=True)


def write_combined_table(run_root: str | Path, parts: Iterable[Mapping[str, object]]) -> Path:
    """Write ``combined_measurements.csv`` into ``run_root`` and return its path."""
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    path = run_root / COMBINED_FILENAME
    build_combined_table(parts).to_csv(path, index=False)
    return path


def merge_tables(sources: Iterable[str | Path]) -> pd.DataFrame:
    """Merge combined tables from several runs into one.

    A method that was re-run in a later source replaces its earlier rows, so a
    run where a backend was unavailable does not leave stale failures behind.
    On a tie, a successful row wins over a failed one.
    """
    frames: list[pd.DataFrame] = []
    for order, source in enumerate(sources):
        path = Path(source)
        if path.is_dir():
            path = path / COMBINED_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"No combined table at {path}")
        frame = pd.read_csv(path)
        frame["_source_order"] = order
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    merged["_ok"] = (merged["status"] == "success").astype(int)
    merged = merged.sort_values(["_ok", "_source_order"], kind="stable")
    merged = merged.drop_duplicates(subset=["subject_id", "anthro_method"], keep="last")
    merged = merged.drop(columns=["_ok", "_source_order"])
    return merged.sort_values(
        ["subject_id", "anthro_method"], kind="stable"
    ).reset_index(drop=True).reindex(columns=COMBINED_COLUMNS)


def summarize_combined(frame: pd.DataFrame) -> str:
    """A one-line-per-method digest for console output."""
    if len(frame) == 0:
        return "combined table is empty"
    lines = []
    for method, group in frame.groupby("anthro_method", sort=True):
        ok = int((group["status"] == "success").sum())
        seconds = float(pd.to_numeric(group["runtime_seconds"], errors="coerce").sum())
        filled = int(group[MEASUREMENT_COLUMNS].notna().any(axis=0).sum())
        lines.append(
            f"  {method:<14} {ok}/{len(group)} ok  {seconds:7.1f}s  {filled} measurement columns"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    """Merge combined tables from several runs into one table."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="unified.combine",
        description="Merge combined_measurements.csv tables from several runs into one.",
    )
    parser.add_argument(
        "sources",
        nargs="+",
        help="combined_measurements.csv files, or run folders containing one. "
             "Later sources win when the same subject and method appear twice.",
    )
    parser.add_argument("--out", required=True, help="Directory to write the merged table into.")
    args = parser.parse_args(argv)

    merged = merge_tables(args.sources)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / COMBINED_FILENAME
    merged.to_csv(path, index=False)

    print(f"Merged {len(args.sources)} tables into {path}")
    print(f"{len(merged)} rows · {merged['subject_id'].nunique()} subjects · "
          f"{merged['anthro_method'].nunique()} methods")
    print(summarize_combined(merged))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
