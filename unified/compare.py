"""Score anthropometry methods against a reference method.

The MATLAB backend runs the original ``Avatar.m`` through the MATLAB Engine, so
it is the natural ground truth for everything else in the pipeline. This module
takes one or more ``combined_measurements.csv`` tables, picks a reference method
out of them, and reports how far every other method sits from it -- per
measurement and per subject.

Only pairs where both the reference and the compared method produced a value are
scored, so a method is never penalised for columns it does not implement. The
per-measurement coverage count makes that explicit.

CLI::

    python -m unified.compare runs/<id>/combined_measurements.csv --reference matlab
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .combine import COMBINED_FILENAME
from .obj2anthro.schema import MEASUREMENT_COLUMNS


DEFAULT_REFERENCE = "matlab"

# Below this absolute delta two values are the same number to print precision.
EXACT_ATOL = 1e-6
# A relative gap this small is floating-point noise, not a methodological one.
EXACT_RTOL = 1e-9


def load_tables(sources: Iterable[str | Path]) -> pd.DataFrame:
    """Read combined tables, or run folders containing one, into a single frame."""
    frames: list[pd.DataFrame] = []
    for source in sources:
        path = Path(source)
        if path.is_dir():
            path = path / COMBINED_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"No combined table at {path}")
        frames.append(pd.read_csv(path))
    if not frames:
        raise ValueError("No comparison sources given")
    return pd.concat(frames, ignore_index=True)


def _successful(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["status"] == "success"]


def long_form(frame: pd.DataFrame, reference: str) -> pd.DataFrame:
    """One row per (subject, method, measurement) with the reference alongside."""
    present = [c for c in MEASUREMENT_COLUMNS if c in frame.columns]
    ok = _successful(frame)

    melted = ok.melt(
        id_vars=["subject_id", "anthro_method"],
        value_vars=present,
        var_name="measurement",
        value_name="value",
    ).dropna(subset=["value"])

    truth = melted[melted["anthro_method"] == reference][
        ["subject_id", "measurement", "value"]
    ].rename(columns={"value": "reference_value"})

    others = melted[melted["anthro_method"] != reference]
    merged = others.merge(truth, on=["subject_id", "measurement"], how="inner")

    merged["delta"] = merged["value"] - merged["reference_value"]
    merged["abs_delta"] = merged["delta"].abs()
    denominator = merged["reference_value"].abs().replace(0.0, np.nan)
    merged["pct_error"] = 100.0 * merged["delta"] / denominator
    merged["abs_pct_error"] = merged["pct_error"].abs()
    merged["matches"] = np.isclose(
        merged["value"], merged["reference_value"], rtol=EXACT_RTOL, atol=EXACT_ATOL
    )
    return merged.rename(columns={"anthro_method": "method"})


def per_measurement(detail: pd.DataFrame) -> pd.DataFrame:
    """Per (method, measurement) agreement with the reference."""
    if detail.empty:
        return pd.DataFrame()

    def stats(group: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n_subjects": len(group),
            "n_exact": int(group["matches"].sum()),
            "mean_abs_error": group["abs_delta"].mean(),
            "median_abs_error": group["abs_delta"].median(),
            "max_abs_error": group["abs_delta"].max(),
            "rmse": float(np.sqrt(np.mean(group["delta"] ** 2))),
            "bias": group["delta"].mean(),
            "mean_abs_pct_error": group["abs_pct_error"].mean(),
            "max_abs_pct_error": group["abs_pct_error"].max(),
        })

    table = detail.groupby(["method", "measurement"], sort=True).apply(
        stats, include_groups=False
    )
    table = table.reset_index()
    table["pct_exact"] = 100.0 * table["n_exact"] / table["n_subjects"]
    return table.sort_values(["method", "mean_abs_pct_error"], ascending=[True, False])


def per_method(detail: pd.DataFrame) -> pd.DataFrame:
    """One headline row per method."""
    if detail.empty:
        return pd.DataFrame()
    grouped = detail.groupby("method", sort=True)
    table = grouped.agg(
        n_comparisons=("abs_delta", "size"),
        n_measurements=("measurement", "nunique"),
        n_subjects=("subject_id", "nunique"),
        n_exact=("matches", "sum"),
        mean_abs_error=("abs_delta", "mean"),
        median_abs_error=("abs_delta", "median"),
        max_abs_error=("abs_delta", "max"),
        mean_abs_pct_error=("abs_pct_error", "mean"),
        median_abs_pct_error=("abs_pct_error", "median"),
    ).reset_index()
    table["pct_exact"] = 100.0 * table["n_exact"] / table["n_comparisons"]
    return table.sort_values("mean_abs_pct_error")


def per_subject(detail: pd.DataFrame) -> pd.DataFrame:
    """Per (subject, method) agreement, to spot a single bad mesh."""
    if detail.empty:
        return pd.DataFrame()
    table = detail.groupby(["subject_id", "method"], sort=True).agg(
        n_measurements=("abs_delta", "size"),
        n_exact=("matches", "sum"),
        mean_abs_pct_error=("abs_pct_error", "mean"),
        max_abs_pct_error=("abs_pct_error", "max"),
        worst_measurement=("abs_pct_error", "idxmax"),
    ).reset_index()
    table["worst_measurement"] = table["worst_measurement"].map(
        detail["measurement"]
    )
    table["pct_exact"] = 100.0 * table["n_exact"] / table["n_measurements"]
    return table


def coverage(frame: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Which measurements each method fills, and whether the reference has them."""
    present = [c for c in MEASUREMENT_COLUMNS if c in frame.columns]
    ok = _successful(frame)
    filled = (
        ok.groupby("anthro_method")[present]
        .apply(lambda g: g.notna().any())
        .T
    )
    filled.index.name = "measurement"
    filled = filled.reset_index()
    if reference in filled.columns:
        filled["in_reference"] = filled[reference]
    return filled


def write_tables(out_dir: str | Path, frame: pd.DataFrame, reference: str) -> dict[str, Path]:
    """Write every comparison table into ``out_dir``; return the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detail = long_form(frame, reference)
    outputs = {
        "detail": detail,
        "by_measurement": per_measurement(detail),
        "by_method": per_method(detail),
        "by_subject": per_subject(detail),
        "coverage": coverage(frame, reference),
    }
    paths: dict[str, Path] = {}
    for name, table in outputs.items():
        path = out_dir / f"comparison_{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    return paths


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="unified.compare",
        description="Score anthropometry methods against a reference method.",
    )
    parser.add_argument(
        "sources",
        nargs="+",
        help="combined_measurements.csv files, or run folders containing one.",
    )
    parser.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE,
        help=f"Method treated as ground truth. Defaults to {DEFAULT_REFERENCE}.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Directory for the comparison tables. Defaults to the first source's folder.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    frame = load_tables(args.sources)

    methods = sorted(_successful(frame)["anthro_method"].unique())
    if args.reference not in methods:
        print(
            f"Reference method {args.reference!r} has no successful rows. "
            f"Available: {', '.join(methods) or '(none)'}"
        )
        return 1

    first = Path(args.sources[0])
    out_dir = Path(args.out) if args.out else (first if first.is_dir() else first.parent)
    paths = write_tables(out_dir, frame, args.reference)

    detail = long_form(frame, args.reference)
    summary = per_method(detail)
    print(f"Reference: {args.reference}")
    print(summary.to_string(index=False))
    for name, path in paths.items():
        print(f"  {name:<14} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
