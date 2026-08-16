from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from unified.pipeline import allocate_run_root

from .pipeline import AUTO_BACKENDS, EXTRA_BACKENDS, run_pipeline


METHOD_CHOICES = ("auto", *AUTO_BACKENDS, *EXTRA_BACKENDS, "all")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="unified obj2anthro",
        description=(
            "Run OBJ anthropometry using the segmentation, slice, avatar, and MATLAB backends."
        ),
    )
    parser.add_argument("--input", required=True, help="Input OBJ file or directory.")
    parser.add_argument("--method", choices=METHOD_CHOICES, default="auto")
    parser.add_argument(
        "--backend",
        choices=METHOD_CHOICES,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--recursive", action="store_true", default=True, help="Search directories recursively.")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Search only one directory level.")
    parser.add_argument("--units", choices=("auto", "mm", "cm", "dm", "m"), default="auto")
    parser.add_argument(
        "--out",
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Directory for the timestamped CSV and raw backend artifacts. Defaults to runs/<UTC timestamp>/obj2anthro.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="CSV stem and raw folder name. Defaults to UTC YYYYMMDDTHHMMSSZ_full_anthro.",
    )
    parser.add_argument("--n-slices", type=int, default=200, help="Slice backend slice count.")
    parser.add_argument(
        "--height-scale-to-cm",
        type=float,
        default=None,
        help="Explicit mesh-unit to cm scale for the slice backend.",
    )
    parser.add_argument("--no-images", action="store_true", help="Disable image/diagnostic output.")
    parser.add_argument("--no-aligned-obj", action="store_true", help="Disable slice aligned OBJ output.")
    parser.add_argument("--show", action="store_true", help="Open segmentation backend interactive mesh view.")
    return parser.parse_args(argv)


def main(argv=None):
    from unified.combine import summarize_combined, write_combined_table

    args = parse_args(argv)
    method = args.backend or args.method
    if args.output_dir:
        output_dir = Path(args.output_dir)
        run_root = output_dir
    else:
        run_root = allocate_run_root()
        output_dir = run_root / "obj2anthro"
    df = run_pipeline(
        input_path=Path(args.input),
        backend=method,
        recursive=args.recursive,
        units=args.units,
        output_dir=output_dir,
        run_id=args.run_id,
        n_slices=args.n_slices,
        save_images=not args.no_images,
        save_aligned_obj=not args.no_aligned_obj,
        height_scale_to_cm=args.height_scale_to_cm,
        show=args.show,
    )
    print(f"Wrote {len(df)} rows to {df.attrs.get('output_csv')}")
    print(f"Raw artifacts: {df.attrs.get('raw_output_dir')}")

    combined_path = write_combined_table(
        run_root,
        [{"frame": df, "run_id": run_root.name, "source_method": "direct", "branch_dir": str(output_dir)}],
    )
    print(f"Combined table: {combined_path}")
    print(summarize_combined(pd.read_csv(combined_path)))

    statuses = [str(value) for value in df.get("status", [])]
    return 0 if statuses and all(status == "success" for status in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
