from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import CANONICAL_TEST_SET_DIR, DEFAULT_OUTPUT_DIR, run_pipeline


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="unified obj2anthro",
        description=(
            "Run OBJ anthropometry using segmentation (relocated Python_Fall2025 "
            "implementation) and slice (existing Python_slice_2026 implementation)."
        ),
    )
    parser.add_argument(
        "--input",
        default=str(CANONICAL_TEST_SET_DIR),
        help="Input OBJ file or directory. Defaults to the standard OBJ test set.",
    )
    parser.add_argument("--method", choices=("auto", "segmentation", "slice", "all"), default="auto")
    parser.add_argument(
        "--backend",
        choices=("auto", "segmentation", "slice", "all"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--recursive", action="store_true", default=True, help="Search directories recursively.")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Search only one directory level.")
    parser.add_argument("--units", choices=("auto", "mm", "cm", "dm", "m"), default="cm")
    parser.add_argument(
        "--out",
        "--output-dir",
        dest="output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for the timestamped CSV and raw backend artifacts.",
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
    args = parse_args(argv)
    method = args.backend or args.method
    df = run_pipeline(
        input_path=Path(args.input),
        backend=method,
        recursive=args.recursive,
        units=args.units,
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
        n_slices=args.n_slices,
        save_images=not args.no_images,
        save_aligned_obj=not args.no_aligned_obj,
        height_scale_to_cm=args.height_scale_to_cm,
        show=args.show,
    )
    print(f"Wrote {len(df)} rows to {df.attrs.get('output_csv')}")
    print(f"Raw artifacts: {df.attrs.get('raw_output_dir')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
