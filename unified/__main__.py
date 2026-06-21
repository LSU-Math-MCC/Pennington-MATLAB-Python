from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import CANONICAL_TEST_SET_DIR, DEFAULT_OUTPUT_DIR, run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run the unified OBJ measurement pipeline.")
    parser.add_argument(
        "--input",
        default=str(CANONICAL_TEST_SET_DIR),
        help="Input OBJ file or directory. Defaults to the standard OBJ test set.",
    )
    parser.add_argument("--backend", choices=("segmentation", "slice", "all"), default="all")
    parser.add_argument("--recursive", action="store_true", help="Search directories recursively.")
    parser.add_argument("--units", choices=("auto", "mm", "cm", "dm", "m"), default="cm")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Versionable directory for the timestamped CSV.",
    )
    parser.add_argument(
        "--raw-output-root",
        default=None,
        help="Git-ignored raw artifact root. Defaults to OUTPUT_DIR/raw.",
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
    return parser.parse_args()


def main():
    args = parse_args()
    df = run_pipeline(
        input_path=Path(args.input),
        backend=args.backend,
        recursive=args.recursive,
        units=args.units,
        output_dir=Path(args.output_dir),
        raw_output_root=Path(args.raw_output_root) if args.raw_output_root else None,
        run_id=args.run_id,
        n_slices=args.n_slices,
        save_images=not args.no_images,
        save_aligned_obj=not args.no_aligned_obj,
        height_scale_to_cm=args.height_scale_to_cm,
        show=args.show,
    )
    print(f"Wrote {len(df)} rows to {df.attrs.get('output_csv')}")
    print(f"Raw artifacts: {df.attrs.get('raw_output_dir')}")


if __name__ == "__main__":
    main()
