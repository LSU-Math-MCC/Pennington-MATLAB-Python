from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="unified", description="Run the staged Pennington tool.")
    parser.add_argument("--input", required=True, help="Image, OBJ, or directory to process.")
    parser.add_argument("--image-method", default="auto", help="Image-to-OBJ method. Defaults to auto.")
    parser.add_argument("--anthro-method", default="auto", help="Anthropometry method. Defaults to auto.")
    parser.add_argument("--units", choices=("auto", "mm", "cm", "dm", "m"), default="auto")
    parser.add_argument("--out", default=None, help="Run output root. Defaults to runs/<UTC timestamp>.")
    return parser


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "img2obj":
        from . import img2obj

        return img2obj.main(argv[1:])
    if argv and argv[0] == "obj2anthro":
        from . import obj2anthro

        return obj2anthro.main(argv[1:])
    if argv and argv[0] == "ml":
        from . import ml

        return ml.main(argv[1:])

    args = build_parser().parse_args(argv)
    manifest = run_pipeline(
        input_path=Path(args.input),
        image_method=args.image_method,
        anthro_method=args.anthro_method,
        units=args.units,
        out=args.out,
    )
    print(f"Run {manifest['run_id']} wrote {Path(args.out) if args.out else 'runs/' + manifest['run_id']}")
    return 0 if manifest.get("status") == "success" else 1
