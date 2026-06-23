from __future__ import annotations

import argparse
import sys
from pathlib import Path

from unified.pipeline import SUBJECT_DIR_RE, allocate_run_root, is_supported_image


IMG2OBJ_ROOT = Path(__file__).resolve().parent
IMG2OBJ_SRC = IMG2OBJ_ROOT / "src"


def _ensure_native_importable() -> None:
    src = str(IMG2OBJ_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


def native_main(argv=None) -> int:
    """Call the relocated image preprocessor's existing CLI."""
    _ensure_native_importable()
    from pipeline.run import main as run_main

    return run_main(argv)


def _native_mode(input_path: Path) -> tuple[str, str]:
    if input_path.is_file() and is_supported_image(input_path):
        return "single", "--image"
    if input_path.is_dir() and SUBJECT_DIR_RE.match(input_path.name):
        return "subject", "--subject"
    if input_path.is_dir():
        return "folder", "--images"
    raise ValueError(f"Input is not a supported image file or directory: {input_path}")


def _find_obj_handoffs(out_dir: Path, method: str) -> list[dict[str, str]]:
    manifest = out_dir / "manifest.json"
    handoffs = []
    for obj_path in sorted(out_dir.rglob("*.obj")):
        handoffs.append(
            {
                "subject_id": obj_path.stem,
                "method": method,
                "obj_path": str(obj_path),
                "native_output_dir": str(out_dir),
                "native_manifest_path": str(manifest) if manifest.exists() else "",
            }
        )
    return handoffs


def _relative(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(path.name)


def _nearest_subject_dir(file_path: Path, root: Path) -> Path | None:
    current = file_path.parent
    matches: list[Path] = []
    while True:
        if SUBJECT_DIR_RE.match(current.name):
            matches.append(current)
        if current == root or root not in current.parents:
            break
        current = current.parent
    return matches[-1] if matches else None


def _native_invocations(input_path: Path, out_dir: Path, native_method: str, native_args: list[str] | None) -> list[list[str]]:
    mode, input_flag = _native_mode(input_path)
    if not input_path.is_dir() or mode == "subject":
        return [[mode, input_flag, str(input_path), "--out", str(out_dir), "--backend", native_method, *(native_args or [])]]

    images = sorted(p for p in input_path.rglob("*") if p.is_file() and is_supported_image(p))
    subject_dirs = sorted({p for image in images if (p := _nearest_subject_dir(image, input_path)) is not None})
    if not subject_dirs:
        return [["folder", "--images", str(input_path), "--out", str(out_dir), "--backend", native_method, *(native_args or [])]]

    grouped_images = {image for subject_dir in subject_dirs for image in subject_dir.rglob("*") if image.is_file() and is_supported_image(image)}
    invocations: list[list[str]] = []
    for subject_dir in subject_dirs:
        rel = _relative(subject_dir, input_path)
        invocations.append(
            ["subject", "--subject", str(subject_dir), "--out", str(out_dir / rel), "--backend", native_method, *(native_args or [])]
        )
    for image in images:
        if image in grouped_images:
            continue
        rel = _relative(image.with_suffix(""), input_path)
        invocations.append(
            ["single", "--image", str(image), "--out", str(out_dir / rel), "--backend", native_method, *(native_args or [])]
        )
    return invocations


def run(input_path: str | Path, method: str = "auto", out: str | Path | None = None, native_args: list[str] | None = None) -> dict[str, object]:
    input_path = Path(input_path)
    out_dir = Path(out) if out is not None else allocate_run_root() / "img2obj"
    out_dir.mkdir(parents=True, exist_ok=True)
    native_method = "auto" if method == "auto" else method
    invocations = _native_invocations(input_path, out_dir, native_method, native_args)
    statuses = [native_main(argv) for argv in invocations]
    return {
        "status": "success" if all(status == 0 for status in statuses) else "failed",
        "native_mode": "mixed" if len(invocations) > 1 else invocations[0][0],
        "native_invocations": invocations,
        "native_output_dir": str(out_dir),
        "native_manifest_path": str(out_dir / "manifest.json"),
        "obj_handoffs": _find_obj_handoffs(out_dir, method),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="unified img2obj", description="Run the relocated image-to-OBJ preprocessor.")
    parser.add_argument("--input", required=True, help="Image file or directory.")
    parser.add_argument("--method", default="auto", help="Native image method/backend. Defaults to auto.")
    parser.add_argument("--out", default=None, help="Output root. Defaults to runs/<UTC timestamp>/img2obj.")
    return parser


def main(argv=None) -> int:
    args, native_args = build_parser().parse_known_args(argv)
    result = run(args.input, method=args.method, out=args.out, native_args=native_args)
    print(result["native_output_dir"])
    return 0 if result["status"] == "success" else 1
