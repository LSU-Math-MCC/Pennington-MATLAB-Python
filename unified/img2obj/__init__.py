from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from unified.pipeline import SUBJECT_DIR_RE, allocate_run_root, is_supported_image, sanitize_id


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


def _read_native_manifest(out_dir: Path) -> dict[str, object]:
    manifest = out_dir / "manifest.json"
    if not manifest.exists():
        return {}
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _manifest_inputs(out_dir: Path) -> list[str]:
    inputs = _read_native_manifest(out_dir).get("inputs", [])
    return [str(item) for item in inputs] if isinstance(inputs, list) else []


def _write_obj(path: Path, vertices, faces) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# exported by unified.img2obj handoff bridge\n")
        for vertex in vertices:
            handle.write(f"v {float(vertex[0]):.8f} {float(vertex[1]):.8f} {float(vertex[2]):.8f}\n")
        for face in faces:
            values = [int(index) + 1 for index in face[:3]]
            handle.write(f"f {values[0]} {values[1]} {values[2]}\n")


def _export_npz_people(out_dir: Path) -> list[Path]:
    exported: list[Path] = []
    for npz_path in sorted(out_dir.rglob("*.npz")):
        try:
            data = __import__("numpy").load(npz_path, allow_pickle=True)
            people = data["people"]
            faces = data["faces"]
        except Exception:
            continue
        for index, vertices in enumerate(people):
            if len(vertices) == 0:
                continue
            suffix = f"_p{index}" if len(people) > 1 else ""
            obj_path = npz_path.with_name(f"{npz_path.stem}{suffix}.obj")
            _write_obj(obj_path, vertices, faces)
            exported.append(obj_path)
    return exported


def _is_mesh_candidate(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() == ".glb":
        return True
    if path.suffix.lower() == ".ply":
        return "mesh" in name or "proxy" in name
    return False


def _export_mesh_artifacts(out_dir: Path) -> list[Path]:
    exported: list[Path] = []
    for mesh_path in sorted(path for path in out_dir.rglob("*") if path.is_file() and _is_mesh_candidate(path)):
        obj_path = mesh_path.with_suffix(".obj")
        if obj_path.exists():
            continue
        try:
            import trimesh

            mesh = trimesh.load(mesh_path, process=False, force="mesh")
            if mesh is None or len(getattr(mesh, "vertices", [])) == 0 or len(getattr(mesh, "faces", [])) == 0:
                continue
            mesh.export(obj_path)
        except Exception:
            continue
        exported.append(obj_path)
    return exported


def _ensure_obj_exports(out_dir: Path) -> list[str]:
    before = {path.resolve() for path in out_dir.rglob("*.obj")}
    exported = _export_npz_people(out_dir)
    exported.extend(_export_mesh_artifacts(out_dir))
    return [str(path) for path in exported if path.resolve() not in before]


def _find_obj_handoffs(out_dir: Path, method: str) -> list[dict[str, object]]:
    manifest = out_dir / "manifest.json"
    source_images = _manifest_inputs(out_dir)
    handoffs = []
    for obj_path in sorted(out_dir.rglob("*.obj")):
        if any(part in {".cache", "debug"} for part in obj_path.parts):
            continue
        handoffs.append(
            {
                "subject_id": sanitize_id(obj_path.stem),
                "method": method,
                "obj_path": str(obj_path),
                "native_output_dir": str(out_dir),
                "native_manifest_path": str(manifest) if manifest.exists() else "",
                "source_images": source_images,
                "selected_instance": obj_path.stem,
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
    return matches[0] if matches else None


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
    exported = _ensure_obj_exports(out_dir)
    obj_handoffs = _find_obj_handoffs(out_dir, method)
    warnings = []
    errors = []
    if all(status == 0 for status in statuses) and not obj_handoffs:
        warnings.append("Image backend completed but produced no OBJ handoff artifacts.")
    if any(status != 0 for status in statuses):
        errors.append("One or more native image invocations failed.")
    if all(status == 0 for status in statuses) and obj_handoffs:
        status = "success"
    elif obj_handoffs:
        status = "partial"
    else:
        status = "failed"
    return {
        "status": status,
        "native_mode": "mixed" if len(invocations) > 1 else invocations[0][0],
        "native_invocations": invocations,
        "native_statuses": statuses,
        "native_output_dir": str(out_dir),
        "native_manifest_path": str(out_dir / "manifest.json"),
        "exported_obj_paths": exported,
        "obj_handoffs": obj_handoffs,
        "warnings": warnings,
        "errors": errors,
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
