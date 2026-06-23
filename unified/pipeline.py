from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SUBJECT_DIR_RE = re.compile(r"^(s|subject)\d+$", re.IGNORECASE)
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs"


@dataclass(frozen=True)
class ObjHandoff:
    subject_id: str
    method: str
    obj_path: Path
    native_output_dir: Path | None = None
    native_manifest_path: Path | None = None


def canonical_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def allocate_run_root(out: str | Path | None = None) -> Path:
    path = Path(out) if out is not None else RUNS_ROOT / canonical_run_id()
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def _repo_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _relative(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _nearest_subject_dir(file_path: Path, input_root: Path) -> Path | None:
    current = file_path.parent
    root = input_root if input_root.is_dir() else input_root.parent
    matches: list[Path] = []
    while True:
        if SUBJECT_DIR_RE.match(current.name):
            matches.append(current)
        if current == root or root not in current.parents:
            break
        current = current.parent
    return matches[-1] if matches else None


def classify_input(input_path: str | Path) -> dict[str, object]:
    path = Path(input_path)
    plan: dict[str, object] = {
        "input": str(path),
        "images": [],
        "objs": [],
        "unsupported": [],
        "subject_groups": {},
    }
    if path.is_file():
        if is_supported_image(path):
            plan["images"] = [str(path)]
        elif path.suffix.lower() == ".obj":
            plan["objs"] = [str(path)]
        else:
            plan["unsupported"] = [str(path)]
        return plan
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    images: list[Path] = []
    objs: list[Path] = []
    unsupported: list[Path] = []
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        if is_supported_image(item):
            images.append(item)
        elif item.suffix.lower() == ".obj":
            objs.append(item)
        else:
            unsupported.append(item)

    groups: dict[str, list[str]] = {}
    for image in images:
        subject_dir = _nearest_subject_dir(image, path)
        if subject_dir is None:
            key = _relative(image.with_suffix(""), path)
        else:
            key = _relative(subject_dir, path)
        groups.setdefault(key, []).append(_relative(image, path))

    plan["images"] = [str(p) for p in images]
    plan["objs"] = [str(p) for p in objs]
    plan["unsupported"] = [str(p) for p in unsupported]
    plan["subject_groups"] = groups
    return plan


def selected_anthro_methods(method: str) -> list[str]:
    if method in {"auto", "all"}:
        return ["segmentation", "slice"]
    return [method]


def expand_branches(obj_handoffs: Iterable[ObjHandoff], anthro_methods: Iterable[str]) -> list[tuple[ObjHandoff, str]]:
    # TODO: Central policy point for inner-vs-outer product reconciliation.
    # Current behavior is the lossless full outer product of upstream artifacts
    # and selected downstream methods.
    return [(handoff, method) for handoff in obj_handoffs for method in anthro_methods]


def _write_manifest(run_root: Path, manifest: dict[str, object]) -> None:
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def _obj_handoffs_from_paths(paths: Iterable[Path], method: str) -> list[ObjHandoff]:
    return [ObjHandoff(subject_id=path.stem, method=method, obj_path=path) for path in paths]


def run_pipeline(
    input_path: str | Path,
    image_method: str = "auto",
    anthro_method: str = "auto",
    units: str = "auto",
    out: str | Path | None = None,
) -> dict[str, object]:
    from . import img2obj, obj2anthro

    started = datetime.now(timezone.utc)
    run_root = allocate_run_root(out)
    input_path = Path(input_path)
    plan = classify_input(input_path)
    manifest: dict[str, object] = {
        "run_id": run_root.name,
        "repository_head": _repo_head(),
        "started_at": started.isoformat(),
        "input": str(input_path),
        "input_plan": plan,
        "selected_methods": {"image": image_method, "anthropometry": anthro_method},
        "stages": {},
        "warnings": [],
        "status": "running",
    }

    obj_handoffs = _obj_handoffs_from_paths((Path(p) for p in plan["objs"]), "direct")

    if plan["images"]:
        image_out = run_root / "img2obj"
        image_result = img2obj.run(input_path, method=image_method, out=image_out)
        manifest["stages"]["img2obj"] = image_result
        obj_handoffs.extend(
            ObjHandoff(
                subject_id=item.get("subject_id", Path(item["obj_path"]).stem),
                method=item.get("method", image_method),
                obj_path=Path(item["obj_path"]),
                native_output_dir=Path(item["native_output_dir"]) if item.get("native_output_dir") else None,
                native_manifest_path=Path(item["native_manifest_path"]) if item.get("native_manifest_path") else None,
            )
            for item in image_result.get("obj_handoffs", [])
        )
        if not image_result.get("obj_handoffs"):
            manifest["warnings"].append("Image processing produced no OBJ handoff artifacts; anthropometry skipped for images.")

    if obj_handoffs:
        obj_out = run_root / "obj2anthro"
        methods = selected_anthro_methods(anthro_method)
        branches = expand_branches(obj_handoffs, methods)
        anthro_outputs = []
        for handoff, method in branches:
            df = obj2anthro.run_pipeline(
                handoff.obj_path,
                backend=method,
                units=units,
                output_dir=obj_out / method,
            )
            anthro_outputs.append(
                {
                    "source_obj": str(handoff.obj_path),
                    "method": method,
                    "output_csv": df.attrs.get("output_csv"),
                    "raw_output_dir": df.attrs.get("raw_output_dir"),
                    "rows": len(df),
                }
            )
        manifest["stages"]["obj2anthro"] = anthro_outputs
    else:
        manifest["stages"]["obj2anthro"] = {"skipped": True, "reason": "no OBJ inputs or handoffs"}

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "success"
    _write_manifest(run_root, manifest)
    return manifest
