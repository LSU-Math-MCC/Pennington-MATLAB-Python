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
    source_images: tuple[str, ...] = ()
    selected_instance: str | None = None


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


def sanitize_id(value: str | Path) -> str:
    raw = Path(value).stem if isinstance(value, Path) else str(value)
    raw = raw.replace("\\", "/")
    raw = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("._-")
    return raw or "subject"


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
    return matches[0] if matches else None


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
    return [ObjHandoff(subject_id=sanitize_id(path), method=method, obj_path=path) for path in paths]


def _handoff_from_mapping(item: dict[str, object], fallback_method: str) -> ObjHandoff:
    obj_path = Path(str(item["obj_path"]))
    source_images = item.get("source_images") or item.get("source_image") or ()
    if isinstance(source_images, str):
        source_images = (source_images,)
    return ObjHandoff(
        subject_id=sanitize_id(str(item.get("subject_id") or obj_path.stem)),
        method=str(item.get("method") or fallback_method),
        obj_path=obj_path,
        native_output_dir=Path(str(item["native_output_dir"])) if item.get("native_output_dir") else None,
        native_manifest_path=Path(str(item["native_manifest_path"])) if item.get("native_manifest_path") else None,
        source_images=tuple(str(p) for p in source_images),
        selected_instance=str(item["selected_instance"]) if item.get("selected_instance") else None,
    )


def _handoff_manifest(handoff: ObjHandoff) -> dict[str, object]:
    return {
        "subject_id": handoff.subject_id,
        "source_method": handoff.method,
        "obj_path": str(handoff.obj_path),
        "native_output_dir": str(handoff.native_output_dir) if handoff.native_output_dir else None,
        "native_manifest_path": str(handoff.native_manifest_path) if handoff.native_manifest_path else None,
        "source_images": list(handoff.source_images),
        "selected_instance": handoff.selected_instance,
    }


def _unique_branch_dir(base: Path, handoff: ObjHandoff, anthro_method: str, used: set[Path]) -> Path:
    root = base / sanitize_id(handoff.method) / sanitize_id(handoff.subject_id) / sanitize_id(anthro_method)
    path = root
    index = 2
    while path in used:
        path = root.with_name(f"{root.name}_{index}")
        index += 1
    used.add(path)
    return path


def _branch_status(df) -> str:
    if len(df) == 0:
        return "failed"
    statuses = [str(value) for value in df.get("status", [])]
    if statuses and all(status == "success" for status in statuses):
        return "success"
    if any(status == "success" for status in statuses):
        return "partial"
    return "failed"


def _root_status(manifest: dict[str, object], branch_outputs: list[dict[str, object]]) -> str:
    if not branch_outputs:
        return "failed"
    branch_statuses = [str(item.get("status", "failed")) for item in branch_outputs]
    useful = any(status in {"success", "partial"} and int(item.get("rows", 0)) > 0 for status, item in zip(branch_statuses, branch_outputs))
    any_problem = bool(manifest["warnings"] or manifest["errors"] or any(status != "success" for status in branch_statuses))
    if not useful:
        return "failed"
    return "partial" if any_problem else "success"


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
    manifest: dict[str, object] = {
        "run_id": run_root.name,
        "repository_head": _repo_head(),
        "started_at": started.isoformat(),
        "input": str(input_path),
        "input_plan": {},
        "selected_methods": {"image": image_method, "anthropometry": anthro_method},
        "stages": {},
        "warnings": [],
        "errors": [],
        "image_artifact_roots": [],
        "obj_handoffs": [],
        "manifest_path": str(run_root / "manifest.json"),
        "status": "running",
    }

    branch_outputs: list[dict[str, object]] = []
    try:
        plan = classify_input(input_path)
        manifest["input_plan"] = plan
        obj_handoffs = _obj_handoffs_from_paths((Path(p) for p in plan["objs"]), "direct")

        if not plan["images"] and not plan["objs"]:
            if plan["unsupported"]:
                manifest["errors"].append("Input contains only unsupported files.")
            else:
                manifest["errors"].append("Input produced no image or OBJ work.")

        if plan["images"]:
            image_out = run_root / "img2obj"
            try:
                image_result = img2obj.run(input_path, method=image_method, out=image_out)
            except Exception as exc:  # noqa: BLE001
                image_result = {
                    "status": "failed",
                    "native_output_dir": str(image_out),
                    "obj_handoffs": [],
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }
            manifest["stages"]["img2obj"] = image_result
            manifest["image_artifact_roots"] = [image_result.get("native_output_dir", str(image_out))]
            for message in image_result.get("warnings", []):
                manifest["warnings"].append(str(message))
            for message in image_result.get("errors", []):
                manifest["errors"].append(str(message))
            obj_handoffs.extend(
                _handoff_from_mapping(item, image_method)
                for item in image_result.get("obj_handoffs", [])
            )
            if not image_result.get("obj_handoffs"):
                manifest["warnings"].append("Image processing produced no OBJ handoff artifacts; anthropometry skipped for images.")

        manifest["obj_handoffs"] = [_handoff_manifest(handoff) for handoff in obj_handoffs]

        if obj_handoffs:
            obj_out = run_root / "obj2anthro"
            methods = selected_anthro_methods(anthro_method)
            branches = expand_branches(obj_handoffs, methods)
            used_dirs: set[Path] = set()
            for handoff, method in branches:
                branch_dir = _unique_branch_dir(obj_out, handoff, method, used_dirs)
                try:
                    df = obj2anthro.run_pipeline(
                        handoff.obj_path,
                        backend=method,
                        units=units,
                        output_dir=branch_dir,
                        run_id="results",
                    )
                    status = _branch_status(df)
                    error = ""
                except Exception as exc:  # noqa: BLE001
                    df = None
                    status = "failed"
                    error = f"{type(exc).__name__}: {exc}"
                branch_outputs.append(
                    {
                        "source_obj": str(handoff.obj_path),
                        "source_method": handoff.method,
                        "subject_id": handoff.subject_id,
                        "anthro_method": method,
                        "branch_dir": str(branch_dir),
                        "output_csv": df.attrs.get("output_csv") if df is not None else None,
                        "raw_output_dir": df.attrs.get("raw_output_dir") if df is not None else None,
                        "raw_artifact_dir": df.attrs.get("raw_output_dir") if df is not None else None,
                        "status": status,
                        "rows": len(df) if df is not None else 0,
                        "error": error,
                    }
                )
                if error:
                    manifest["errors"].append(error)
            manifest["stages"]["obj2anthro"] = branch_outputs
        else:
            manifest["stages"]["obj2anthro"] = {"skipped": True, "reason": "no OBJ inputs or handoffs"}

        manifest["status"] = _root_status(manifest, branch_outputs)
    except Exception as exc:  # noqa: BLE001
        manifest["errors"].append(f"{type(exc).__name__}: {exc}")
        manifest["status"] = "failed"
    finally:
        if manifest["status"] == "running":
            manifest["status"] = _root_status(manifest, branch_outputs)
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(run_root, manifest)
    return manifest
