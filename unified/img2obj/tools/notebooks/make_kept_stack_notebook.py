"""Generate the kept-stack tooling notebook."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB_PATH = REPO / "notebooks" / "kept_stack_tooling_demo.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


cells = [
    md(
        """
# MeshMap Kept-Stack Tooling Demo

This notebook replaces the broad exploratory demos after cleanup. It only covers the tools we
decided to keep:

| Product need | Kept tool family |
|---|---|
| Anthropometric body shape | CameraHMR, SHAPY benchmark/reference |
| Close-range perspective + SMPL-X body | BLADE |
| Whole-body/canonical support | LHM / Multi-HMR |
| Hands/fingers | HaMeR / MANO |
| Face fidelity | DECA / FLAME |
| Rendering and visual audit | Blender |
| Default lightweight pipeline | YOLOv8n, MediaPipe Tasks, Depth-Anything, core `src/pipeline` |

Removed families such as 3DDFA, GFPGAN, PiFUHD, SMPLitex, and HMR2 are intentionally absent.
"""
    ),
    code(
        """
import json
import os
import subprocess
import sys
from pathlib import Path

from IPython.display import HTML, Markdown, display
from PIL import Image, ImageDraw

REPO = Path.cwd().resolve()
while REPO != REPO.parent and not (REPO / "pyproject.toml").exists():
    REPO = REPO.parent

OUT = REPO / "notebooks" / "_kept_stack_outputs"
OUT.mkdir(parents=True, exist_ok=True)

def html_table(rows, columns=None):
    rows = list(rows)
    columns = columns or (list(rows[0]) if rows else [])
    def esc(x):
        return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{esc(r.get(c, ''))}</td>" for c in columns) + "</tr>" for r in rows)
    display(HTML(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"))

def mib(path):
    p = Path(path)
    if not p.exists():
        return None
    if p.is_file():
        return round(p.stat().st_size / 1024 / 1024, 2)
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return round(total / 1024 / 1024, 2)

print("repo:", REPO)
print("output:", OUT)
"""
    ),
    md("## 1. Kept Stack Inventory"),
    code(
        """
kept = [
    {"area": "default_pipeline", "local": "src/ + models/*.task + yolov8n*.pt", "external": "HF Depth-Anything cache", "status": "kept"},
    {"area": "body_shape", "local": "tools/hmr/camerahmr/*.py, tools/smplx", "external": "~/CameraHMR + camerahmr env", "status": "kept"},
    {"area": "perspective_smplx", "local": "tools/hmr/blade/run_blade.py, BLADE setup docs", "external": "~/blade + blade_env", "status": "kept"},
    {"area": "anthropometry_reference", "local": "tools/hmr/shapy/run_shapy.py, tools/benchmark/bench_all.py", "external": "~/shapy + shapy env", "status": "kept"},
    {"area": "whole_body_avatar", "local": "src/pipeline/backends/lhm_backend.py", "external": "~/LHM + lhm env", "status": "kept"},
    {"area": "hands", "local": "tools/hands/run_hamer_slim.py, tools/hands/weld_hands.py", "external": "~/hamer", "status": "kept"},
    {"area": "face", "local": "tools/face/deca_full.py, tools/face/integrate_deca_face.py, tools/face/make_flame_disp.py", "external": "~/DECA + FLAME/SMPL-X correspondence", "status": "kept"},
    {"area": "rendering", "local": "vendor/blender, tools/render/blender_render.py", "external": "none", "status": "kept"},
]
html_table(kept, ["area", "local", "external", "status"])
"""
    ),
    md("## 2. Local Footprint After Cleanup"),
    code(
        """
local_paths = [
    "src", "tools", "tests", "benchmarks", "notebooks", "models", "datasets", "vendor",
    "yolov8n-seg.pt", "yolov8n-pose.pt",
]
rows = [{"path": p, "exists": Path(p).exists(), "MiB": mib(REPO / p)} for p in local_paths]
html_table(rows, ["path", "exists", "MiB"])
"""
    ),
    md("## 3. External WSL Installs Kept"),
    code(
        """
external = [
    "~/LHM",
    "~/hamer",
    "~/CameraHMR",
    "~/blade",
    "~/shapy",
    "~/DECA",
    "~/miniconda3/envs/lhm",
    "~/miniconda3/envs/camerahmr",
    "~/miniconda3/envs/blade_env",
    "~/miniconda3/envs/shapy",
]
cmd = "for p in " + " ".join(external) + "; do if [ -e \\"$p\\" ]; then du -sh \\"$p\\"; else echo MISSING \\"$p\\"; fi; done"
try:
    cp = subprocess.run(["wsl", "-e", "bash", "-lc", cmd], capture_output=True, text=True, timeout=120)
    print(cp.stdout)
    if cp.stderr:
        print(cp.stderr)
except Exception as e:
    print("WSL status check skipped:", repr(e))
"""
    ),
    md("## 4. Removed Families Are Absent"),
    code(
        """
dropped = [
    "gfpgan", "yolov8x.pt", "yolov8x-seg.pt",
    "tools/run_3ddfa_face.py", "tools/gfpgan_enhance.py", "tools/run_pifuhd.sh",
    "tools/smplitex_complete.py", "tools/run_hmr2.py",
    "benchmarks/configs/meshmap_hmr2.json", "benchmarks/configs/meshmap_hmr2_clip.json",
]
rows = [{"path": p, "exists": (REPO / p).exists()} for p in dropped]
html_table(rows, ["path", "exists"])
assert not any(r["exists"] for r in rows), "A removed-family path still exists"
"""
    ),
    md("## 5. Product DAG"),
    code(
        """
boxes = [
    ("Images / Video Frames", 40, 40),
    ("YOLO + MediaPipe + Depth", 330, 40),
    ("CameraHMR\\nshape prior", 40, 160),
    ("BLADE\\nperspective SMPL-X", 270, 160),
    ("LHM / Multi-HMR\\nwhole-body support", 520, 160),
    ("SHAPY\\nmeasurement reference", 40, 310),
    ("HaMeR\\nhands", 270, 310),
    ("DECA / FLAME\\nface", 520, 310),
    ("Blender / viewers\\nrender + audit", 270, 460),
    ("MeshMap\\ncanonical anthropometry", 520, 460),
]
img = Image.new("RGB", (820, 600), (248, 248, 248))
d = ImageDraw.Draw(img)
for text, x, y in boxes:
    d.rounded_rectangle((x, y, x + 210, y + 82), radius=8, fill=(255, 255, 255), outline=(40, 80, 110), width=2)
    d.multiline_text((x + 12, y + 18), text, fill=(20, 35, 45), spacing=4)
arrows = [
    ((250, 81), (330, 81)), ((435, 122), (145, 160)), ((435, 122), (375, 160)), ((435, 122), (625, 160)),
    ((145, 242), (145, 310)), ((375, 242), (375, 310)), ((625, 242), (625, 310)),
    ((145, 392), (520, 501)), ((375, 392), (520, 501)), ((625, 392), (625, 460)), ((480, 501), (520, 501)),
]
for a, b in arrows:
    d.line((a, b), fill=(70, 100, 130), width=3)
    d.ellipse((b[0]-4, b[1]-4, b[0]+4, b[1]+4), fill=(70, 100, 130))
display(img)
"""
    ),
    md("## 6. Kept Tool Commands"),
    code(
        """
commands = [
    {"tool": "default_pipeline_single", "command": "python -m pipeline.run single --backend real --image datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_000029.png --out runs/ssp3d_bodybuilder_single"},
    {"tool": "CameraHMR SSP-3D", "command": "conda run -n camerahmr python tools/hmr/camerahmr/camerahmr_ssp3d.py"},
    {"tool": "CameraHMR subjects", "command": "conda run -n camerahmr python tools/hmr/camerahmr/camerahmr_subjects.py"},
    {"tool": "BLADE overlay/schema", "command": "conda run -n blade_env python tools/hmr/blade/run_blade.py --images <img> --out <out>"},
    {"tool": "SHAPY subject", "command": "conda run -n shapy python tools/hmr/shapy/run_shapy.py <subject_dir> <out_dir>"},
    {"tool": "SHAPY benchmark summary", "command": "python tools/benchmark/bench_all.py --dataset ssp3d --methods camerahmr_sota meshmap_full published_shapy"},
    {"tool": "LHM backend", "command": "PYTHONPATH=src python -m pipeline.run single --backend lhm --image <img> --out <out>"},
    {"tool": "HaMeR slim", "command": "bash tools/hands/run_hamer.sh <image> <out_dir>"},
    {"tool": "DECA face", "command": "conda run -n camerahmr python tools/face/deca_full.py <image> <out_prefix>"},
    {"tool": "DECA->body face integration", "command": "python tools/face/integrate_deca_face.py <subject> <body_glb> <out_prefix>"},
    {"tool": "Blender render", "command": "python tools/render/blender_render.py <mesh_or_scene> <out_png>"},
]
html_table(commands, ["tool", "command"])
"""
    ),
    md("## 7. Benchmark Harness Smoke"),
    code(
        """
bench_out = OUT / "ssp3d_bench_smoke"
cmd = [sys.executable, str(REPO / "tools" / "benchmark" / "bench_all.py"), "--dataset", "ssp3d", "--out", str(bench_out)]
cp = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=120)
print(cp.stdout)
if cp.stderr:
    print(cp.stderr)
print("returncode:", cp.returncode)
metrics = bench_out / "metrics.json"
if metrics.exists():
    display(Markdown("### metrics.json"))
    print(metrics.read_text())
assert cp.returncode == 0
"""
    ),
    md("## 8. Local Unit Test Smoke"),
    code(
        """
cmd = [sys.executable, "-m", "pytest", "tests/test_face_mapping.py", "tests/test_cli_smoke.py", "-q"]
cp = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=180)
print(cp.stdout)
if cp.stderr:
    print(cp.stderr)
print("returncode:", cp.returncode)
assert cp.returncode == 0
"""
    ),
    md("## 9. Notebook Summary"),
    code(
        """
print("Kept stack notebook executed.")
print("Project record (tools audit §12):", REPO / "docs" / "PROJECT.md")
print("Benchmark config:", REPO / "benchmarks" / "configs" / "meshmap_full.json")
print("Generated notebook outputs:", OUT)
"""
    ),
]


def main() -> None:
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(NB_PATH)


if __name__ == "__main__":
    main()
