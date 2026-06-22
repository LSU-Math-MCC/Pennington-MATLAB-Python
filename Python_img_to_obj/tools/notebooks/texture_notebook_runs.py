"""Callable notebook runners for texture-fit demos.

These functions keep notebooks executable without embedding long shell recipes
in every cell. Heavy model code still runs inside the WSL conda envs that own
those dependencies.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _find_repo() -> Path:
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return p.parents[2]


def _to_wsl(path) -> str:
    r"""C:\a\b -> /mnt/c/a/b (identity if the path is already POSIX, e.g. on a WSL kernel)."""
    p = os.path.abspath(str(path))
    drive, rest = os.path.splitdrive(p)
    return ("/mnt/" + drive[0].lower() + rest.replace("\\", "/")) if drive else p.replace("\\", "/")


REPO = _find_repo()
WSL_REPO = _to_wsl(REPO)            # path the WSL conda envs see, derived from REPO (no hardcoding)
SSP3D_SINGLE = "datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_000029.png"
SSP3D_PERSON = "datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_*.png"


def run_wsl(command: str, env: str = "lhm", timeout: int | None = None) -> None:
    full = (
        "source ~/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate {env} && "
        f"cd {WSL_REPO} && "
        f"{command}"
    )
    print(f"\n$ wsl bash -lc {full}")
    subprocess.run(["wsl", "bash", "-lc", full], check=True, timeout=timeout)


def ensure_texture_bake(
    subject: str,
    out: str,
    atlas: int,
    extra_args: str = "",
    rebuild: bool = False,
    timeout: int = 60 * 60,
) -> Path:
    out_path = REPO / out
    expected = out_path / "uv_report.json"
    if rebuild or not expected.exists():
        run_wsl(
            f"python tools/texture/texture_uv_bake.py --subject '{subject}' "
            f"--out {out} --atlas {atlas} {extra_args}",
            env="lhm",
            timeout=timeout,
        )
    else:
        print("bake already present:", expected)
    return out_path


def ensure_texture_overlay(
    subject: str,
    bake: str,
    out: str,
    limit: int,
    atlas_mode: str = "auto",
    rebuild: bool = False,
    timeout: int = 60 * 60,
) -> Path:
    out_path = REPO / out
    expected = out_path / "TEXTURED_MESH_OVERLAY_CONTACT_SHEET.png"
    if rebuild or not expected.exists():
        run_wsl(
            f"python tools/texture/render_canonical_texture_overlays.py "
            f"--subject '{subject}' --bake {bake} --out {out} "
            f"--limit {limit} --atlas-mode {atlas_mode}",
            env="lhm",
            timeout=timeout,
        )
    else:
        print("overlay already present:", expected)
    return out_path


def ensure_singleton_texture_runs(rebuild: bool = False) -> dict[str, Path]:
    specs = {
        "SSP-3D bodybuilder observed 1024": (
            "runs/singleton_texture_ssp3d_bodybuilder_observed_1024",
            1024,
            "--face-repair off --unmapped-fill grey --no-coherent-face",
            "observed",
        ),
        "SSP-3D bodybuilder observed 2048": (
            "runs/singleton_texture_ssp3d_bodybuilder_observed_2048",
            2048,
            "--face-repair off --unmapped-fill grey --no-coherent-face",
            "observed",
        ),
        "SSP-3D bodybuilder repaired/completed failure 1024": (
            "runs/singleton_texture_ssp3d_bodybuilder_uvfix_1024",
            1024,
            "--face-repair face-head --unmapped-fill skin",
            "full",
        ),
    }
    runs: dict[str, Path] = {}
    for label, (out, atlas, args, mode) in specs.items():
        runs[label] = ensure_texture_bake(SSP3D_SINGLE, out, atlas, args, rebuild=rebuild)
        ensure_texture_overlay(
            SSP3D_SINGLE,
            out,
            f"{out}/photo_overlays",
            limit=1,
            atlas_mode=mode,
            rebuild=rebuild,
        )
    return runs


def ensure_multiview_texture_run(rebuild: bool = False) -> dict[str, Path]:
    bake = "runs/mesh_texture_notebook_ssp3d_bodybuilder_hq"
    ensure_texture_bake(
        SSP3D_PERSON,
        bake,
        2048,
        "--face-repair face-head --face-occlusion-clean conservative",
        rebuild=rebuild,
        timeout=90 * 60,
    )

    deca_prefix = "runs/decafull_ssp3d_bodybuilder"
    deca_expected = REPO / f"{deca_prefix}_verts.npy"
    if rebuild or not deca_expected.exists():
        run_wsl(
            f"python tools/face/deca_full.py {SSP3D_SINGLE} {deca_prefix} cuda",
            env="camerahmr",
            timeout=60 * 60,
        )
    else:
        print("DECA already present:", deca_expected)

    hybrid = REPO / bake / "deca_hybrid_body_deca_face.glb"
    if rebuild or not hybrid.exists():
        run_wsl(
            "python tools/face/integrate_deca_face.py ssp3d_bodybuilder "
            f"{bake}/apose_textured_uv.glb {bake}/deca_hybrid",
            env="lhm",
            timeout=60 * 60,
        )
    else:
        print("hybrid already present:", hybrid)

    preview = REPO / bake / "hybrid_preview"
    if rebuild or not (preview / "render_front.png").exists() or not (preview / "render_face.png").exists():
        run_wsl(
            f"python tools/texture/render_texture_preview.py "
            f"{bake}/deca_hybrid_body_deca_face.glb {bake}/hybrid_preview",
            env="lhm",
            timeout=30 * 60,
        )
    else:
        print("hybrid preview already present:", preview)

    ensure_texture_overlay(
        SSP3D_PERSON,
        bake,
        f"{bake}/photo_overlays_viewuv",
        limit=7,
        rebuild=rebuild,
        timeout=60 * 60,
    )

    return {
        "bake": REPO / bake,
        "overlays": REPO / bake / "photo_overlays_viewuv",
        "deca_prefix": REPO / deca_prefix,
    }
