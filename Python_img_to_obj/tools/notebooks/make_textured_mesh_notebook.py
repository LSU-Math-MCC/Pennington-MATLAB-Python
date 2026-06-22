"""Build the textured mesh pipeline notebook as an executable real-pipeline recipe."""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


REPO = Path(__file__).resolve().parents[2]
NB = REPO / "notebooks" / "textured_mesh_pipeline_demo.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


cells = [
    md(
        """
        # Canonical Mesh Textures + Photo Overlay Acceptance

        This notebook uses the real texture pipeline, not a 2D texture-card demo:

        1. bake subject photos into canonical SMPL-X UV coordinates,
        2. export an actual textured `apose_textured_uv.glb`,
        3. optionally graft a DECA/FLAME face after cutting away the old SMPL-X head,
        4. render the textured posed mesh back over each original photo in the same style as the HMR overlay notebook.

        The acceptance sheet includes a **projective ceiling** column. If that column is invisible, pose/camera projection is good enough; remaining visible error in the canonical atlas column is texture fusion/atlas quality.
        """
    ),
    code(
        r"""
        from pathlib import Path
        import json, os, sys, time
        from IPython.display import Image, display, Markdown
        from PIL import Image as PILImage

        REPO = Path.cwd().resolve()
        while REPO != REPO.parent and not (REPO / "pyproject.toml").exists():
            REPO = REPO.parent
        SSP3D_IMAGES = REPO / "datasets" / "SSP-3D" / "ssp_3d" / "images"
        SSP3D_PERSON = "bodybuilding_vid_009_clip_000_person_001_frame_*.png"
        SUBJECT = str(SSP3D_IMAGES / SSP3D_PERSON)
        DECA_IMAGE = SSP3D_IMAGES / "bodybuilding_vid_009_clip_000_person_001_frame_000029.png"
        BAKE = REPO / "runs" / "mesh_texture_notebook_ssp3d_bodybuilder_hq"
        OVERLAYS = BAKE / "photo_overlays_viewuv"
        print("repo:", REPO)
        print("subject:", SUBJECT)
        print("deca image:", DECA_IMAGE)
        print("bake:", BAKE)
        print("overlay:", OVERLAYS)
        REBUILD = False
        sys.path.insert(0, str(REPO))
        from tools.notebooks.texture_notebook_runs import ensure_multiview_texture_run
        """
    ),
    md(
        """
        ## Commands Actually Run

        These are the commands used for the current artifacts. They are intentionally kept explicit because the expensive cells should be auditable and reproducible from the notebook.
        """
    ),
    code(
        r"""
        outputs = ensure_multiview_texture_run(rebuild=REBUILD)
        print(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2))
        """
    ),
    md("## Artifact Check"),
    code(
        r"""
        required = [
            BAKE / "apose_textured_uv.glb",
            BAKE / "atlas.png",
            BAKE / "atlas_normal.png",
            BAKE / "uv_report.json",
            BAKE / "deca_hybrid_body_deca_face.glb",
            BAKE / "hybrid_preview" / "render_face.png",
            BAKE / "hybrid_preview" / "render_front.png",
            OVERLAYS / "TEXTURED_MESH_OVERLAY_CONTACT_SHEET.png",
            OVERLAYS / "overlay_report.json",
        ]
        rows = []
        for p in required:
            rows.append({
                "file": str(p.relative_to(REPO)),
                "exists": p.exists(),
                "size_mb": round(p.stat().st_size / 1024 / 1024, 3) if p.exists() else None,
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)) if p.exists() else None,
            })
        print(json.dumps(rows, indent=2))
        assert all(r["exists"] for r in rows)
        """
    ),
    md("## UV Bake Report"),
    code(
        r"""
        report = json.loads((BAKE / "uv_report.json").read_text())
        display(Markdown(
            "| metric | value |\n|---|---:|\n" +
            "\n".join(f"| `{k}` | `{v}` |" for k, v in report.items())
        ))
        assert report["atlas"] == 2048
        assert report["coverage_texels"] > 0.45
        """
    ),
    md("## Canonical UV Atlas"),
    code(
        r"""
        display(Image(filename=str(BAKE / "atlas.png"), width=700))
        """
    ),
    md("## Face Mesh Quality"),
    code(
        r"""
        display(Markdown("**DECA/FLAME photo-colored face after cutting the old SMPL-X head away:**"))
        display(Image(filename=str(BAKE / "hybrid_preview" / "render_face.png"), width=520))
        """
    ),
    md("## Full Textured Mesh"),
    code(
        r"""
        display(Image(filename=str(BAKE / "hybrid_preview" / "render_front.png"), width=420))
        print("GLB:", BAKE / "deca_hybrid_body_deca_face.glb")
        """
    ),
    md(
        """
        ## Original Photo Overlay Acceptance

        Columns:

        - **original**: source photo.
        - **posed textured mesh**: the canonical atlas rendered on the posed mesh only.
        - **projective ceiling / photo-perfect overlay**: source photo projected through the same posed mesh. This is the "you cannot tell a mesh is overlaid" target.
        - **view UV atlas overlay**: the same source photo first baked into canonical UV coordinates for that view, then rendered back.
        - **global atlas overlay**: the fused canonical atlas over the original photo; this is the failure mode to debug.
        """
    ),
    code(
        r"""
        display(Image(filename=str(OVERLAYS / "TEXTURED_MESH_OVERLAY_CONTACT_SHEET.png"), width=1100))
        """
    ),
    md("## Overlay Metrics"),
    code(
        r"""
        overlay_report_raw = json.loads((OVERLAYS / "overlay_report.json").read_text())
        if isinstance(overlay_report_raw, dict):
            overlay_report = overlay_report_raw["rows"]
            atlas_mode = overlay_report_raw.get("atlas_mode", "unknown")
        else:
            overlay_report = overlay_report_raw
            atlas_mode = "legacy_rgb_full"
        ok = [r for r in overlay_report if r.get("ok")]
        display(Markdown(f"Overlay atlas mode: `{atlas_mode}`"))
        display(Markdown(
            "| image | coverage | photo-perfect diff | view-UV diff | global-atlas diff |\n|---|---:|---:|---:|---:|\n" +
            "\n".join(
                f"| `{Path(r['image']).name}` | {r['coverage_px']:.3f} | "
                f"{r['mean_projective_absdiff']:.2f} | "
                f"{r['mean_view_conditioned_absdiff']:.2f} | "
                f"{r['mean_overlay_absdiff']:.2f} |"
                for r in ok
            )
        ))
        assert len(ok) == 7
        assert max(r["mean_projective_absdiff"] for r in ok) < 1.1
        """
    ),
    md(
        """
        ## Current Read

        The projection path is strong: the projective ceiling is visually near-invisible and numerically below 1 px-scale RGB diff.

        The per-view UV atlas confirms the canonical coordinate path works, but still shows UV discretization/visibility artifacts. The global atlas remains the bad output: it mixes incompatible views, lighting, crossed-limb occlusions, garment boundaries, and watermark pixels into one static surface. The notebook now exposes that failure directly instead of hiding it in a standalone render.
        """
    ),
]


def main():
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    NB.parent.mkdir(parents=True, exist_ok=True)
    NB.write_text(nbf.writes(nb), encoding="utf-8")
    print(NB)


if __name__ == "__main__":
    main()
