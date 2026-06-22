"""Build the singleton texture diagnostics notebook.

The notebook is intentionally evidence-first: it runs the singleton texture
experiments, displays the contact sheets, and asserts the metric thresholds that
separate projection/mapping from completion/fusion.
"""
from pathlib import Path
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "singleton_texture_diagnostics.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


cells = [
    md(
        """
        # Singleton Texture Diagnostics

        This notebook isolates the texture problem on single images before
        returning to multi-view fusion. The key distinction is:

        - **projective ceiling**: the source photo projected through the posed mesh.
        - **view UV atlas overlay**: one source view baked into canonical UV, then rendered back.
        - **observed global atlas overlay**: the saved canonical atlas rendered with alpha only
          where texels were actually observed.

        If the projective and observed atlas overlays are close but completed
        textures look grotesque, the failure is completion/repair/fusion rather
        than the basic model projection or UV map.
        """
    ),
    code(
        r"""
        from pathlib import Path
        import json
        import sys
        from IPython.display import Image, Markdown, display

        ROOT = Path.cwd().resolve()
        if ROOT.name == "notebooks":
            ROOT = ROOT.parent
        SSP3D_IMAGE = ROOT / "datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_000029.png"
        RUNS = {
            "SSP-3D bodybuilder observed 1024": ROOT / "runs/singleton_texture_ssp3d_bodybuilder_observed_1024",
            "SSP-3D bodybuilder observed 2048": ROOT / "runs/singleton_texture_ssp3d_bodybuilder_observed_2048",
            "SSP-3D bodybuilder repaired/completed failure 1024": ROOT / "runs/singleton_texture_ssp3d_bodybuilder_uvfix_1024",
        }
        print("source image:", SSP3D_IMAGE)
        assert SSP3D_IMAGE.exists(), SSP3D_IMAGE
        REBUILD = False
        sys.path.insert(0, str(ROOT))
        from tools.notebooks.texture_notebook_runs import ensure_singleton_texture_runs

        RUNS = ensure_singleton_texture_runs(rebuild=REBUILD)

        for label, run in RUNS.items():
            assert run.exists(), f"missing run: {label} -> {run}"
            assert (run / "uv_report.json").exists(), f"missing uv_report: {label}"
            assert (run / "photo_overlays/TEXTURED_MESH_OVERLAY_CONTACT_SHEET.png").exists(), label
        """
    ),
    md("## Metrics"),
    code(
        r"""
        rows = []
        for label, run in RUNS.items():
            overlay_raw = json.loads((run / "photo_overlays/overlay_report.json").read_text())
            if isinstance(overlay_raw, dict):
                atlas_mode = overlay_raw.get("atlas_mode", "unknown")
                overlay_rows = overlay_raw["rows"]
            else:
                atlas_mode = "legacy_rgb_full"
                overlay_rows = overlay_raw
            uv = json.loads((run / "uv_report.json").read_text())
            ok = [r for r in overlay_rows if r.get("ok")]
            assert ok, label
            r = ok[0]
            rows.append({
                "run": label,
                "atlas": uv["atlas"],
                "mode": atlas_mode,
                "coverage_texels": uv["coverage_texels"],
                "coherent_face": uv.get("coherent_face_enabled", "legacy"),
                "repair": uv.get("face_repair_mode", "legacy"),
                "fill": uv.get("unmapped_fill_mode", "legacy"),
                "projective": r["mean_projective_absdiff"],
                "view_uv": r["mean_view_conditioned_absdiff"],
                "global": r["mean_overlay_absdiff"],
            })

        table = "| run | atlas | mode | texel coverage | coherent face | repair | fill | projective diff | view-UV diff | global/observed diff |\n"
        table += "|---|---:|---|---:|---|---|---|---:|---:|---:|\n"
        for r in rows:
            table += (
                f"| {r['run']} | {r['atlas']} | `{r['mode']}` | {r['coverage_texels']:.3f} | "
                f"{r['coherent_face']} | {r['repair']} | {r['fill']} | "
                f"{r['projective']:.2f} | {r['view_uv']:.2f} | {r['global']:.2f} |\n"
            )
        display(Markdown(table))

        observed = [r for r in rows if r["mode"] == "observed_rgba"]
        assert max(r["projective"] for r in observed) < 1.0
        assert max(r["global"] for r in observed) < 7.0
        """
    ),
    md("## Fixed Singleton: SSP-3D Bodybuilder 1024"),
    code(
        r"""
        display(Image(filename=str(RUNS["SSP-3D bodybuilder observed 1024"] / "photo_overlays/TEXTURED_MESH_OVERLAY_CONTACT_SHEET.png"), width=1200))
        display(Image(filename=str(RUNS["SSP-3D bodybuilder observed 1024"] / "atlas.png"), width=700))
        """
    ),
    md("## Resolution Check: 1024 vs 2048"),
    code(
        r"""
        display(Image(filename=str(RUNS["SSP-3D bodybuilder observed 2048"] / "photo_overlays/TEXTURED_MESH_OVERLAY_CONTACT_SHEET.png"), width=1200))
        display(Image(filename=str(RUNS["SSP-3D bodybuilder observed 2048"] / "atlas.png"), width=700))
        """
    ),
    md("## Failure Case: Completion/Repair Rendered As Truth"),
    code(
        r"""
        display(Image(filename=str(RUNS["SSP-3D bodybuilder repaired/completed failure 1024"] / "photo_overlays/TEXTURED_MESH_OVERLAY_CONTACT_SHEET.png"), width=1200))
        """
    ),
    md(
        """
        ## Read

        Singleton mapping is basically working. Raising atlas resolution from
        1024 to 2048 improves the raw observed overlay, but the big visual
        failure is not resolution. The grotesque outputs appear when fabricated
        completion, symmetric fill, and the coherent-face warp are rendered as
        if they were real observed texels.

        For the next multi-view pass, the production rule should be: preserve
        observed texels separately, never use completion as acceptance truth,
        keep coherent-face disabled unless the face warp passes its own overlay
        metric, and make fusion operate on per-texel confidence rather than
        filling all unmapped regions into a photo overlay.
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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
