import json
from pathlib import Path

from pipeline.run import main


def test_single_smoke(tmp_path):
    out = tmp_path / "smoke"
    fixture = Path(__file__).parent / "fixtures" / "person_stub.png"
    rc = main(["single", "--image", str(fixture), "--out", str(out),
               "--backend", "dummy", "--quick"])
    assert rc == 0
    assert (out / "manifest.json").exists()
    assert (out / "index.html").exists()
    assert (out / "canonical_splats.ply").exists()
    debug = out / "debug"
    for name in ["input.png", "mask.png", "pose_overlay.png", "depth_preview.png",
                 "selected_depth_overlay.png", "splat_projection_overlay.png",
                 "canonical_preview.png"]:
        assert (debug / name).exists(), f"missing debug/{name}"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "success"
