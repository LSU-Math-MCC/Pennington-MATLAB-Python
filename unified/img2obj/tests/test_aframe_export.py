import numpy as np
from pipeline.export import aframe
from pipeline.types import SplatCloud


def _dummy_splats(n=20):
    rng = np.random.default_rng(0)
    return SplatCloud(centers=rng.normal(size=(n, 3)), scales=np.full((n, 3), 0.01),
                      rotations=np.tile([1.0, 0, 0, 0], (n, 1)), opacities=np.ones(n),
                      colors=rng.uniform(0, 1, (n, 3)))


def test_index_html_created(tmp_path):
    p = aframe.write_viewer(tmp_path, splats=_dummy_splats(), label="t")
    assert p.exists()


def test_html_contains_a_scene(tmp_path):
    aframe.write_viewer(tmp_path, splats=_dummy_splats(), label="t")
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "<a-scene" in html


def test_viewer_from_dummy_splats_has_points(tmp_path):
    aframe.write_viewer(tmp_path, splats=_dummy_splats(7), label="t")
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "SCENE_DATA" in html
    assert "point-cloud" in html


def test_aggregate_links_exist(tmp_path):
    entries = [{"label": "a", "href": "images/a/index.html", "status": "ok"}]
    p = aframe.write_aggregate(tmp_path, entries)
    assert p.exists()
    assert "images/a/index.html" in p.read_text(encoding="utf-8")
