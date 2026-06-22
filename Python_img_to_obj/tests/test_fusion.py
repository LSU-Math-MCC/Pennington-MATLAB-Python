import numpy as np
from pipeline.geometry.fusion import fuse_clouds
from pipeline.types import SplatCloud


def _cloud(centers, conf=1.0, region=0, color=0.5):
    n = centers.shape[0]
    return SplatCloud(centers=centers, scales=np.full((n, 3), 0.01),
                      rotations=np.tile([1.0, 0, 0, 0], (n, 1)),
                      opacities=np.ones(n), colors=np.full((n, 3), color),
                      extras={"confidence": np.full(n, conf),
                              "region": np.full(n, region, int)})


def test_voxel_merge_reduces_count():
    rng = np.random.default_rng(0)
    base = np.zeros((50, 3))
    a = _cloud(base + rng.normal(scale=0.001, size=(50, 3)))
    b = _cloud(base + rng.normal(scale=0.001, size=(50, 3)))
    fused, rep = fuse_clouds([a, b], body_voxel=0.05)
    assert rep["output"] < rep["input"]
    assert rep["output"] >= 1


def test_weighted_center_near_mean():
    a = _cloud(np.array([[0.0, 0, 0]]), conf=1.0)
    b = _cloud(np.array([[0.02, 0, 0]]), conf=1.0)
    fused, rep = fuse_clouds([a, b], body_voxel=0.1)
    assert rep["output"] == 1
    assert np.allclose(fused.centers[0], [0.01, 0, 0], atol=1e-6)


def test_low_confidence_dropped():
    a = _cloud(np.array([[0.0, 0, 0]]), conf=0.9)
    b = _cloud(np.array([[1.0, 0, 0]]), conf=0.05)
    fused, rep = fuse_clouds([a, b], body_voxel=0.05, conf_thresh=0.2)
    assert rep["output"] == 1


def test_report_has_counts():
    a = _cloud(np.zeros((3, 3)))
    fused, rep = fuse_clouds([a])
    assert "input" in rep and "output" in rep


def test_face_uses_smaller_voxel():
    # two face splats 0.008 apart: merge at body voxel 0.02 but separate at face 0.006
    face = _cloud(np.array([[0.0, 0, 0], [0.008, 0, 0]]), region=1)
    fused, rep = fuse_clouds([face], body_voxel=0.02, face_voxel=0.006)
    assert rep["face_out"] == 2
