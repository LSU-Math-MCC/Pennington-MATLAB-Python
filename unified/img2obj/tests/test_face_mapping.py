import numpy as np
from pipeline.geometry import camera as camlib
from pipeline.geometry import face_mapping as fm
from pipeline.types import Face2DResult, SplatCloud


def _scene():
    H = W = 20
    person = np.zeros((H, W), bool)
    person[2:18, 5:15] = True            # body region
    depth = np.where(person, 3.0, np.nan)
    depth[2:8, 7:13] = 2.0               # face area closer
    cam = camlib.default_camera(W, H)
    # face landmarks (eyes, nose, mouth, chin) in upper region
    def L(x, y):
        return (float(x), float(y), 0.9)
    landmarks = {
        "left_eye": L(8, 4), "right_eye": L(12, 4),
        "left_eye_outer": L(7.5, 4), "right_eye_outer": L(12.5, 4),
        "nose_tip": L(10, 5), "mouth_left": L(9, 6), "mouth_right": L(11, 6),
        "chin": L(10, 7),
    }
    face = Face2DResult(face_mask=None, landmarks=landmarks,
                        bbox=(7, 3, 13, 8), confidence=0.9)
    return person, depth, cam, face


def test_face_region_intersects_person():
    person, depth, cam, face = _scene()
    region = fm.build_face_region(face, person, margin_px=1)
    assert region.shape == person.shape
    # region must be a subset of the person mask
    assert np.all(person[region])
    assert region.sum() > 0


def test_face_depth_from_face_region_only():
    person, depth, cam, face = _scene()
    region = fm.build_face_region(face, person, margin_px=0)
    s = fm.select_face_depth(region, depth, cam)
    # all selected pixels inside region
    for u, v in s.pixels:
        assert region[int(v), int(u)]


def test_3d_anchors_recovered():
    person, depth, cam, face = _scene()
    region = fm.build_face_region(face, person, margin_px=1)
    anchors = fm.lift_face_anchors(face.landmarks, region, depth, cam)
    assert "nose_tip" in anchors
    assert "left_eye" in anchors and "right_eye" in anchors


def test_face_frame_axes_stable():
    person, depth, cam, face = _scene()
    region = fm.build_face_region(face, person, margin_px=1)
    anchors = fm.lift_face_anchors(face.landmarks, region, depth, cam)
    T, scale, conf = fm.face_canonical_frame(anchors)
    assert T is not None
    R = T[:3, :3] / scale
    # rows orthonormal -> right handed
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)


def test_face_splits_use_smaller_voxel_and_conf_filter():
    from pipeline.config import Config
    cfg = Config()
    assert cfg.face_voxel < cfg.body_voxel
    # low-confidence face frames excluded: face_canonical_frame returns conf 0 when
    # landmarks insufficient
    T, scale, conf = fm.face_canonical_frame({})
    assert conf == 0.0
