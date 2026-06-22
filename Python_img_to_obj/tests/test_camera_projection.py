import numpy as np
from pipeline.geometry import camera as camlib


def test_project_center_pixel():
    cam = camlib.default_camera(100, 100)
    # a point straight ahead on the optical axis projects to principal point
    X = np.array([[0.0, 0.0, 5.0]])
    uv, valid = camlib.project(cam, X)
    assert valid[0]
    assert np.allclose(uv[0], [50, 50], atol=1e-6)


def test_backproject_roundtrip():
    cam = camlib.default_camera(120, 90)
    X = np.array([[0.3, -0.2, 4.0]])
    uv, valid = camlib.project(cam, X)
    assert valid[0]
    depth = X[0, 2]
    Xr = camlib.backproject(cam, uv, np.array([depth]))
    assert np.allclose(Xr[0], X[0], atol=1e-6)


def test_reject_behind_camera():
    cam = camlib.default_camera(100, 100)
    X = np.array([[0.0, 0.0, -2.0]])
    uv, valid = camlib.project(cam, X)
    assert not valid[0]


def test_reject_out_of_bounds():
    cam = camlib.default_camera(100, 100)
    X = np.array([[100.0, 0.0, 1.0]])  # huge x -> off image
    uv, valid = camlib.project(cam, X)
    assert not valid[0]
