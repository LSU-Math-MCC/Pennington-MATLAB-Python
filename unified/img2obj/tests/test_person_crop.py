import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "render"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from person_crop import largest_box_excluding_others  # noqa: E402


def test_largest_box_excluding_others_uses_image_when_subject_is_alone():
    boxes = np.array([[40, 40, 80, 120]], dtype=float)

    assert largest_box_excluding_others(boxes, 200, 160, 0) == (0, 0, 200, 160)


def test_largest_box_excluding_others_cuts_to_nearest_subject_edges():
    boxes = np.array(
        [
            [40, 40, 80, 120],
            [0, 20, 25, 130],
            [130, 10, 180, 150],
            [30, 0, 95, 22],
            [35, 140, 90, 160],
        ],
        dtype=float,
    )

    assert largest_box_excluding_others(boxes, 200, 160, 0) == (25, 22, 130, 140)


def test_largest_box_excluding_others_ignores_unavoidable_target_overlap():
    boxes = np.array(
        [
            [40, 40, 80, 120],
            [70, 80, 110, 130],
            [130, 10, 180, 150],
        ],
        dtype=float,
    )

    assert largest_box_excluding_others(boxes, 200, 160, 0) == (0, 0, 130, 160)
