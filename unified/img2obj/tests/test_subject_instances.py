import numpy as np
from pipeline.geometry import camera as camlib
from pipeline.geometry.mask_depth_select import select_masked_depth
from pipeline.geometry.splat_assign import assign_splats_to_instance, resolve_ambiguous
from pipeline.geometry.subject_instances import associate_tracks, mask_iou
from pipeline.types import SplatCloud, SubjectInstance


def _two_person_scene():
    H, W = 40, 60
    m0 = np.zeros((H, W), bool); m0[10:30, 8:24] = True     # left person
    m1 = np.zeros((H, W), bool); m1[10:30, 36:52] = True    # right person
    depth = np.full((H, W), np.nan)
    depth[m0] = 3.0
    depth[m1] = 3.0
    cam = camlib.default_camera(W, H)
    return m0, m1, depth, cam


def _splat_at(cam, u, v, d):
    return camlib.backproject_to_world(cam, np.array([[u, v]]), np.array([d]))


def test_two_instances_independent_selection():
    m0, m1, depth, cam = _two_person_scene()
    s0 = select_masked_depth(m0, depth, cam)
    s1 = select_masked_depth(m1, depth, cam)
    assert len(s0) == int(m0.sum())
    assert len(s1) == int(m1.sum())
    # disjoint
    assert mask_iou(m0, m1) == 0.0


def test_nonoverlap_splats_assigned_correctly():
    m0, m1, depth, cam = _two_person_scene()
    s0 = select_masked_depth(m0, depth, cam)
    s1 = select_masked_depth(m1, depth, cam)
    c0 = _splat_at(cam, 16, 20, 3.0)   # inside person 0
    c1 = _splat_at(cam, 44, 20, 3.0)   # inside person 1
    splats = SplatCloud(np.concatenate([c0, c1]), np.full((2, 3), .01),
                        np.tile([1., 0, 0, 0], (2, 1)), np.ones(2), np.full((2, 3), .5))
    _, sc0 = assign_splats_to_instance(splats, cam, m0, depth, s0)
    _, sc1 = assign_splats_to_instance(splats, cam, m1, depth, s1)
    assign, amb = resolve_ambiguous([sc0, sc1])
    assert assign[0] == 0
    assert assign[1] == 1
    assert not amb[0] and not amb[1]


def test_overlap_splat_marked_ambiguous():
    # construct scores directly: a splat both people claim similarly
    sc0 = np.array([0.6])
    sc1 = np.array([0.62])
    assign, amb = resolve_ambiguous([sc0, sc1], margin=0.15)
    assert amb[0]


def test_tracks_kept_separate_when_uncertain():
    # two images, each with one clearly different-position person -> stays >=1 track,
    # bystander not merged into target
    insts_img0 = [SubjectInstance("a_p0", "a.jpg", np.zeros((1, 1), bool), (0, 0, 10, 30),
                                  association_confidence=0.8)]
    insts_img1 = [SubjectInstance("b_p0", "b.jpg", np.zeros((1, 1), bool), (2, 0, 12, 32),
                                  association_confidence=0.8),
                  SubjectInstance("b_p1", "b.jpg", np.zeros((1, 1), bool), (200, 0, 210, 20),
                                  association_confidence=0.4)]
    tracks = associate_tracks([insts_img0, insts_img1], same_subject=True)
    # the recurring/primary subject track should not include the far bystander
    target = tracks[0]
    ids = [i.instance_id for i in target.instances]
    assert "b_p1" not in ids
