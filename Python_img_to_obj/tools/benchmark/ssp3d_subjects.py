"""Shared SSP-3D subject selection so the teaser and the per-method runners agree on WHICH subjects
to evaluate. Diverse selection (farthest-point sampling in GT-shape space) keeps the most extreme
body type but also spreads across slim / average / tall / muscular, so the figure isn't all one
body type. Pure-numpy so it imports in any env (camerahmr / shapy / blade_env)."""
import numpy as np


def subject_groups(gt):
    """Group SSP-3D frame indices by identical GT betas -> list of member-index lists."""
    keys = [tuple(np.round(gt[i, :10], 3)) for i in range(len(gt))]
    g = {}
    for i, k in enumerate(keys):
        g.setdefault(k, []).append(i)
    return list(g.values())


def select_diverse(gt, n):
    """Farthest-point sampling over subject GT betas. Seeds with the most extreme subject (so the
    big-shape examples are kept), then repeatedly adds the subject farthest from those chosen."""
    subj = subject_groups(gt)
    B = np.array([gt[m[0], :10] for m in subj], np.float64)
    sel = [int(np.argmax(np.linalg.norm(B, axis=1)))]
    while len(sel) < min(n, len(subj)):
        d = np.min(np.linalg.norm(B[:, None, :] - B[None, sel, :], axis=2), axis=1)
        d[sel] = -1.0
        sel.append(int(np.argmax(d)))
    return [subj[i] for i in sel]


def _build_metric(gt, smpl_dir):
    """Dimensionless build metric per subject: GT-mesh volume / height^3 (small = skinny, large =
    stocky). Pose-invariant (neutral T-pose, betas only) and deterministic across envs."""
    import torch, smplx, trimesh
    m = smplx.SMPL(smpl_dir, gender="neutral"); faces = m.faces
    subj = subject_groups(gt)
    vals = []
    for mem in subj:
        with torch.no_grad():
            v = m(betas=torch.tensor(gt[mem[0], :10]).float().unsqueeze(0)).vertices[0].numpy()
        h = v[:, 1].max() - v[:, 1].min()
        vals.append(abs(trimesh.Trimesh(v, faces, process=False).volume) / (h ** 3))
    return subj, np.array(vals)


def select_skinny(gt, n, smpl_dir):
    """The n skinniest subjects (smallest volume/height^3)."""
    subj, vals = _build_metric(gt, smpl_dir)
    return [subj[i] for i in np.argsort(vals)[:n]]


def select_rep_frames(gt, n, skinny=False, smpl_dir=None):
    """One representative frame index per selected subject (diverse by default, or skinniest)."""
    subj = select_skinny(gt, n, smpl_dir) if skinny else select_diverse(gt, n)
    return [m[0] for m in subj]
