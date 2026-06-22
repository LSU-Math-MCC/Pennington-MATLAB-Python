"""Export meshmap's CLIP-corrected A-pose mesh + EXACT SMPL-X-semantic 17 markers for the
Pennington team. Their heuristic collapses arm markers (shoulder==wrist) in A-pose; ours are
exact SMPL-X joints/landmarks so wrists/shoulders/ankles never collapse.

Writes runs/penn_integration/<s>_apose.obj and <s>_markers_smplx.json (same coord frame).
Run (WSL lhm): python tools/workflows/export_for_penn.py [s1 ...]
"""

# --- tool-path bootstrap (capability-tree reorg): make sibling tool modules importable ---
import os as _os, sys as _sys
_repo = _os.path.dirname(_os.path.abspath(__file__))
while _repo != _os.path.dirname(_repo) and not _os.path.exists(_os.path.join(_repo, "pyproject.toml")):
    _repo = _os.path.dirname(_repo)
for _sub in ("smplx", "texture", "benchmark", "geometry", "anthro", "render"):
    _p = _os.path.join(_repo, "tools", _sub)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import os
import sys
import json

import numpy as np

REPO = _repo
OUT = REPO + "/runs/penn_integration"
sys.path.insert(0, REPO + "/tools")


def smplx_markers(v, j, named):
    """17 Pennington markers from SMPL-X semantics. j: (22,3) joints (Y up, Z forward)."""
    P = {n: np.array(p) for n, p in named.items()}
    Yup = 1; Zf = 2; X = 0
    head = P["head"]
    # nose tip: most-forward vertex within the head band
    band = v[v[:, Yup] > head[Yup] - 0.06]
    nose = band[np.argmax(band[:, Zf])] if len(band) else head
    # crotch: lowest central vertex just below pelvis
    pelvisY = P["pelvis"][Yup]
    cen = v[(np.abs(v[:, X]) < 0.035) & (v[:, Yup] < pelvisY) & (v[:, Yup] > pelvisY - 0.28)]
    crotch = cen[np.argmin(cen[:, Yup])] if len(cen) else P["pelvis"]
    # armpits: highest torso-side vertex just below & medial to each shoulder
    def armpit(sh, side):
        sel = v[(np.sign(v[:, X]) == np.sign(sh[X])) & (v[:, Yup] < sh[Yup]) &
                (v[:, Yup] > sh[Yup] - 0.18) & (np.abs(v[:, X]) < abs(sh[X]) * 0.85)]
        return sel[np.argmax(sel[:, Yup])] if len(sel) else sh
    lsh, rsh = P["left_shoulder"], P["right_shoulder"]
    # hips: LATERAL pelvis SURFACE points (widest) at hip-joint height — not internal joints,
    # and NOT the hands (which hang near hip height in A-pose), so cap |X| to ~2.6x the joint
    # offset to stay on the pelvis surface and exclude the arm/wrist vertices.
    def hip_surface(hipj):
        cap = abs(hipj[X]) * 2.6 + 0.02
        sel = v[(np.sign(v[:, X]) == np.sign(hipj[X])) & (np.abs(v[:, Yup] - hipj[Yup]) < 0.05) &
                (np.abs(v[:, X]) < cap)]
        return sel[np.argmax(np.abs(sel[:, X]))] if len(sel) else hipj
    lhip, rhip = hip_surface(P["left_hip"]), hip_surface(P["right_hip"])
    # shoulder-top (arm highest point): vertex above shoulder joint on that side
    def shoulder_top(sh):
        sel = v[(np.sign(v[:, X]) == np.sign(sh[X])) & (np.abs(v[:, X] - sh[X]) < 0.08) &
                (v[:, Yup] > sh[Yup] - 0.02)]
        return sel[np.argmax(sel[:, Yup])] if len(sel) else sh
    return [
        ("head", "tip of nose", nose),
        ("trunk", "crotch", crotch),
        ("trunk", "left armpit", armpit(lsh, "left")),
        ("trunk", "right armpit", armpit(rsh, "right")),
        ("trunk", "left hip", lhip),
        ("trunk", "right hip", rhip),
        ("trunk", "collar", (P["left_collar"] + P["right_collar"]) / 2),
        ("left arm", "highest point of arm", shoulder_top(lsh)),
        ("left arm", "shoulder", lsh),
        ("left arm", "wrist", P["left_wrist"]),
        ("right arm", "highest point of arm", shoulder_top(rsh)),
        ("right arm", "shoulder", rsh),
        ("right arm", "wrist", P["right_wrist"]),
        ("left leg", "foot", P["left_foot"]),
        ("left leg", "ankle", P["left_ankle"]),
        ("right leg", "foot", P["right_foot"]),
        ("right leg", "ankle", P["right_ankle"]),
    ]


def main():
    import trimesh
    import lhm_anthropometry as A
    subjects = sys.argv[1:] or ["s1", "s2", "s3", "s4", "s5"]
    os.makedirs(OUT, exist_ok=True)
    for s in subjects:
        bp = f"{REPO}/runs/fit_{s}/fused_betas.npy"
        if not os.path.exists(bp):
            print("skip", s); continue
        betas = np.load(bp)[:10]
        v, j, faces, named = A.smplx_apose(betas, gender="female")
        mesh = trimesh.Trimesh(v, faces, process=False)
        mesh.export(f"{OUT}/{s}_apose.obj")
        marks = smplx_markers(v, j, named)
        out = {"coord": "Yup_Zfwd_meters", "markers": [
            {"part": p, "name": n, "xyz": [float(x) for x in pt]} for p, n, pt in marks]}
        json.dump(out, open(f"{OUT}/{s}_markers_smplx.json", "w"), indent=2)
        print(f"exported {s}: {OUT}/{s}_apose.obj + 17 SMPL-X markers")
    print("EXPORT_FOR_PENN_DONE")


if __name__ == "__main__":
    main()
