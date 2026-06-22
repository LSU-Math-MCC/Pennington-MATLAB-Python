"""Multi-view SMPL-X shape fusion -> anthropometric measurements.

Runs INSIDE the WSL LHM conda env (has Multi-HMR pose estimator + smplx + model files).

Pipeline (the metric/measurement path, NOT gaussian generation):
  1. For each image of a subject, regress SMPL-X betas with LHM's Multi-HMR estimator.
  2. Robustly fuse betas across views (weighted by full-body visibility + inlier trim).
  3. Forward neutral SMPL-X in canonical pose -> metric mesh (meters).
  4. Measure: stature, shoulder breadth, arm span, chest/waist/hip girth, inseam, limb
     lengths -- via trimesh planar sections + joint distances.
  5. Report measurements with uncertainty propagated from cross-view beta spread.

Usage (in WSL):
  cd ~/LHM && python <this> --subject "/mnt/c/.../datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_*.png" --out /mnt/c/.../runs/anthro_ssp3d_bodybuilder

Absolute scale from uncalibrated 2D is pinned by Multi-HMR's metric prior; pass
--stature-cm to rescale the whole body to a known height (best practice for legacy media).
"""
import argparse
import json
import os
import sys
import glob

import numpy as np

LHM_ROOT = os.path.expanduser("~/LHM")
if os.path.isdir(LHM_ROOT) and LHM_ROOT not in sys.path:
    sys.path.insert(0, LHM_ROOT)
for _p in (os.path.join(LHM_ROOT, "engine", "pose_estimation"),
           os.path.join(LHM_ROOT, "engine", "pose_estimation", "blocks")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

HUMAN_MODELS = os.path.join(LHM_ROOT, "pretrained_models", "human_model_files")


_EST = None
_CLIP = None


def _estimator():
    global _EST
    if _EST is None:
        from engine.pose_estimation.pose_estimator import PoseEstimator
        _EST = PoseEstimator(HUMAN_MODELS, device="cuda")
    return _EST


def regress_betas(images):
    """Return list of dicts {image, beta, is_full_body} using LHM Multi-HMR."""
    est = _estimator()
    out = []
    for img in images:
        try:
            r = est(img)
            beta = getattr(r, "beta", None)
            if beta is None:
                print(f"  [skip] {os.path.basename(img)}: {getattr(r,'msg','no beta')}")
                continue
            out.append({"image": img, "beta": np.asarray(beta).reshape(-1),
                        "is_full_body": bool(getattr(r, "is_full_body", True))})
            print(f"  [ok]   {os.path.basename(img)} full_body={out[-1]['is_full_body']} "
                  f"beta0={out[-1]['beta'][0]:.3f}")
        except Exception as e:
            print(f"  [err]  {os.path.basename(img)}: {repr(e)[:120]}")
    return out


def fuse_betas(per_view):
    """Robust fused betas: weight full-body higher, trim per-dim outliers (MAD)."""
    if not per_view:
        return None, None, 0
    B = np.stack([d["beta"] for d in per_view], 0)          # V x num_betas
    w = np.array([1.0 if d["is_full_body"] else 0.4 for d in per_view])
    # per-dimension robust mean with MAD outlier trimming
    med = np.median(B, 0)
    mad = np.median(np.abs(B - med), 0) + 1e-6
    inlier = np.abs(B - med) <= 3.0 * mad                   # V x D bool
    fused = np.zeros(B.shape[1]); spread = np.zeros(B.shape[1])
    for d in range(B.shape[1]):
        m = inlier[:, d]
        wd = w[m]
        fused[d] = np.average(B[m, d], weights=wd) if wd.sum() > 0 else med[d]
        spread[d] = np.std(B[m, d]) if m.sum() > 1 else 0.0
    return fused, spread, B.shape[0]


def estimate_gender(images, thresh=0.62):
    """Algorithmic gender allocation from the face/appearance via CLIP zero-shot,
    fused across views by summing per-view log-posteriors (independent-view MLE).

    Returns (gender in {male,female,neutral}, dict with fused probs + per-view debug).
    Neutral = abstain when fused confidence < thresh (don't force a wrong SMPL-X model).
    """
    try:
        import torch
        from PIL import Image
        global _CLIP
        if _CLIP is None:
            from transformers import CLIPProcessor, CLIPModel
            _CLIP = (CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
                     CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"))
        model, proc = _CLIP
        prompts = ["a photo of a man", "a photo of a woman"]
        logsum = np.zeros(2)
        per_view = []
        for img in images:
            im = Image.open(img).convert("RGB")
            inp = proc(text=prompts, images=im, return_tensors="pt", padding=True)
            with torch.no_grad():
                p = model(**inp).logits_per_image.softmax(-1)[0].cpu().numpy()
            logsum += np.log(p + 1e-9)
            per_view.append({"image": os.path.basename(img),
                             "p_male": float(p[0]), "p_female": float(p[1])})
        post = np.exp(logsum - logsum.max()); post /= post.sum()
        idx = int(post.argmax()); conf = float(post[idx])
        gender = ("male", "female")[idx] if conf >= thresh else "neutral"
        return gender, {"fused_p_male": float(post[0]), "fused_p_female": float(post[1]),
                        "confidence": conf, "decision": gender, "per_view": per_view}
    except Exception as e:
        return "neutral", {"error": repr(e)[:160], "decision": "neutral"}


# SMPL-X body-joint names (joints 0..21); body_pose covers joints 1..21 (idx = j-1).
SMPLX_JOINTS = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
                "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
                "neck", "left_collar", "right_collar", "head", "left_shoulder",
                "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]


def apose_body_pose(arm_deg=45.0, forearm_deg=0.0):
    """Construct an SMPL-X body_pose (21x3 axis-angle) for a canonical A-pose:
    arms lowered from the T-pose rest by `arm_deg` about the forward (z) axis."""
    bp = np.zeros((21, 3), np.float32)
    a = np.deg2rad(arm_deg)
    # body_pose index = joint_index - 1 ; collars 13/14 -> idx 12/13, shoulders 16/17 -> 15/16.
    # SMPL-X rest arms are near-horizontal; ADDUCT (lower) them to A-pose. Sign chosen so
    # arms go DOWN (verified in render). Split rotation across collar+shoulder for a natural A.
    bp[12] = [0, 0, -a * 0.35]   # left collar
    bp[13] = [0, 0, +a * 0.35]   # right collar
    bp[15] = [0, 0, -a]          # left shoulder: lower arm
    bp[16] = [0, 0, +a]          # right shoulder: lower arm (mirror)
    return bp.reshape(-1)


def smplx_apose(betas, gender="neutral", arm_deg=45.0, expression=None, jaw_pose=None):
    """Forward SMPL-X in the canonical A-pose. Returns (verts, joints, faces, names),
    pelvis-centered with +Y up. Joints are the metric A-pose joint placements.
    expression(10)/jaw_pose(3): optional FLAME face params (from the view textured onto the
    face) so the DISPLAYED head geometry matches the baked face texture (mouth/expression)."""
    import torch
    import smplx
    nb = len(betas)
    ne = 10 if expression is not None else 10
    model = smplx.create(HUMAN_MODELS, model_type="smplx", gender=gender,
                         num_betas=nb, num_expression_coeffs=ne, use_pca=False, flat_hand_mean=True)
    bp = torch.tensor(apose_body_pose(arm_deg), dtype=torch.float32).unsqueeze(0)
    kw = {}
    if expression is not None:
        kw["expression"] = torch.tensor(np.asarray(expression)[:ne], dtype=torch.float32).unsqueeze(0)
    if jaw_pose is not None:
        kw["jaw_pose"] = torch.tensor(np.asarray(jaw_pose).reshape(3), dtype=torch.float32).unsqueeze(0)
    out = model(betas=torch.tensor(betas, dtype=torch.float32).unsqueeze(0),
                body_pose=bp, **kw)
    v = out.vertices[0].detach().cpu().numpy()
    j = out.joints[0].detach().cpu().numpy()[:22]
    pelvis = j[0].copy()
    v = v - pelvis; j = j - pelvis
    named = {SMPLX_JOINTS[i]: [float(x) for x in j[i]] for i in range(22)}
    return v, j, model.faces, named


def smplx_measure_mesh(betas, gender="neutral", raise_deg=78.0):
    """Mesh with arms RAISED ~horizontal so torso cross-sections at any height contain NO
    arm vertices (torso shape is pose-invariant -> valid for abdomen contours/girths)."""
    import torch
    import smplx
    nb = len(betas)
    model = smplx.create(HUMAN_MODELS, model_type="smplx", gender=gender,
                         num_betas=nb, use_pca=False, flat_hand_mean=True)
    bp = np.zeros((21, 3), np.float32)
    a = np.deg2rad(raise_deg)
    bp[12] = [0, 0, +a * 0.3]; bp[13] = [0, 0, -a * 0.3]   # collars
    bp[15] = [0, 0, +a]; bp[16] = [0, 0, -a]               # shoulders: raise (opp. of A-pose)
    out = model(betas=torch.tensor(betas, dtype=torch.float32).unsqueeze(0),
                body_pose=torch.tensor(bp.reshape(-1), dtype=torch.float32).unsqueeze(0))
    return (out.vertices[0].detach().cpu().numpy(),
            out.joints[0].detach().cpu().numpy()[:22], model.faces)


def smplx_mesh(betas, num_betas=None, gender="neutral"):
    import torch
    import smplx
    nb = len(betas) if num_betas is None else num_betas
    model = smplx.create(HUMAN_MODELS, model_type="smplx", gender=gender,
                         num_betas=nb, use_pca=False, flat_hand_mean=True)
    out = model(betas=torch.tensor(betas[:nb], dtype=torch.float32).unsqueeze(0))
    v = out.vertices[0].detach().cpu().numpy()
    j = out.joints[0].detach().cpu().numpy()
    return v, j, model.faces


# SMPL-X body joint indices (standard)
J = {"pelvis": 0, "left_hip": 1, "right_hip": 2, "spine1": 3, "left_knee": 4,
     "right_knee": 5, "spine2": 6, "left_ankle": 7, "right_ankle": 8, "spine3": 9,
     "neck": 12, "left_shoulder": 16, "right_shoulder": 17, "left_elbow": 18,
     "right_elbow": 19, "left_wrist": 20, "right_wrist": 21, "head": 15}


def girth(mesh, y, axis=1):
    """Perimeter of the body cross-section at height y (largest single loop)."""
    try:
        sec = mesh.section(plane_origin=[0, y, 0], plane_normal=[0, 1, 0])
        if sec is None:
            return None
        planar, _ = sec.to_planar()
        if len(planar.polygons_full) == 0:
            return float(planar.length)
        # largest polygon perimeter (torso loop, not stray loops)
        return float(max(p.length for p in planar.polygons_full))
    except Exception:
        return None


def girth_profile(mesh, named, n=72):
    """Central section curve: girth (perimeter) as a function of height across the torso.
    Returns (ys, gs) with gs in metres (nan where the section is undefined)."""
    pelvis_y = float(named["pelvis"][1]); neck_y = float(named["neck"][1])
    # stay within the torso (pelvis..neck): going below pelvis catches the thighs/legs and
    # corrupts the section. Hip girth is the max within the lower torso band.
    ys = np.linspace(pelvis_y, neck_y, n)
    gs = np.array([(g if g is not None else np.nan) for g in
                   (girth(mesh, float(y)) for y in ys)], dtype=float)
    return ys, gs


def anatomical_levels(ys, gs, named):
    """Pick anthropometric heights from the section curve EXTREMA, not landmarks:
       waist = girth MINIMUM (natural waist), hip = girth MAX in pelvic band,
       chest/bust = girth MAX in upper torso, abdomen = girth MAX between hip and waist."""
    ys = np.asarray(ys, float); g = np.asarray(gs, float)
    pelvis_y = float(named["pelvis"][1]); spine2_y = float(named["spine2"][1])
    spine3_y = float(named["spine3"][1])

    def pick(band, fn):
        idx = np.where(band & np.isfinite(g))[0]
        if idx.size == 0:
            return None
        return float(ys[idx[fn(g[idx])]])

    waist_y = pick((ys >= pelvis_y) & (ys <= spine2_y), np.argmin)        # narrowest torso
    if waist_y is None:
        waist_y = 0.5 * (pelvis_y + spine2_y)
    hip_y = pick(ys <= waist_y, np.argmax)                                # widest pelvic
    # chest/bust = widest band BELOW the armpit (cap at spine3) so arms never enter the section
    chest_y = pick((ys >= waist_y) & (ys <= spine3_y), np.argmax)
    abdomen_y = pick((ys >= (hip_y if hip_y else pelvis_y)) & (ys <= waist_y), np.argmax)
    return {"waist_y": waist_y, "hip_y": hip_y or pelvis_y,
            "chest_y": chest_y or spine2_y, "abdomen_y": abdomen_y or waist_y}


def measure(verts, joints, faces):
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    stature = float(verts[:, 1].max() - verts[:, 1].min())

    def Y(name):
        return float(joints[J[name], 1])

    pelvis_y = Y("pelvis")
    ys, gs = girth_profile(mesh, {n: list(joints[J[n]]) for n in J})
    lv = anatomical_levels(ys, gs, {n: list(joints[J[n]]) for n in J})
    chest_y, waist_y, hip_y = lv["chest_y"], lv["waist_y"], lv["hip_y"]

    m = {
        "stature_m": stature,
        "shoulder_breadth_m": float(np.linalg.norm(joints[J["left_shoulder"]] - joints[J["right_shoulder"]])),
        "arm_span_m": float(np.linalg.norm(verts[verts[:, 0].argmin()] - verts[verts[:, 0].argmax()])),
        "upper_arm_len_m": float(np.linalg.norm(joints[J["left_shoulder"]] - joints[J["left_elbow"]])),
        "forearm_len_m": float(np.linalg.norm(joints[J["left_elbow"]] - joints[J["left_wrist"]])),
        "thigh_len_m": float(np.linalg.norm(joints[J["left_hip"]] - joints[J["left_knee"]])),
        "shank_len_m": float(np.linalg.norm(joints[J["left_knee"]] - joints[J["left_ankle"]])),
        "inseam_m": float(Y("pelvis") - verts[:, 1].min()),
        "chest_girth_m": girth(mesh, chest_y),
        "waist_girth_m": girth(mesh, waist_y),
        "hip_girth_m": girth(mesh, hip_y),
    }
    return m, mesh


def _torso_halfwidth(verts, y, default=0.25):
    """Half-width of the torso at height y, excluding hanging arms: find the first large
    gap in sorted |x| (torso cluster -> gap -> arm cluster)."""
    band = verts[np.abs(verts[:, 1] - y) <= 0.02]
    if band.shape[0] < 10:
        return default
    x = np.sort(np.abs(band[:, 0]))
    gaps = np.diff(x)
    big = np.where(gaps > 0.05)[0]                         # >5cm gap => arm separation
    return float(x[big[0]] + 0.01) if big.size else float(x.max())


def abdomen_contours(verts, faces, named, n_levels=24):
    """Extract the abdomen contours from the A-pose mesh (the prior to be MLE-corrected):
      transverse_xz : horizontal cross-section at navel height -> ordered (x, z) loop
      coronal_yx    : frontal silhouette (lateral extent vs height) over the abdomen band
    Frame: pelvis-centered, +Y up.
    """
    import trimesh
    pelvis_y = float(named["pelvis"][1])
    spine2_y = float(named["spine2"][1])
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    # anatomical abdomen level from the central section curve (girth minimum = waist),
    # not an arbitrary landmark fraction
    ys, gs = girth_profile(m, named)
    lv = anatomical_levels(ys, gs, named)
    navel_y = lv["abdomen_y"]

    # transverse cross-section at navel: vertices in a thin y-band, TORSO ONLY (exclude the
    # arms, which hang at navel height in SMPL-X rest pose), ordered by angle in x-z.
    torso_x = _torso_halfwidth(verts, navel_y)
    band = verts[(np.abs(verts[:, 1] - navel_y) <= 0.012) & (np.abs(verts[:, 0]) <= torso_x)]
    xz = []
    if band.shape[0] >= 8:
        cx, cz = band[:, 0].mean(), band[:, 2].mean()
        ang = np.arctan2(band[:, 2] - cz, band[:, 0] - cx)
        b = band[np.argsort(ang)]
        xz = [[round(float(x), 5), round(float(z), 5)] for x, z in zip(b[:, 0], b[:, 2])]
        xz.append(xz[0])                                  # close loop

    y0, y1 = pelvis_y - 0.02, spine2_y
    band = verts[(verts[:, 1] >= y0) & (verts[:, 1] <= y1)]
    yx = []
    if band.shape[0] > 10:
        edges = np.linspace(y0, y1, n_levels)
        for i in range(len(edges) - 1):
            yc = 0.5 * (edges[i] + edges[i + 1])
            txw = _torso_halfwidth(verts, yc)
            sl = band[(band[:, 1] >= edges[i]) & (band[:, 1] < edges[i + 1])
                      & (np.abs(band[:, 0]) <= txw)]        # torso only (exclude arms)
            if sl.shape[0] < 3:
                continue
            y = float(0.5 * (edges[i] + edges[i + 1]))
            yx.append({"y": round(y, 5),
                       "x_left": round(float(sl[:, 0].min()), 5),
                       "x_right": round(float(sl[:, 0].max()), 5),
                       "width": round(float(sl[:, 0].max() - sl[:, 0].min()), 5)})
    return {"navel_y": round(navel_y, 5), "transverse_xz": xz, "coronal_yx": yx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stature-cm", type=float, default=None,
                    help="rescale body to a known stature (cm) for absolute calibration")
    ap.add_argument("--from-betas", default=None,
                    help="skip Multi-HMR; re-measure from a saved fused_betas.npy (fast)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.from_betas:
        fused = np.load(args.from_betas)
        bj = os.path.join(os.path.dirname(args.from_betas), "betas.json")
        spread = (np.array(json.load(open(bj))["beta_spread"])
                  if os.path.exists(bj) else np.zeros_like(fused))
        n = json.load(open(bj)).get("n_views_fused", 0) if os.path.exists(bj) else 0
        gj = os.path.join(args.out, "gender.json")
        gender = json.load(open(gj))["decision"] if os.path.exists(gj) else "neutral"
    else:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
        if any(ch in args.subject for ch in "*?[]"):
            images = glob.glob(args.subject)
        elif os.path.isdir(args.subject):
            images = []
            for root, _, _ in os.walk(args.subject):
                for e in exts:
                    images += glob.glob(os.path.join(root, e))
        else:
            images = [args.subject]
        images = sorted(images)
        print(f"subject views: {len(images)}")
        per_view = regress_betas(images)
        fused, spread, n = fuse_betas(per_view)
        if fused is None:
            print("ANTHRO_FAIL: no betas regressed")
            return
        np.save(os.path.join(args.out, "fused_betas.npy"), fused)
        json.dump({"per_view": [{"image": os.path.basename(d["image"]),
                                 "is_full_body": d["is_full_body"],
                                 "beta": d["beta"].tolist()} for d in per_view],
                   "fused_beta": fused.tolist(), "beta_spread": spread.tolist(),
                   "n_views_fused": n},
                  open(os.path.join(args.out, "betas.json"), "w"), indent=2)
        gender, gdbg = estimate_gender(images)
        json.dump(gdbg, open(os.path.join(args.out, "gender.json"), "w"), indent=2)
        print(f"GENDER {gender} (p_f={gdbg.get('fused_p_female')}, conf={gdbg.get('confidence')})")

    print(f"using SMPL-X gender model: {gender}")
    verts, joints, faces = smplx_mesh(fused, gender=gender)
    meas, mesh = measure(verts, joints, faces)

    # optional absolute calibration to a known stature
    if args.stature_cm:
        scale = (args.stature_cm / 100.0) / meas["stature_m"]
        verts2 = verts * scale
        verts2[:, 1] -= verts2[:, 1].min()
        meas, mesh = measure(verts2, joints * scale, faces)
        meas["calibrated_to_stature_cm"] = args.stature_cm

    # uncertainty: perturb betas by +/- spread, re-measure stature/girths
    import torch  # noqa
    unc = {}
    for sgn in (+1, -1):
        v2, j2, f2 = smplx_mesh(fused + sgn * spread, gender=gender)
        mm, _ = measure(v2, j2, f2)
        for k, val in mm.items():
            if isinstance(val, (int, float)):
                unc.setdefault(k, []).append(val)
    meas_cm = {}
    for k, v in meas.items():
        if isinstance(v, (int, float)) and k.endswith("_m"):
            cm = v * 100.0
            band = unc.get(k, [])
            half = (max(band) - min(band)) * 100.0 / 2 if len(band) == 2 else None
            meas_cm[k.replace("_m", "_cm")] = {"value": round(cm, 1),
                                               "uncertainty_cm": round(half, 1) if half else None}
        elif not k.endswith("_m"):
            meas_cm[k] = v
    meas_cm["gender_model"] = gender
    json.dump(meas_cm, open(os.path.join(args.out, "measurements.json"), "w"), indent=2)

    # export the measured canonical mesh
    try:
        mesh.export(os.path.join(args.out, "smplx_canonical.obj"))
    except Exception as e:
        print("mesh export err", e)

    # ---- A-POSE deliverable: mesh + named joint placements + abdomen contours ----
    try:
        av, aj, af, named = smplx_apose(fused, gender=gender, arm_deg=45.0)
        import trimesh
        amesh = trimesh.Trimesh(vertices=av, faces=af, process=False)
        amesh.export(os.path.join(args.out, "apose_mesh.obj"))
        json.dump({"gender": gender, "frame": "pelvis-centered, +Y up, A-pose",
                   "joints": named},
                  open(os.path.join(args.out, "apose_joints.json"), "w"), indent=2)
        # abdomen contour measured on the arms-clear T-pose mesh (shape is pose-invariant)
        contours = abdomen_contours(verts, faces, named)
        json.dump(contours, open(os.path.join(args.out, "abdomen_contours.json"), "w"), indent=2)
        print("APOSE_OK joints=%d abdomen_xz_pts=%d yx_pts=%d" %
              (len(named), len(contours["transverse_xz"]), len(contours["coronal_yx"])))
    except Exception as e:
        import traceback; print("APOSE_ERR", traceback.format_exc()[-500:])

    print("ANTHRO_OK")
    print(json.dumps(meas_cm, indent=2))


if __name__ == "__main__":
    main()
