"""Reproduce SHAPY's ATTRIBUTES->SHAPE mechanism with an OPEN submodel (CLIP), to break the
mode-collapse-to-average that monocular SMPL(-X) regressors tend to suffer (observed with Multi-HMR).

SHAPY's key result: semantic body attributes predict accurate METRIC shape. We do exactly that
with CLIP zero-shot (which, unlike the regressors, CAN tell slim from heavy):

  1. CLIP scores each image over an ordinal BUILD ladder (very slim .. very heavy) and a
     MUSCLE axis, fused across views (independent-view log-posterior, same as gender).
  2. build score -> target BMI -> target MASS (= BMI * height_m^2). This injects the absolute
     heaviness that monocular depth/scale cannot resolve.
  3. Fit 10 SMPL-X betas so the shaped mesh hits {target_mass, current height} while preserving
     the SILHOUETTE-derived PROPORTIONS (normalized girth profile of the existing fused mesh).
     -> CLIP sets absolute heaviness, silhouette sets shape -> clip_betas.npy (fusable, sigma
     small because attributes are the de-biasing signal).

Run (WSL lhm):  python tools/geometry/clip_shape.py [s1 s2 ...]
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
import glob
import json

import numpy as np
import torch

REPO = _repo
sys.path.insert(0, REPO + "/tools")
import shapy_measure as SM            # noqa: E402  (measure / plane_perimeter / mesh_volume)

SX_LM = dict(HEAD_TOP=8976, LEFT_HEEL=8847, LEFT_NIPPLE=3572, BELLY_BUTTON=5939, PUBIC_BONE=5949)
DENSITY = 985.0
# ordinal build ladder -> representative BMI (kg/m^2). CLIP gives a soft posterior over these.
BUILD = [("a photo of a very thin, slim, skinny person", 17.5),
         ("a photo of a slim, lean, fit person", 20.5),
         ("a photo of an average build person", 23.5),
         ("a photo of a curvy, heavyset, overweight person", 28.5),
         ("a photo of a very heavy, obese, fat person", 34.0)]


def clip_build_bmi(images):
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    if not hasattr(clip_build_bmi, "m"):
        clip_build_bmi.m = (CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
                            CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"))
    model, proc = clip_build_bmi.m
    prompts = [b[0] for b in BUILD]; bmis = np.array([b[1] for b in BUILD])
    logsum = np.zeros(len(BUILD)); per = []
    for ip in images:
        im = Image.open(ip).convert("RGB")
        inp = proc(text=prompts, images=im, return_tensors="pt", padding=True)
        with torch.no_grad():
            p = model(**inp).logits_per_image.softmax(-1)[0].cpu().numpy()
        logsum += np.log(p + 1e-9); per.append(p.tolist())
    post = np.exp(logsum - logsum.max()); post /= post.sum()
    bmi = float((post * bmis).sum())          # expected BMI under CLIP posterior
    return bmi, post.tolist()


def fit_betas(model, faces, target_mass, target_height, ref_norm_girth, prop_rel, x0,
              y_band=(0.05, 0.92), n=16):
    """Fit 10 SMPL-X betas to {CLIP target mass, height, silhouette proportions}, anchored to a
    NEUTRAL prior and bounded so an UNRELIABLE silhouette (e.g. a non-standing subject) cannot
    produce a degenerate/grotesque body. prop_rel in [0,1] gates how much we trust the
    silhouette proportions; low prop_rel -> a clean neutral body at the CLIP mass."""
    from scipy.optimize import least_squares

    def shaped(b):
        with torch.no_grad():
            return model(betas=torch.tensor(b, dtype=torch.float32).unsqueeze(0)).vertices[0].numpy()

    def resid(b):
        v = shaped(b)
        y0, y1 = v[:, 1].min(), v[:, 1].max(); h = y1 - y0
        mass = SM.mesh_volume(v, faces) * DENSITY
        g = np.array([SM.plane_perimeter(v, faces, y0 + (y1 - y0) * (y_band[0] + (y_band[1] - y_band[0]) * (i + 0.5) / n))
                      for i in range(n)])
        gn = g / (g.sum() + 1e-6)
        return np.concatenate([
            [(mass - target_mass) * 0.6],             # CLIP absolute heaviness (drives slimming)
            [(h - target_height) * 8.0],              # keep height
            (gn - ref_norm_girth) * (5.0 * prop_rel), # silhouette proportions, gated by reliability
            b * (0.35 + 0.5 * (1 - prop_rel)),        # NEUTRAL anchor (stronger when unreliable)
        ])

    # init from the reference shape (proportions already right) and descend to the CLIP mass;
    # clip init into bounds so a wild silhouette ref can't seed a degenerate start.
    x0 = np.clip(x0, -2.0, 2.0)
    sol = least_squares(resid, x0, method="trf", max_nfev=250, diff_step=0.1,
                        bounds=(-2.2 * np.ones(10), 2.2 * np.ones(10)))
    return sol.x


def main():
    import smplx
    import lhm_anthropometry as A
    subjects = sys.argv[1:] or ["ssp3d_bodybuilder"]
    ssp_glob = f"{REPO}/datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_*.png"
    model = smplx.create(A.HUMAN_MODELS, model_type="smplx", gender="neutral", num_betas=10)
    faces = model.faces.astype(np.int64)
    agg = {}
    for s in subjects:
        subject_glob = s if any(ch in s for ch in "*?[]/\\") else ssp_glob
        imgs = sorted(glob.glob(subject_glob))[:5]
        # ORIGINAL silhouette-fit betas (heavy backup if we already promoted), as proportion ref
        heavy = f"{REPO}/runs/fit_{s}/fused_betas_heavy.npy"
        bp = heavy if os.path.exists(heavy) else f"{REPO}/runs/fit_{s}/fused_betas.npy"
        if not imgs or not os.path.exists(bp):
            print(f"skip {s}"); continue
        bmi, post = clip_build_bmi(imgs)
        sil = np.load(bp)[:10].astype(np.float32)
        # reliability of the SILHOUETTE refinement: it matches a width-profile by image row, so
        # it only holds for a STANDING subject; extreme betas flag a non-standing/bad fit.
        sil_rel = float(np.clip(1.4 - np.abs(sil).max() / 2.0, 0.1, 1.0))
        # fall back to Multi-HMR's POSE-INVARIANT shape (prior_betas) when silhouette is bad.
        priorp = f"{REPO}/runs/fit_{s}/prior_betas.npy"
        if sil_rel < 0.5 and os.path.exists(priorp):
            ref = np.load(priorp)[:10].astype(np.float32); prop_rel = 0.6; src = "MultiHMR-prior"
        else:
            ref = sil; prop_rel = sil_rel; src = "silhouette"
        with torch.no_grad():
            vref = model(betas=torch.tensor(ref, dtype=torch.float32).unsqueeze(0)).vertices[0].numpy()
        y0, y1 = vref[:, 1].min(), vref[:, 1].max(); h = y1 - y0
        gref = np.array([SM.plane_perimeter(vref, faces, y0 + (y1 - y0) * (0.05 + 0.87 * (i + 0.5) / 16))
                         for i in range(16)])
        gref /= (gref.sum() + 1e-6)
        # height from a NEUTRAL standing body (sil height is unreliable if pose is bad)
        with torch.no_grad():
            vneut = model(betas=torch.zeros(1, 10)).vertices[0].numpy()
        h_neutral = vneut[:, 1].max() - vneut[:, 1].min()
        h_use = h if prop_rel > 0.5 else h_neutral
        target_mass = bmi * h_use * h_use
        b = fit_betas(model, faces, target_mass, h_use, gref, prop_rel, x0=ref)
        np.save(f"{REPO}/runs/fit_{s}/clip_betas.npy", b)
        # report resulting measurements
        with torch.no_grad():
            vb = model(betas=torch.tensor(b, dtype=torch.float32).unsqueeze(0)).vertices[0].numpy()
        m = SM.plane_perimeter(vb, faces, vb[SX_LM["BELLY_BUTTON"], 1])
        mass = SM.mesh_volume(vb, faces) * DENSITY
        agg[s] = dict(clip_bmi=round(bmi, 1), target_mass=round(target_mass, 1),
                      result_mass=round(mass, 1), result_waist_cm=round(m * 100, 1), post=post)
        print(f"CLIP_SHAPE {s}: BMI~{bmi:.1f} ref={src}(rel={prop_rel:.2f}) "
              f"target_mass={target_mass:.1f}kg -> betas[:4]={np.round(b[:4],2)} "
              f"result_mass={mass:.1f}kg waist={m*100:.1f}cm")
    json.dump(agg, open(f"{REPO}/runs/CLIP_shape.json", "w"), indent=2)
    print("CLIP_SHAPE_DONE")


if __name__ == "__main__":
    main()
