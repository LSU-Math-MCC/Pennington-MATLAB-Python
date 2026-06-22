"""Beta-space MLE fusion: combine SMPL-X shape estimators by inverse-variance.

Each estimator is a noisy observation of the true betas with a calibrated sigma:
  - Multi-HMR prior  (prior_betas.npy): broad prior, sigma_prior (fixed, loose)
  - silhouette-fit   (fitted_betas.npy): image-observation, sigma_fit small when the fit
                       improved a lot (sigma_fit = base / sqrt(max(improvement_pct,1)))
  - SHAPY (optional, shapy_betas.npy): metric regression, sigma_shapy
Fused = sum(b_i / sigma_i^2) / sum(1/sigma_i^2);  posterior var = 1/sum(1/sigma_i^2).
Per the framework: information adds across estimators; calibrated sigma is load-bearing.

Usage: python tools/geometry/fuse_betas.py <fit_dir>   (e.g. runs/fit_s1) -> writes fused_betas.npy
"""
import sys
import os
import json

import numpy as np

SIGMA_PRIOR = 1.0          # Multi-HMR prior: loose
SIGMA_FIT_BASE = 1.2       # silhouette-fit base; scaled by improvement
SIGMA_SHAPY = 0.5          # SHAPY metric regression: tight (when available)


def fuse(fit_dir):
    obs = []
    prior = os.path.join(fit_dir, "prior_betas.npy")
    fitp = os.path.join(fit_dir, "fitted_betas.npy")
    rep = os.path.join(fit_dir, "fit_report.json")
    if os.path.exists(prior):
        obs.append((np.load(prior), SIGMA_PRIOR))
    if os.path.exists(fitp):
        imp = 1.0
        if os.path.exists(rep):
            imp = max(json.load(open(rep)).get("improvement_pct", 1.0), 1.0)
        sig = SIGMA_FIT_BASE / np.sqrt(imp)        # better fit -> smaller sigma -> more weight
        obs.append((np.load(fitp), sig))
    shp = os.path.join(fit_dir, "shapy_betas.npy")
    if os.path.exists(shp):
        obs.append((np.load(shp), SIGMA_SHAPY))
    if not obs:
        print("no estimators in", fit_dir); return None
    n = min(len(b) for b, _ in obs)
    info = np.zeros(n); acc = np.zeros(n)
    for b, s in obs:
        w = 1.0 / s ** 2
        info += w; acc += w * b[:n]
    fused = acc / info
    var = 1.0 / info
    np.save(os.path.join(fit_dir, "fused_betas.npy"), fused)
    json.dump({"n_estimators": len(obs),
               "sigmas": [float(s) for _, s in obs],
               "fused_beta": fused.tolist(),
               "posterior_sigma": np.sqrt(var).tolist()},
              open(os.path.join(fit_dir, "fuse_report.json"), "w"), indent=2)
    print(f"FUSE_OK {os.path.basename(fit_dir)} estimators={len(obs)} "
          f"sigmas={[round(s,2) for _,s in obs]}")
    return fused


if __name__ == "__main__":
    dirs = sys.argv[1:] or [f"runs/fit_s{i}" for i in range(1, 6)]
    for d in dirs:
        if os.path.isdir(d):
            fuse(d)
    print("FUSE_ALL_DONE")
