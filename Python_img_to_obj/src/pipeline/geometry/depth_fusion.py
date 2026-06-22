"""Maximum-likelihood (information-filter) fusion of many depth estimators.

Treat every (estimator n, pixel m) as an independent measurement of the true metric
depth d*(m):

    z_nm = a_n * d_n(m) + b_n + e_nm ,   e_nm ~ N(0, sigma_nm^2)

Monocular estimators are affine-ambiguous, so (a_n, b_n) are calibrated by robust
alignment to a common reference (the SMPL-X metric prior when available, else the
inverse-variance ensemble). After alignment each measurement is z'_nm with calibrated
noise sigma_nm. The MLE of d*(m) given independent Gaussian measurements is the
inverse-variance weighted mean, and its posterior variance is the inverse of the summed
Fisher information:

    info(m)   = sum_n 1 / sigma_nm^2          (+ prior precision)
    d_hat(m)  = (sum_n z'_nm / sigma_nm^2) / info(m)
    var(m)    = 1 / info(m)

This is exactly a scalar information filter / Kalman measurement update per pixel. Adding
estimators or views only adds non-negative terms to info(m) -> the posterior variance can
only shrink. Calibration of sigma_nm is the load-bearing step (a priori sigma), so it is
explicit and testable here.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------- calibration ---
def robust_affine(d: np.ndarray, ref: np.ndarray, mask: np.ndarray, iters: int = 5):
    """Fit a, b minimizing Huber( a*d + b - ref ) over mask. Returns (a, b, resid_std).

    resid_std is a robust (MAD-based) estimate of the post-alignment residual scale =
    the estimator's calibrated *global* sigma in metric units.
    """
    m = np.asarray(mask, bool)
    x = np.asarray(d, float)[m]
    y = np.asarray(ref, float)[m]
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size < 10:
        return 1.0, 0.0, 1.0
    w = np.ones_like(x)
    a, b = 1.0, 0.0
    for _ in range(iters):
        # weighted least squares for [a, b]
        sw = w.sum()
        mx = (w * x).sum() / sw
        my = (w * y).sum() / sw
        cov = (w * (x - mx) * (y - my)).sum()
        var = (w * (x - mx) ** 2).sum() + 1e-12
        a = cov / var
        b = my - a * mx
        r = a * x + b - y
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        c = 1.345 * s                                   # Huber threshold
        ar = np.abs(r)
        w = np.where(ar <= c, 1.0, c / (ar + 1e-12))
    r = a * x + b - y
    resid_std = float(1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9)
    return float(a), float(b), resid_std


def per_pixel_sigma(aligned: np.ndarray, ref: np.ndarray, mask: np.ndarray,
                    sigma_global: float, ensemble_std: np.ndarray | None = None):
    """A-priori sigma map for one aligned estimator.

    Base = sigma_global, inflated where depth is unreliable:
      * high local depth gradient (silhouette / occlusion edges)
      * near the mask boundary
      * (optional) where estimators disagree (ensemble_std)
    """
    a = np.asarray(aligned, float)
    H, W = a.shape
    # local gradient magnitude (edges) -> normalized 0..1 inflation
    gy, gx = np.gradient(np.nan_to_num(a))
    grad = np.hypot(gx, gy)
    g = grad / (np.percentile(grad[np.asarray(mask, bool)], 90) + 1e-9)
    g = np.clip(g, 0, 3)
    # distance to mask boundary (small dist -> less reliable)
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(np.asarray(mask, bool))
    edge = np.exp(-dist / 6.0)                          # ~1 at boundary, ->0 inside
    inflate = 1.0 + 1.5 * g + 1.0 * edge
    sig = sigma_global * inflate
    if ensemble_std is not None:
        sig = np.sqrt(sig ** 2 + (0.5 * ensemble_std) ** 2)
    sig = np.where(np.isfinite(a), sig, np.inf)         # missing -> infinite sigma
    return sig


# ------------------------------------------------------------------- fusion ----
def information_fuse(depths, sigmas, prior=None, prior_sigma=None):
    """Per-pixel information-filter MLE fusion.

    depths : list of HxW aligned depth maps (metric, same frame)
    sigmas : list of HxW per-pixel a-priori sigmas (same shapes); inf where missing
    prior  : optional HxW prior mean (e.g. SMPL-X rendered depth)
    prior_sigma : optional HxW or scalar prior sigma

    Returns (d_hat HxW, var HxW, info HxW). var = posterior variance = 1/info.
    """
    depths = [np.asarray(d, float) for d in depths]
    sigmas = [np.asarray(s, float) for s in sigmas]
    H, W = depths[0].shape
    info = np.zeros((H, W))
    acc = np.zeros((H, W))
    for d, s in zip(depths, sigmas):
        prec = 1.0 / np.clip(s ** 2, 1e-12, np.inf)
        prec = np.where(np.isfinite(d) & np.isfinite(s), prec, 0.0)
        info += prec
        acc += np.where(prec > 0, d * prec, 0.0)
    if prior is not None and prior_sigma is not None:
        ps = np.full((H, W), prior_sigma, float) if np.isscalar(prior_sigma) else np.asarray(prior_sigma, float)
        pprec = 1.0 / np.clip(ps ** 2, 1e-12, np.inf)
        pprec = np.where(np.isfinite(prior), pprec, 0.0)
        info += pprec
        acc += np.where(pprec > 0, np.nan_to_num(prior) * pprec, 0.0)
    var = 1.0 / np.where(info > 0, info, np.nan)
    d_hat = np.where(info > 0, acc / np.where(info > 0, info, 1.0), np.nan)
    return d_hat, var, info


def fuse_estimators(depth_maps, mask, reference=None, ref_sigma=None):
    """End-to-end: calibrate N raw depth maps to a common frame and MLE-fuse them.

    depth_maps : list of HxW raw (possibly affine-ambiguous) depth maps
    mask       : HxW person mask
    reference  : optional HxW metric reference (SMPL-X prior). If None, the median of a
                 first inverse-variance pass bootstraps the reference (relative scale).

    Returns dict with fused depth, posterior var/sigma, info, per-estimator (a,b,sigma).
    """
    mask = np.asarray(mask, bool)
    # bootstrap reference if none: align all to the first, take median
    if reference is None:
        base = depth_maps[0]
        aligned0 = []
        for d in depth_maps:
            a, b, _ = robust_affine(d, base, mask)
            aligned0.append(a * d + b)
        reference = np.nanmedian(np.stack(aligned0, 0), axis=0)
        ref_sigma = None

    aligned, sigmas, params = [], [], []
    for d in depth_maps:
        a, b, sg = robust_affine(d, reference, mask)
        al = a * d + b
        aligned.append(al)
        params.append({"a": a, "b": b, "sigma_global": sg})
    ens_std = np.nanstd(np.stack(aligned, 0), axis=0)
    for al, p in zip(aligned, params):
        sigmas.append(per_pixel_sigma(al, reference, mask, p["sigma_global"], ens_std))

    prior = reference if ref_sigma is not None else None
    d_hat, var, info = information_fuse(aligned, sigmas, prior=prior, prior_sigma=ref_sigma)
    # restrict to mask
    d_hat = np.where(mask, d_hat, np.nan)
    sigma = np.sqrt(var)
    return {"depth": d_hat, "sigma": sigma, "var": var, "info": info,
            "aligned": aligned, "sigmas": sigmas, "params": params,
            "ensemble_std": ens_std, "reference": reference}
