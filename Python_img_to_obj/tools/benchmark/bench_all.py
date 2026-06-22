"""Kept-stack SHAPY-style benchmark summary.

This runner intentionally ignores removed legacy methods (HMR2, PiFUHD, 3DDFA,
GFPGAN, SMPLitex). It summarizes the anthropometry-relevant stack we kept:

* CameraHMR: strongest measured SSP-3D body-shape prior.
* BLADE: close-range/perspective SMPL-X candidate when collected.
* SHAPY: published anthropometry baseline and optional local regressor.
* MeshMap full: fused CameraHMR/BLADE/LHM/SHAPY/silhouette/CLIP evidence.

It is conservative: HBW remains the real SHAPY-style victory condition. SSP-3D
is a sanity benchmark and cannot by itself prove HBW superiority.
"""

from __future__ import annotations
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


import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO + "/tools")
import bench_discover  # noqa: E402

SHAPY = json.load(open(REPO + "/benchmarks/configs/shapy_published.json"))["targets"]


def env_info():
    try:
        import torch

        return {"torch": torch.__version__, "cuda": torch.cuda.is_available()}
    except Exception:
        return {}


def run_ssp3d(methods):
    res = {"dataset": "ssp3d", "methods": {}, "shapy_target": SHAPY["ssp3d"]}
    if "camerahmr_sota" in methods:
        res["methods"]["camerahmr_sota"] = {
            "pve_t_sc_mm": 11.6,
            "setting": "published/reproduced single-image CameraHMR SSP-3D baseline",
            "source": "benchmarks/configs/ssp3d_sota.json",
        }

    skf = REPO + "/runs/CAMERAHMR_fusion_results.json"
    if os.path.exists(skf) and "meshmap_camerahmr_skf" in methods:
        s = json.load(open(skf))
        res["methods"]["meshmap_camerahmr_skf"] = {
            "pve_t_sc_mm": s.get("best"),
            "single_image_pve_mm": s.get("single_image"),
            "sota_single_image_pve_mm": s.get("sota_single_image"),
            "beats_single_image_sota": s.get("beats_sota"),
            "n_images": s.get("n_images"),
            "n_subjects": s.get("n_subjects"),
            "setting": "multi-view SKF-style fusion on official CameraHMR betas",
            "source": "runs/CAMERAHMR_fusion_results.json",
        }

    blade = REPO + "/runs/BLADE_ssp3d.npz"
    if os.path.exists(blade) and "blade_smplx" in methods:
        res["methods"]["blade_smplx"] = {
            "setting": "BLADE SMPL-X betas collected on SSP-3D; run shared evaluator before claiming rank",
            "source": "runs/BLADE_ssp3d.npz",
        }

    full = REPO + "/runs/SSP3D_fullpipe.json"
    if os.path.exists(full) and "meshmap_full" in methods:
        f = json.load(open(full))
        err = dict(zip(f.get("keys", []), f.get("err_cm", [])))
        res["methods"]["meshmap_full"] = {
            "meas_err_cm": err,
            "mean_meas_err_cm": round(float(np.mean(list(err.values()))), 2) if err else None,
            "n": f.get("n"),
            "setting": "canonical A-pose anthropometric measurement error",
            "source": "runs/SSP3D_fullpipe.json",
        }

    res["methods"]["published_shapy"] = {"pve_t_sc_mm": SHAPY["ssp3d"]["pve_t_sc_mm"]}
    pve_methods = {
        k: v.get("pve_t_sc_mm")
        for k, v in res["methods"].items()
        if isinstance(v, dict) and v.get("pve_t_sc_mm") is not None and k != "published_shapy"
    }
    best_name = min(pve_methods, key=pve_methods.get) if pve_methods else None
    res["best_pve_t_sc"] = {"method": best_name, "mm": pve_methods.get(best_name) if best_name else None}
    res["beats_shapy_ssp3d"] = bool(best_name and pve_methods[best_name] < SHAPY["ssp3d"]["pve_t_sc_mm"])
    res["note"] = (
        "SSP-3D PVE-T-SC is scale-corrected and is only a sanity benchmark for this product. "
        "HBW/MMTS-style measurement error owns the real anthropometry verdict."
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hbw", "mmts", "ssp3d"])
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["camerahmr_sota", "meshmap_camerahmr_skf", "blade_smplx", "meshmap_full", "published_shapy"],
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_id = args.dataset + "_" + time.strftime("%Y%m%d_%H%M%S")
    out = args.out or (REPO + "/benchmarks/results/" + run_id)
    os.makedirs(out, exist_ok=True)

    found = bench_discover.discover()
    ds = found[args.dataset]
    json.dump(found, open(out + "/discovery.json", "w"), indent=2)
    json.dump(
        {"environment": env_info(), "methods": args.methods, "dataset": args.dataset, "time": run_id},
        open(out + "/config.json", "w"),
        indent=2,
    )

    metrics = {
        "status": "ok",
        "dataset": args.dataset,
        "methods": args.methods,
        "baseline": "published_shapy",
        "baseline_source": found["baseline_source"],
        "shapy_targets": SHAPY.get(args.dataset, {}),
    }
    if not ds["present"]:
        metrics.update(
            status="blocked_missing_dataset",
            beats_shapy=False,
            verdict="insufficient_ground_truth",
            searched_paths=ds["searched"],
        )
    elif args.dataset == "ssp3d":
        metrics["ssp3d"] = run_ssp3d(args.methods)
        metrics["beats_shapy"] = metrics["ssp3d"]["beats_shapy_ssp3d"]
        metrics["verdict"] = "beats_shapy_only_on_ssp3d" if metrics["beats_shapy"] else "insufficient_ground_truth"
        metrics["decision_rule"] = "SSP-3D is a sanity benchmark; HBW owns the real verdict."
    else:
        metrics.update(status="dataset_present_runner_todo", beats_shapy=False, verdict="does_not_beat_shapy_on_hbw")

    json.dump(metrics, open(out + "/metrics.json", "w"), indent=2)
    with open(out + "/summary.md", "w") as f:
        f.write(f"# Kept-stack SHAPY benchmark - {args.dataset}\n\n")
        f.write(f"- run: `{run_id}`\n")
        f.write(f"- methods: `{', '.join(args.methods)}`\n")
        f.write(f"- dataset present: **{ds['present']}** path: `{ds.get('path')}`\n")
        f.write(f"- baseline source: **{found['baseline_source']}**\n\n")
        f.write(f"## Verdict: `{metrics.get('verdict')}`\n\n")
        if args.dataset == "ssp3d" and ds["present"]:
            best = metrics["ssp3d"].get("best_pve_t_sc", {})
            f.write(
                f"Best available SSP-3D PVE-T-SC: **{best.get('method')} {best.get('mm')} mm** "
                f"vs published SHAPY **{SHAPY['ssp3d']['pve_t_sc_mm']} mm**.\n\n"
            )
            f.write(f"> {metrics['ssp3d']['note']}\n")
    print(f"VERDICT={metrics.get('verdict')} -> {out}/metrics.json")


if __name__ == "__main__":
    main()
