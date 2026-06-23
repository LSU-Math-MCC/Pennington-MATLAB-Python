"""Dataset + SHAPY-checkpoint discovery for the SHAPY benchmark (per docs/PROJECT.md §9).
Searches the documented locations, NEVER asks for credentials. Returns a manifest dict.
"""
import os
import glob
from pathlib import Path

HOME = os.path.expanduser("~")
REPO = str(Path(__file__).resolve().parents[2])

DATASET_PATHS = {
    "hbw": [f"{REPO}/datasets/HBW", f"{REPO}/datasets/hbw", "datasets/HBW", "datasets/hbw",
            f"{HOME}/HBW", f"{HOME}/datasets/HBW",
            "/mnt/d/datasets/HBW", "/mnt/e/datasets/HBW"],
    "mmts": [f"{REPO}/datasets/MMTS", "datasets/MMTS", f"{HOME}/MMTS", f"{HOME}/datasets/MMTS"],
    "ssp3d": [f"{REPO}/datasets/SSP-3D", f"{REPO}/datasets/SSP-3D/data_ext/ssp_3d",
              "datasets/SSP-3D", "datasets/SSP-3D/data_ext/ssp_3d",
              f"{HOME}/SSP-3D", f"{HOME}/SSP-3D/data_ext/ssp_3d", f"{HOME}/datasets/SSP-3D",
              "/mnt/d/datasets/SSP-3D", "/mnt/e/datasets/SSP-3D"],
}
SHAPY_CKPT_PATHS = [f"{HOME}/shapy/data/trained_models", f"{HOME}/shapy/data/expose_release",
                    f"{HOME}/shapy/data/utility_files", "vendor/shapy/data/trained_models",
                    "external/shapy/data/trained_models"]


def discover():
    found = {}
    for ds, paths in DATASET_PATHS.items():
        hit = next((p for p in paths if os.path.exists(p)), None)
        found[ds] = {"present": hit is not None, "path": hit, "searched": paths}
    # ssp3d: also locate the labels/images under the found dir
    if found["ssp3d"]["present"]:
        ext = glob.glob(found["ssp3d"]["path"] + "/**/labels.npz", recursive=True)
        found["ssp3d"]["labels"] = ext[0] if ext else None
        found["ssp3d"]["data_root"] = os.path.dirname(ext[0]) if ext else found["ssp3d"]["path"]
    shapy = next((p for p in SHAPY_CKPT_PATHS if os.path.isdir(p) and os.listdir(p)), None)
    found["shapy_local_checkpoints"] = {"present": shapy is not None, "path": shapy,
                                        "searched": SHAPY_CKPT_PATHS}
    found["baseline_source"] = "local_shapy" if shapy else "published"
    return found


if __name__ == "__main__":
    import json
    d = discover()
    print(json.dumps(d, indent=2))
