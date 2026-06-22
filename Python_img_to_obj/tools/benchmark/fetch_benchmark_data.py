"""Fetch/open setup instructions for benchmark datasets.

Automates only public, ungated downloads. HBW requires SHAPY registration/license
acceptance, so this script prints the official steps and verifies a manually placed
folder instead of scraping credentials or gated files.

Usage:
  python tools/benchmark/fetch_benchmark_data.py --ssp3d
  python tools/benchmark/fetch_benchmark_data.py --hbw-info
  python tools/benchmark/fetch_benchmark_data.py --all
"""
from __future__ import annotations

import argparse
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO / "datasets"
SSP3D_URL = "https://github.com/akashsengupta1997/SSP-3D/raw/master/ssp_3d.zip"
HBW_DATA_URL = "https://shapy.is.tue.mpg.de/datasets.html"
HBW_EVAL_URL = "https://github.com/muelea/shapy/tree/master/regressor/hbw_evaluation"


def _download(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {url}")
    print(f"    -> {dst}")
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)


def fetch_ssp3d(data_root: Path, force: bool = False):
    target = data_root / "SSP-3D"
    labels = list(target.glob("**/labels.npz")) if target.exists() else []
    if labels and not force:
        print(f"SSP3D_PRESENT labels={labels[0]}")
        return labels[0]

    data_root.mkdir(parents=True, exist_ok=True)
    zip_path = data_root / "ssp_3d.zip"
    if force or not zip_path.exists():
        _download(SSP3D_URL, zip_path)

    if target.exists() and force:
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)

    labels = list(target.glob("**/labels.npz"))
    if not labels:
        raise SystemExit(f"SSP3D_FAIL no labels.npz after extracting {zip_path}")
    data_dir = labels[0].parent
    print(f"SSP3D_OK root={target} labels={labels[0]}")
    print("Set SSP3D_ROOT to this path if running old scripts:")
    print(f"  SSP3D_ROOT={data_dir}")
    return labels[0]


def hbw_info(data_root: Path):
    target = data_root / "HBW"
    present = target.exists()
    print("HBW is gated by the SHAPY license/registration flow.")
    print(f"Official dataset page: {HBW_DATA_URL}")
    print(f"Official eval docs:    {HBW_EVAL_URL}")
    print("")
    print("What is public/automatable:")
    print("  - validation images and validation GT scans after website registration/license acceptance")
    print("  - test images after website registration/license acceptance")
    print("")
    print("What is not public:")
    print("  - HBW test ground truth. Official test evaluation requires submitting an NPZ")
    print("    with image_name and v_shaped to shapy@tue.mpg.de.")
    print("")
    print("Expected local layout after manual download:")
    print(f"  {target}")
    print("")
    print(f"HBW_PRESENT={present}")
    if present:
        print(f"HBW_PATH={target}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--ssp3d", action="store_true")
    ap.add_argument("--hbw-info", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    if args.all or args.ssp3d:
        fetch_ssp3d(data_root, force=args.force)
    if args.all or args.hbw_info:
        hbw_info(data_root)
    if not (args.all or args.ssp3d or args.hbw_info):
        ap.print_help()


if __name__ == "__main__":
    main()
