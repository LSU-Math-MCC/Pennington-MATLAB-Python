"""LHM geometry backend: single image -> canonical (A-pose) 3D Gaussians via LHM.

LHM runs in a WSL2 conda env with CUDA (see tools/hmr/lhm/wsl_setup_lhm.sh). This backend
shells into that env, runs LHM's mesh/gaussian export (`infer_single_view` ->
`save_ply`) which produces canonical-pose Gaussians, then loads the PLY back into a
SplatCloud in the pipeline's canonical frame.

Until the WSL LHM env is validated, instantiating this backend raises a clear,
actionable error; the registry only builds it when 'lhm' is explicitly selected.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

from ..types import SplatCloud
from ..geometry import camera as camlib
from . import base

# WSL paths
WSL_LHM_DIR = "~/LHM"
WSL_CONDA_ENV = "lhm"


def _wsl(cmd: str, timeout=1800):
    full = ["wsl", "bash", "-lc", cmd]
    return subprocess.run(full, capture_output=True, text=True, timeout=timeout)


def lhm_cache_dir() -> Path:
    d = Path(__file__).resolve().parents[3] / "runs" / ".lhm_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _image_hash(image) -> str:
    import hashlib
    return hashlib.sha1(np.ascontiguousarray(image)).hexdigest()[:16]


def lhm_available() -> bool:
    """True if the WSL LHM env imports its CUDA extensions."""
    try:
        r = _wsl(
            "source ~/miniconda3/etc/profile.d/conda.sh && conda activate lhm && "
            "python -c 'import torch,pytorch3d,diff_gaussian_rasterization,simple_knn; "
            "print(torch.cuda.is_available())'",
            timeout=120,
        )
        return r.returncode == 0 and "True" in r.stdout
    except Exception:
        return False


class LHMReconstructor(base.GSReconstructionBackend):
    name = "lhm-mini"
    version = "1"

    def __init__(self, config=None, model_name="LHM-MINI"):
        self.config = config
        self.model_name = model_name
        if not lhm_available():
            raise RuntimeError(
                "LHM backend not ready: the WSL2 LHM env (CUDA extensions) is not "
                "importable yet. Run tools/hmr/lhm/wsl_setup_lhm.sh + tools/hmr/lhm/wsl_fix_lhm.sh, "
                "then verify with lhm_backend.lhm_available(). "
                "This backend requires a CUDA GPU via WSL2."
            )

    def reconstruct(self, image: np.ndarray, out_dir: Path):
        """Run LHM to export canonical (A-pose) Gaussians; load as a SplatCloud.

        LHM inference is very expensive (Sapiens-1B encoder), so the exported gaussian
        PLY is cached by (image content hash + model). A cache hit skips inference
        entirely and just re-parses the saved PLY.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        H, W = image.shape[:2]

        key = _image_hash(image) + "_" + self.model_name
        cache_ply = lhm_cache_dir() / f"{key}.ply"
        if cache_ply.exists():
            splats = canonicalize_lhm(load_gaussian_ply(cache_ply))
            return splats, camlib.default_camera(W, H)

        from ..io import save_image
        img_path = out_dir / "lhm_input.png"
        save_image(img_path, image)
        wsl_out = _to_wsl_path(out_dir / "lhm_out")
        win_to_wsl = _to_wsl_path(img_path.parent)   # LHM accepts a folder

        cmd = (
            "source ~/miniconda3/etc/profile.d/conda.sh && conda activate lhm && "
            f"cd {WSL_LHM_DIR} && python -m LHM.launch infer.human_lrm "
            f"model_name={self.model_name} image_input='{win_to_wsl}' "
            f"export_mesh=True motion_seqs_dir=None"
        )
        r = _wsl(cmd, timeout=2400)
        if r.returncode != 0:
            raise RuntimeError(f"LHM inference failed: {r.stderr[-800:]}")

        ply = _find_ply(Path(WSL_LHM_DIR)) or _find_ply(out_dir / "lhm_out")
        if ply is None:
            raise RuntimeError("LHM produced no .ply")
        # populate cache
        try:
            import shutil
            cache_ply.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ply, cache_ply)
        except Exception:
            pass
        splats = canonicalize_lhm(load_gaussian_ply(ply))
        return splats, camlib.default_camera(W, H)


def _to_wsl_path(p: Path) -> str:
    p = str(Path(p).resolve())
    # C:\Users\... -> /mnt/c/Users/...
    if len(p) > 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return p.replace("\\", "/")


def _find_ply(d: Path):
    d = Path(d)
    if not d.exists():
        return None
    plys = sorted(d.rglob("*.ply"), key=lambda x: x.stat().st_size, reverse=True)
    return plys[0] if plys else None


def load_gaussian_ply(path) -> SplatCloud:
    """Load a 3DGS PLY (positions + f_dc color + opacity + scale + rot) into SplatCloud.

    Falls back gracefully to xyz(+rgb) for plain point PLYs.
    """
    import struct

    path = Path(path)
    with open(path, "rb") as f:
        head = []
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            head.append(line)
            if line == "end_header":
                break
        props = [l.split()[-1] for l in head if l.startswith("property")]
        n = next((int(l.split()[-1]) for l in head if l.startswith("element vertex")), 0)
        is_ascii = any("ascii" in l for l in head)
        if is_ascii:
            data = np.loadtxt(path, skiprows=len(head), max_rows=n)
        else:
            fmt = "<" + "f" * len(props)
            rec = struct.calcsize(fmt)
            buf = f.read(rec * n)
            data = np.frombuffer(buf, dtype=np.float32).reshape(n, len(props))
    col = {name: i for i, name in enumerate(props)}

    def take(keys, default):
        idx = [col[k] for k in keys if k in col]
        return data[:, idx] if len(idx) == len(keys) else default

    centers = take(["x", "y", "z"], np.zeros((n, 3)))
    if "f_dc_0" in col:
        sh = take(["f_dc_0", "f_dc_1", "f_dc_2"], np.zeros((n, 3)))
        colors = np.clip(0.5 + 0.28209479 * sh, 0, 1)   # SH DC -> RGB
    elif "red" in col:
        colors = take(["red", "green", "blue"], np.full((n, 3), 0.6)) / 255.0
    else:
        colors = np.full((n, 3), 0.6)
    scales = np.exp(take(["scale_0", "scale_1", "scale_2"], np.log(np.full((n, 3), 0.01))))
    rots = take(["rot_0", "rot_1", "rot_2", "rot_3"], np.tile([1.0, 0, 0, 0], (n, 1)))
    op = take(["opacity"], np.full((n, 1), 3.0)).reshape(-1)
    opac = 1.0 / (1.0 + np.exp(-op))                    # sigmoid
    return SplatCloud(centers=centers.astype(np.float64), scales=scales.astype(np.float64),
                      rotations=rots.astype(np.float64), opacities=opac.astype(np.float64),
                      colors=colors.astype(np.float64),
                      extras={"region": np.zeros(n, int), "confidence": np.full(n, 0.95)})


def canonicalize_lhm(splats: SplatCloud) -> SplatCloud:
    """Center on pelvis (cloud centroid proxy), normalize by height, +Y up."""
    c = splats.centers.copy()
    if c.shape[0] == 0:
        return splats
    c = c - np.median(c, axis=0)
    height = float(np.percentile(c[:, 1], 97) - np.percentile(c[:, 1], 3)) or 1.0
    c = c / height
    if c[:, 1].mean() < 0:
        c[:, 1] *= -1
    splats.centers = c
    return splats
