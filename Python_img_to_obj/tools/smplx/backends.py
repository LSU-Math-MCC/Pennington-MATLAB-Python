"""Polymorphic HMR backend registry.

Each backend is the SAME interface -- (name, label, conda env, runner script) -- and is invoked
through its runner with `--images ... --out <dir>/<name>`. The runner is the only env-specific
code; it either reduces native output to schema.npz or writes a backend overlay image. Adding a
third method is one Backend() line plus a runner that emits one of those artifacts.

This module is import-safe in ANY env (pure stdlib) so the compositor can enumerate backends
without importing torch.
"""
import os
import re
import sys
import shutil
import shlex
import subprocess
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
_ON_WIN = os.name == "nt"
# Resolved by the TARGET (WSL) shell, so keep ~ literal (do NOT expanduser on Windows).
CONDA_SH = "~/miniconda3/etc/profile.d/conda.sh"


def _to_wsl(p):
    r"""C:\a\b -> /mnt/c/a/b so a Windows-orchestrated run can hand paths to WSL."""
    p = os.path.abspath(p)
    drive, rest = os.path.splitdrive(p)
    if drive:  # 'C:'
        return "/mnt/" + drive[0].lower() + rest.replace("\\", "/")
    return p.replace("\\", "/")


def _sh(p):
    """A filesystem path as the target shell (WSL on Windows, local bash on Linux) sees it."""
    return _to_wsl(p) if _ON_WIN else p


def _bash(inner):
    """Run a bash login command where the conda envs live: WSL when on Windows, local bash on Linux.

    This is what lets the notebook run from a *Windows* VS Code kernel — the Windows process only
    orchestrates + renders; each backend's heavy CUDA work runs in its WSL conda env via `wsl`.
    """
    cmd = ["wsl", "bash", "-lc", inner] if _ON_WIN else ["bash", "-lc", inner]
    subprocess.run(cmd, check=True)


def reexec_in_wsl(env="camerahmr", cwd_repo=True):
    """If on Windows, re-exec THIS script inside the WSL conda `env` and exit with its code; no-op
    on Linux. Lets a Windows VS Code kernel call heavy tools (torch/smplx/pyrender + SMPL models
    that only exist in WSL) by transparently bouncing the same command into WSL. Args that look
    like Windows paths are translated to /mnt/c; flags/values pass through unchanged."""
    if not _ON_WIN:
        return
    def conv(a):
        return _to_wsl(a) if (re.match(r"^[A-Za-z]:[\\/]", a) or os.path.isabs(a)) else a
    script = _to_wsl(os.path.abspath(sys.argv[0]))
    args = " ".join(shlex.quote(conv(a)) for a in sys.argv[1:])
    cd = f"cd {shlex.quote(_to_wsl(REPO))} && " if cwd_repo else ""
    inner = f"source {CONDA_SH} && conda activate {shlex.quote(env)} && {cd}python {shlex.quote(script)} {args}"
    raise SystemExit(subprocess.run(["wsl", "bash", "-lc", inner]).returncode)


@dataclass(frozen=True)
class Backend:
    name: str          # short id / output subdir
    label: str         # row label in the figure
    env: str           # conda env it must run in
    runner: str        # runner script (relative to repo root)
    cwd: str = ""      # working dir for the runner (some repos require being run from their root)
    crop_first: bool = False
    crop_env: str = "camerahmr"

    def run(self, image_paths, out_root):
        """Invoke the runner inside its conda env; it writes <out_root>/<name>/<stem>.npz per image.
        out_dir is made ABSOLUTE here (in the caller's cwd) because the runner cd's into its repo
        (self.cwd) first -- a relative path would otherwise land under ~/blade, ~/CameraHMR, etc."""
        out_root = os.path.abspath(out_root)
        out_dir = os.path.join(out_root, self.name)
        os.makedirs(out_dir, exist_ok=True)
        runner_abs = os.path.join(REPO, self.runner)
        run_images = list(image_paths)
        if self.crop_first:
            crop_src = os.path.join(out_root, "_crop_src", self.name)
            crop_dir = os.path.join(out_root, "_crops", self.name)
            shutil.rmtree(crop_src, ignore_errors=True)
            shutil.rmtree(crop_dir, ignore_errors=True)
            os.makedirs(crop_src, exist_ok=True)
            run_images = []
            for p in image_paths:
                stem, ext = os.path.splitext(os.path.basename(p))
                pre = os.path.join(REPO, "runs", "ssp3d_target_crops", stem + ".png")
                if os.path.exists(pre):
                    run_images.append(pre)
                    continue
                shutil.copy2(p, os.path.join(crop_src, stem + (ext or ".png")))
                run_images.append(os.path.join(crop_dir, stem + ".png"))
            if os.listdir(crop_src):
                cropper = os.path.join(REPO, "tools", "render", "person_crop.py")
                crop_inner = (
                    f"source {CONDA_SH} && conda activate {shlex.quote(self.crop_env)} && "
                    f"python {shlex.quote(_sh(cropper))} --images-dir {shlex.quote(_sh(crop_src))} "
                    f"--out-dir {shlex.quote(_sh(crop_dir))}"
                )
                print(f"[{self.name}] crop env={self.crop_env}")
                _bash(crop_inner)
        inner = f"source {CONDA_SH} && conda activate {self.env} && "
        if self.cwd:
            inner += f"cd {self.cwd} && "   # ~ expanded by the target shell; these repo dirs have no spaces
        inner += (
            "python "
            + shlex.quote(_sh(runner_abs))
            + " --out "
            + shlex.quote(_sh(out_dir))
            + " --images "
            + " ".join(shlex.quote(_sh(p)) for p in run_images)
        )
        print(f"[{self.name}] conda env={self.env}")
        _bash(inner)
        return out_dir


REGISTRY = {
    "camerahmr": Backend("camerahmr", "CameraHMR", "camerahmr",
                         "tools/hmr/camerahmr/run_camerahmr.py", cwd="~/CameraHMR"),
    "blade":     Backend("blade",     "BLADE",     "blade_env",
                         "tools/hmr/blade/run_blade.py", cwd="~/blade",
                         crop_first=True, crop_env="camerahmr"),
}
