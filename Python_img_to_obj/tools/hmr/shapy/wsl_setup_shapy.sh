#!/usr/bin/env bash
# Set up SHAPY (baseline) in its own conda env. CUDA mesh-intersection + SMPL-X linked.
# The trained checkpoints (shapy_data.zip) are MPI-gated -> flagged at the end.
set -u
REPO="$(d="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"; while [ "$d" != / ] && [ ! -f "$d/pyproject.toml" ]; do d="$(dirname "$d")"; done; echo "$d")"
LOG(){ echo "===SHAPY=== $* ($(date +%H:%M:%S))"; }
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

if ! conda env list | grep -q "^shapy "; then
  LOG "create env shapy (py3.8)"; conda create -n shapy python=3.8 -y || { echo FAIL_env; exit 1; }
fi
conda activate shapy
cd "$HOME/shapy" || exit 1
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/attributes/"

# torch that supports Ampere (sm_86) — SHAPY pins 1.7.1+cu102 (too old for 3080Ti); use cu117.
LOG "install torch cu117"
pip install --quiet torch==1.13.1+cu117 torchvision==0.14.1+cu117 --index-url https://download.pytorch.org/whl/cu117 || echo WARN_torch

LOG "install requirements (best-effort; some pins are old)"
pip install --quiet -r requirements.txt 2>&1 | tail -3 || echo WARN_reqs
pip install --quiet smplx trimesh open3d omegaconf loguru pyyaml yacs || true

LOG "build attributes pkg"
(cd attributes && pip install --quiet -e . 2>&1 | tail -2) || echo WARN_attr

LOG "build mesh-mesh-intersection (CUDA)"
(cd mesh-mesh-intersection && export CUDA_SAMPLES_INC=$(pwd)/include && \
 pip install --quiet -r requirements.txt 2>&1 | tail -2 && \
 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6" pip install --no-build-isolation -e . 2>&1 | tail -3) || echo WARN_mmi

# link SMPL-X / SMPL bodies from the LHM data we already have
LOG "link body models"
mkdir -p data/body_models/smplx data/body_models/smpl
for g in NEUTRAL MALE FEMALE; do
  ln -sf "$HOME/LHM/pretrained_models/human_model_files/smplx/SMPLX_${g}.npz" "data/body_models/smplx/SMPLX_${g}.npz" 2>/dev/null
done
for g in NEUTRAL MALE FEMALE; do
  ln -sf "$REPO/vendor/smpl_extract/SMPL_${g}.pkl" "data/body_models/smpl/SMPL_${g}.pkl" 2>/dev/null
done

LOG "import sanity"
python -c "import torch; print('torch',torch.__version__,'cuda',torch.cuda.is_available())" 2>&1 | tail -1
python -c "import smplx, trimesh; print('smplx ok')" 2>&1 | tail -1

LOG "DONE — REMAINING (gated): shapy_data.zip from https://shapy.is.tue.mpg.de"
echo "  place trained_models/, expose_release/, utility_files/ under ~/shapy/data/"
echo "SHAPY_SETUP_DONE"
