#!/usr/bin/env bash
# Set up LHM (Large Animatable Human Model) inside WSL2 with a matched conda env.
# Logs stage markers so progress/errors are easy to follow from Windows.
# Idempotent-ish: re-running skips steps already done.
set -u
LOG() { echo "===STAGE=== $* ($(date +%H:%M:%S))"; }
FAIL() { echo "===FAILED=== $*"; }

HOME_DIR="$HOME"
MC="$HOME_DIR/miniconda3"
LHM_DIR="$HOME_DIR/LHM"

# 1. miniconda
if [ ! -x "$MC/bin/conda" ]; then
  LOG "installing miniconda"
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh || { FAIL miniconda-download; exit 1; }
  bash /tmp/mc.sh -b -p "$MC" || { FAIL miniconda-install; exit 1; }
else
  LOG "miniconda present"
fi
source "$MC/etc/profile.d/conda.sh"

# accept conda channel Terms of Service (required by conda >=24.x)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# 2. env
if ! conda env list | grep -q "^lhm "; then
  LOG "creating conda env lhm (python 3.10)"
  conda create -n lhm python=3.10 -y || { FAIL conda-create; exit 1; }
else
  LOG "conda env lhm present"
fi
conda activate lhm

# 3. clone LHM
if [ ! -d "$LHM_DIR/.git" ]; then
  LOG "cloning LHM"
  git clone --depth 1 https://github.com/aigc3d/LHM.git "$LHM_DIR" || { FAIL git-clone; exit 1; }
else
  LOG "LHM clone present; pulling"
  git -C "$LHM_DIR" pull --ff-only || true
fi
cd "$LHM_DIR"

# 4. core torch (cu121, matches LHM pin) before the heavy source builds
LOG "installing torch cu121"
pip install --quiet torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121 || { FAIL torch-install; exit 1; }

LOG "torch sanity check"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_built", torch.version.cuda, "avail", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
PY

# 5. run LHM's own installer (compiles pytorch3d, diff-gaussian-rasterization, simple-knn, sam2, basicsr)
LOG "running LHM install_cu121.sh (source builds; this is slow)"
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.6"   # RTX 3080 Ti (Ampere) = sm_86
bash install_cu121.sh
echo "===INSTALL_SH_EXIT=== $?"

# 6. download LHM-MINI weights
LOG "downloading LHM-MINI weights"
pip install --quiet huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download
d = snapshot_download(repo_id='3DAIGC/LHM-MINI', cache_dir='./pretrained_models/huggingface')
print("weights at", d)
PY
echo "===WEIGHTS_EXIT=== $?"

LOG "DONE"
