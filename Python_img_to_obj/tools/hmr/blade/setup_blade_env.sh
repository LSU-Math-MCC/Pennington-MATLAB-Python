#!/bin/bash
# Idempotent BLADE (NVlabs/blade, CVPR2025) install for a CUDA-12 host (RTX 3080 Ti).
# Builds a SELF-CONTAINED CUDA 11.8 + gcc-11 toolchain inside the conda env (system CUDA untouched),
# then the custom ops (mmcv, sapiens, aios, torch-trust-ncg). Follows docs/INSTALL_CUDA13.md +
# README Quick Install. Bodies are already symlinked; weights are fetched by fetch_blade.sh.
#
# Run (WSL):  bash tools/hmr/blade/setup_blade_env.sh 2>&1 | tee ~/blade/setup.log
# NB: no `set -u` -- conda's own activate/deactivate hooks reference unbound vars and would abort.
BLADE=~/blade
source ~/miniconda3/etc/profile.d/conda.sh
log(){ echo "=== [$(date +%H:%M:%S)] $* ==="; }

# ---- env ----
if ! conda env list | grep -q '/blade_env$'; then
  log "create blade_env (py3.9.19)"
  conda create -y -n blade_env python=3.9.19
fi
conda activate blade_env

# ---- self-contained CUDA 11.8 + gcc11 toolchain (no system changes) ----
if [ ! -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" ]; then
  log "install gcc11"
  conda install -y -c conda-forge gcc_linux-64=11 gxx_linux-64=11 sysroot_linux-64=2.17
fi
if [ ! -x "$CONDA_PREFIX/bin/nvcc" ]; then   # guard on nvcc, NOT gcc -- they install separately
  # version-LABEL channel pins EVERY cuda component to 11.8 (plain `-c nvidia cuda-toolkit=11.8.0`
  # leaves cudart/cusparse/nvrtc at 12.x -> link/ABI mismatch with torch cu118).
  log "install consistent CUDA 11.8 toolkit (label channel)"
  conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
fi
export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export NVCCFLAGS="--compiler-bindir=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
# conda's cuda toolkit keeps headers/libs under targets/<arch>/{include,lib}; add both so the
# ops build finds cusparse.h and links -lcudart.
_CINC="$CONDA_PREFIX/targets/x86_64-linux/include"; _CLIB="$CONDA_PREFIX/targets/x86_64-linux/lib"

# thrust/cub headers (needed by some ops builds)
if [ ! -d "$CONDA_PREFIX/include/thrust" ]; then
  log "install thrust/cub 1.16 headers"
  rm -rf /tmp/thrust-1.16
  git clone --recursive --branch 1.16.0 https://github.com/NVIDIA/thrust.git /tmp/thrust-1.16
  rsync -a /tmp/thrust-1.16/thrust "$CONDA_PREFIX/include/"
  rsync -a /tmp/thrust-1.16/dependencies/cub/cub "$CONDA_PREFIX/include/"
fi
export CPATH="$CONDA_PREFIX/include:$_CINC${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:$_CINC${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$_CLIB${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$_CLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ---- python deps (torch 2.0.1+cu118, pytorch3d, the long pip list) ----
if ! python -c "import torch" 2>/dev/null; then
  log "pip: torch 2.0.1+cu118 + pytorch3d"
  pip install tqdm torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
  pip install fvcore iopath numpy==1.24.4 wandb
  pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt201/download.html
fi
log "pip: remaining python deps"
pip install -U pip setuptools wheel        # build backend (pkg_resources) for --no-build-isolation
pip install --no-warn-conflicts matplotlib==3.8.4 colorama requests huggingface-hub safetensors pillow six click openxlab
pip install --no-warn-conflicts scipy munkres tqdm cython fsspec yapf==0.40.1 packaging omegaconf ipdb ftfy regex
pip install --no-build-isolation chumpy || echo "chumpy build skipped (old setup.py; only needed for some SMPL .pkl loads)"
pip install --no-warn-conflicts json_tricks terminaltables modelindex prettytable albumentations
pip install --no-warn-conflicts smplx==0.1.28 debugpy numba yacs scikit-learn filterpy h5py trimesh scikit-image tensorboardx pyrender torchgeometry joblib boto3 easydict pycocotools colormap pytorch-transformers pickle5 plyfile timm pyglet future tensorboard cdflib ftfy einops
pip install --no-warn-conflicts numpy==1.23.1 "mediapipe==0.10.14" xtcocotools shapely terminaltables  # 0.10.14=last with mp.solutions AND protobuf4; xtcocotools/shapely are mmpose/mmdet deps skipped by --no-deps

# ---- project + custom ops ----
# NON-editable (not -e): the env's legacy setuptools build backend lacks the PEP660
# build_editable hook, so editable installs error out. --no-build-isolation so mmcv's CUDA-ops
# build sees the env's torch + nvcc. --no-deps on the mm* forks so they don't pull mmcv from PyPI
# (without ops) and clobber the build we just compiled.
log "build mmcv ops (non-editable)"
( cd "$BLADE/mmcv" && MMCV_WITH_OPS=1 pip install --no-build-isolation --no-warn-conflicts . -v )
log "build sapiens packages (non-editable, --no-deps)"
for p in engine pretrain pose det seg; do ( cd "$BLADE/sapiens/$p" && pip install --no-build-isolation --no-deps --no-warn-conflicts . ); done
( cd "$BLADE/sapiens" && pip install --no-build-isolation --no-deps --no-warn-conflicts . )
( cd "$BLADE" && pip install --no-build-isolation --no-deps --no-warn-conflicts . )
pip install --no-warn-conflicts ffmpeg astropy easydev pandas rtree vedo codecov flake8 interrogate isort pytest surrogate xdoctest setuptools loguru open3d omegaconf
log "build aios ops + torch-trust-ncg"
( cd "$BLADE/aios_repo/models/aios/ops" && python setup.py build install )
( cd "$BLADE/torch-trust-ncg" && python setup.py install )
pip install --no-warn-conflicts numpy==1.23.1

# ---- persist toolchain + EGL into the env's activate hooks ----
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/10_nvrtc11.sh" <<'EOS'
NVRTC_LIB_DIR="$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cuda_nvrtc/lib"
export _OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$NVRTC_LIB_DIR:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EOS
cat > "$CONDA_PREFIX/etc/conda/activate.d/20_cuda11_toolchain.sh" <<'EOS'
export CUDA_HOME="$CONDA_PREFIX"
if [ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" ]; then
  export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
  export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
  export CUDAHOSTCXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
fi
export PYGLET_HEADLESS=True
export PYOPENGL_PLATFORM=egl
[ -f "$CONDA_PREFIX/lib/libEGL.so" ] || ln -sf "$CONDA_PREFIX/lib/libEGL.so.1" "$CONDA_PREFIX/lib/libEGL.so" 2>/dev/null || true
EOS

log "VERIFY"
python -c "import torch,pytorch3d; print('torch',torch.__version__,'cuda',torch.cuda.is_available())" || true
python -c "import mmcv; from mmcv.ops import nms; print('mmcv ops OK', mmcv.__version__)" || echo "mmcv ops FAILED"
log "BLADE_SETUP_DONE"
