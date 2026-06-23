#!/bin/bash
# Fix the blade_env CUDA toolchain to a CONSISTENT 11.8 (the piecemeal `-c nvidia cuda-toolkit=11.8.0`
# left cuda-nvcc at 11.8 but cudart/cusparse/nvrtc at 12.9 -> link/ABI mismatch with torch cu118).
# The version-LABEL channel pins every component to 11.8. Then rebuild the CUDA custom ops.
#
# Run (WSL):  bash tools/hmr/blade/fix_blade_cuda.sh 2>&1 | tee ~/blade/cuda_fix.log
BLADE=~/blade
source ~/miniconda3/etc/profile.d/conda.sh
conda activate blade_env
log(){ echo "=== [$(date +%H:%M:%S)] $* ==="; }

# undo the earlier bad symlinks that shadowed 11.8 headers with 12.x ones from targets/
log "remove stale header symlinks in include/"
find "$CONDA_PREFIX/include" -maxdepth 1 -type l -delete 2>/dev/null

log "install consistent CUDA 11.8 toolkit (label channel) -- downgrades the 12.9 soup"
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# consistent toolchain env (matches setup_blade_env.sh)
export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
CINC="$CONDA_PREFIX/targets/x86_64-linux/include"
CLIB="$CONDA_PREFIX/targets/x86_64-linux/lib"
export CPATH="$CONDA_PREFIX/include:$CINC"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CLIB:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CLIB:$LD_LIBRARY_PATH"

log "sanity: nvcc + cudart version"
nvcc --version | tail -2
ls "$CONDA_PREFIX"/lib/libcudart.so* "$CLIB"/libcudart.so* 2>/dev/null | head
test -f "$CINC/cusparse.h" && echo "cusparse.h present" || echo "cusparse.h MISSING"

log "rebuild mmcv ops (non-editable)"
( cd "$BLADE/mmcv" && rm -rf build && MMCV_WITH_OPS=1 pip install --no-build-isolation --no-warn-conflicts . )
log "rebuild aios ops + torch-trust-ncg"
( cd "$BLADE/aios_repo/models/aios/ops" && python setup.py build install )
( cd "$BLADE/torch-trust-ncg" && python setup.py install )

log "VERIFY"
python - <<'PY'
import importlib
for m in ["torch","pytorch3d","mmcv","smplx","mediapipe","mmpose","mmdet"]:
    try: importlib.import_module(m); print("OK  ", m)
    except Exception as e: print("FAIL", m, "->", type(e).__name__, str(e)[:120])
try:
    from mmcv.ops import nms; import mmcv; print("OK   mmcv.ops  (mmcv", mmcv.__version__, ")")
except Exception as e: print("FAIL mmcv.ops ->", e)
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
PY
log "CUDA_FIX_DONE"
