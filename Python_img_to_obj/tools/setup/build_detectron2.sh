#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate camerahmr
echo "=== locate cusolverDn.h ==="
HDR=$(find "$CONDA_PREFIX" -name cusolverDn.h 2>/dev/null | head -1)
echo "cusolverDn.h -> $HDR"
INCDIR=$(dirname "$HDR")
# Some CUDA headers live under targets/x86_64-linux/include; make sure nvcc/gcc see them.
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
CC11="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
CXX11="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CC="$CC11" CXX="$CXX11"
export TORCH_CUDA_ARCH_LIST="8.6"
export NVCC_PREPEND_FLAGS="-ccbin $CC11"
export CPATH="$INCDIR:$CONDA_PREFIX/include:$CPATH"
export CPLUS_INCLUDE_PATH="$INCDIR:$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH"
echo "CPATH=$CPATH"
echo "=== build detectron2 ==="
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git' 2>&1 | grep -aiE 'building wheel|successfully built|successfully installed|error:|created wheel|fatal error|unsupported' | tail -12
python -c "import detectron2; from detectron2 import _C; print('detectron2', detectron2.__version__, 'CUDA ops OK')"
echo DETECTRON2_DONE
