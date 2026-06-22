#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate camerahmr
python -c "import pytorch3d; print('pytorch3d already', pytorch3d.__version__)" 2>/dev/null && exit 0
echo "=== try prebuilt wheel (py310/torch2.0.0/cu118) ==="
pip install --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt200/download.html 2>&1 | tail -4
python -c "import pytorch3d; print('pytorch3d', pytorch3d.__version__)" 2>/dev/null && exit 0
echo "=== prebuilt failed -> build from source w/ env CUDA 11.8 + gcc 11 ==="
export CUDA_HOME="$CONDA_PREFIX"; export PATH="$CONDA_PREFIX/bin:$PATH"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export TORCH_CUDA_ARCH_LIST="8.6"; export NVCC_PREPEND_FLAGS="-ccbin $CC"
HDR=$(find "$CONDA_PREFIX" -name cusolverDn.h 2>/dev/null | head -1); export CPATH="$(dirname "$HDR"):$CONDA_PREFIX/include:$CPATH"
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable" 2>&1 | grep -aiE 'building|successfully|error|created wheel' | tail -8
python -c "import pytorch3d; print('pytorch3d', pytorch3d.__version__)"
echo PYTORCH3D_DONE
