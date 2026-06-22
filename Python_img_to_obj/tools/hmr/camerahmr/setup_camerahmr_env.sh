#!/bin/bash
# Full official CameraHMR env per their ReadMe — no shortcuts (detectron2 built from source).
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda env list | grep -q camerahmr || conda create -y -n camerahmr python=3.10
conda activate camerahmr
cd ~/CameraHMR
# torch already installed (2.0.0+cu118). Build detectron2 against it WITHOUT isolation so torch is visible.
echo "=== detectron2 (no-build-isolation, CUDA ops) ==="
pip install ninja 2>&1 | tail -1
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git' 2>&1 | tail -6
echo "=== rest of requirements (detectron2 line removed) ==="
grep -v 'detectron2' requirements.txt > /tmp/cam_reqs.txt
pip install -r /tmp/cam_reqs.txt 2>&1 | tail -6
echo "=== verify imports ==="
python -c "import torch, detectron2, smplx, pytorch_lightning; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| detectron2', detectron2.__version__)"
echo CAMERAHMR_ENV_DONE
