#!/bin/bash
# Fetch BLADE checkpoint + preprocessing backbones into ~/blade/pretrained.
# BLADE / Depth-Anything-V2-metric / AiOS are public on HF; facebook/sapiens-* are GATED
# (accept the license on the HF page + `huggingface-cli login` first, else those 4 lines 403).
# Bodies are already symlinked by hand; we do NOT re-download SMPL/SMPL-X.
#
# Run (WSL):  bash tools/hmr/blade/fetch_blade.sh
BLADE=~/blade
source ~/miniconda3/etc/profile.d/conda.sh
conda activate blade_env || { echo "blade_env missing -- run setup_blade_env.sh first"; exit 1; }
cd "$BLADE"
command -v hf >/dev/null 2>&1 || pip install -q "huggingface_hub[cli]"
dl(){ # repo  file  dest_dir
  if compgen -G "$2/$(basename $2)" >/dev/null 2>&1; then echo "skip $2 (have it)"; return; fi
  echo "=== hf download $1 $2 -> $3 ==="
  hf download "$1" "$2" --local-dir "$3" || echo "!! FAILED $1/$2 (gated? run huggingface-cli login / accept license)"
}
# BLADE checkpoint (public). McMvMc/BLADE, fallback McMvMc/BLADE_backup
if [ ! -f pretrained/epoch_2.pth ]; then
  hf download McMvMc/BLADE epoch_2.pth --local-dir pretrained \
    || hf download McMvMc/BLADE_backup epoch_2.pth --local-dir pretrained \
    || echo "!! BLADE epoch_2.pth FAILED"
fi
dl depth-anything/Depth-Anything-V2-Metric-Hypersim-Large depth_anything_v2_metric_hypersim_vitl.pth pretrained/model_init_weights
dl ttxskk/AiOS aios_checkpoint.pth pretrained/model_init_weights
# GATED (facebook): accept license + login or these 403
dl facebook/sapiens-pose-bbox-detector rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth pretrained/rtmpose
dl facebook/sapiens-pose-1b sapiens_1b_goliath_best_goliath_AP_639.pth pretrained/pose
echo "=== pretrained tree ==="
find pretrained -maxdepth 2 -type f ! -name ".gitkeep" -exec ls -lh {} \;
