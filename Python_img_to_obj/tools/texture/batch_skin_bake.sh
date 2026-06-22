#!/bin/bash
# Skin-only textured A-pose bake for all subjects (honest body appearance, no garments/bg).
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lhm
M="$(d="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"; while [ "$d" != / ] && [ ! -f "$d/pyproject.toml" ]; do d="$(dirname "$d")"; done; echo "$d")"
cd "$M"
export PYTHONPATH=src
subjects=("$@")
if [ ${#subjects[@]} -eq 0 ]; then
  subjects=(ssp3d_bodybuilder)
fi
for s in "${subjects[@]}"; do
  echo "=== $s ==="
  python tools/texture/texture_uv_bake.py \
    --subject "$M/datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_*.png" \
    --out "$M/runs/uv_${s}_skin" \
    --betas "$M/runs/fit_${s}/fused_betas.npy" \
    --skin-only 2>&1 | tail -8
done
echo ALLBAKE_DONE
