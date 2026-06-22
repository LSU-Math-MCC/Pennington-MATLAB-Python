#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lhm
R="$(d="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"; while [ "$d" != / ] && [ ! -f "$d/pyproject.toml" ]; do d="$(dirname "$d")"; done; echo "$d")"
cd "$R"
for S in ssp3d_bodybuilder; do
  echo "==== bake $S ===="
  python tools/texture/texture_uv_bake.py --subject "$R/datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_000029.png" \
    --out "$R/runs/uv_${S}_1v" --betas "$R/runs/camerahmr_${S}_smplx_betas.npy" --atlas 2048 2>&1 \
    | grep -aiE 'texels written|coverage|symmetry|UVBAKE_OK|Error' | tail -4
done
echo BAKE_BATCH_DONE
