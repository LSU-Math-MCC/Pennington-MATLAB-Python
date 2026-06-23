#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
R="$(d="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"; while [ "$d" != / ] && [ ! -f "$d/pyproject.toml" ]; do d="$(dirname "$d")"; done; echo "$d")"
cd "$R"
subjects=("$@")
if [ ${#subjects[@]} -eq 0 ]; then
  subjects=(ssp3d_bodybuilder)
fi
for S in "${subjects[@]}"; do
  echo "==== FLAME face $S ===="
  conda activate camerahmr
  # DECA geometry from the ORIGINAL best-frontal frame (smaller -> fast); texture from enhanced in bake
  [ -f "runs/flame_${S}_verts.npy" ] || python tools/face/deca_flame.py "datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_000029.png" "runs/flame_${S}_verts.npy" 2>&1 | grep -aiE 'FLAME verts|Error' | tail -1
  conda activate lhm
  python tools/face/make_flame_disp.py "$S" 2>&1 | grep -aiE 'FLAME-welded|Error' | tail -1
  python tools/texture/texture_uv_bake.py --subject "$R/runs/enhanced_$S.jpg" --out "$R/runs/uv_${S}_1v" \
    --betas "$R/runs/camerahmr_${S}_smplx_betas.npy" --atlas 2048 --face-disp "$R/runs/flame_disp_$S.npy" 2>&1 \
    | grep -aiE 'UVBAKE_OK|Error' | tail -1
  python tools/texture/fill_grey_texture.py "$S" 2>&1 | grep -aiE 'dark-patch' | tail -1
done
echo FLAME_REBAKE_DONE
