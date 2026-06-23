#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lhm
R="$(d="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"; while [ "$d" != / ] && [ ! -f "$d/pyproject.toml" ]; do d="$(dirname "$d")"; done; echo "$d")"
cd "$R"
S="${1:-ssp3d_bodybuilder}"
SUBJECT="$R/datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_*.png"
OUT="$R/runs/uv_${S}_hq"
python tools/texture/texture_uv_bake.py --subject "$SUBJECT" --out "$OUT" \
  --betas "$R/runs/camerahmr_${S}_smplx_betas.npy" --atlas 2048 2>&1 \
  | grep -aivE 'Warning|warn|FutureWarning|will be|hasattr|Setting' | tail -16
if [ -f "$R/runs/decafull_${S}_verts.npy" ]; then
  python tools/face/integrate_deca_face.py "$S" "$OUT/apose_textured_uv.glb" "$OUT/deca_hybrid" 2>&1 \
    | grep -aivE 'Warning|warn|FutureWarning|hasattr' | tail -4
  echo "FINAL_GLB $OUT/deca_hybrid_body_deca_face.glb"
else
  echo "FINAL_GLB $OUT/apose_textured_uv.glb"
fi
echo "BAKE_DONE $S"
