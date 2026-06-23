#!/bin/bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lhm

R="$(d="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"; while [ "$d" != / ] && [ ! -f "$d/pyproject.toml" ]; do d="$(dirname "$d")"; done; echo "$d")"
cd "$R"

ATLAS="${ATLAS:-512}"
SUBJECTS="${SUBJECTS:-ssp3d_bodybuilder}"
SSP_BODYBUILDER="$R/datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_*.png"
MODES="${MODES:-off face face-head}"
FACE_OCCLUSION_CLEAN="${FACE_OCCLUSION_CLEAN:-conservative}"
WITH_DECA_FACE="${WITH_DECA_FACE:-1}"
OUT_ROOT="${OUT_ROOT:-$R/runs/texture_eval_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_ROOT"

echo "TEXTURE_EVAL out=$OUT_ROOT atlas=$ATLAS subjects=[$SUBJECTS] modes=[$MODES] face_occlusion_clean=$FACE_OCCLUSION_CLEAN with_deca_face=$WITH_DECA_FACE"
for S in $SUBJECTS; do
  IMG="$SSP_BODYBUILDER"
  BETAS="$R/runs/camerahmr_${S}_smplx_betas.npy"
  for MODE in $MODES; do
    OUT="$OUT_ROOT/${S}_${MODE}"
    mkdir -p "$OUT"
    echo "==== bake $S mode=$MODE ===="
    python tools/texture/texture_uv_bake.py \
      --subject "$IMG" \
      --out "$OUT" \
      --betas "$BETAS" \
      --atlas "$ATLAS" \
      --face-repair "$MODE" \
      --face-occlusion-clean "$FACE_OCCLUSION_CLEAN" \
      2>&1 | tee "$OUT/bake.log" | grep -aiE 'texels written|coherent face|face repaired|coverage|UVBAKE_OK|Error' || true
    python tools/texture/render_texture_preview.py "$OUT/apose_textured_uv.glb" "$OUT" 2>&1 | tee "$OUT/render.log"
  done
  if [ "$WITH_DECA_FACE" = "1" ] && [ -f "$R/runs/decafull_${S}_verts.npy" ]; then
    BODY_MODE="face-head"
    if [ ! -f "$OUT_ROOT/${S}_${BODY_MODE}/apose_textured_uv.glb" ]; then
      BODY_MODE="$(echo "$MODES" | awk '{print $NF}')"
    fi
    OUT="$OUT_ROOT/${S}_deca"
    mkdir -p "$OUT"
    echo "==== deca hybrid $S body_mode=$BODY_MODE ===="
    python tools/face/integrate_deca_face.py "$S" \
      "$OUT_ROOT/${S}_${BODY_MODE}/apose_textured_uv.glb" \
      "$OUT/hybrid" \
      2>&1 | tee "$OUT/integrate.log" | grep -aiE 'integrated|Error' || true
    python tools/texture/render_texture_preview.py "$OUT/hybrid_body_deca_face.glb" "$OUT" \
      2>&1 | tee "$OUT/render.log"
  fi
done

python tools/texture/galleries.py eval "$OUT_ROOT"
echo "TEXTURE_EVAL_DONE $OUT_ROOT"
