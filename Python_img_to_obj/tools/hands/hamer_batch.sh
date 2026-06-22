#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lhm
cd "$(d="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"; while [ "$d" != / ] && [ ! -f "$d/pyproject.toml" ]; do d="$(dirname "$d")"; done; echo "$d")"
for s in ssp3d_bodybuilder; do
  echo "==== $s ===="
  python tools/hands/run_hamer_slim.py "datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_000029.png" "runs/hamer_$s" 2>&1 \
    | grep -aiE 'hand boxes|saved|DONE|Error|no hands' | tail -4
done
