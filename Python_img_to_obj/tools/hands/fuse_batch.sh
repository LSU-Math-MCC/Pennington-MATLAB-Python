#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lhm
cd "$(d="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"; while [ "$d" != / ] && [ ! -f "$d/pyproject.toml" ]; do d="$(dirname "$d")"; done; echo "$d")"
for s in s2 s3 s4; do
  echo "==== $s ===="
  python tools/hands/fuse_hands_generic.py "runs/uv_$s/apose_textured_uv.glb" "runs/hamer_$s" "runs/hamer_$s/composite" 2>&1 \
    | grep -aiE 'placed|rendered|skip|Error' | tail -4
done
