#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lhm
cd "$(d="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"; while [ "$d" != / ] && [ ! -f "$d/pyproject.toml" ]; do d="$(dirname "$d")"; done; echo "$d")"
for S in s1 s2 s3 s4 s5; do
  python tools/texture/render_textured_with_hands.py "$S" 2>&1 | grep -aiE 'rendered|Error' | tail -1
done
echo RENDER_ALL_DONE
