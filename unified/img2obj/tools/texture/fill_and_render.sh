#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lhm
cd "$(d="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"; while [ "$d" != / ] && [ ! -f "$d/pyproject.toml" ]; do d="$(dirname "$d")"; done; echo "$d")"
for S in s1 s2 s3 s4 s5; do
  python tools/texture/fill_grey_texture.py "$S" 2>&1 | grep -aiE 'grey texels|overwrote|no texture' | tail -2
  python tools/texture/render_textured_with_hands.py "$S" 2>&1 | grep -aiE 'rendered' | tail -1
done
echo FILL_RENDER_DONE
