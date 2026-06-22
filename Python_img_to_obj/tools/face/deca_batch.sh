#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate camerahmr
cd "$(d="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"; while [ "$d" != / ] && [ ! -f "$d/pyproject.toml" ]; do d="$(dirname "$d")"; done; echo "$d")"
for S in ssp3d_bodybuilder; do
  python tools/face/deca_full.py "datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_000029.png" "runs/decafull_$S" 2>&1 | grep -aiE 'DECA done|Error' | tail -1
done
python - <<'PY'
import numpy as np, imageio
from PIL import Image, ImageDraw
rows=[]
for s in ['ssp3d_bodybuilder']:
    a=np.array(Image.open(f'runs/decafull_{s}_input.png').convert('RGB').resize((300,330)))
    b=np.array(Image.open(f'runs/decafull_{s}_render.png').convert('RGB').resize((300,330)))
    pair=np.concatenate([a,b],axis=1); im=Image.fromarray(pair); ImageDraw.Draw(im).text((6,6),s+' photo|render',fill=(220,0,0)); rows.append(np.array(im))
imageio.imwrite('runs/DECA_COMPARE.png',np.concatenate(rows,axis=0)); print('grid ok')
PY
echo DECA_BATCH_DONE
