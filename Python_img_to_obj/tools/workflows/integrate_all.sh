#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lhm
cd "$(d="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"; while [ "$d" != / ] && [ ! -f "$d/pyproject.toml" ]; do d="$(dirname "$d")"; done; echo "$d")"
for S in s1 s2 s3 s4 s5; do
  python tools/face/integrate_deca_face.py "$S" 2>&1 | grep -aiE 'integrated|Error' | tail -1
done
python - <<'PY'
import numpy as np, imageio
from PIL import Image, ImageDraw
rows=[]
for s in ['s1','s2','s3','s4','s5']:
    a=np.array(Image.open(f'runs/decafull_{s}_input.png').convert('RGB').resize((300,330)))
    b=np.array(Image.open(f'runs/final_{s}_face.png').convert('RGB').resize((300,330)))
    pair=np.concatenate([a,b],axis=1); im=Image.fromarray(pair); ImageDraw.Draw(im).text((6,6),s+' photo|on-body',fill=(220,0,0)); rows.append(np.array(im))
imageio.imwrite('runs/FINAL_FACES.png',np.concatenate(rows,axis=0)); print('grid ok')
PY
echo INTEGRATE_DONE
