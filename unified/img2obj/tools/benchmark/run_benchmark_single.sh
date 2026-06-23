#!/bin/bash
# Focused single-image benchmark: face (jawline/nose) + NON-GENERIC abdomen contours from ONE
# image. Usage: bash tools/benchmark/run_benchmark_single.sh <image_path>
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lhm
M="$(d="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"; while [ "$d" != / ] && [ ! -f "$d/pyproject.toml" ]; do d="$(dirname "$d")"; done; echo "$d")"
cd "$M"; export PYTHONPATH=src
export IMG="${1:-$M/datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_000029.png}"
OUT="$M/runs/bench_single"
mkdir -p "$OUT"
echo "=== 1. silhouette-fit betas ==="
python tools/geometry/fit_silhouette.py --subject "$IMG" --out "$OUT" --gender female || true
echo "=== 2. CLIP attribute de-bias (uses prior/silhouette) ==="
# fuse prior+fitted then CLIP-shape needs a fit dir; reuse OUT
python tools/geometry/fuse_betas.py "$OUT" || true
python - <<'PY' || true
import sys, numpy as np, os, torch
sys.path[:0]=['tools/geometry','tools/anthro']; import clip_shape as CS, shapy_measure as SM, lhm_anthropometry as A, smplx
out=os.path.join(os.getcwd(), "runs", "bench_single")
img=os.environ["IMG"]
bmi,_=CS.clip_build_bmi([img]); print("CLIP BMI", round(bmi,1))
model=smplx.create(A.HUMAN_MODELS,model_type="smplx",gender="female",num_betas=10)
faces=model.faces.astype(np.int64)
sil=np.load(out+"/fused_betas.npy")[:10].astype(np.float32)
rel=float(np.clip(1.4-np.abs(sil).max()/2.0,0.1,1.0))
with torch.no_grad(): vref=model(betas=torch.tensor(sil).unsqueeze(0)).vertices[0].numpy()
y0,y1=vref[:,1].min(),vref[:,1].max(); h=y1-y0
gref=np.array([SM.plane_perimeter(vref,faces,y0+(y1-y0)*(0.05+0.87*(k+0.5)/16)) for k in range(16)]); gref/=gref.sum()+1e-6
b=CS.fit_betas(model,faces,bmi*h*h,h,gref,rel,x0=sil)
np.save(out+"/fused_betas.npy", b)   # CLIP-debiased shape for the bake
print("CLIP-debiased betas[:4]", np.round(b[:4],2))
PY
echo "=== 3. textured A-pose (face landmark-aligned, single image) ==="
python tools/texture/texture_uv_bake.py --subject "$IMG" --out "$OUT" \
  --betas "$OUT/fused_betas.npy" --skin-only 2>&1 | tail -6
echo "=== 4. abdomen contours + measurements ==="
python tools/anthro/shapy_measure.py bench_single 2>&1 | tail -3 || true
echo BENCH_DONE
