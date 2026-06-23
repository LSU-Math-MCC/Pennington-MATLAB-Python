#!/bin/bash
# Run HaMeR (SOTA hand mesh recovery) on an image -> per-hand MANO meshes (finger-level).
set -e
REPO="$(d="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"; while [ "$d" != / ] && [ ! -f "$d/pyproject.toml" ]; do d="$(dirname "$d")"; done; echo "$d")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lhm
cd ~/hamer
# MANO into HaMeR's expected location
mkdir -p _DATA/data/mano
[ -f _DATA/data/mano/MANO_RIGHT.pkl ] || cp ~/LHM/pretrained_models/human_model_files/mano/MANO_RIGHT.pkl _DATA/data/mano/ 2>/dev/null || true
echo "=== install hamer (best-effort) ==="
python -c "import hamer" 2>/dev/null || pip install -e . 2>&1 | tail -3
echo "=== run demo ==="
IMG="${1:-$REPO/datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_000029.png}"
OUT="${2:-$REPO/runs/hamer_out}"
mkdir -p /tmp/hamer_in && cp "$IMG" /tmp/hamer_in/
python demo.py --img_folder /tmp/hamer_in --out_folder "$OUT" --batch_size=1 --save_mesh --full_frame 2>&1 | tail -8
echo "=== outputs ==="; ls "$OUT" 2>/dev/null | head
echo HAMER_RUN_DONE
