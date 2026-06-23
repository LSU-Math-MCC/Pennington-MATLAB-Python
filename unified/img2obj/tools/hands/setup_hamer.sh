#!/bin/bash
# HaMeR: SOTA single-image hand mesh recovery (MANO, finger-level), 2024, geopavlakos.
# Public auto-download weights. Gives the finger-level detail to fuse into SMPL-X MANO hands.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
cd ~
[ -d hamer ] || git clone --recursive https://github.com/geopavlakos/hamer.git
cd hamer
echo "=== fetch HaMeR data/checkpoints (~1-2GB, public) ==="
[ -f _DATA/hamer_ckpts/checkpoints/hamer.ckpt ] || bash fetch_demo_data.sh 2>&1 | tail -6
echo "=== MANO check (HaMeR needs MANO_RIGHT.pkl) ==="
find ~ -iname 'MANO_RIGHT.pkl' 2>/dev/null | head -3
ls _DATA/ 2>/dev/null
echo HAMER_FETCH_DONE
