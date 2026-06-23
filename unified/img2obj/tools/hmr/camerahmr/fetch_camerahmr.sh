#!/bin/bash
# Non-interactive CameraHMR weight fetch. Reads creds from env so nothing is hard-coded.
# Usage:  set CAMUSER and CAMPASS in your shell, then run this script.
set -e
urle () { local LANG=C i x; for (( i=0; i<${#1}; i++ )); do x="${1:i:1}"; [[ "$x" == [a-zA-Z0-9.~-] ]] && echo -n "$x" || printf '%%%02X' "'$x"; done; echo; }
U=$(urle "$CAMUSER"); P=$(urle "$CAMPASS")
cd ~/CameraHMR
mkdir -p data/models/SMPL data/pretrained-models
base='https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile='
get () { wget --post-data "username=$U&password=$P" "${base}$1" -O "$2" --no-check-certificate --continue -q --show-progress; }
get SMPL_NEUTRAL.pkl                  data/models/SMPL/SMPL_NEUTRAL.pkl
get cam_model_cleaned.ckpt            data/pretrained-models/cam_model_cleaned.ckpt
get camerahmr_checkpoint_cleaned.ckpt data/pretrained-models/camerahmr_checkpoint_cleaned.ckpt
get model_final_f05665.pkl            data/pretrained-models/model_final_f05665.pkl
get smpl_mean_params.npz              data/smpl_mean_params.npz
echo "=== sizes (a few hundred MB total; ckpt should be >100MB, not a tiny HTML error page) ==="
ls -lh data/pretrained-models/ data/models/SMPL/ data/smpl_mean_params.npz
