#!/bin/bash
# Back up the heavy silhouette-fused betas, promote the CLIP-corrected betas as the uniform shape.
cd "$(d="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"; while [ "$d" != / ] && [ ! -f "$d/pyproject.toml" ]; do d="$(dirname "$d")"; done; echo "$d")"
for s in s1 s2 s3 s4 s5; do
  d="runs/fit_$s"
  [ -f "$d/clip_betas.npy" ] || { echo "no clip_betas for $s"; continue; }
  [ -f "$d/fused_betas_heavy.npy" ] || cp "$d/fused_betas.npy" "$d/fused_betas_heavy.npy"
  cp "$d/clip_betas.npy" "$d/fused_betas.npy"
  echo "promoted $s"
done
echo PROMOTE_DONE
