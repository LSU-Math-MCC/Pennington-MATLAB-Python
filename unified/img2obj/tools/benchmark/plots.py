"""Focused single-image benchmark figure: SOURCE photo | textured A-pose | face close-up |
abdomen-contour overlay | metrics. For brutally-honest comparison of face (jawline/nose) and
NON-GENERIC abdomen contours against the source. Reads runs/bench_single/."""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = os.path.dirname(os.path.abspath(__file__))
while REPO != os.path.dirname(REPO) and not os.path.exists(os.path.join(REPO, "pyproject.toml")):
    REPO = os.path.dirname(REPO)
R = REPO
B = f"{R}/runs/bench_single"
SRC = sys.argv[1] if len(sys.argv) > 1 else f"{R}/datasets/SSP-3D/ssp_3d/images/bodybuilding_vid_009_clip_000_person_001_frame_000029.png"

panels = [("source", SRC),
          ("textured A-pose", f"{B}/blender_studio.png"),
          ("face close-up", f"{B}/blender_face.png"),
          ("silhouette/contour fit", f"{B}/silhouette_overlay.png")]
fig, axes = plt.subplots(1, len(panels), figsize=(20, 6))
for ax, (title, p) in zip(axes, panels):
    if os.path.exists(p):
        ax.imshow(np.asarray(Image.open(p).convert("RGB")))
    else:
        ax.text(0.5, 0.5, "(missing)\n" + os.path.basename(p), ha="center", va="center")
    ax.set_title(title, fontsize=11); ax.axis("off")
fig.suptitle("Single-image benchmark — s7 (Cape Town): face + non-generic abdomen", fontsize=14)
fig.savefig(f"{R}/runs/BENCH_s7.png", dpi=100, bbox_inches="tight")
print("saved runs/BENCH_s7.png")
