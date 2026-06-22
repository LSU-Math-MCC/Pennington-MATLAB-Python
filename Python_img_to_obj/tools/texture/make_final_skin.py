"""Montage the 5 skin-only textured A-poses into runs/FINAL_skin.png. Coverage is read from
each run's uv_report.json (REAL value, never hardcoded)."""
import json
import os
import numpy as np
from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.abspath(__file__))
while REPO != os.path.dirname(REPO) and not os.path.exists(os.path.join(REPO, "pyproject.toml")):
    REPO = os.path.dirname(REPO)
R = os.path.join(REPO, "runs")
tiles = []
for s in ["s1", "s2", "s3", "s4", "s5"]:
    p = f"{R}/uv_{s}_skin/blender_studio.png"
    rep = f"{R}/uv_{s}_skin/uv_report.json"
    cov = "?"
    if os.path.exists(rep):
        cov = f"{json.load(open(rep)).get('coverage_texels', 0) * 100:.0f}%"
    im = Image.open(p).convert("RGB").resize((360, 360))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 360, 22], fill=(20, 20, 24))
    d.text((6, 5), f"{s}  skin-coverage {cov}", fill=(120, 230, 160))
    tiles.append(np.asarray(im))
grid = np.concatenate(tiles, 1)
Image.fromarray(grid).save(f"{R}/FINAL_skin.png")
print("FINAL_skin.png", grid.shape)
