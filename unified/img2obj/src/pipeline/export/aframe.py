"""Self-contained A-Frame viewer generation.

Point data is embedded inline in the HTML so the viewer works when opened directly
from the local filesystem (file://) without a web server. A-Frame + orbit-controls
load from CDN; if offline, the scene/axes still render and points are built via a
small custom THREE.Points component.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..types import SplatCloud

_AFRAME = "https://aframe.io/releases/1.5.0/aframe.min.js"
_ORBIT = "https://unpkg.com/aframe-orbit-controls@1.3.2/dist/aframe-orbit-controls.min.js"


def _downsample(points, colors, max_points):
    n = points.shape[0]
    if max_points and n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
        return points[idx], (colors[idx] if colors is not None else None)
    return points, colors


def write_viewer(
    out_dir,
    splats: SplatCloud | None = None,
    points: np.ndarray | None = None,
    colors: np.ndarray | None = None,
    joints: dict | None = None,
    label: str = "subject",
    title: str = "3DGS Canonical Viewer",
    extra_links: list | None = None,
    max_points: int = 60000,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if splats is not None:
        points = splats.centers
        colors = splats.colors
    if points is None:
        points = np.zeros((0, 3))
    points = np.asarray(points, dtype=np.float64)
    if colors is None:
        colors = np.full((points.shape[0], 3), 0.7)
    colors = np.asarray(colors, dtype=np.float64)
    if colors.size and colors.max() > 1.01:
        colors = colors / 255.0

    points, colors = _downsample(points, colors, max_points)

    pts_flat = [round(float(v), 5) for v in points.reshape(-1)]
    col_flat = [round(float(v), 4) for v in colors.reshape(-1)] if colors.size else []

    joints_list = []
    if joints:
        for name, v in joints.items():
            p = v[:3] if not hasattr(v, "shape") else (v[:3, 3] if v.shape == (4, 4) else v[:3])
            joints_list.append({"name": name, "p": [round(float(x), 4) for x in np.asarray(p).reshape(-1)[:3]]})

    links_html = ""
    if extra_links:
        items = "".join(f'<li><a href="{l["href"]}">{l["label"]}</a></li>' for l in extra_links)
        links_html = f'<div id="links"><b>Reconstructions</b><ul>{items}</ul></div>'

    data = {"points": pts_flat, "colors": col_flat, "joints": joints_list}
    data_json = json.dumps(data)

    html = _TEMPLATE.format(
        aframe=_AFRAME, orbit=_ORBIT, title=title, label=label,
        data_json=data_json, links_html=links_html,
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    return out_dir / "index.html"


_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="{aframe}"></script>
<script src="{orbit}"></script>
<style>
  body {{ margin:0; font-family: system-ui, sans-serif; }}
  #hud {{ position:fixed; top:8px; left:8px; z-index:10; background:rgba(0,0,0,.6);
         color:#fff; padding:8px 12px; border-radius:6px; font-size:13px; max-width:280px; }}
  #links a {{ color:#9cf; }}
  #links ul {{ margin:4px 0; padding-left:18px; }}
</style>
</head>
<body>
<div id="hud">
  <div><b>{title}</b></div>
  <div>subject: {label}</div>
  <div id="count"></div>
  {links_html}
</div>
<script>
const SCENE_DATA = {data_json};

AFRAME.registerComponent('point-cloud', {{
  init: function () {{
    const d = SCENE_DATA;
    const n = d.points.length / 3;
    document.getElementById('count').textContent = n + ' points';
    const geom = new THREE.BufferGeometry();
    const pos = new Float32Array(d.points);
    geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    if (d.colors && d.colors.length === d.points.length) {{
      const col = new Float32Array(d.colors);
      geom.setAttribute('color', new THREE.BufferAttribute(col, 3));
    }}
    const mat = new THREE.PointsMaterial({{ size: 0.015, vertexColors: d.colors.length>0 }});
    if (d.colors.length === 0) mat.color = new THREE.Color(0.7,0.7,0.7);
    const pts = new THREE.Points(geom, mat);
    this.el.setObject3D('cloud', pts);
    // joints as spheres
    d.joints.forEach(j => {{
      const s = document.createElement('a-sphere');
      s.setAttribute('radius', 0.02);
      s.setAttribute('color', '#ff5050');
      s.setAttribute('position', j.p.join(' '));
      this.el.sceneEl.appendChild(s);
    }});
  }}
}});
</script>
<a-scene background="color: #101018">
  <a-entity point-cloud></a-entity>
  <a-grid></a-grid>
  <a-entity line="start: 0 0 0; end: 0.5 0 0; color: #ff4d4d"></a-entity>
  <a-entity line="start: 0 0 0; end: 0 0.5 0; color: #4dff4d"></a-entity>
  <a-entity line="start: 0 0 0; end: 0 0 0.5; color: #4d8bff"></a-entity>
  <a-entity light="type: ambient; intensity: 0.9"></a-entity>
  <a-entity light="type: directional; intensity: 0.6" position="1 2 1"></a-entity>
  <a-entity camera="fov: 55" position="0 0.8 3"
            orbit-controls="target: 0 0.4 0; initialPosition: 0 0.8 3; minDistance: 0.3; maxDistance: 50">
  </a-entity>
</a-scene>
</body>
</html>
"""


def write_aggregate(out_dir, entries: list, title="Folder reconstructions"):
    """entries: list of {label, href}. Writes a selector index.html linking each run."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = "".join(
        f'<li><a href="{e["href"]}">{e["label"]}</a> '
        f'<span style="color:#888">{e.get("status","")}</span></li>'
        for e in entries
    )
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;background:#14141c;color:#eee}}
a{{color:#9cf}} li{{margin:6px 0}}</style></head>
<body><h2>{title}</h2><ul>{items}</ul></body></html>"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    return out_dir / "index.html"
