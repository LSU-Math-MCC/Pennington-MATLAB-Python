"""Create a fresh face+body texture demo from local images.

This is a lightweight visual demo for the kept stack: MediaPipe face landmarks
from the default backend are used to warp real face pixels into a canonical face
texture atlas. YOLO segmentation is used when available to isolate the body and
build a front-body texture card. The output includes per-subject texture panels
and an HTML canvas viewer that wraps face and body textures onto shaded surfaces.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.backends.real import RealFace, RealSeg  # noqa: E402

OUT = REPO / "notebooks" / "face_texture_demo"
CANON = 640
BODY_W, BODY_H = 512, 896


CANONICAL = {
    "forehead_center": (0.50, 0.15),
    "left_face_contour": (0.22, 0.52),
    "right_face_contour": (0.78, 0.52),
    "chin": (0.50, 0.88),
    "left_eye_outer": (0.36, 0.36),
    "left_eye": (0.42, 0.35),
    "left_eye_inner": (0.47, 0.36),
    "right_eye_inner": (0.53, 0.36),
    "right_eye": (0.58, 0.35),
    "right_eye_outer": (0.64, 0.36),
    "nose_bridge": (0.50, 0.45),
    "nose_tip": (0.50, 0.55),
    "mouth_left": (0.39, 0.70),
    "upper_lip": (0.50, 0.68),
    "lower_lip": (0.50, 0.75),
    "mouth_right": (0.61, 0.70),
}


def _font(size=18):
    for name in ("arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def load_image(path: Path, max_edge: int = 1100) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    scale = min(1.0, max_edge / max(img.size))
    if scale < 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    return np.asarray(img)


def detect_face(image: np.ndarray):
    face = RealFace().estimate_face(image, OUT)
    return face if face.landmarks else None


def oval_mask(size: int) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    nx = (x - size * 0.5) / (size * 0.40)
    ny = (y - size * 0.53) / (size * 0.49)
    m = (nx * nx + ny * ny) <= 1.0
    mask = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(8))
    return np.asarray(mask, np.float32) / 255.0


def make_texture(image: np.ndarray, face, size: int = CANON):
    needed = ("left_eye", "right_eye", "nose_tip", "mouth_left", "mouth_right")
    if not all(k in face.landmarks for k in needed):
        return None

    def pt(name):
        return np.array(face.landmarks[name][:2], np.float32)

    left_eye = np.mean([pt(k) for k in ("left_eye_outer", "left_eye", "left_eye_inner") if k in face.landmarks], axis=0)
    right_eye = np.mean([pt(k) for k in ("right_eye_inner", "right_eye", "right_eye_outer") if k in face.landmarks], axis=0)
    eye_mid = (left_eye + right_eye) * 0.5
    mouth = (pt("mouth_left") + pt("mouth_right")) * 0.5
    eye_vec = right_eye - left_eye
    angle = float(np.degrees(np.arctan2(eye_vec[1], eye_vec[0])))
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D(tuple(eye_mid), -angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)

    def transform(p):
        return np.array([M[0, 0] * p[0] + M[0, 1] * p[1] + M[0, 2], M[1, 0] * p[0] + M[1, 1] * p[1] + M[1, 2]], np.float32)

    eye_mid_r = transform(eye_mid)
    mouth_r = transform(mouth)
    x0, y0, x1, y1 = face.bbox
    corners = np.array([transform(np.array(p, np.float32)) for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))])
    bx0, by0 = corners.min(axis=0)
    bx1, by1 = corners.max(axis=0)
    face_w = max(float(bx1 - bx0) * 1.72, float(np.linalg.norm(eye_vec)) * 3.4, 120.0)
    face_h = max(face_w * 1.18, float(mouth_r[1] - eye_mid_r[1]) * 2.45, float(by1 - by0) * 1.35)
    cx = float(eye_mid_r[0])
    cy = float(eye_mid_r[1] + face_h * 0.20)
    crop_x0, crop_y0 = int(round(cx - face_w * 0.5)), int(round(cy - face_h * 0.5))
    crop_x1, crop_y1 = int(round(cx + face_w * 0.5)), int(round(cy + face_h * 0.5))
    pad_l, pad_t = max(0, -crop_x0), max(0, -crop_y0)
    pad_r, pad_b = max(0, crop_x1 - w), max(0, crop_y1 - h)
    padded = cv2.copyMakeBorder(rotated, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101)
    crop = padded[crop_y0 + pad_t:crop_y1 + pad_t, crop_x0 + pad_l:crop_x1 + pad_l]
    if crop.size == 0:
        return None

    fitted = Image.fromarray(crop).resize((int(size * 0.80), int(size * 0.97)), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (32, 34, 36))
    paste_xy = ((size - fitted.width) // 2, int(size * 0.035))
    canvas.paste(fitted, paste_xy)
    tex = np.asarray(canvas).astype(np.float32)
    mask = oval_mask(size)
    skin_region = mask > 0.65
    skin = np.median(tex[skin_region], axis=0) if skin_region.any() else np.array([190, 150, 130], np.float32)
    bg = np.zeros_like(tex) + skin
    tex = tex * mask[:, :, None] + bg * (1.0 - mask[:, :, None])
    tex = cv2.bilateralFilter(np.clip(tex, 0, 255).astype(np.uint8), 5, 24, 24).astype(np.float32)

    # Gentle relighting over the face oval: preserves identity pixels but makes the
    # atlas read as a coherent texture instead of a flat crop.
    yy, xx = np.mgrid[0:size, 0:size]
    nx = (xx - size * 0.5) / (size * 0.43)
    ny = (yy - size * 0.52) / (size * 0.54)
    dome = np.clip(1.0 - 0.28 * nx * nx - 0.16 * ny * ny + 0.08 * (-nx - ny), 0.72, 1.12)
    tex = tex * dome[:, :, None]
    tex = tex * mask[:, :, None] + np.array([32, 34, 36], np.float32) * (1.0 - mask[:, :, None])
    return np.clip(tex, 0, 255).astype(np.uint8), mask


def face_selection_quality(face) -> tuple[bool, str]:
    needed = ("left_eye", "right_eye", "nose_tip")
    if not all(k in face.landmarks for k in needed):
        return False, "missing core landmarks"
    x0, y0, x1, y1 = face.bbox
    bw, bh = float(x1 - x0), float(y1 - y0)
    if bw < 90 or bh < 100:
        return False, f"small face bbox {bw:.0f}x{bh:.0f}"
    left_eye = np.array(face.landmarks["left_eye"][:2], np.float32)
    right_eye = np.array(face.landmarks["right_eye"][:2], np.float32)
    nose = np.array(face.landmarks["nose_tip"][:2], np.float32)
    eye_vec = right_eye - left_eye
    angle = float(np.degrees(np.arctan2(eye_vec[1], eye_vec[0])))
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    nose_offset = abs(float(nose[0]) - (x0 + x1) * 0.5) / max(1.0, bw)
    if abs(angle) > 28:
        return False, f"tilted face {angle:.1f}deg"
    if nose_offset > 0.28:
        return False, f"profile-ish nose offset {nose_offset:.2f}"
    return True, f"face {bw:.0f}x{bh:.0f}, angle {angle:.1f}, nose offset {nose_offset:.2f}"


def person_mask(image: np.ndarray):
    try:
        segs = RealSeg().segment_people(image, OUT)
        if segs:
            return np.asarray(segs[0].person_mask, bool), segs[0].bbox
    except Exception as e:
        print("person segmentation fallback:", repr(e)[:120])
    h, w = image.shape[:2]
    return np.ones((h, w), bool), (0, 0, w - 1, h - 1)


def make_body_texture(image: np.ndarray, mask: np.ndarray, bbox, size=(BODY_W, BODY_H)):
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    h, w = image.shape[:2]
    pad_x = int(0.10 * max(1, x1 - x0))
    pad_y = int(0.04 * max(1, y1 - y0))
    x0, x1 = max(0, x0 - pad_x), min(w, x1 + pad_x)
    y0, y1 = max(0, y0 - pad_y), min(h, y1 + pad_y)
    crop = image[y0:y1, x0:x1]
    cmask = mask[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    # Keep aspect honest: paste the person crop into a fixed canonical card
    # instead of stretching width/height independently.
    card_w, card_h = size
    scale = min(card_w / crop.shape[1], card_h / crop.shape[0])
    new_w = max(1, int(crop.shape[1] * scale))
    new_h = max(1, int(crop.shape[0] * scale))
    crop_img = Image.fromarray(crop).resize((new_w, new_h), Image.LANCZOS)
    mask_img = Image.fromarray((cmask * 255).astype(np.uint8)).resize((new_w, new_h), Image.NEAREST)
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(2))

    bg = Image.new("RGB", size, (31, 33, 35))
    alpha = Image.new("L", size, 0)
    x = (card_w - new_w) // 2
    y = (card_h - new_h) // 2
    bg.paste(crop_img, (x, y), mask_img)
    alpha.paste(mask_img, (x, y))

    arr = np.asarray(bg).astype(np.float32)
    yy, xx = np.mgrid[0:card_h, 0:card_w]
    nx = (xx - card_w * 0.5) / (card_w * 0.55)
    ny = (yy - card_h * 0.48) / (card_h * 0.58)
    shade = np.clip(1.05 - 0.14 * nx * nx - 0.08 * ny * ny + 0.05 * (-nx), 0.78, 1.12)
    alpha_arr = np.asarray(alpha).astype(np.float32) / 255.0
    arr = arr * (shade[:, :, None] * alpha_arr[:, :, None] + (1 - alpha_arr[:, :, None]))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), alpha


def body_preview(body_tex: Image.Image, alpha: Image.Image, yaw: float = 0.0, size=(300, 520)) -> Image.Image:
    tex = body_tex.resize(size, Image.LANCZOS)
    a = alpha.resize(size, Image.LANCZOS)
    arr = np.asarray(tex).astype(np.float32)
    aa = np.asarray(a).astype(np.float32) / 255.0
    h, w = size[1], size[0]
    yy, xx = np.mgrid[0:h, 0:w]
    nx = (xx - w * 0.5) / (w * 0.52)
    ny = (yy - h * 0.50) / (h * 0.58)
    z = np.sqrt(np.clip(1.0 - np.minimum(1.0, nx * nx), 0, 1))
    yr = np.radians(yaw)
    shade = np.clip(0.54 + 0.38 * (z * np.cos(yr) - nx * np.sin(yr)) + 0.06 * (-ny), 0.24, 1.10)
    out = np.zeros((h, w, 3), np.float32) + np.array([28, 30, 32])
    out = arr * shade[:, :, None] * aa[:, :, None] + out * (1 - aa[:, :, None])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def landmark_overlay(image: np.ndarray, face, mask=None) -> Image.Image:
    im = Image.fromarray(image).convert("RGB")
    d = ImageDraw.Draw(im)
    if mask is not None:
        overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.bitmap((0, 0), Image.fromarray((mask * 110).astype(np.uint8)), fill=(255, 180, 70, 55))
        im = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
        d = ImageDraw.Draw(im)
    if face.face_mask is not None:
        overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        ys, xs = np.where(face.face_mask)
        if xs.size:
            od.bitmap((0, 0), Image.fromarray((face.face_mask * 130).astype(np.uint8)), fill=(60, 160, 255, 80))
        im = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
        d = ImageDraw.Draw(im)
    for name, (x, y, c) in face.landmarks.items():
        if name not in CANONICAL:
            continue
        col = (60, 255, 120) if c > 0.5 else (255, 150, 60)
        d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=col)
    if face.bbox:
        d.rectangle(face.bbox, outline=(255, 240, 90), width=2)
    return im


def shaded_preview(texture: np.ndarray, yaw: float = 0.0, size: int = 420) -> Image.Image:
    tex = Image.fromarray(texture).resize((size, size), Image.LANCZOS)
    arr = np.asarray(tex).astype(np.float32)
    y, x = np.mgrid[0:size, 0:size]
    nx = (x - size * 0.5) / (size * 0.41)
    ny = (y - size * 0.53) / (size * 0.49)
    face = nx * nx + ny * ny <= 1.0
    z = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0, 1))
    yaw_rad = np.radians(yaw)
    shade = np.clip(0.50 + 0.45 * (z * np.cos(yaw_rad) - nx * np.sin(yaw_rad)) + 0.10 * (-ny), 0.18, 1.12)
    out = np.zeros((size, size, 3), np.float32) + np.array([28, 30, 32])
    out[face] = arr[face] * shade[face, None]
    rim = Image.fromarray((face.astype(np.uint8) * 255)).filter(ImageFilter.GaussianBlur(4))
    rgb = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    alpha = np.asarray(rim, np.float32) / 255.0
    comp = np.asarray(rgb).astype(np.float32) * alpha[:, :, None] + np.array([28, 30, 32]) * (1 - alpha[:, :, None])
    return Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8))


def make_panel(label: str, image: np.ndarray, face, texture: np.ndarray, body_tex: Image.Image, body_alpha: Image.Image, mask) -> Image.Image:
    src = landmark_overlay(image, face, mask)
    src.thumbnail((360, 420))
    tex = Image.fromarray(texture).resize((320, 320), Image.LANCZOS)
    face_prevs = [shaded_preview(texture, yaw, 220) for yaw in (-24, 24)]
    body_card = body_tex.resize((230, 402), Image.LANCZOS)
    body_prevs = [body_preview(body_tex, body_alpha, yaw, (220, 402)) for yaw in (-18, 18)]
    panel = Image.new("RGB", (360 + 320 + 2 * 220 + 230 + 2 * 220 + 8 * 12, 470), (22, 24, 26))
    d = ImageDraw.Draw(panel)
    d.text((12, 10), label, font=_font(22), fill=(245, 245, 235))
    x = 14
    panel.paste(src, (x, 44))
    d.text((x, 430), "input + body mask + face anchors", font=_font(15), fill=(190, 210, 225))
    x += 360 + 12
    panel.paste(tex, (x, 74))
    d.text((x, 430), "face texture atlas", font=_font(15), fill=(190, 210, 225))
    x += 320 + 12
    for yaw, prev in zip((-24, 24), face_prevs):
        panel.paste(prev, (x, 74))
        d.text((x + 38, 430), f"face wrap {yaw:+d}", font=_font(15), fill=(190, 210, 225))
        x += 220 + 12
    panel.paste(body_card, (x, 44))
    d.text((x + 22, 430), "body texture card", font=_font(15), fill=(190, 210, 225))
    x += 230 + 12
    for yaw, prev in zip((-18, 18), body_prevs):
        panel.paste(prev, (x, 44))
        d.text((x + 36, 430), f"body wrap {yaw:+d}", font=_font(15), fill=(190, 210, 225))
        x += 220 + 12
    return panel


def image_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def write_html(items):
    options = "\n".join(f'<button data-idx="{i}">{it["label"]}</button>' for i, it in enumerate(items))
    face_textures = json.dumps([image_data_uri(Path(it["texture"])) for it in items])
    body_textures = json.dumps([image_data_uri(Path(it["body_texture"])) for it in items])
    body_alphas = json.dumps([image_data_uri(Path(it["body_alpha"])) for it in items])
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>MeshMap Face + Body Texture Demo</title>
<style>
body {{ margin:0; background:#16181a; color:#f3efe7; font-family:Segoe UI, Arial, sans-serif; }}
main {{ display:grid; grid-template-columns:330px 1fr; min-height:100vh; }}
aside {{ padding:24px; background:#202326; border-right:1px solid #353a3f; }}
h1 {{ font-size:26px; margin:0 0 8px; }}
p {{ color:#b9c2c9; line-height:1.45; }}
button {{ display:block; width:100%; margin:8px 0; padding:10px 12px; background:#30363d; color:#fff; border:1px solid #4b535c; border-radius:6px; text-align:left; cursor:pointer; }}
button.active {{ background:#546a7b; border-color:#8fb6cf; }}
.stage {{ display:flex; align-items:center; justify-content:center; position:relative; overflow:hidden; }}
canvas {{ width:min(82vw,1000px); height:min(82vh,900px); image-rendering:auto; }}
.strip {{ position:absolute; left:28px; right:28px; bottom:24px; color:#cad6df; font-size:13px; display:flex; gap:14px; align-items:center; }}
.pill {{ padding:6px 9px; background:#252a2e; border:1px solid #46515a; border-radius:999px; }}
</style>
</head>
<body>
<main>
<aside>
<h1>Face + Body Texture Wrap</h1>
<p>Fresh YOLO/MediaPipe signals extract body and face pixels, canonicalize them, then the browser wraps both over shaded rotating surfaces.</p>
{options}
</aside>
<section class="stage">
<canvas id="c" width="900" height="900"></canvas>
<div class="strip"><span class="pill">real pixels</span><span class="pill">face atlas</span><span class="pill">body card</span><span class="pill">live wrap</span><span class="pill">no removed-tool cache</span></div>
</section>
</main>
<script>
const faceTextures = {face_textures};
const bodyTextures = {body_textures};
const bodyAlphas = {body_alphas};
let idx = 0;
const c = document.getElementById('c');
const ctx = c.getContext('2d');
const faceImg = new Image();
const bodyImg = new Image();
const alphaImg = new Image();
const faceOff = document.createElement('canvas');
faceOff.width = faceOff.height = 640;
const faceCtx = faceOff.getContext('2d', {{willReadFrequently:true}});
const bodyOff = document.createElement('canvas');
bodyOff.width = 512; bodyOff.height = 896;
const bodyCtx = bodyOff.getContext('2d', {{willReadFrequently:true}});
const alphaOff = document.createElement('canvas');
alphaOff.width = 512; alphaOff.height = 896;
const alphaCtx = alphaOff.getContext('2d', {{willReadFrequently:true}});
let facePixels = null, bodyPixels = null, alphaPixels = null, pendingLoads = 0;
function load(i) {{
  idx = i;
  facePixels = bodyPixels = alphaPixels = null;
  pendingLoads = 3;
  document.querySelectorAll('button').forEach((b,j)=>b.classList.toggle('active', j===idx));
  faceImg.onload = () => {{ faceCtx.drawImage(faceImg,0,0,640,640); facePixels = faceCtx.getImageData(0,0,640,640).data; pendingLoads--; }};
  bodyImg.onload = () => {{ bodyCtx.drawImage(bodyImg,0,0,512,896); bodyPixels = bodyCtx.getImageData(0,0,512,896).data; pendingLoads--; }};
  alphaImg.onload = () => {{ alphaCtx.drawImage(alphaImg,0,0,512,896); alphaPixels = alphaCtx.getImageData(0,0,512,896).data; pendingLoads--; }};
  faceImg.src = faceTextures[idx];
  bodyImg.src = bodyTextures[idx];
  alphaImg.src = bodyAlphas[idx];
}}
document.querySelectorAll('button').forEach((b,i)=>b.onclick=()=>load(i));
function sampleFace(u,v) {{
  u = Math.max(0, Math.min(639, u|0));
  v = Math.max(0, Math.min(639, v|0));
  const k = (v*640+u)*4;
  return [facePixels[k], facePixels[k+1], facePixels[k+2], 255];
}}
function sampleBody(u,v) {{
  u = Math.max(0, Math.min(511, u|0));
  v = Math.max(0, Math.min(895, v|0));
  const k = (v*512+u)*4;
  return [bodyPixels[k], bodyPixels[k+1], bodyPixels[k+2], alphaPixels[k]];
}}
function draw(t) {{
  requestAnimationFrame(draw);
  if (!facePixels || !bodyPixels || !alphaPixels || pendingLoads > 0) return;
  const W=900,H=900,cx=610,cy=430,rx=192,ry=232;
  const yaw = Math.sin(t*0.001)*0.42;
  ctx.fillStyle = '#16181a'; ctx.fillRect(0,0,W,H);
  const id = ctx.createImageData(W,H);
  const data = id.data;
  for (let y=150; y<700; y+=2) {{
    for (let x=360; x<850; x+=2) {{
      const nx=(x-cx)/rx, ny=(y-cy)/ry;
      const r2=nx*nx+ny*ny;
      if (r2>1) continue;
      const z=Math.sqrt(Math.max(0,1-r2));
      const xr = nx*Math.cos(yaw)+z*Math.sin(yaw);
      const zr = z*Math.cos(yaw)-nx*Math.sin(yaw);
      if (zr<0.03) continue;
      const u=(xr*0.40+0.50)*640;
      const v=(ny*0.49+0.53)*640;
      const col=sampleFace(u,v);
      const shade=Math.max(0.20, Math.min(1.14, 0.48+0.48*zr+0.10*(-ny)-0.06*xr));
      for (let yy=0; yy<2; yy++) for (let xx=0; xx<2; xx++) {{
        const k=((y+yy)*W+(x+xx))*4;
        data[k]=col[0]*shade; data[k+1]=col[1]*shade; data[k+2]=col[2]*shade; data[k+3]=255;
      }}
    }}
  }}
  ctx.putImageData(id,0,0);
  ctx.globalCompositeOperation='screen';
  const g=ctx.createRadialGradient(360,270,20,450,440,430);
  g.addColorStop(0,'rgba(255,255,255,.22)');
  g.addColorStop(1,'rgba(255,255,255,0)');
  ctx.fillStyle=g; ctx.fillRect(0,0,W,H);
  ctx.globalCompositeOperation='source-over';
  // Body surface samples the separate YOLO-derived canonical body texture card.
  ctx.fillStyle='rgba(0,0,0,.30)';
  ctx.beginPath(); ctx.ellipse(285,470,105,245,0,0,Math.PI*2); ctx.fill();
  const yaw2 = Math.sin(t*0.001+1.4)*0.25;
  for (let y=150; y<795; y+=3) {{
    for (let x=95; x<475; x+=3) {{
      const nx=(x-285)/142, ny=(y-470)/320;
      if (nx*nx+ny*ny>1) continue;
      const u=(nx*.48+.5)*512, v=(ny*.50+.5)*896;
      const col=sampleBody(u,v);
      if (col[3] < 20) continue;
      const z=Math.sqrt(Math.max(0,1-nx*nx));
      const shade=Math.max(.20, Math.min(1.08,.48+.40*(z*Math.cos(yaw2)-nx*Math.sin(yaw2))+.05*(-ny)));
      ctx.fillStyle=`rgb(${{col[0]*shade}},${{col[1]*shade}},${{col[2]*shade}})`;
      ctx.fillRect(x,y,3,3);
    }}
  }}
}}
load(0); requestAnimationFrame(draw);
</script>
</body>
</html>"""
    (OUT / "index.html").write_text(html, encoding="utf-8")


def texture_quality(texture: np.ndarray, body_tex: Image.Image, alpha: Image.Image) -> dict:
    mask = oval_mask(texture.shape[0]) > 0.5
    face_px = texture[mask]
    body_arr = np.asarray(body_tex)
    a = np.asarray(alpha) > 20
    body_px = body_arr[a]
    def stats(px):
        if px.size == 0:
            return {"mean_luma": 0, "contrast": 0, "sat": 0}
        lum = 0.2126 * px[:, 0] + 0.7152 * px[:, 1] + 0.0722 * px[:, 2]
        sat = px.max(1) - px.min(1)
        return {
            "mean_luma": round(float(lum.mean()), 2),
            "contrast": round(float(lum.std()), 2),
            "sat": round(float(sat.mean()), 2),
        }
    q = {"face": stats(face_px), "body": stats(body_px), "body_coverage": round(float(a.mean()), 3)}
    return q


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.iterdir():
        if old.is_file() and old.suffix.lower() in {".png", ".json", ".html"}:
            old.unlink()
    preferred = [
        REPO / "images" / "subject" / "s1" / "1.jpg",
        REPO / "images" / "subject" / "s2" / "3.jpg",
        REPO / "images" / "subject" / "s3" / "1.jpg",
        REPO / "images" / "subject" / "s4" / "7.jpg",
        REPO / "images" / "subject" / "s5" / "3.jpg",
        REPO / "images" / "subject" / "s4" / "1.jpg",
        REPO / "images" / "single" / "s3.jpg",
        REPO / "images" / "single" / "s6.jpg",
        REPO / "images" / "single" / "s7.jpg",
        REPO / "images" / "single" / "s12.jpg",
    ]
    discovered = sorted(
        p for p in (REPO / "images").rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    seen = set()
    candidates = []
    for path in preferred + discovered:
        if path.exists() and path not in seen:
            candidates.append(path)
            seen.add(path)
    items = []
    panels = []
    for path in candidates:
        if len(items) >= 6:
            break
        if not path.exists():
            continue
        try:
            image = load_image(path)
        except Exception as e:
            print(f"skip unreadable: {path} ({e})")
            continue
        face = detect_face(image)
        if face is None:
            print(f"skip no face: {path}")
            continue
        keep, why = face_selection_quality(face)
        if not keep:
            print(f"skip weak texture face: {path} ({why})")
            continue
        mask, bbox = person_mask(image)
        made = make_texture(image, face)
        if made is None:
            print(f"skip weak landmarks: {path}")
            continue
        texture, _ = made
        body_made = make_body_texture(image, mask, bbox)
        if body_made is None:
            print(f"skip weak body: {path}")
            continue
        body_tex, body_alpha = body_made
        label = path.parent.name + "/" + path.name
        stem = path.parent.name + "_" + path.stem
        tex_path = OUT / f"{stem}_texture.png"
        body_path = OUT / f"{stem}_body_texture.png"
        body_alpha_path = OUT / f"{stem}_body_alpha.png"
        panel_path = OUT / f"{stem}_panel.png"
        Image.fromarray(texture).save(tex_path)
        body_tex.save(body_path)
        body_alpha.save(body_alpha_path)
        panel = make_panel(label, image, face, texture, body_tex, body_alpha, mask)
        panel.save(panel_path)
        panels.append(panel)
        q = texture_quality(texture, body_tex, body_alpha)
        items.append({
            "label": label, "texture": str(tex_path), "body_texture": str(body_path),
            "body_alpha": str(body_alpha_path), "panel": str(panel_path), "bbox": face.bbox,
            "quality": q,
        })
        print(f"ok {label}: landmarks={len(face.landmarks)} body_coverage={q['body_coverage']} -> {tex_path.name}, {body_path.name}")

    if not items:
        raise SystemExit("No face textures generated")
    gap = 12
    sheet_w = max(p.width for p in panels)
    sheet_h = sum(p.height for p in panels) + gap * (len(panels) - 1)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (18, 20, 22))
    y = 0
    for p in panels:
        sheet.paste(p, (0, y))
        y += p.height + gap
    sheet.save(OUT / "FACE_TEXTURE_CONTACT_SHEET.png")
    write_html(items)
    (OUT / "manifest.json").write_text(json.dumps(items, indent=2), encoding="utf-8")
    print(f"saved {OUT / 'FACE_TEXTURE_CONTACT_SHEET.png'}")
    print(f"saved {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
