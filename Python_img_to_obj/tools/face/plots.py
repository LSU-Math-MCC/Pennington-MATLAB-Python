"""Create a side-by-side face mapping comparison sheet."""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size=18):
    for name in ("arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def fit(im: Image.Image, size=(300, 360)):
    out = Image.new("RGB", size, (245, 245, 245))
    im = im.convert("RGB")
    im.thumbnail((size[0] - 16, size[1] - 38), Image.LANCZOS)
    out.paste(im, ((size[0] - im.width) // 2, 34 + (size[1] - 38 - im.height) // 2))
    return out


def main():
    if len(sys.argv) < 4 or len(sys.argv[2:]) % 2 != 0:
        raise SystemExit("usage: make_face_mapping_comparison.py <out.png> <label> <image> ...")
    out_path = Path(sys.argv[1])
    pairs = list(zip(sys.argv[2::2], sys.argv[3::2]))
    tile_w, tile_h = 300, 360
    sheet = Image.new("RGB", (tile_w * len(pairs), tile_h), (20, 22, 24))
    draw = ImageDraw.Draw(sheet)
    for i, (label, path) in enumerate(pairs):
        im = fit(Image.open(path), (tile_w, tile_h))
        x = i * tile_w
        sheet.paste(im, (x, 0))
        draw.rectangle([x, 0, x + tile_w, 32], fill=(20, 22, 24))
        draw.text((x + 10, 7), label, fill=(235, 238, 240), font=font(17))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
