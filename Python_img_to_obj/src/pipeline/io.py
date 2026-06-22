"""Image and array I/O helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Register HEIF/AVIF openers if available (some inputs are AVIF with a .jpg name).
try:
    import pillow_avif  # noqa: F401  (registers an AVIF opener with PIL on import)
except Exception:
    pass
try:
    from pi_heif import register_heif_opener, register_avif_opener
    register_heif_opener()
    try:
        register_avif_opener()
    except Exception:
        pass
except Exception:
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except Exception:
        pass

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".avif", ".heic", ""}


def load_image(path: str | Path, max_edge: int = 0) -> np.ndarray:
    """Load an image as HxWx3 uint8 RGB. Optionally downscale the longest edge.

    The s2 fixture in examples has no extension; PIL sniffs the format from bytes.
    """
    img = Image.open(path).convert("RGB")
    if max_edge and max(img.size) > max_edge:
        w, h = img.size
        s = max_edge / float(max(w, h))
        img = img.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def save_image(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = arr
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    Image.fromarray(a).save(path)


def is_image_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() in IMAGE_EXTS:
        # try to actually open ambiguous (extensionless) files
        if path.suffix == "":
            try:
                Image.open(path).verify()
                return True
            except Exception:
                return False
        return True
    return False


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    out = [p for p in sorted(folder.iterdir()) if is_image_file(p)]
    return out


def content_hash(path: str | Path) -> str:
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
