from __future__ import annotations

import os
from pathlib import Path


def camerahmr_root() -> Path:
    """Return the local CameraHMR checkout root."""
    return Path(os.environ.get("CAMERAHMR_ROOT", "~/CameraHMR")).expanduser()
