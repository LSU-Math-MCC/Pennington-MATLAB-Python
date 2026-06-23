"""Tiny content-addressed cache for expensive backend outputs.

Cache key = sha1(input content hash + backend name + backend version + config hash + tag).
Stored as files under <out>/.cache/. Caching is best-effort: a miss or load error just
re-computes.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Callable


class Cache:
    def __init__(self, root: str | Path, enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def key(self, *parts: str) -> str:
        h = hashlib.sha1("||".join(str(p) for p in parts).encode()).hexdigest()[:20]
        return h

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.pkl"

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        p = self._path(key)
        if not p.exists():
            return None
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def put(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        try:
            with open(self._path(key), "wb") as f:
                pickle.dump(value, f)
        except Exception:
            pass

    def memoize(self, key: str, fn: Callable[[], Any]) -> tuple[Any, bool]:
        """Return (value, cached). cached=True if served from cache."""
        cached = self.get(key)
        if cached is not None:
            return cached, True
        val = fn()
        self.put(key, val)
        return val, False
