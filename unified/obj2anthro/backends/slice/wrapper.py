from __future__ import annotations

from ...backend_registry import SliceBackend


def run(obj_file, options):
    return SliceBackend().run(obj_file, options)
