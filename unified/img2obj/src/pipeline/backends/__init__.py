"""Backend interfaces + factory.

All model-specific code lives behind these interfaces so the geometry/fusion core
(and its tests) can run with dummy/synthetic data. The factory selects dummy or real
implementations based on config.backend ("dummy" | "real" | "auto").
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BackendSet:
    gs: object
    segment: object
    pose: object
    face: object
    depth: object
    association: object
    versions: dict


def get_backends(config) -> BackendSet:
    """Build one backend per stage from the per-stage registry.

    Selection = preset (config.backend: dummy|real|lhm|auto) overlaid with any
    per-stage overrides in config.stage_impls. See backends/registry.py.
    """
    from .registry import build_backend_set
    return build_backend_set(config)
