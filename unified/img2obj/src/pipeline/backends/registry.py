"""Per-stage backend registry: pick and choose an implementation for each stage.

Stages (independently selectable):
    gs     - scene/geometry source (dummy | depth-lift | lhm)
    seg    - person instance segmentation (dummy | yolo)
    pose   - 2D/3D pose markers (dummy | mediapipe | yolo)
    face   - face landmarks (dummy | facemesh)
    depth  - depth source (dummy | depth-anything)
    assoc  - subject association embedding (dummy | color-hist)

A preset (`--backend dummy|real|lhm`) sets defaults for every stage; any stage can
then be overridden individually (`--pose yolo`, `--depth dummy`, ...). Each stage
factory is lazy, and in "auto"/preset mode a stage that fails to construct falls back
to its dummy implementation so the pipeline still runs.
"""
from __future__ import annotations


def _dummy():
    from . import dummy
    return dummy


def _real():
    from . import real
    return real


# stage -> { impl_name -> factory(config) -> backend_instance }
def build_registry():
    d = _dummy

    def real_attr(name):
        def factory(cfg):
            return getattr(_real(), name)()
        return factory

    return {
        "gs": {
            "dummy": lambda cfg: d().DummyGS(),
            "depth-lift": real_attr("RealGS"),
            "depth-lift-large": lambda cfg: _real().RealGS(depth_size="large"),
            "lhm": lambda cfg: _lhm_gs(cfg, "LHM-MINI"),
            "lhm-500m": lambda cfg: _lhm_gs(cfg, "LHM-500M"),
            "lhm-1b": lambda cfg: _lhm_gs(cfg, "LHM-1B"),
        },
        "seg": {
            "dummy": lambda cfg: d().DummySeg(),
            "yolo": real_attr("RealSeg"),
        },
        "pose": {
            "dummy": lambda cfg: d().DummyPose(),
            "mediapipe": real_attr("RealPose"),
            "yolo": lambda cfg: _real().RealPose(),  # RealPose falls back to YOLO internally
        },
        "face": {
            "dummy": lambda cfg: d().DummyFace(),
            "facemesh": real_attr("RealFace"),
        },
        "depth": {
            "dummy": lambda cfg: d().DummyDepth(),
            "depth-anything": lambda cfg: _real().RealDepth(size="small"),
            "depth-anything-base": lambda cfg: _real().RealDepth(size="base"),
            "depth-anything-large": lambda cfg: _real().RealDepth(size="large"),
        },
        "assoc": {
            "dummy": lambda cfg: d().DummyAssoc(),
            "color-hist": real_attr("RealAssoc"),
        },
    }


def _lhm_gs(cfg, model_name="LHM-MINI"):
    from . import lhm_backend
    return lhm_backend.LHMReconstructor(cfg, model_name=model_name)


# preset -> per-stage impl name
PRESETS = {
    "dummy": {"gs": "dummy", "seg": "dummy", "pose": "dummy", "face": "dummy",
              "depth": "dummy", "assoc": "dummy"},
    "real": {"gs": "depth-lift", "seg": "yolo", "pose": "mediapipe", "face": "facemesh",
             "depth": "depth-anything", "assoc": "color-hist"},
    "lhm": {"gs": "lhm", "seg": "yolo", "pose": "mediapipe", "face": "facemesh",
            "depth": "depth-anything", "assoc": "color-hist"},
}
PRESETS["auto"] = PRESETS["real"]

STAGES = ["gs", "seg", "pose", "face", "depth", "assoc"]


def resolve_stage_plan(config) -> dict:
    """Return {stage: impl_name} from preset + per-stage overrides on the config."""
    preset = getattr(config, "backend", "auto")
    plan = dict(PRESETS.get(preset, PRESETS["auto"]))
    overrides = getattr(config, "stage_impls", None) or {}
    for stage, impl in overrides.items():
        if impl:
            plan[stage] = impl
    return plan


def build_backend_set(config):
    """Instantiate one backend per stage according to the resolved plan.

    In dummy mode (or when a chosen real impl can't load), fall back to dummy per
    stage unless the impl was explicitly requested (then raise).
    """
    from . import BackendSet
    reg = build_registry()
    plan = resolve_stage_plan(config)
    overrides = getattr(config, "stage_impls", None) or {}
    strict = getattr(config, "backend", "auto") == "dummy"

    instances = {}
    versions = {}
    for stage in STAGES:
        impl = plan[stage]
        factory = reg[stage].get(impl)
        explicit = stage in overrides
        try:
            if factory is None:
                raise KeyError(f"unknown impl '{impl}' for stage '{stage}'")
            instances[stage] = factory(config)
            versions[stage] = impl
        except Exception as e:  # noqa: BLE001
            if strict or explicit:
                raise
            # fall back to dummy for this stage
            instances[stage] = reg[stage]["dummy"](config)
            versions[stage] = f"dummy(fallback:{impl}:{type(e).__name__})"

    return BackendSet(
        gs=instances["gs"], segment=instances["seg"], pose=instances["pose"],
        face=instances["face"], depth=instances["depth"],
        association=instances["assoc"], versions=versions,
    )
