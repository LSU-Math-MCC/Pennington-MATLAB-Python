"""CLI entry point and pipeline orchestration for single / folder / subject modes."""
from __future__ import annotations

import argparse
import glob
import json
import time
import traceback
from pathlib import Path

import numpy as np

from .config import Config
from . import io as pio
from .cache import Cache
from .parallel import resolve_workers, run_jobs
from .backends import get_backends
from .geometry import camera as camlib
from .geometry.mask_depth_select import select_masked_depth, save_samples
from .geometry.splat_assign import assign_splats_to_instance, resolve_ambiguous
from .geometry import face_mapping as fm
from .geometry.canonicalize import (lift_joints_3d, estimate_canonical_frame,
                                     frame_from_points, canonicalize_splats)
from .geometry.visibility import analyze_visibility
from .geometry.anchor_graph import build_anchor_graph, anchor_tier
from .geometry import fusion as fuse
from .geometry import triangulation as tri
from .geometry.transforms import procrustes, make_T, apply_T
from .geometry.subject_instances import associate_tracks
from .types import SubjectInstance, SplatCloud
from .export import ply as plyexp
from .export import aframe, debug_viz, glb
from .export import render3d
from .export import relight


# ---------------------------------------------------------------- per image ---

def process_image(image_path, out_dir, config: Config, backends, cache: Cache | None = None):
    """Run the single-image DAG. Returns a result dict with per-instance canonical
    clouds, joints, and a status. Raises on hard failure (caller records it)."""
    out_dir = Path(out_dir)
    debug = out_dir / "debug"
    debug.mkdir(parents=True, exist_ok=True)
    timings = {}
    t_all = time.time()

    image = pio.load_image(image_path, max_edge=config.max_image_edge if config.quick else 0)
    H, W = image.shape[:2]
    debug_viz.save(debug / "input.png", image)

    chash = pio.content_hash(image_path)

    def stage(name, fn):
        t = time.time()
        cached = False
        if cache is not None:
            key = cache.key(chash, name, config.hash(),
                            getattr(backends, "versions", {}).get(name.split("_")[0], "?"))
            val = cache.get(key)
            if val is not None:
                cached = True
            else:
                val = fn()
                cache.put(key, val)
        else:
            val = fn()
        timings[name] = {"seconds": round(time.time() - t, 4), "cached": cached}
        return val

    # --- scene + depth + people ---
    splats, cam = stage("gs", lambda: backends.gs.reconstruct(image, out_dir))
    if cam is None:
        cam = camlib.default_camera(W, H)
    seg_results = stage("seg", lambda: backends.segment.segment_people(image, out_dir))
    depthres = stage("depth", lambda: backends.depth.estimate_or_render_depth(image, splats, cam, out_dir))
    depth = depthres.depth
    debug_viz.save(debug / "depth_preview.png", debug_viz.depth_preview(depth))

    if not seg_results:
        raise RuntimeError("No person detected")

    # combined person mask debug
    union = np.zeros((H, W), bool)
    for s in seg_results:
        union |= np.asarray(s.person_mask, bool)
    debug_viz.save(debug / "mask.png", debug_viz.overlay_mask(image, union))

    # --- per instance: build SubjectInstance + assignment scores ---
    instances = []
    score_arrays = []
    base_name = Path(image_path).stem
    for i, seg in enumerate(seg_results):
        mask = np.asarray(seg.person_mask, bool)
        bbox = seg.bbox or _mask_bbox(mask)
        pose = backends.pose.estimate_pose(image, out_dir, bbox=bbox)
        face = backends.face.estimate_face(image, out_dir, bbox=bbox)
        vis = analyze_visibility(mask, bbox, pose, face, image.shape)

        samples = select_masked_depth(
            mask, depth, cam, confidence=depthres.confidence,
            depth_min=config.depth_min, depth_max=config.depth_max,
            conf_thresh=config.depth_conf_thresh,
            subsample=20000 if config.quick else None,
        )
        med = float(np.median(samples.depths)) if len(samples) else None
        idx, scores = assign_splats_to_instance(
            splats, cam, mask, depth, samples,
            depth_tau=config.depth_tau, tau_3d=config.tau_3d,
            person_threshold=config.person_threshold, median_person_depth=med,
        )
        inst = SubjectInstance(
            instance_id=f"{base_name}_p{i}", image_path=str(image_path), mask=mask,
            bbox=tuple(bbox), pose_2d=pose, face_2d=face, visibility=vis,
            selected_samples=samples, splat_indices=idx, splat_scores=scores,
            association_confidence=vis.quality_score,
        )
        instances.append(inst)
        score_arrays.append(scores)

    # --- resolve ambiguous splats across instances ---
    assignment, ambiguous = resolve_ambiguous(score_arrays, margin=config.ambiguous_margin)

    # --- per instance: canonicalize ---
    inst_outputs = []
    for i, inst in enumerate(instances):
        # owned = above-threshold person splats for this instance that won the
        # cross-instance assignment and are not ambiguous between people.
        thresholded = set(int(j) for j in (inst.splat_indices
                          if inst.splat_indices is not None else np.zeros(0, int)).tolist())
        won = np.nonzero((assignment == i) & (~ambiguous))[0]
        owned = np.array([j for j in won if int(j) in thresholded], dtype=int)
        if owned.size == 0 and thresholded:
            owned = np.array(sorted(thresholded), dtype=int)
        inst_splats = _subset(splats, owned)

        joints = lift_joints_3d(inst.pose_2d, depth, inst.mask, cam)
        ct = estimate_canonical_frame(joints)
        if ct.confidence < 0.3 and len(inst.selected_samples) > 5:
            ct = frame_from_points(inst.selected_samples.points_world)

        # face region / anchors / frame
        face_region = fm.build_face_region(inst.face_2d, inst.mask, config.face_margin_px)
        face_samples = fm.select_face_depth(face_region, depth, cam,
                                            confidence=depthres.confidence,
                                            depth_min=config.depth_min, depth_max=config.depth_max)
        face_anchors = fm.lift_face_anchors(inst.face_2d.landmarks, face_region, depth, cam)
        face_T, face_scale, face_conf = fm.face_canonical_frame(face_anchors)
        face_idx, face_scores = fm.assign_face_splats(
            splats, cam, face_region, depth, face_samples, owned,
            tau_face_depth=config.tau_face_depth, tau_face_3d=config.tau_face_3d,
            median_face_depth=float(np.median(face_samples.depths)) if len(face_samples) else None,
        )

        inst.canonical_transform = ct
        graph = build_anchor_graph(joints, face_anchors, source_image=str(image_path))
        tier = anchor_tier(graph)

        # canonicalize the owned splats
        can = canonicalize_splats(inst_splats, ct)
        # tag region: face splats among owned
        owned_set = {int(o): k for k, o in enumerate(owned)}
        region = np.zeros(len(can), np.int64)
        conf = np.full(len(can), float(inst.visibility.quality_score) + 0.2)
        for fi in face_idx:
            if int(fi) in owned_set:
                region[owned_set[int(fi)]] = 1
                conf[owned_set[int(fi)]] = max(conf[owned_set[int(fi)]], 0.9)
        can.extras["region"] = region
        can.extras["confidence"] = np.clip(conf, 0, 1)

        # save per-instance artifacts
        idir = out_dir / "instances" / inst.instance_id
        idir.mkdir(parents=True, exist_ok=True)
        save_samples(idir / "selected_depth_samples.npz", inst.selected_samples)
        plyexp.save_point_ply(idir / "selected_pointcloud.ply",
                              inst.selected_samples.points_world)
        plyexp.save_splat_ply(idir / "person_splats.ply", inst_splats)
        np.save(idir / "person_splat_indices.npy", owned)
        plyexp.save_splat_ply(idir / "canonical_splats.ply", can)
        _write_joints_json(idir / "canonical_joints.json", joints, ct)
        _write_confidence_json(idir / "confidence.json", inst, ct, face_conf, tier)

        # debug overlays for this instance
        if i == 0:
            _instance_debug(debug, image, inst, depth, face_region, splats, cam, owned)

        inst_outputs.append({
            "instance_id": inst.instance_id, "canonical": can, "joints": joints,
            "transform": ct, "tier": tier, "face_conf": face_conf,
            "n_owned": int(owned.size), "ambiguous": int(ambiguous.sum()),
            "visible_regions": inst.visibility.visible_regions,
            "quality": inst.visibility.quality_score,
        })

    # --- per-image canonical export: combine instances side by side ---
    combined = _concat_clouds([o["canonical"] for o in inst_outputs])
    plyexp.save_splat_ply(out_dir / "canonical_splats.ply", combined)
    joints_for_view = inst_outputs[0]["joints"] if inst_outputs else {}
    can_joints_view = _joints_canonical(joints_for_view, inst_outputs[0]["transform"]) if inst_outputs else {}
    aframe.write_viewer(out_dir, splats=combined, joints=can_joints_view,
                        label=base_name, title=f"Reconstruction: {base_name}")
    # 3D plot + orthographic A-pose views of the canonical cloud
    try:
        render3d.render_canonical(debug, combined, joints=can_joints_view, label=base_name)
        import shutil
        if (debug / "canonical_3d.png").exists():
            shutil.copy(debug / "canonical_3d.png", debug / "canonical_preview.png")
    except Exception as e:  # noqa: BLE001
        debug_viz.save(debug / "canonical_preview.png", debug_viz.depth_preview(depth))
        timings.setdefault("render_error", {})["msg"] = str(e)

    # hi-fi relightable bundle for this image's canonical A-pose splats
    hifi = _write_hifi(out_dir, combined, label=base_name,
                       poisson_depth=9 if config.quick else 11)

    timings["total"] = {"seconds": round(time.time() - t_all, 4), "cached": False}
    return {
        "status": "success",
        "image": str(image_path),
        "n_instances": len(instances),
        "instances": inst_outputs,
        "timings": timings,
        "combined": combined,
        "out_dir": str(out_dir),
    }


# -------------------------------------------------------------- small utils ---

def _write_hifi(out_dir, splats, label, poisson_depth=10):
    """Emit the hi-fidelity relightable bundle: full 3DGS PLY + Poisson GLB mesh
    (with normals) + dynamic relight GUI. Each piece is best-effort."""
    out_dir = Path(out_dir)
    info = {}
    try:
        relight.save_gaussian_ply_full(out_dir / "apose_splats.ply", splats)
        info["splats_ply"] = True
    except Exception as e:  # noqa: BLE001
        info["splats_ply_error"] = str(e)
    try:
        mesh = relight.build_relight_mesh(splats, poisson_depth=poisson_depth)
        if mesh is not None and relight.save_glb(mesh, out_dir / "apose_mesh.glb"):
            info["mesh_glb"] = True
            info["mesh_vertices"] = int(len(mesh.vertices))
    except Exception as e:  # noqa: BLE001
        info["mesh_error"] = str(e)
    try:
        relight.write_relight_viewer(out_dir, label=label)
        info["relight_html"] = True
    except Exception as e:  # noqa: BLE001
        info["relight_error"] = str(e)
    return info


def _warmup_if_real(backends):
    """Load heavy model singletons before parallel jobs to avoid first-use races."""
    try:
        from .backends import real
        if isinstance(backends.gs, real.RealGS):
            real.warmup_models()
    except Exception:
        pass


def _mask_bbox(mask):
    vs, us = np.nonzero(mask)
    if us.size == 0:
        return (0, 0, mask.shape[1] - 1, mask.shape[0] - 1)
    return (int(us.min()), int(vs.min()), int(us.max()), int(vs.max()))


def _subset(splats: SplatCloud, idx) -> SplatCloud:
    idx = np.asarray(idx, int)
    return SplatCloud(
        centers=splats.centers[idx], scales=splats.scales[idx],
        rotations=splats.rotations[idx], opacities=splats.opacities[idx],
        colors=splats.colors[idx], extras={},
    )


def _concat_clouds(clouds):
    clouds = [c for c in clouds if len(c) > 0]
    if not clouds:
        return SplatCloud(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 4)),
                          np.zeros(0), np.zeros((0, 3)))
    return SplatCloud(
        centers=np.concatenate([c.centers for c in clouds]),
        scales=np.concatenate([c.scales for c in clouds]),
        rotations=np.concatenate([c.rotations for c in clouds]),
        opacities=np.concatenate([c.opacities for c in clouds]),
        colors=np.concatenate([c.colors for c in clouds]),
        extras={
            "region": np.concatenate([c.extras.get("region", np.zeros(len(c), int)) for c in clouds]),
            "confidence": np.concatenate([c.extras.get("confidence", np.ones(len(c))) for c in clouds]),
        },
    )


def _joints_canonical(joints, ct):
    out = {}
    T = ct.world_to_canonical
    for name, v in joints.items():
        out[name] = apply_T(T, np.array(v[:3]))[0].tolist()
    return out


def _write_joints_json(path, joints, ct):
    data = {"world": {k: list(map(float, v)) for k, v in joints.items()},
            "canonical": _joints_canonical(joints, ct),
            "scale": ct.scale, "anchor": ct.anchor_used, "confidence": ct.confidence}
    Path(path).write_text(json.dumps(data, indent=2))


def _write_confidence_json(path, inst, ct, face_conf, tier):
    data = {
        "instance_id": inst.instance_id,
        "visible_regions": {k: float(v) for k, v in inst.visibility.visible_regions.items()},
        "occlusion": {k: bool(v) for k, v in inst.visibility.occlusion_flags.items()},
        "crop": {k: bool(v) for k, v in inst.visibility.crop_flags.items()},
        "quality": float(inst.visibility.quality_score),
        "canonical_anchor": ct.anchor_used, "canonical_confidence": float(ct.confidence),
        "anchor_tier": tier, "face_confidence": float(face_conf),
    }
    Path(path).write_text(json.dumps(data, indent=2))


def _instance_debug(debug, image, inst, depth, face_region, splats, cam, owned):
    debug_viz.save(debug / "pose_overlay.png",
                   debug_viz.draw_pose(image, inst.pose_2d.keypoints, inst.pose_2d.skeleton_edges))
    debug_viz.save(debug / "selected_depth_overlay.png",
                   debug_viz.draw_points(image, inst.selected_samples.pixels[::5]
                                         if len(inst.selected_samples) else np.zeros((0, 2))))
    from .geometry.projection import project_centers
    if owned.size:
        sub = owned if owned.size <= 4000 else owned[np.linspace(0, owned.size - 1, 4000).astype(int)]
        uv, _, _ = project_centers(cam, splats.centers[sub])
        debug_viz.save(debug / "splat_projection_overlay.png",
                       debug_viz.draw_points(image, uv, color=(255, 220, 60), radius=2))
    else:
        debug_viz.save(debug / "splat_projection_overlay.png", image)
    # face debug
    debug_viz.save(debug / "face_region_mask.png",
                   debug_viz.overlay_mask(image, face_region, color=(60, 120, 255)))
    debug_viz.save(debug / "face_landmarks_overlay.png",
                   debug_viz.draw_pose(image, inst.face_2d.landmarks))


# ------------------------------------------------------------------- modes ---

def _list_images_arg(path_or_glob):
    value = str(path_or_glob)
    if any(ch in value for ch in "*?[]"):
        return [Path(p) for p in sorted(glob.glob(value)) if pio.is_image_file(Path(p))]
    path = Path(value)
    if path.is_dir():
        return pio.list_images(path)
    return [path] if pio.is_image_file(path) else []

def run_single(image_path, out_dir, config: Config):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backends = get_backends(config)
    cache = Cache(out_dir / ".cache", enabled=True)
    manifest = _new_manifest("single", [str(image_path)], config, backends)
    try:
        res = process_image(image_path, out_dir, config, backends, cache)
        manifest["status"] = "success"
        manifest["outputs"] = [str(out_dir / "index.html"),
                               str(out_dir / "canonical_splats.ply")]
        manifest["timings"] = res["timings"]
        manifest["instances"] = [_inst_summary(o) for o in res["instances"]]
    except Exception as e:  # noqa: BLE001
        manifest["status"] = "failed"
        manifest["failures"].append({"image": str(image_path), "stage": "process",
                                     "error": str(e), "trace": traceback.format_exc()})
    _write_manifest(out_dir, manifest)
    return manifest


def run_folder(images_dir, out_dir, config: Config):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backends = get_backends(config)
    images = _list_images_arg(images_dir)
    manifest = _new_manifest("folder", [str(p) for p in images], config, backends)
    workers = resolve_workers(config.workers)
    _warmup_if_real(backends)

    def job(img):
        idir = out_dir / "images" / Path(img).stem
        cache = Cache(idir / ".cache", enabled=True)
        return process_image(img, idir, config, backends, cache)

    results = run_jobs(images, job, workers)
    entries = []
    n_ok = n_fail = 0
    for img, res in zip(images, results):
        stem = Path(img).stem
        if isinstance(res, Exception):
            n_fail += 1
            manifest["failures"].append({"image": str(img), "stage": "process",
                                         "error": str(res)})
            entries.append({"label": stem, "href": f"images/{stem}/index.html",
                            "status": "FAILED"})
        else:
            n_ok += 1
            entries.append({"label": stem, "href": f"images/{stem}/index.html",
                            "status": f'{res["n_instances"]} ppl'})
            manifest["timings"][stem] = res["timings"].get("total", {})
    aframe.write_aggregate(out_dir, entries, title=f"Folder: {Path(images_dir).name}")
    manifest["status"] = "success" if n_fail == 0 and n_ok > 0 else (
        "partial" if n_ok > 0 else "failed")
    manifest["outputs"] = [str(out_dir / "index.html")]
    _write_manifest(out_dir, manifest)
    return manifest


def run_subject(subject_dir, out_dir, config: Config, target_hint=None):
    out_dir = Path(out_dir)
    (out_dir / "assets").mkdir(parents=True, exist_ok=True)
    (out_dir / "debug").mkdir(parents=True, exist_ok=True)
    backends = get_backends(config)
    subject_dir = Path(subject_dir)
    images = _list_images_arg(subject_dir)
    manifest = _new_manifest("subject", [str(p) for p in images], config, backends)
    workers = resolve_workers(config.workers)
    _warmup_if_real(backends)

    def job(img):
        idir = out_dir / "views" / img.parent.name / img.stem
        cache = Cache(idir / ".cache", enabled=True)
        return process_image(img, idir, config, backends, cache)

    results = run_jobs(images, job, workers)

    # build per-image instance lists for association
    per_image_instances = []
    res_by_inst = {}
    for img, res in zip(images, results):
        if isinstance(res, Exception):
            manifest["failures"].append({"image": str(img), "error": str(res)})
            continue
        img_insts = []
        for o in res["instances"]:
            si = SubjectInstance(
                instance_id=o["instance_id"], image_path=str(img),
                mask=np.zeros((1, 1), bool), bbox=(0, 0, 1, 1),
                association_confidence=o["quality"],
            )
            img_insts.append(si)
            res_by_inst[o["instance_id"]] = o
        if img_insts:
            per_image_instances.append(img_insts)

    tracks = associate_tracks(per_image_instances, same_subject=True, target_hint=target_hint)
    alignment_report = {"tracks": len(tracks)}

    # target = first track (already sorted by recurrence/size in same_subject mode)
    fused = None
    fusion_report = {}
    if tracks:
        target = tracks[0]
        clouds = []
        align_T = []
        ref_joints = None
        for si in target.instances:
            o = res_by_inst.get(si.instance_id)
            if o is None:
                continue
            cloud = o["canonical"]
            # refine via Procrustes on shared canonical joints to the reference frame
            jc = o["joints"]
            if ref_joints is None:
                ref_joints = _joints_canonical(jc, o["transform"])
                T = np.eye(4)
            else:
                T = _align_to_reference(jc, o["transform"], ref_joints)
            cloud2 = _apply_T_cloud(cloud, T)
            clouds.append(cloud2)
            align_T.append(T.tolist())
        alignment_report["target_subject"] = target.subject_id
        alignment_report["n_views"] = len(clouds)
        alignment_report["transforms"] = align_T

        fused, fusion_report = fuse.fuse_clouds(
            clouds, body_voxel=config.body_voxel, face_voxel=config.face_voxel)
        fused = fuse.remove_outliers(fused, radius=0.2, min_neighbors=1)

    if fused is None or len(fused) == 0:
        fused = SplatCloud(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 4)),
                           np.zeros(0), np.zeros((0, 3)),
                           extras={"region": np.zeros(0, int), "confidence": np.zeros(0)})

    plyexp.save_splat_ply(out_dir / "assets" / "fused_canonical_splats.ply", fused)
    # proxy mesh
    try:
        mesh = tri.proxy_mesh_from_points(fused.centers, fused.colors)
        if mesh is not None:
            tri.save_mesh(mesh, path_glb=out_dir / "assets" / "fused_proxy_mesh.glb",
                          path_ply=out_dir / "assets" / "fused_proxy_mesh.ply")
    except Exception as e:  # noqa: BLE001
        fusion_report["mesh_error"] = str(e)

    (out_dir / "debug" / "fusion_report.json").write_text(json.dumps(fusion_report, indent=2))
    (out_dir / "debug" / "alignment_report.json").write_text(json.dumps(alignment_report, indent=2))
    (out_dir / "assets" / "canonical_joints.json").write_text(
        json.dumps(ref_joints or {}, indent=2))

    aframe.write_viewer(out_dir, splats=fused, joints=ref_joints or {},
                        label=subject_dir.name, title=f"Fused subject: {subject_dir.name}")
    try:
        render3d.render_canonical(out_dir / "debug", fused, joints=ref_joints or {},
                                  label=f"{subject_dir.name} fused")
    except Exception as e:  # noqa: BLE001
        fusion_report["render_error"] = str(e)
    # hi-fi relightable bundle (full 3DGS PLY + Poisson GLB + relight GUI)
    fusion_report["hifi"] = _write_hifi(out_dir / "assets", fused,
                                        label=f"{subject_dir.name} (fused A-pose)")
    relight.write_relight_viewer(out_dir, glb_name="assets/apose_mesh.glb",
                                 splat_name="assets/apose_splats.ply",
                                 label=f"{subject_dir.name} (fused A-pose)")

    manifest["status"] = "success" if len(fused) > 0 else "partial"
    manifest["outputs"] = [str(out_dir / "index.html"),
                           str(out_dir / "assets" / "fused_canonical_splats.ply")]
    manifest["fusion"] = fusion_report
    manifest["alignment"] = alignment_report
    _write_manifest(out_dir, manifest)
    return manifest


def _align_to_reference(joints, ct, ref_joints):
    """Procrustes from this view's canonical joints to reference canonical joints."""
    cur = _joints_canonical(joints, ct)
    common = [k for k in cur if k in ref_joints]
    if len(common) < 3:
        return np.eye(4)
    src = np.array([cur[k] for k in common])
    dst = np.array([ref_joints[k] for k in common])
    try:
        R, t, s = procrustes(src, dst, with_scale=True)
        return make_T(R, t, scale=s)
    except Exception:
        return np.eye(4)


def _apply_T_cloud(cloud: SplatCloud, T):
    if len(cloud) == 0 or np.allclose(T, np.eye(4)):
        return cloud
    centers = apply_T(T, cloud.centers)
    return SplatCloud(centers, cloud.scales, cloud.rotations, cloud.opacities,
                      cloud.colors, extras=dict(cloud.extras))


# ---------------------------------------------------------------- manifest ---

def _new_manifest(mode, inputs, config, backends):
    return {
        "status": "pending", "mode": mode, "inputs": inputs, "outputs": [],
        "timings": {}, "failures": [], "instances": [],
        "backend_versions": getattr(backends, "versions", {}),
        "config": config.to_dict(),
    }


def _inst_summary(o):
    return {"instance_id": o["instance_id"], "n_owned_splats": o["n_owned"],
            "ambiguous_splats": o["ambiguous"], "anchor_tier": o["tier"],
            "face_confidence": o["face_conf"], "quality": o["quality"],
            "visible_regions": {k: float(v) for k, v in o["visible_regions"].items()}}


def _write_manifest(out_dir, manifest):
    (Path(out_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


# -------------------------------------------------------------------- CLI ----

def build_parser():
    p = argparse.ArgumentParser(prog="pipeline.run")
    sub = p.add_subparsers(dest="mode", required=True)

    def common(sp):
        sp.add_argument("--out", required=True)
        sp.add_argument("--backend", default="auto",
                        choices=["auto", "dummy", "real", "lhm"],
                        help="preset of per-stage defaults")
        sp.add_argument("--quick", action="store_true")
        sp.add_argument("--workers", default="auto")
        sp.add_argument("--depth-tau", type=float, default=None)
        sp.add_argument("--max-edge", type=int, default=None)
        # per-stage overrides (modular: pick an impl for any stage)
        sp.add_argument("--gs", default=None,
                        choices=["dummy", "depth-lift", "depth-lift-large",
                                 "lhm", "lhm-500m", "lhm-1b"], help="geometry/scene stage")
        sp.add_argument("--seg", default=None, choices=["dummy", "yolo"])
        sp.add_argument("--pose", default=None, choices=["dummy", "mediapipe", "yolo"])
        sp.add_argument("--face", default=None, choices=["dummy", "facemesh"])
        sp.add_argument("--depth", default=None,
                        choices=["dummy", "depth-anything", "depth-anything-base",
                                 "depth-anything-large"])
        sp.add_argument("--assoc", default=None, choices=["dummy", "color-hist"])

    sp = sub.add_parser("single"); common(sp); sp.add_argument("--image", required=True)
    sp = sub.add_parser("folder"); common(sp); sp.add_argument("--images", required=True)
    sp = sub.add_parser("subject"); common(sp)
    sp.add_argument("--subject", required=True)
    sp.add_argument("--target-hint", default=None)
    return p


def config_from_args(args):
    cfg = Config(backend=args.backend, quick=bool(args.quick), workers=args.workers)
    stage_map = {"gs": "gs", "seg": "seg", "pose": "pose", "face": "face",
                 "depth": "depth", "assoc": "assoc"}
    for argname, stage in stage_map.items():
        val = getattr(args, argname, None)
        if val is not None:
            cfg.stage_impls[stage] = val
    if getattr(args, "depth_tau", None) is not None:
        cfg.depth_tau = args.depth_tau
    if getattr(args, "max_edge", None) is not None:
        cfg.max_image_edge = args.max_edge
    return cfg


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    if args.mode == "single":
        m = run_single(args.image, args.out, cfg)
    elif args.mode == "folder":
        m = run_folder(args.images, args.out, cfg)
    elif args.mode == "subject":
        m = run_subject(args.subject, args.out, cfg, target_hint=args.target_hint)
    else:
        raise SystemExit(2)
    print(json.dumps({"status": m["status"], "mode": m["mode"],
                      "out": args.out, "failures": len(m["failures"])}))
    return 0 if m["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
