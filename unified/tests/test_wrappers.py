from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from unified.pipeline import ObjHandoff, classify_input, expand_branches, run_pipeline
from unified.img2obj import _ensure_obj_exports, _find_obj_handoffs, _native_invocations


def run_help(*args):
    return subprocess.run(
        [sys.executable, "-m", *args, "--help"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )


def test_unified_help():
    result = run_help("unified")
    assert result.returncode == 0
    assert "--input" in result.stdout


def test_img2obj_help():
    result = run_help("unified", "img2obj")
    assert result.returncode == 0
    assert "--method" in result.stdout


def test_obj2anthro_help():
    result = run_help("unified", "obj2anthro")
    assert result.returncode == 0
    assert "segmentation" in result.stdout


def test_direct_obj2anthro_help():
    result = run_help("unified.obj2anthro")
    assert result.returncode == 0
    assert "--method" in result.stdout


def test_direct_img2obj_help():
    result = run_help("unified.img2obj")
    assert result.returncode == 0
    assert "--method" in result.stdout


def test_obj_file_allocation_has_no_img_stage(tmp_path):
    obj = tmp_path / "person.obj"
    obj.write_text("# obj\n", encoding="utf-8")

    plan = classify_input(obj)

    assert plan["objs"] == [str(obj)]
    assert plan["images"] == []


def test_image_file_allocation(tmp_path):
    image = tmp_path / "person.png"
    image.write_bytes(b"stub")

    plan = classify_input(image)

    assert plan["images"] == [str(image)]
    assert plan["objs"] == []


def test_subject_grouping_names(tmp_path):
    for name in ["s1", "S22", "subject3", "Subject104"]:
        subject = tmp_path / name
        subject.mkdir()
        (subject / "front.jpg").write_bytes(b"stub")
        (subject / "side.jpg").write_bytes(b"stub")

    plan = classify_input(tmp_path)

    assert sorted(plan["subject_groups"]) == ["S22", "Subject104", "s1", "subject3"]
    assert all(len(paths) == 2 for paths in plan["subject_groups"].values())


def test_nested_subject_group_uses_nearest_subject_directory(tmp_path):
    outer = tmp_path / "subject1"
    inner = outer / "subject2"
    inner.mkdir(parents=True)
    (inner / "front.jpg").write_bytes(b"stub")

    plan = classify_input(tmp_path)

    assert sorted(plan["subject_groups"]) == ["subject1/subject2"]


def test_ordinary_directories_do_not_group(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    (photos / "front.jpg").write_bytes(b"stub")
    (photos / "side.jpg").write_bytes(b"stub")

    plan = classify_input(tmp_path)

    assert sorted(plan["subject_groups"]) == ["photos/front", "photos/side"]


def test_mixed_directory_preserves_relative_paths(tmp_path):
    (tmp_path / "person_a.png").write_bytes(b"stub")
    (tmp_path / "person_b.obj").write_text("# obj\n", encoding="utf-8")
    subject = tmp_path / "subject12"
    subject.mkdir()
    (subject / "front.jpg").write_bytes(b"stub")
    (subject / "side.jpg").write_bytes(b"stub")
    (tmp_path / "notes.txt").write_text("ignore\n", encoding="utf-8")

    plan = classify_input(tmp_path)

    assert str(tmp_path / "person_b.obj") in plan["objs"]
    assert "subject12" in plan["subject_groups"]
    assert sorted(plan["subject_groups"]["subject12"]) == ["subject12/front.jpg", "subject12/side.jpg"]
    assert str(tmp_path / "notes.txt") in plan["unsupported"]


def test_full_factorial_branch_expansion():
    handoffs = [
        ObjHandoff("a", "dummy", Path("a.obj")),
        ObjHandoff("b", "dummy", Path("b.obj")),
    ]

    branches = expand_branches(handoffs, ["segmentation", "slice"])

    assert [(h.subject_id, method) for h, method in branches] == [
        ("a", "segmentation"),
        ("a", "slice"),
        ("b", "segmentation"),
        ("b", "slice"),
    ]


def test_img2obj_mixed_directory_native_invocations(tmp_path):
    (tmp_path / "loose.png").write_bytes(b"stub")
    subject = tmp_path / "s1"
    subject.mkdir()
    (subject / "front.jpg").write_bytes(b"stub")
    (subject / "side.jpg").write_bytes(b"stub")

    invocations = _native_invocations(tmp_path, tmp_path / "out", "dummy", None)

    assert [argv[0] for argv in invocations] == ["subject", "single"]
    assert str(subject) in invocations[0]
    assert str(tmp_path / "loose.png") in invocations[1]


def test_img2obj_nested_subject_invocation_uses_nearest_subject_directory(tmp_path):
    inner = tmp_path / "subject1" / "subject2"
    inner.mkdir(parents=True)
    (inner / "front.jpg").write_bytes(b"stub")

    invocations = _native_invocations(tmp_path, tmp_path / "out", "dummy", None)

    assert len(invocations) == 1
    assert invocations[0][0] == "subject"
    assert str(inner) in invocations[0]


def test_breadcrumbs_exist():
    assert Path("Python_img_to_obj/README.md").exists()
    assert Path("Python_Fall2025/README.md").exists()
    assert Path("Python_ML_2021/README.md").exists()
    assert Path("unified/RELOCATION_MAP.md").exists()


def _success_frame(output_dir: Path, run_id: str = "results") -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    df = pd.DataFrame([{"status": "success", "subject_id": "subject"}])
    df.to_csv(csv_path, index=False)
    df.attrs["output_csv"] = str(csv_path)
    df.attrs["raw_output_dir"] = str(raw_dir)
    df.attrs["run_id"] = run_id
    return df


def test_image_handoff_calls_obj2anthro(monkeypatch, tmp_path):
    import unified.img2obj as img2obj
    import unified.obj2anthro as obj2anthro

    image = tmp_path / "person.png"
    image.write_bytes(b"stub")
    obj = tmp_path / "person.obj"
    obj.write_text("# obj\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(
        img2obj,
        "run",
        lambda input_path, method, out: {
            "status": "success",
            "native_output_dir": str(out),
            "obj_handoffs": [{"subject_id": "person", "method": "fake", "obj_path": str(obj), "source_images": [str(image)]}],
        },
    )

    def fake_obj2anthro(obj_path, backend, units, output_dir, run_id):
        calls.append((obj_path, backend, units, output_dir, run_id))
        return _success_frame(output_dir, run_id)

    monkeypatch.setattr(obj2anthro, "run_pipeline", fake_obj2anthro)

    manifest = run_pipeline(image, anthro_method="slice", out=tmp_path / "run")

    assert manifest["status"] == "success"
    assert len(calls) == 1
    assert calls[0][0] == obj
    assert calls[0][1] == "slice"
    assert calls[0][4] == "results"
    assert manifest["stages"]["obj2anthro"][0]["source_method"] == "fake"


def test_image_auto_camerahmr_uses_slice_anthro(monkeypatch, tmp_path):
    import unified.img2obj as img2obj
    import unified.obj2anthro as obj2anthro

    image = tmp_path / "person.png"
    image.write_bytes(b"stub")
    obj = tmp_path / "person.obj"
    obj.write_text("# obj\n", encoding="utf-8")
    backends = []

    monkeypatch.setattr(
        img2obj,
        "run",
        lambda input_path, method, out: {
            "status": "success",
            "native_output_dir": str(out),
            "obj_handoffs": [{"subject_id": "person", "method": "camerahmr", "obj_path": str(obj), "source_images": [str(image)]}],
        },
    )

    def fake_obj2anthro(obj_path, backend, units, output_dir, run_id):
        backends.append(backend)
        return _success_frame(output_dir, run_id)

    monkeypatch.setattr(obj2anthro, "run_pipeline", fake_obj2anthro)

    manifest = run_pipeline(image, image_method="auto", anthro_method="auto", out=tmp_path / "run")

    assert manifest["status"] == "success"
    assert backends == ["slice"]
    assert manifest["selected_methods"]["resolved_anthropometry"] == ["slice"]


def test_direct_obj_input_skips_img2obj(monkeypatch, tmp_path):
    import unified.img2obj as img2obj
    import unified.obj2anthro as obj2anthro

    obj = tmp_path / "person.obj"
    obj.write_text("# obj\n", encoding="utf-8")
    monkeypatch.setattr(img2obj, "run", lambda *args, **kwargs: pytest.fail("img2obj should not run for direct OBJ"))
    monkeypatch.setattr(obj2anthro, "run_pipeline", lambda obj_path, backend, units, output_dir, run_id: _success_frame(output_dir, run_id))

    manifest = run_pipeline(obj, anthro_method="slice", out=tmp_path / "run")

    assert "img2obj" not in manifest["stages"]
    assert manifest["status"] == "success"
    assert manifest["stages"]["obj2anthro"][0]["source_method"] == "direct"


def test_mixed_directory_runs_direct_and_image_handoffs(monkeypatch, tmp_path):
    import unified.img2obj as img2obj
    import unified.obj2anthro as obj2anthro

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    image = input_dir / "image.png"
    image.write_bytes(b"stub")
    direct = input_dir / "direct.obj"
    direct.write_text("# direct\n", encoding="utf-8")
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    derived = generated_dir / "derived.obj"
    derived.write_text("# derived\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        img2obj,
        "run",
        lambda input_path, method, out: {
            "status": "success",
            "native_output_dir": str(out),
            "obj_handoffs": [{"subject_id": "derived", "method": "fake", "obj_path": str(derived)}],
        },
    )

    def fake_obj2anthro(obj_path, backend, units, output_dir, run_id):
        calls.append(Path(obj_path).name)
        return _success_frame(output_dir, run_id)

    monkeypatch.setattr(obj2anthro, "run_pipeline", fake_obj2anthro)

    manifest = run_pipeline(input_dir, anthro_method="slice", out=tmp_path / "run")

    assert manifest["status"] == "success"
    assert sorted(calls) == ["derived.obj", "direct.obj"]


def test_duplicate_handoffs_use_distinct_branch_dirs(monkeypatch, tmp_path):
    import unified.img2obj as img2obj
    import unified.obj2anthro as obj2anthro

    image = tmp_path / "person.png"
    image.write_bytes(b"stub")
    obj_a = tmp_path / "a.obj"
    obj_b = tmp_path / "b.obj"
    obj_a.write_text("# a\n", encoding="utf-8")
    obj_b.write_text("# b\n", encoding="utf-8")
    output_dirs = []
    monkeypatch.setattr(
        img2obj,
        "run",
        lambda input_path, method, out: {
            "status": "success",
            "native_output_dir": str(out),
            "obj_handoffs": [
                {"subject_id": "same", "method": "fake", "obj_path": str(obj_a)},
                {"subject_id": "same", "method": "fake", "obj_path": str(obj_b)},
            ],
        },
    )

    def fake_obj2anthro(obj_path, backend, units, output_dir, run_id):
        output_dirs.append(output_dir)
        return _success_frame(output_dir, run_id)

    monkeypatch.setattr(obj2anthro, "run_pipeline", fake_obj2anthro)

    run_pipeline(image, anthro_method="slice", out=tmp_path / "run")

    assert len(output_dirs) == 2
    assert output_dirs[0] != output_dirs[1]


def test_image_backend_failure_sets_failed_status(monkeypatch, tmp_path):
    import unified.img2obj as img2obj

    image = tmp_path / "person.png"
    image.write_bytes(b"stub")
    monkeypatch.setattr(
        img2obj,
        "run",
        lambda input_path, method, out: {"status": "failed", "native_output_dir": str(out), "obj_handoffs": [], "errors": ["backend failed"]},
    )

    manifest = run_pipeline(image, out=tmp_path / "run")

    assert manifest["status"] == "failed"
    assert manifest["errors"] == ["backend failed"]


def test_image_success_without_obj_handoff_is_not_success(monkeypatch, tmp_path):
    import unified.img2obj as img2obj

    image = tmp_path / "person.png"
    image.write_bytes(b"stub")
    monkeypatch.setattr(
        img2obj,
        "run",
        lambda input_path, method, out: {"status": "success", "native_output_dir": str(out), "obj_handoffs": []},
    )

    manifest = run_pipeline(image, out=tmp_path / "run")

    assert manifest["status"] == "failed"
    assert "no OBJ handoff" in manifest["warnings"][0]


def test_top_level_cli_nonzero_for_partial(monkeypatch, tmp_path):
    import unified.cli as cli

    monkeypatch.setattr(cli, "run_pipeline", lambda **kwargs: {"run_id": "r", "status": "partial"})

    assert cli.main(["--input", str(tmp_path / "anything.obj")]) == 1


def test_obj2anthro_stage_requires_input_and_defaults_to_auto_units():
    from unified.obj2anthro.cli import parse_args

    with pytest.raises(SystemExit):
        parse_args([])
    args = parse_args(["--input", "person.obj"])
    assert args.units == "auto"
    assert args.output_dir is None


def test_npz_people_export_creates_obj_handoff(tmp_path):
    import numpy as np

    people = np.empty(1, dtype=object)
    people[0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    np.savez(tmp_path / "person.npz", people=people, faces=np.array([[0, 1, 2]], dtype=int))

    exported = _ensure_obj_exports(tmp_path)
    handoffs = _find_obj_handoffs(tmp_path, "camerahmr")

    assert len(exported) == 1
    assert Path(exported[0]).suffix == ".obj"
    assert handoffs[0]["obj_path"] == exported[0]


def test_relocation_docs_have_no_mojibake_or_duplicate_banners():
    docs = [
        Path("README.md"),
        Path("unified/README.md"),
        Path("unified/RELOCATION_MAP.md"),
        Path("Python_img_to_obj/README.md"),
        Path("Python_Fall2025/README.md"),
        Path("Python_ML_2021/README.md"),
        Path("unified/img2obj/README.md"),
        Path("unified/obj2anthro/backends/segmentation/README.md"),
        Path("unified/ml/experiment/README.md"),
    ]
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        assert "\u00e2" not in text
        assert "\u00c2" not in text
        assert "\ufffd" not in text
        assert text.count("Relocation note:") <= 1
