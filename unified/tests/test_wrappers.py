from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from unified.pipeline import ObjHandoff, classify_input, expand_branches
from unified.img2obj import _native_invocations


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


def test_breadcrumbs_exist():
    assert Path("Python_img_to_obj/README.md").exists()
    assert Path("Python_Fall2025/README.md").exists()
    assert Path("Python_ML_2021/README.md").exists()
    assert Path("unified/RELOCATION_MAP.md").exists()
