import csv
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
TRUNK_KEYS = (
    "Trunk Length",
    "Crotch Height",
    "Chest Circumference",
    "Waist Circumference",
    "Stomach Peak Circumference",
    "Hip Circumference",
)
MEASUREMENT_KEYS = (
    "Collar to Scalp Length",
    *TRUNK_KEYS,
    "Left Arm Length",
    "Left Arm Wrist Girth",
    "Left Arm Forearm Girth",
    "Left Arm Bicep Girth",
    "Right Arm Length",
    "Right Arm Wrist Girth",
    "Right Arm Forearm Girth",
    "Right Arm Bicep Girth",
    "Left Leg Length",
    "Left Leg Ankle Girth",
    "Left Leg Calf Girth",
    "Left Leg Thigh Girth",
    "Right Leg Length",
    "Right Leg Ankle Girth",
    "Right Leg Calf Girth",
    "Right Leg Thigh Girth",
)


def mesh_files():
    files = sorted(ROOT.joinpath("model_files").glob("*.obj"))
    files += sorted(ROOT.joinpath("model_files", "OBJ").glob("*.obj"))
    return [path for path in files if path.stem != "penn-mesh-1"]


def case_name(path):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.relative_to(ROOT).with_suffix("").as_posix())


def units_for(path):
    return "dm" if path.name == "man.obj" else "mm"


def parse_measurements(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    values = {key: None for key in MEASUREMENT_KEYS}
    section = None
    side = None
    for line in text.splitlines():
        if "HEAD MEASUREMENTS" in line:
            section, side = "Head", None
        elif "TRUNK MEASUREMENTS" in line:
            section, side = "Trunk", None
        elif "ARM MEASUREMENTS" in line:
            section, side = "Arm", None
        elif "LEG MEASUREMENTS" in line:
            section, side = "Leg", None
        elif "LEFT ARM" in line:
            section, side = "Arm", "Left"
        elif "RIGHT ARM" in line:
            section, side = "Arm", "Right"
        elif "LEFT LEG" in line:
            section, side = "Leg", "Left"
        elif "RIGHT LEG" in line:
            section, side = "Leg", "Right"

        match = re.search(r"^\s*([A-Za-z ]+):\s+([0-9.]+) cm", line)
        if not match:
            continue
        label, value = match.group(1).strip(), float(match.group(2))
        if section == "Head":
            key = label
        elif section == "Trunk":
            key = label
        elif section in ("Arm", "Leg") and side:
            key = f"{side} {section} {label}"
        else:
            continue
        if key in values:
            values[key] = value
    return values


def parse_height(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Mesh height:\s+([0-9.]+) cm", text)
    return float(match.group(1)) if match else None


def run_case(mesh, out_dir):
    name = case_name(mesh)
    log = out_dir / "logs" / f"{name}.txt"
    image = out_dir / "images" / f"{name}.png"
    log.parent.mkdir(parents=True, exist_ok=True)
    image.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "src.main",
        str(mesh.relative_to(ROOT)),
        "--units",
        units_for(mesh),
        "--diary",
        str(log.relative_to(ROOT)),
        "--save-image",
        str(image.relative_to(ROOT)),
    ]
    completed = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return name, completed.returncode, log, image, completed.stdout


def ratio_issue(values, left_key, right_key, threshold, label):
    left, right = values.get(left_key), values.get(right_key)
    if not left or not right:
        return None
    ratio = max(left, right) / min(left, right)
    return f"{label}_asym_{ratio:.2f}x" if ratio > threshold else None


def flag_issues(row, values, image):
    issues = []
    if row["returncode"] != 0:
        issues.append("run_failed")
    if not image.exists():
        issues.append("missing_image")
    missing = [key for key, value in values.items() if value is None]
    if missing:
        issues.append(f"missing_{len(missing)}_measurements")
    zeroish = [key for key, value in values.items() if value is not None and value <= 1.0]
    if zeroish:
        issues.append(f"zeroish_{len(zeroish)}_measurements")
    for maybe in (
        ratio_issue(values, "Left Arm Length", "Right Arm Length", 1.30, "arm_length"),
        ratio_issue(values, "Left Arm Wrist Girth", "Right Arm Wrist Girth", 1.30, "wrist"),
        ratio_issue(values, "Left Arm Forearm Girth", "Right Arm Forearm Girth", 1.30, "forearm"),
        ratio_issue(values, "Left Arm Bicep Girth", "Right Arm Bicep Girth", 1.30, "bicep"),
        ratio_issue(values, "Left Leg Length", "Right Leg Length", 1.20, "leg_length"),
    ):
        if maybe:
            issues.append(maybe)
    hip = values.get("Hip Circumference")
    stomach = values.get("Stomach Peak Circumference")
    if hip and stomach and stomach / hip > 1.25:
        issues.append(f"stomach_hip_{stomach / hip:.2f}x")
    return ";".join(issues)


def run_benchmark(out_dir="output/bench/testset"):
    out_dir = ROOT / out_dir
    rows = []
    for mesh in mesh_files():
        name, code, log, image, stdout = run_case(mesh, out_dir)
        values = parse_measurements(log) if log.exists() else {key: None for key in MEASUREMENT_KEYS}
        row = {
            "case": name,
            "returncode": code,
            "height_cm": parse_height(log) if log.exists() else None,
            "log": str(log.relative_to(ROOT)),
            "image": str(image.relative_to(ROOT)),
        }
        row.update(values)
        row["issues"] = flag_issues(row, values, image)
        rows.append(row)
        status = "FAIL" if code else ("CHECK" if row["issues"] else "OK")
        print(f"{status} {name}" + (f" [{row['issues']}]" if row["issues"] else ""))
        if code != 0:
            print(stdout[-2000:])
    write_summary(out_dir, rows)
    write_contact_sheet(out_dir, rows)
    return rows


def write_summary(out_dir, rows):
    path = out_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        fields = ("case", "returncode", "height_cm", "log", "image", *MEASUREMENT_KEYS, "issues")
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path.relative_to(ROOT)}")


def write_contact_sheet(out_dir, rows, thumb_width=520):
    images = []
    for row in rows:
        path = ROOT / row["image"]
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        scale = thumb_width / image.width
        thumb = image.resize((thumb_width, int(image.height * scale)))
        label_height = 42
        tile = Image.new("RGB", (thumb.width, thumb.height + label_height), "white")
        tile.paste(thumb, (0, label_height))
        draw = ImageDraw.Draw(tile)
        color = "red" if row["issues"] else "black"
        draw.text((6, 4), row["case"], fill=color)
        if row["issues"]:
            draw.text((6, 22), row["issues"][:100], fill=color)
        images.append(tile)
    if not images:
        return
    cols = min(3, len(images))
    rows_count = (len(images) + cols - 1) // cols
    width = cols * max(image.width for image in images)
    height = rows_count * max(image.height for image in images)
    sheet = Image.new("RGB", (width, height), "white")
    for index, image in enumerate(images):
        x = (index % cols) * image.width
        y = (index // cols) * image.height
        sheet.paste(image, (x, y))
    path = out_dir / "contact_sheet.jpg"
    sheet.save(path, quality=92)
    print(f"Wrote {path.relative_to(ROOT)}")
