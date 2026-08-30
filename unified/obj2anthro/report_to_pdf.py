#!/usr/bin/env python3
"""Print a report page to PDF with headless Chrome or Edge.

The page carries an ``@media print`` block, so the browser's own print path
gives a properly set document -- forced light palette, no shadows, figures and
tables kept off page breaks, table headers repeated. Nothing else on this
machine renders MathML, which the report uses for the girth definitions.

    python -m unified.obj2anthro.report_to_pdf docs/geometry_report.html
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def find_browser() -> str | None:
    for name in ("chrome", "google-chrome", "chromium", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    return next((c for c in CANDIDATES if Path(c).is_file()), None)


def to_pdf(html: Path, pdf: Path, timeout: int = 180) -> Path:
    browser = find_browser()
    if browser is None:
        raise SystemExit(
            "No Chrome or Edge found. Install one, or open the HTML and use the "
            "browser's own Print to PDF -- the page has a print stylesheet.")
    html, pdf = html.resolve(), pdf.resolve()
    pdf.parent.mkdir(parents=True, exist_ok=True)
    # file:// wants forward slashes even on Windows.
    url = "file:///" + str(html).replace("\\", "/").lstrip("/")
    subprocess.run(
        [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
         # Give the webfonts and the inline data-URI figures time to land, and
         # let compositing finish, or the first pages print unstyled.
         "--virtual-time-budget=30000", "--run-all-compositor-stages-before-draw",
         f"--print-to-pdf={pdf}", url],
        check=True, timeout=timeout,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if not pdf.is_file() or pdf.stat().st_size < 100_000:
        raise SystemExit(f"{pdf} came out empty or tiny; the page likely did not load.")
    return pdf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("html", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    pdf = args.out or args.html.with_suffix(".pdf")
    written = to_pdf(args.html, pdf)
    print(f"Wrote {written}  ({written.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
