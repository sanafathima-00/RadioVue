"""Package the code needed by the Colab notebooks into a zip, with no data, weights or images.

Usage:  python tools/make_colab_bundle.py [--out dist/radiovue_colab_bundle.zip]
Upload the zip to Google Drive at ``MyDrive/RadioVue/code/radiovue_colab_bundle.zip``.

An allow-list (not a deny-list) decides what goes in, so clinical data, model
weights, notebooks containing patient information, and local archives cannot
leak into the bundle by accident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRS = ["radiovue_pipeline", "xray_standardization", "Parallax", "Volumea", "notebooks", "tests", "validation_data/lidc_split"]
INCLUDE_FILES = ["requirements-standardization.txt", "requirements-pipeline.txt", "validation_data/public_cc0/PROVENANCE.md"]
CODE_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".txt", ".md", ".cpp", ".cu", ".h", ".ipynb", ".cfg", ".toml", ".sh", ".pth"}
EXCLUDE_PARTS = {"models", "X-ray data", "X-ray data_sample", "__pycache__", "outputs", "logs", "checkpoints", "dist", "build", "assets", "demo"}
MAX_FILE_BYTES = 20 * 1024 * 1024  # keeps the small LPIPS/VGG .pth files that Volumea imports; excludes weights


def wanted(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDE_PARTS for part in rel.parts) or path.suffix.lower() not in CODE_SUFFIXES:
        return False
    if path.suffix == ".ipynb" and rel.parts[0] != "notebooks":
        return False  # only the generated, PHI-free notebooks
    return path.stat().st_size <= MAX_FILE_BYTES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "dist" / "radiovue_colab_bundle.zip"))
    args = parser.parse_args()
    files = [p for d in INCLUDE_DIRS for p in sorted((ROOT / d).rglob("*")) if p.is_file() and wanted(p)]
    files += [ROOT / f for f in INCLUDE_FILES if (ROOT / f).is_file()]
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout.strip() != ""
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": head, "working_tree_dirty": dirty,
        "files": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p, str(p.relative_to(ROOT)).replace("\\", "/"))
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest, indent=1))
    print(f"{out}  files={len(files)}  bytes={out.stat().st_size}")


if __name__ == "__main__":
    main()
