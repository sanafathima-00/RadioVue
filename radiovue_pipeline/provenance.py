"""Provenance and synthetic-image labelling.

Two copies are written for every synthetic image:

* ``SYNTHETIC_<name>.png``: the exact model-ready pixels.  This copy is not
  visually stamped, because a burned-in banner would corrupt downstream model
  input.  It is labelled by filename, by PNG text chunks, and by a JSON
  sidecar.
* ``SYNTHETIC_<name>_display.png``: the same image with a burned-in
  "SYNTHETIC" banner.  Use it for every human-facing view.

Original inputs are copied unchanged and hashed; they are never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin

SYNTHETIC_BANNER = "SYNTHETIC - NOT AN ACQUIRED RADIOGRAPH"
ORIGIN_ORIGINAL = "ORIGINAL"
ORIGIN_SYNTHETIC = "SYNTHETIC"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ImageRecord:
    path: str
    origin: str
    view: str
    sha256: str
    width: int
    height: int
    generator: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    display_path: str | None = None
    created_utc: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preserve_original(source: str | Path, out_dir: Path, view: str) -> ImageRecord:
    """Copy the uploaded file byte-for-byte and record its hash."""
    source = Path(source)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"ORIGINAL_input{source.suffix.lower()}"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite preserved original: {target}")
    shutil.copy2(source, target)
    digest = sha256_file(target)
    if digest != sha256_file(source):
        raise IOError("Preserved original does not match the source bytes.")
    width = height = 0
    try:
        with Image.open(target) as image:
            width, height = image.size
    except Exception:  # DICOM: dimensions come from the rendered copy instead
        pass
    return ImageRecord(str(target), ORIGIN_ORIGINAL, view, digest, width, height)


def stamp_synthetic(image: Image.Image, extra_line: str | None = None) -> Image.Image:
    """Return an RGB copy with a high-contrast burned-in synthetic banner."""
    stamped = image.convert("RGB")
    draw = ImageDraw.Draw(stamped)
    w, h = stamped.size
    band = max(14, h // 18)
    draw.rectangle([0, 0, w, band], fill=(200, 0, 0))
    draw.text((4, 2), SYNTHETIC_BANNER, fill=(255, 255, 255))
    if extra_line:
        draw.rectangle([0, h - band, w, h], fill=(200, 0, 0))
        draw.text((4, h - band + 2), extra_line, fill=(255, 255, 255))
    return stamped


def save_synthetic(
    image: Image.Image,
    out_dir: Path,
    name: str,
    view: str,
    generator: str,
    parameters: dict[str, Any],
) -> ImageRecord:
    """Write model-ready + display copies and a JSON sidecar for a synthetic view."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"SYNTHETIC_{name}"
    model_path = out_dir / f"{stem}.png"
    display_path = out_dir / f"{stem}_display.png"
    if model_path.exists():
        raise FileExistsError(f"Refusing to overwrite synthetic output: {model_path}")

    info = PngImagePlugin.PngInfo()
    info.add_text("RadioVue-Origin", ORIGIN_SYNTHETIC)
    info.add_text("RadioVue-View", view)
    info.add_text("RadioVue-Generator", generator)
    info.add_text("RadioVue-Warning", SYNTHETIC_BANNER)
    image.save(model_path, pnginfo=info)
    stamp_synthetic(image, f"{view} | {generator}").save(display_path, pnginfo=info)

    record = ImageRecord(
        path=str(model_path),
        origin=ORIGIN_SYNTHETIC,
        view=view,
        sha256=sha256_file(model_path),
        width=image.size[0],
        height=image.size[1],
        generator=generator,
        parameters=parameters,
        display_path=str(display_path),
    )
    (out_dir / f"{stem}.provenance.json").write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    return record


def read_origin(path: str | Path) -> str | None:
    """Return the RadioVue origin tag of a PNG written by this module, if any."""
    with Image.open(path) as image:
        return image.info.get("RadioVue-Origin")
