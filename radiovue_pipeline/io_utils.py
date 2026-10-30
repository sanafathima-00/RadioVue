"""Input loading that reuses the validated DICOM rendering of ``xray_standardization``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from xray_standardization.intensity import render_dicom, render_raster
from xray_standardization.loaders import load_input
from xray_standardization.pipeline import _apply_authoritative_orientation
from xray_standardization.validation import _canonical_projection


@dataclass
class LoadedXray:
    image: Image.Image  # 8-bit grayscale display pixels
    kind: str  # "dicom" | "raster"
    metadata: dict[str, Any]  # allow-listed technical metadata only
    metadata_view: str | None  # PA | AP | LATERAL | OBLIQUE | None
    transformations: list[str] = field(default_factory=list)


def load_xray(path: str | Path) -> LoadedXray:
    """Load DICOM/PNG/JPEG as grayscale without contrast enhancement.

    DICOM display semantics follow ``xray_standardization.intensity``.  Raster
    files are treated as already-rendered display pixels.  Only allow-listed
    technical metadata is kept (no patient identifiers).
    """
    loaded = load_input(path)
    if loaded.kind == "dicom":
        image, details = render_dicom(loaded.dataset, loaded.metadata)
        image, operations = _apply_authoritative_orientation(image, loaded.metadata)
        transformations = [f"dicom_render:{details['uint8_mapping']}", *operations]
    else:
        image, details = render_raster(loaded.image)
        transformations = [f"raster:{details['uint8_mapping']}"]
    metadata_view = _canonical_projection(loaded.metadata.get("ViewPosition")) or _canonical_projection(
        loaded.metadata.get("ViewCodeMeaning")
    )
    return LoadedXray(image.convert("L"), loaded.kind, loaded.metadata, metadata_view, transformations)


def to_float01(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def from_float01(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8), mode="L")


def pad_to_square(image: Image.Image, fill: int = 0) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Centre-pad to a square.  Returns the padded image and (left, top, right, bottom) padding.

    Matches the MedGemma 1.5 localization preprocessing (np.pad with zeros,
    extra pixel on the trailing side) without its uint8 overflow.
    """
    w, h = image.size
    side = max(w, h)
    left = (side - w) // 2
    top = (side - h) // 2
    canvas = Image.new(image.mode, (side, side), color=fill)
    canvas.paste(image, (left, top))
    return canvas, (left, top, side - w - left, side - h - top)
